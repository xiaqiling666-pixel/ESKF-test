from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from .adapters import (
    diagnostic_truth_view,
    generate_demo_dataset,
    load_dataset_from_config,
    observation_view,
    row_to_frame,
)
from .analysis import (
    export_pipeline_results,
)
from .analysis.quality import (
    CovarianceHealthTracker,
    SensorFreshnessTracker,
)
from .analysis.state_machine import ModeStateTracker, ModeThresholds
from .config import load_config, project_path
from .core import ImuInitializationSample, OfflineESKF
from .core.math_utils import ensure_dir, quat_to_euler
from .core.filter import PredictDiagnostics
from .measurements import (
    BarometerMeasurement,
    GnssPositionMeasurement,
    GnssVelocityMeasurement,
    MagYawMeasurement,
    MeasurementManager,
)
from .pipeline.initialization_controller import (
    assess_initialization_status as _assess_initialization_status,
    bootstrap_initialize_from_position_pair as _bootstrap_initialize_from_position_pair,
    initialize_filter as _initialize_filter,
    preinit_sample_from_frame as _preinit_sample_from_frame,
    summarize_initialization_status as _summarize_initialization_status,
)
from .pipeline.initialization_pass import InitializationRuntimeState, run_initialization_pass
from .pipeline.lever_arm_context import evaluate_gnss_lever_arm_diagnostics
from .pipeline.mode_context import evaluate_mode_context
from .pipeline.measurement_pass import run_measurement_pass
from .pipeline.record_builder import (
    build_preinit_record as _build_preinit_record,
    build_runtime_record as _build_runtime_record,
)


def _build_measurements(config) -> list:
    measurement_models = [GnssPositionMeasurement(), GnssVelocityMeasurement()]
    if config.use_baro:
        measurement_models.append(BarometerMeasurement())
    if config.use_mag:
        measurement_models.append(MagYawMeasurement())
    return measurement_models


def _mode_thresholds_for_measurements(filter_engine, measurement_models) -> ModeThresholds:
    gnss_trigger_streaks: list[int] = []
    auxiliary_trigger_streaks: list[int] = []
    for model in measurement_models:
        trigger_streak = max(1, int(model.policy(filter_engine).recovery_trigger_reject_streak))
        if model.name in {"gnss_pos", "gnss_vel"}:
            gnss_trigger_streaks.append(trigger_streak)
        elif model.name in {"baro", "mag"}:
            auxiliary_trigger_streaks.append(trigger_streak)

    gnss_threshold = min(gnss_trigger_streaks, default=3)
    auxiliary_threshold = min(auxiliary_trigger_streaks, default=3)
    return ModeThresholds(
        gnss_reject_streak_degraded=gnss_threshold,
        gnss_reject_bypass_streak_degraded=max(2, gnss_threshold - 1),
        gnss_adaptive_streak_degraded=gnss_threshold,
        gnss_skip_streak_degraded=gnss_threshold,
        auxiliary_reject_streak_degraded=auxiliary_threshold,
        auxiliary_reject_bypass_streak_degraded=max(2, auxiliary_threshold),
        auxiliary_adaptive_streak_degraded=auxiliary_threshold,
        auxiliary_skip_streak_degraded=auxiliary_threshold,
    )


def run_pipeline(config_path: str | None = None) -> pd.DataFrame:
    pipeline_start = time.perf_counter()
    config = load_config(config_path)
    dataset_path = project_path(config.dataset_path)
    results_root = ensure_dir(project_path(config.results_dir))

    if not dataset_path.exists():
        generate_demo_dataset(dataset_path)

    dataset_load_result = load_dataset_from_config(config)
    sensor_df = dataset_load_result.dataframe
    if dataset_load_result.navigation_reference_override is not None:
        config.navigation_environment.reference_lat_deg = dataset_load_result.navigation_reference_override.reference_lat_deg
        config.navigation_environment.reference_lon_deg = dataset_load_result.navigation_reference_override.reference_lon_deg
        config.navigation_environment.reference_height_m = dataset_load_result.navigation_reference_override.reference_height_m
    measurement_models = _build_measurements(config)
    filter_engine = OfflineESKF(config)
    measurement_manager = MeasurementManager()
    freshness_tracker = SensorFreshnessTracker()
    covariance_tracker = CovarianceHealthTracker()
    mode_tracker = ModeStateTracker()
    mode_thresholds = _mode_thresholds_for_measurements(filter_engine, measurement_models)
    initialization_runtime = InitializationRuntimeState()
    last_time: float | None = None
    records: list[dict[str, Any]] = []

    for _, row in sensor_df.iterrows():
        frame = row_to_frame(row)
        measurement_frame = observation_view(frame)
        truth_frame = diagnostic_truth_view(frame)
        initialization_pass = run_initialization_pass(
            filter_engine=filter_engine,
            raw_frame=frame,
            measurement_frame=measurement_frame,
            truth_frame=truth_frame,
            runtime_state=initialization_runtime,
        )
        initialization_runtime = initialization_pass.runtime_state
        initialized_this_frame = initialization_pass.initialized_this_frame
        if initialization_pass.preinit_record is not None:
            records.append(initialization_pass.preinit_record)
            continue
        if initialized_this_frame:
            last_time = frame.time

        if initialized_this_frame:
            predict_diag = PredictDiagnostics(
                raw_dt=0.0,
                applied_dt=0.0,
                skipped=True,
                warning=False,
                reason="initialization_completed_this_frame",
            )
            filter_engine.last_predict_diagnostics = predict_diag
        else:
            dt = 0.0 if last_time is None else max(0.0, frame.time - last_time)
            filter_engine.predict(frame.accel, frame.gyro, dt)
            predict_diag = filter_engine.last_predict_diagnostics

        current_mode = None if mode_tracker.current_decision is None else mode_tracker.current_decision.mode
        measurement_pass = run_measurement_pass(
            filter_engine=filter_engine,
            measurement_manager=measurement_manager,
            freshness_tracker=freshness_tracker,
            measurement_models=measurement_models,
            measurement_frame=measurement_frame,
            current_mode=current_mode,
        )
        innovations = measurement_pass.innovation_norms
        mode_context = evaluate_mode_context(
            filter_engine=filter_engine,
            freshness_tracker=freshness_tracker,
            covariance_tracker=covariance_tracker,
            mode_tracker=mode_tracker,
            mode_thresholds=mode_thresholds,
            current_time=measurement_frame.time,
            pos_innovation_norm=innovations["gnss_pos"],
            vel_innovation_norm=innovations["gnss_vel"],
            baro_innovation_abs=innovations["baro"],
            yaw_innovation_abs_deg=innovations["mag"],
        )
        est_roll, est_pitch, est_yaw = quat_to_euler(filter_engine.state.quaternion)
        nav_env = filter_engine.current_navigation_environment
        gnss_lever_arm_diagnostics = evaluate_gnss_lever_arm_diagnostics(filter_engine, measurement_frame)
        record = _build_runtime_record(
            frame=measurement_frame,
            truth_frame=truth_frame,
            predict_diag=predict_diag,
            filter_engine=filter_engine,
            nav_env=nav_env,
            est_roll=est_roll,
            est_pitch=est_pitch,
            est_yaw=est_yaw,
            sensor_status=mode_context.sensor_status,
            covariance_health=mode_context.covariance_health,
            quality_score=mode_context.quality_score,
            mode_state=mode_context.mode_state,
            measurement_trace=measurement_pass.trace,
            pos_sigma_norm_m=mode_context.pos_sigma_norm_m,
            vel_sigma_norm_mps=mode_context.vel_sigma_norm_mps,
            att_sigma_norm_deg=mode_context.att_sigma_norm_deg,
            covariance_diagonal=mode_context.covariance_diagonal,
            gnss_lever_arm_diagnostics=gnss_lever_arm_diagnostics,
            initialization_summary=initialization_runtime.initialization_summary,
            initialized_this_frame=initialized_this_frame,
            bootstrap_anchor_frame=initialization_runtime.bootstrap_anchor_frame,
        )
        records.append(record)
        last_time = frame.time

    result_df = pd.DataFrame(records)
    export_pipeline_results(
        result_df=result_df,
        results_root=results_root,
        config=config,
        source_summary=dataset_load_result.source_summary,
        navigation_reference_override=dataset_load_result.navigation_reference_override,
        initialization_summary=initialization_runtime.initialization_summary,
        extra_metrics={
            "processed_rows": float(len(result_df)),
            "pipeline_runtime_s": time.perf_counter() - pipeline_start,
            **(dataset_load_result.input_quality_metrics or {}),
        },
    )
    return result_df


def main(config_path: str | None = None) -> None:
    config = load_config(config_path)
    dataset_path = project_path(config.dataset_path)
    results_root = project_path(config.results_dir)
    result_df = run_pipeline(config_path)

    print("ESKF demo finished.")
    print(f"Processed rows: {len(result_df)}")
    print(f"Dataset: {dataset_path}")
    print(f"Figures: {results_root / 'figures'}")
    print(f"Metrics: {results_root / 'metrics'}")
