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
    classify_covariance_health,
    compute_quality_score,
    summarize_measurement_support,
)
from .analysis.state_machine import ModeStateTracker, ModeThresholds, determine_mode
from .config import load_config, project_path
from .core import ImuInitializationSample, OfflineESKF
from .core.math_utils import ensure_dir, quat_to_euler
from .core.state import ERROR_STATE
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


def _truth_record(frame) -> dict[str, float]:
    return {
        "truth_x": np.nan if frame.truth_pos is None else frame.truth_pos[0],
        "truth_y": np.nan if frame.truth_pos is None else frame.truth_pos[1],
        "truth_z": np.nan if frame.truth_pos is None else frame.truth_pos[2],
        "truth_vx": np.nan if frame.truth_vel is None else frame.truth_vel[0],
        "truth_vy": np.nan if frame.truth_vel is None else frame.truth_vel[1],
        "truth_vz": np.nan if frame.truth_vel is None else frame.truth_vel[2],
        "truth_yaw": np.nan if frame.truth_yaw is None else frame.truth_yaw,
    }


def _initialization_summary_record_fields(initialization_summary: dict[str, Any]) -> dict[str, Any]:
    static_alignment_ready_text = str(initialization_summary.get("static_alignment_ready", "")).strip().lower()
    if static_alignment_ready_text == "true":
        static_alignment_ready = True
    elif static_alignment_ready_text == "false":
        static_alignment_ready = False
    else:
        static_alignment_ready = np.nan

    wait_text = initialization_summary.get("initialization_wait_s")
    wait_value = np.nan
    if wait_text not in (None, ""):
        wait_value = float(wait_text)

    return {
        "initialization_phase": str(initialization_summary.get("initialization_phase", "")).strip(),
        "initialization_reason": str(initialization_summary.get("initialization_reason", "")).strip(),
        "initialization_ready_mode": str(initialization_summary.get("initialization_ready_mode", "")).strip(),
        "initialization_heading_source": str(initialization_summary.get("heading_source", "")).strip(),
        "initialization_static_alignment_ready": static_alignment_ready,
        "initialization_static_alignment_reason": str(initialization_summary.get("static_alignment_reason", "")).strip(),
        "initialization_wait_s": wait_value,
        "initialization_pending": str(initialization_summary.get("initialization_phase", "")).strip() != "INITIALIZED",
        "bootstrap_anchor_age_s": np.nan,
        "bootstrap_anchor_horizontal_displacement_m": np.nan,
    }


def _bootstrap_anchor_diagnostics(frame, bootstrap_anchor_frame) -> tuple[float, float]:
    bootstrap_anchor_age_s = np.nan
    bootstrap_anchor_horizontal_displacement_m = np.nan
    if bootstrap_anchor_frame is not None:
        bootstrap_anchor_age_s = max(0.0, float(frame.time - bootstrap_anchor_frame.time))
        if frame.gnss_pos is not None and bootstrap_anchor_frame.gnss_pos is not None:
            displacement = frame.gnss_pos - bootstrap_anchor_frame.gnss_pos
            bootstrap_anchor_horizontal_displacement_m = float(np.linalg.norm(displacement[:2]))
    return bootstrap_anchor_age_s, bootstrap_anchor_horizontal_displacement_m


def _preinit_record(frame, truth_frame, initialization_status, bootstrap_anchor_frame) -> dict[str, Any]:
    bootstrap_anchor_age_s, bootstrap_anchor_horizontal_displacement_m = _bootstrap_anchor_diagnostics(
        frame,
        bootstrap_anchor_frame,
    )
    gnss_pos_available = frame.gnss_pos is not None
    gnss_vel_available = frame.gnss_vel is not None
    baro_available = frame.baro_h is not None
    mag_available = frame.mag_yaw is not None

    record: dict[str, Any] = {
        "time": frame.time,
        "dt_raw_s": 0.0,
        "dt_applied_s": 0.0,
        "predict_skipped": True,
        "predict_warning": False,
        "predict_reason": "not_initialized",
        "est_x": np.nan,
        "est_y": np.nan,
        "est_z": np.nan,
        "est_vx": np.nan,
        "est_vy": np.nan,
        "est_vz": np.nan,
        "est_roll": np.nan,
        "est_pitch": np.nan,
        "est_yaw": np.nan,
        "available_gnss_pos": gnss_pos_available,
        "available_gnss_vel": gnss_vel_available,
        "available_baro": baro_available,
        "available_mag": mag_available,
        "used_gnss_pos": False,
        "used_gnss_vel": False,
        "used_baro": False,
        "used_mag": False,
        "gnss_pos_rejected": False,
        "gnss_vel_rejected": False,
        "baro_rejected": False,
        "mag_rejected": False,
        "gnss_pos_reject_bypassed": False,
        "gnss_vel_reject_bypassed": False,
        "baro_reject_bypassed": False,
        "mag_reject_bypassed": False,
        "gnss_pos_management_mode": "pending_init" if gnss_pos_available else "unavailable",
        "gnss_vel_management_mode": "pending_init" if gnss_vel_available else "unavailable",
        "baro_management_mode": "pending_init" if baro_available else "unavailable",
        "mag_management_mode": "pending_init" if mag_available else "unavailable",
        "gnss_pos_innovation_norm": np.nan,
        "gnss_vel_innovation_norm": np.nan,
        "baro_innovation_abs": np.nan,
        "yaw_innovation_abs_deg": np.nan,
        "gnss_pos_nis": np.nan,
        "gnss_vel_nis": np.nan,
        "baro_nis": np.nan,
        "mag_nis": np.nan,
        "quality_score": np.nan,
        "mode": "UNINITIALIZED",
        "mode_reason": initialization_status.reason,
        "mode_target": "UNINITIALIZED",
        "mode_target_reason": initialization_status.reason,
        "mode_candidate": "",
        "mode_candidate_reason": "",
        "mode_hold_s": 0.0,
        "mode_candidate_hold_s": 0.0,
        "mode_transition_pending": False,
        "initialization_phase": initialization_status.phase,
        "initialization_reason": initialization_status.reason,
        "initialization_ready_mode": "" if initialization_status.ready_mode is None else initialization_status.ready_mode,
        "initialization_heading_source": initialization_status.heading_source,
        "initialization_static_alignment_ready": initialization_status.static_alignment_ready,
        "initialization_static_alignment_reason": initialization_status.static_alignment_reason,
        "initialization_wait_s": initialization_status.wait_time_s,
        "initialization_pending": initialization_status.phase != "INITIALIZED",
        "initialization_completed_this_frame": False,
        "bootstrap_anchor_age_s": bootstrap_anchor_age_s,
        "bootstrap_anchor_horizontal_displacement_m": bootstrap_anchor_horizontal_displacement_m,
    }
    record.update(_truth_record(truth_frame))
    return record


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
    bootstrap_anchor_frame = None
    initialization_samples: list[ImuInitializationSample] = []
    initialization_summary: dict[str, str] = {
        "initialization_mode": "uninitialized",
        "initialization_phase": "WAITING_GNSS",
        "initialization_reason": "no_gnss_position",
        "initialization_ready_mode": "",
        "heading_source": "none",
        "static_coarse_alignment_used": "false",
        "static_alignment_ready": "false",
        "static_alignment_reason": "not_evaluated",
    }
    last_time: float | None = None
    records: list[dict[str, Any]] = []

    for _, row in sensor_df.iterrows():
        frame = row_to_frame(row)
        initialized_this_frame = False
        if not filter_engine.initialized:
            initialization_samples.append(_preinit_sample_from_frame(frame))
        measurement_frame = observation_view(frame)
        truth_frame = diagnostic_truth_view(frame)
        if not filter_engine.initialized:
            initialization_status, _ = _assess_initialization_status(
                filter_engine,
                measurement_frame,
                initialization_samples,
                bootstrap_anchor_frame,
            )
            _summarize_initialization_status(initialization_status, initialization_summary)
            if not _initialize_filter(
                filter_engine,
                measurement_frame,
                initialization_samples,
                initialization_summary,
                bootstrap_anchor_frame,
            ):
                if measurement_frame.gnss_pos is not None:
                    if bootstrap_anchor_frame is None:
                        bootstrap_anchor_frame = measurement_frame
                    elif _bootstrap_initialize_from_position_pair(
                        filter_engine,
                        bootstrap_anchor_frame,
                        measurement_frame,
                        initialization_samples,
                        initialization_summary,
                    ):
                        last_time = frame.time
                        initialized_this_frame = True
                    else:
                        displacement = measurement_frame.gnss_pos - bootstrap_anchor_frame.gnss_pos
                        horizontal_displacement = float(np.linalg.norm(displacement[:2]))
                        if measurement_frame.time - bootstrap_anchor_frame.time > 1.0 and horizontal_displacement < 0.5:
                            bootstrap_anchor_frame = measurement_frame
                if not initialized_this_frame:
                    records.append(
                        _preinit_record(
                            measurement_frame,
                            truth_frame,
                            initialization_status,
                            bootstrap_anchor_frame,
                        )
                    )
                    continue
            last_time = frame.time
            initialized_this_frame = True

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

        innovations = {"gnss_pos": 0.0, "gnss_vel": 0.0, "baro": 0.0, "mag": 0.0}
        innovation_outputs = {"gnss_pos": np.nan, "gnss_vel": np.nan, "baro": np.nan, "mag": np.nan}
        available_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        used_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        nis_values = {"gnss_pos": np.nan, "gnss_vel": np.nan, "baro": np.nan, "mag": np.nan}
        rejected_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        reject_bypassed_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        adaptation_scales = {"gnss_pos": 1.0, "gnss_vel": 1.0, "baro": 1.0, "mag": 1.0}
        recovery_scales = {"gnss_pos": 1.0, "gnss_vel": 1.0, "baro": 1.0, "mag": 1.0}
        mode_scales = {"gnss_pos": 1.0, "gnss_vel": 1.0, "baro": 1.0, "mag": 1.0}
        applied_r_scales = {"gnss_pos": 1.0, "gnss_vel": 1.0, "baro": 1.0, "mag": 1.0}
        management_modes = {"gnss_pos": "unavailable", "gnss_vel": "unavailable", "baro": "unavailable", "mag": "unavailable"}
        nis_adapt_thresholds = {"gnss_pos": np.nan, "gnss_vel": np.nan, "baro": np.nan, "mag": np.nan}
        nis_reject_thresholds = {"gnss_pos": np.nan, "gnss_vel": np.nan, "baro": np.nan, "mag": np.nan}
        current_mode = None if mode_tracker.current_decision is None else mode_tracker.current_decision.mode

        for model in measurement_models:
            policy = model.policy(filter_engine)
            nis_adapt_thresholds[model.name] = np.nan if policy.adapt_threshold is None else policy.adapt_threshold
            nis_reject_thresholds[model.name] = np.nan if policy.reject_threshold is None else policy.reject_threshold
            result = measurement_manager.process(filter_engine, model, measurement_frame, current_mode=current_mode)
            available_flags[result.name] = result.available
            used_flags[result.name] = result.used
            if result.available:
                innovations[result.name] = result.innovation_value
                innovation_outputs[result.name] = result.innovation_value
            nis_values[result.name] = np.nan if result.nis is None else result.nis
            rejected_flags[result.name] = result.rejected
            reject_bypassed_flags[result.name] = result.reject_bypassed
            adaptation_scales[result.name] = result.adaptation_scale
            recovery_scales[result.name] = result.recovery_scale
            mode_scales[result.name] = result.mode_scale
            applied_r_scales[result.name] = result.applied_r_scale
            management_modes[result.name] = result.management_mode
            freshness_tracker.note_result(
                result.name,
                measurement_frame.time,
                available=result.available,
                used=result.used,
                rejected=result.rejected,
                reject_bypassed=result.reject_bypassed,
                management_mode=result.management_mode,
                adaptation_scale=result.adaptation_scale,
            )

        pos_sigma_norm_m = float(np.sqrt(np.trace(filter_engine.P[ERROR_STATE.position, ERROR_STATE.position])))
        vel_sigma_norm_mps = float(np.sqrt(np.trace(filter_engine.P[ERROR_STATE.velocity, ERROR_STATE.velocity])))
        att_sigma_norm_deg = float(
            np.rad2deg(np.sqrt(np.trace(filter_engine.P[ERROR_STATE.attitude, ERROR_STATE.attitude])))
        )
        covariance_diagonal = np.diag(filter_engine.P)
        sensor_status = freshness_tracker.snapshot(measurement_frame.time)
        covariance_health = classify_covariance_health(
            pos_sigma_norm_m=pos_sigma_norm_m,
            vel_sigma_norm_mps=vel_sigma_norm_mps,
            att_sigma_norm_deg=att_sigma_norm_deg,
        )
        covariance_health = covariance_tracker.step(frame.time, covariance_health)
        support_summary = summarize_measurement_support(
            sensor_status,
            gnss_skip_streak_degraded=mode_thresholds.gnss_skip_streak_degraded,
            auxiliary_reject_streak_degraded=mode_thresholds.auxiliary_reject_streak_degraded,
            auxiliary_reject_bypass_streak_degraded=mode_thresholds.auxiliary_reject_bypass_streak_degraded,
            auxiliary_adaptive_streak_degraded=mode_thresholds.auxiliary_adaptive_streak_degraded,
            auxiliary_skip_streak_degraded=mode_thresholds.auxiliary_skip_streak_degraded,
        )
        quality_score = compute_quality_score(
            sensor_status=sensor_status,
            pos_innovation_norm=innovations["gnss_pos"],
            vel_innovation_norm=innovations["gnss_vel"],
            baro_innovation_abs=innovations["baro"],
            yaw_innovation_abs_deg=innovations["mag"],
            pos_sigma_norm_m=pos_sigma_norm_m,
            vel_sigma_norm_mps=vel_sigma_norm_mps,
            att_sigma_norm_deg=att_sigma_norm_deg,
            support_summary=support_summary,
        )
        target_mode_decision = determine_mode(
            sensor_status,
            quality_score,
            covariance_health,
            thresholds=mode_thresholds,
            support_summary=support_summary,
        )
        mode_state = mode_tracker.step(measurement_frame.time, target_mode_decision)
        est_roll, est_pitch, est_yaw = quat_to_euler(filter_engine.state.quaternion)
        nav_env = filter_engine.current_navigation_environment

        record = {
            "time": frame.time,
            "dt_raw_s": predict_diag.raw_dt,
            "dt_applied_s": predict_diag.applied_dt,
            "predict_skipped": predict_diag.skipped,
            "predict_warning": predict_diag.warning,
            "predict_reason": predict_diag.reason,
            "est_x": filter_engine.state.position[0],
            "est_y": filter_engine.state.position[1],
            "est_z": filter_engine.state.position[2],
            "est_vx": filter_engine.state.velocity[0],
            "est_vy": filter_engine.state.velocity[1],
            "est_vz": filter_engine.state.velocity[2],
            "est_roll": est_roll,
            "est_pitch": est_pitch,
            "est_yaw": est_yaw,
            "bg_x": filter_engine.state.gyro_bias[0],
            "bg_y": filter_engine.state.gyro_bias[1],
            "bg_z": filter_engine.state.gyro_bias[2],
            "ba_x": filter_engine.state.accel_bias[0],
            "ba_y": filter_engine.state.accel_bias[1],
            "ba_z": filter_engine.state.accel_bias[2],
            "nav_lat_deg": np.rad2deg(nav_env.current_lat_rad),
            "nav_lon_deg": np.rad2deg(nav_env.current_lon_rad),
            "nav_height_m": nav_env.current_height_m,
            "meridian_radius_m": nav_env.meridian_radius_m,
            "prime_vertical_radius_m": nav_env.prime_vertical_radius_m,
            "gravity_n_x": nav_env.gravity_vector[0],
            "gravity_n_y": nav_env.gravity_vector[1],
            "gravity_n_z": nav_env.gravity_vector[2],
            "gravity_gradient_norm": np.linalg.norm(nav_env.gravity_gradient_nav),
            "coriolis_position_gradient_norm": np.linalg.norm(filter_engine.current_coriolis_position_jacobian),
            "coriolis_velocity_gradient_norm": np.linalg.norm(filter_engine.current_coriolis_velocity_jacobian),
            "earth_rate_n_x": nav_env.earth_rate_nav[0],
            "earth_rate_n_y": nav_env.earth_rate_nav[1],
            "earth_rate_n_z": nav_env.earth_rate_nav[2],
            "transport_rate_n_x": nav_env.transport_rate_nav[0],
            "transport_rate_n_y": nav_env.transport_rate_nav[1],
            "transport_rate_n_z": nav_env.transport_rate_nav[2],
            "available_gnss_pos": available_flags["gnss_pos"],
            "available_gnss_vel": available_flags["gnss_vel"],
            "available_baro": available_flags["baro"],
            "available_mag": available_flags["mag"],
            "used_gnss_pos": used_flags["gnss_pos"],
            "used_gnss_vel": used_flags["gnss_vel"],
            "used_baro": used_flags["baro"],
            "used_mag": used_flags["mag"],
            "recent_gnss_pos": sensor_status.recent_gnss_pos,
            "recent_gnss_vel": sensor_status.recent_gnss_vel,
            "recent_baro": sensor_status.recent_baro,
            "recent_mag": sensor_status.recent_mag,
            "recent_available_gnss_pos": sensor_status.recent_available_gnss_pos,
            "recent_available_gnss_vel": sensor_status.recent_available_gnss_vel,
            "recent_available_baro": sensor_status.recent_available_baro,
            "recent_available_mag": sensor_status.recent_available_mag,
            "gnss_pos_reject_streak": sensor_status.gnss_pos_reject_streak,
            "gnss_vel_reject_streak": sensor_status.gnss_vel_reject_streak,
            "baro_reject_streak": sensor_status.baro_reject_streak,
            "mag_reject_streak": sensor_status.mag_reject_streak,
            "gnss_pos_skip_streak": sensor_status.gnss_pos_skip_streak,
            "gnss_vel_skip_streak": sensor_status.gnss_vel_skip_streak,
            "baro_skip_streak": sensor_status.baro_skip_streak,
            "mag_skip_streak": sensor_status.mag_skip_streak,
            "gnss_pos_reject_bypass_streak": sensor_status.gnss_pos_reject_bypass_streak,
            "gnss_vel_reject_bypass_streak": sensor_status.gnss_vel_reject_bypass_streak,
            "baro_reject_bypass_streak": sensor_status.baro_reject_bypass_streak,
            "mag_reject_bypass_streak": sensor_status.mag_reject_bypass_streak,
            "gnss_pos_adaptive_streak": sensor_status.gnss_pos_adaptive_streak,
            "gnss_vel_adaptive_streak": sensor_status.gnss_vel_adaptive_streak,
            "baro_adaptive_streak": sensor_status.baro_adaptive_streak,
            "mag_adaptive_streak": sensor_status.mag_adaptive_streak,
            "gnss_pos_outage_s": sensor_status.gnss_pos_outage_s,
            "gnss_vel_outage_s": sensor_status.gnss_vel_outage_s,
            "gnss_outage_s": sensor_status.gnss_outage_s,
            "baro_outage_s": sensor_status.baro_outage_s,
            "mag_outage_s": sensor_status.mag_outage_s,
            "auxiliary_outage_s": sensor_status.auxiliary_outage_s,
            "gnss_pos_available_outage_s": sensor_status.gnss_pos_available_outage_s,
            "gnss_vel_available_outage_s": sensor_status.gnss_vel_available_outage_s,
            "baro_available_outage_s": sensor_status.baro_available_outage_s,
            "mag_available_outage_s": sensor_status.mag_available_outage_s,
            "auxiliary_available_outage_s": sensor_status.auxiliary_available_outage_s,
            "gnss_pos_innovation_norm": innovation_outputs["gnss_pos"],
            "gnss_vel_innovation_norm": innovation_outputs["gnss_vel"],
            "baro_innovation_abs": innovation_outputs["baro"],
            "yaw_innovation_abs_deg": innovation_outputs["mag"],
            "gnss_pos_nis": nis_values["gnss_pos"],
            "gnss_vel_nis": nis_values["gnss_vel"],
            "baro_nis": nis_values["baro"],
            "mag_nis": nis_values["mag"],
            "gnss_pos_nis_adapt_threshold": nis_adapt_thresholds["gnss_pos"],
            "gnss_vel_nis_adapt_threshold": nis_adapt_thresholds["gnss_vel"],
            "baro_nis_adapt_threshold": nis_adapt_thresholds["baro"],
            "mag_nis_adapt_threshold": nis_adapt_thresholds["mag"],
            "gnss_pos_nis_reject_threshold": nis_reject_thresholds["gnss_pos"],
            "gnss_vel_nis_reject_threshold": nis_reject_thresholds["gnss_vel"],
            "baro_nis_reject_threshold": nis_reject_thresholds["baro"],
            "mag_nis_reject_threshold": nis_reject_thresholds["mag"],
            "gnss_pos_rejected": rejected_flags["gnss_pos"],
            "gnss_vel_rejected": rejected_flags["gnss_vel"],
            "baro_rejected": rejected_flags["baro"],
            "mag_rejected": rejected_flags["mag"],
            "gnss_pos_reject_bypassed": reject_bypassed_flags["gnss_pos"],
            "gnss_vel_reject_bypassed": reject_bypassed_flags["gnss_vel"],
            "baro_reject_bypassed": reject_bypassed_flags["baro"],
            "mag_reject_bypassed": reject_bypassed_flags["mag"],
            "gnss_pos_management_mode": management_modes["gnss_pos"],
            "gnss_vel_management_mode": management_modes["gnss_vel"],
            "baro_management_mode": management_modes["baro"],
            "mag_management_mode": management_modes["mag"],
            "gnss_pos_r_scale": applied_r_scales["gnss_pos"],
            "gnss_vel_r_scale": applied_r_scales["gnss_vel"],
            "baro_r_scale": applied_r_scales["baro"],
            "mag_r_scale": applied_r_scales["mag"],
            "gnss_pos_adaptation_scale": adaptation_scales["gnss_pos"],
            "gnss_vel_adaptation_scale": adaptation_scales["gnss_vel"],
            "baro_adaptation_scale": adaptation_scales["baro"],
            "mag_adaptation_scale": adaptation_scales["mag"],
            "gnss_pos_recovery_scale": recovery_scales["gnss_pos"],
            "gnss_vel_recovery_scale": recovery_scales["gnss_vel"],
            "baro_recovery_scale": recovery_scales["baro"],
            "mag_recovery_scale": recovery_scales["mag"],
            "gnss_pos_mode_scale": mode_scales["gnss_pos"],
            "gnss_vel_mode_scale": mode_scales["gnss_vel"],
            "baro_mode_scale": mode_scales["baro"],
            "mag_mode_scale": mode_scales["mag"],
            "gnss_pos_applied_r_scale": applied_r_scales["gnss_pos"],
            "gnss_vel_applied_r_scale": applied_r_scales["gnss_vel"],
            "baro_applied_r_scale": applied_r_scales["baro"],
            "mag_applied_r_scale": applied_r_scales["mag"],
            "pos_sigma_norm_m": pos_sigma_norm_m,
            "vel_sigma_norm_mps": vel_sigma_norm_mps,
            "att_sigma_norm_deg": att_sigma_norm_deg,
            "covariance_health": covariance_health.level,
            "covariance_health_reason": covariance_health.reason,
            "covariance_caution": covariance_health.caution,
            "covariance_unhealthy": covariance_health.unhealthy,
            "covariance_pos_excess_m": covariance_health.pos_excess_m,
            "covariance_vel_excess_mps": covariance_health.vel_excess_mps,
            "covariance_att_excess_deg": covariance_health.att_excess_deg,
            "covariance_caution_duration_s": covariance_health.caution_duration_s,
            "covariance_unhealthy_duration_s": covariance_health.unhealthy_duration_s,
            "cov_trace": float(np.trace(filter_engine.P)),
            "cov_min_diag": float(np.min(covariance_diagonal)),
            "quality_score": quality_score,
            "mode": mode_state.mode,
            "mode_reason": mode_state.reason,
            "mode_target": mode_state.target_mode,
            "mode_target_reason": mode_state.target_reason,
            "mode_candidate": "" if mode_state.candidate_mode is None else mode_state.candidate_mode,
            "mode_candidate_reason": "" if mode_state.candidate_reason is None else mode_state.candidate_reason,
            "mode_hold_s": mode_state.mode_hold_s,
            "mode_candidate_hold_s": mode_state.candidate_hold_s,
            "mode_transition_pending": mode_state.transition_pending,
        }
        record.update(_initialization_summary_record_fields(initialization_summary))
        record["initialization_completed_this_frame"] = initialized_this_frame
        if initialized_this_frame:
            (
                record["bootstrap_anchor_age_s"],
                record["bootstrap_anchor_horizontal_displacement_m"],
            ) = _bootstrap_anchor_diagnostics(measurement_frame, bootstrap_anchor_frame)
        record.update(_truth_record(truth_frame))
        records.append(record)
        last_time = frame.time

    result_df = pd.DataFrame(records)
    export_pipeline_results(
        result_df=result_df,
        results_root=results_root,
        config=config,
        source_summary=dataset_load_result.source_summary,
        navigation_reference_override=dataset_load_result.navigation_reference_override,
        initialization_summary=initialization_summary,
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
