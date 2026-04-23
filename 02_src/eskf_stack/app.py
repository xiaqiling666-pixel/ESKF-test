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
    compute_metrics,
    save_covariance_plot,
    save_error_plot,
    save_metrics,
    save_navigation_plot,
    save_quality_plot,
    save_state_machine_summary_plot,
    save_trajectory_plot,
)
from .analysis.quality import (
    CovarianceHealthTracker,
    SensorFreshnessTracker,
    classify_covariance_health,
    compute_quality_score,
)
from .analysis.state_machine import ModeStateTracker, determine_mode
from .config import load_config, project_path
from .core import OfflineESKF
from .core.math_utils import ensure_dir, quat_to_euler
from .core.state import ERROR_STATE
from .measurements import (
    BarometerMeasurement,
    GnssPositionMeasurement,
    GnssVelocityMeasurement,
    MagYawMeasurement,
)


def _build_measurements(config) -> list:
    measurement_models = [GnssPositionMeasurement(), GnssVelocityMeasurement()]
    if config.use_baro:
        measurement_models.append(BarometerMeasurement())
    if config.use_mag:
        measurement_models.append(MagYawMeasurement())
    return measurement_models


def _resolve_initial_velocity(frame) -> np.ndarray:
    if frame.gnss_vel is not None:
        return frame.gnss_vel
    return np.zeros(3, dtype=float)


def _resolve_initial_yaw(frame) -> float:
    if frame.mag_yaw is not None:
        return frame.mag_yaw
    if frame.gnss_vel is not None and float(np.linalg.norm(frame.gnss_vel[:2])) > 1e-6:
        return float(np.arctan2(frame.gnss_vel[1], frame.gnss_vel[0]))
    return 0.0


def _initialize_filter(filter_engine: OfflineESKF, frame) -> bool:
    if frame.gnss_pos is None:
        return False
    if frame.gnss_vel is None and frame.mag_yaw is None:
        return False
    filter_engine.initialize(frame.gnss_pos, _resolve_initial_velocity(frame), _resolve_initial_yaw(frame))
    return True


def _bootstrap_initialize_from_position_pair(
    filter_engine: OfflineESKF,
    anchor_frame,
    frame,
    min_dt_s: float = 0.15,
    min_horizontal_displacement_m: float = 0.5,
) -> bool:
    if anchor_frame is None or anchor_frame.gnss_pos is None or frame.gnss_pos is None:
        return False

    dt = float(frame.time - anchor_frame.time)
    if dt < min_dt_s:
        return False

    displacement = frame.gnss_pos - anchor_frame.gnss_pos
    horizontal_displacement = float(np.linalg.norm(displacement[:2]))
    if horizontal_displacement < min_horizontal_displacement_m:
        return False

    velocity = displacement / dt
    yaw = frame.mag_yaw if frame.mag_yaw is not None else float(np.arctan2(velocity[1], velocity[0]))
    filter_engine.initialize(frame.gnss_pos, velocity, yaw)
    return True


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


def _save_dataset_source_summary(
    metrics_dir,
    config,
    source_summary: dict[str, str] | None,
    navigation_reference_override,
) -> None:
    lines = [
        "Dataset Source Summary",
        "",
        f"config_path: {config.config_path}",
        f"config_profile: {config.config_metadata.profile}",
        f"config_name: {config.config_metadata.name}",
        f"config_purpose: {config.config_metadata.purpose}",
        "",
    ]
    if source_summary:
        for key, value in source_summary.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("adapter_kind: unknown")
    if navigation_reference_override is not None:
        lines.extend(
            [
                "",
                "navigation_reference_override:",
                f"reference_lat_deg: {navigation_reference_override.reference_lat_deg:.9f}",
                f"reference_lon_deg: {navigation_reference_override.reference_lon_deg:.9f}",
                f"reference_height_m: {navigation_reference_override.reference_height_m:.6f}",
            ]
        )
    (metrics_dir / "dataset_source_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_pipeline(config_path: str | None = None) -> pd.DataFrame:
    pipeline_start = time.perf_counter()
    config = load_config(config_path)
    dataset_path = project_path(config.dataset_path)
    results_root = ensure_dir(project_path(config.results_dir))
    figure_dir = ensure_dir(results_root / "figures")
    metrics_dir = ensure_dir(results_root / "metrics")

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
    freshness_tracker = SensorFreshnessTracker()
    covariance_tracker = CovarianceHealthTracker()
    mode_tracker = ModeStateTracker()
    bootstrap_anchor_frame = None
    last_time: float | None = None
    records: list[dict[str, Any]] = []

    for _, row in sensor_df.iterrows():
        frame = row_to_frame(row)
        measurement_frame = observation_view(frame)
        truth_frame = diagnostic_truth_view(frame)
        if not filter_engine.initialized:
            if not _initialize_filter(filter_engine, measurement_frame):
                if measurement_frame.gnss_pos is not None:
                    if bootstrap_anchor_frame is None:
                        bootstrap_anchor_frame = measurement_frame
                    elif _bootstrap_initialize_from_position_pair(
                        filter_engine,
                        bootstrap_anchor_frame,
                        measurement_frame,
                    ):
                        last_time = frame.time
                    else:
                        displacement = measurement_frame.gnss_pos - bootstrap_anchor_frame.gnss_pos
                        horizontal_displacement = float(np.linalg.norm(displacement[:2]))
                        if measurement_frame.time - bootstrap_anchor_frame.time > 1.0 and horizontal_displacement < 0.5:
                            bootstrap_anchor_frame = measurement_frame
                continue
            last_time = frame.time

        dt = 0.0 if last_time is None else max(0.0, frame.time - last_time)
        filter_engine.predict(frame.accel, frame.gyro, dt)

        innovations = {"gnss_pos": 0.0, "gnss_vel": 0.0, "baro": 0.0, "mag": 0.0}
        available_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        used_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        nis_values = {"gnss_pos": np.nan, "gnss_vel": np.nan, "baro": np.nan, "mag": np.nan}
        rejected_flags = {"gnss_pos": False, "gnss_vel": False, "baro": False, "mag": False}
        adaptation_scales = {"gnss_pos": 1.0, "gnss_vel": 1.0, "baro": 1.0, "mag": 1.0}

        for model in measurement_models:
            result = model.apply(filter_engine, measurement_frame)
            available_flags[result.name] = result.available
            used_flags[result.name] = result.used
            innovations[result.name] = result.innovation_value
            nis_values[result.name] = np.nan if result.nis is None else result.nis
            rejected_flags[result.name] = result.rejected
            adaptation_scales[result.name] = result.adaptation_scale
            freshness_tracker.note_result(
                result.name,
                measurement_frame.time,
                available=result.available,
                used=result.used,
                rejected=result.rejected,
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
        quality_score = compute_quality_score(
            sensor_status=sensor_status,
            pos_innovation_norm=innovations["gnss_pos"],
            vel_innovation_norm=innovations["gnss_vel"],
            yaw_innovation_abs_deg=innovations["mag"],
            pos_sigma_norm_m=pos_sigma_norm_m,
            vel_sigma_norm_mps=vel_sigma_norm_mps,
            att_sigma_norm_deg=att_sigma_norm_deg,
        )
        target_mode_decision = determine_mode(
            sensor_status,
            quality_score,
            covariance_health,
        )
        mode_state = mode_tracker.step(measurement_frame.time, target_mode_decision)
        est_roll, est_pitch, est_yaw = quat_to_euler(filter_engine.state.quaternion)
        nav_env = filter_engine.current_navigation_environment

        record = {
            "time": frame.time,
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
            "gnss_pos_reject_streak": sensor_status.gnss_pos_reject_streak,
            "gnss_vel_reject_streak": sensor_status.gnss_vel_reject_streak,
            "gnss_pos_outage_s": sensor_status.gnss_pos_outage_s,
            "gnss_vel_outage_s": sensor_status.gnss_vel_outage_s,
            "gnss_outage_s": sensor_status.gnss_outage_s,
            "gnss_pos_innovation_norm": innovations["gnss_pos"],
            "gnss_vel_innovation_norm": innovations["gnss_vel"],
            "baro_innovation_abs": innovations["baro"],
            "yaw_innovation_abs_deg": innovations["mag"],
            "gnss_pos_nis": nis_values["gnss_pos"],
            "gnss_vel_nis": nis_values["gnss_vel"],
            "baro_nis": nis_values["baro"],
            "mag_nis": nis_values["mag"],
            "gnss_pos_rejected": rejected_flags["gnss_pos"],
            "gnss_vel_rejected": rejected_flags["gnss_vel"],
            "baro_rejected": rejected_flags["baro"],
            "mag_rejected": rejected_flags["mag"],
            "gnss_pos_r_scale": adaptation_scales["gnss_pos"],
            "gnss_vel_r_scale": adaptation_scales["gnss_vel"],
            "baro_r_scale": adaptation_scales["baro"],
            "mag_r_scale": adaptation_scales["mag"],
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
        record.update(_truth_record(truth_frame))
        records.append(record)
        last_time = frame.time

    result_df = pd.DataFrame(records)
    fusion_output_path = metrics_dir / "fusion_output.csv"
    result_df.to_csv(fusion_output_path, index=False)
    _save_dataset_source_summary(
        metrics_dir,
        config,
        dataset_load_result.source_summary,
        dataset_load_result.navigation_reference_override,
    )

    save_trajectory_plot(result_df, figure_dir / "trajectory.png")
    save_error_plot(result_df, figure_dir / "error_summary.png")
    save_quality_plot(result_df, figure_dir / "quality_mode.png")
    save_navigation_plot(result_df, figure_dir / "navigation_diagnostics.png")
    save_covariance_plot(result_df, figure_dir / "covariance_diagnostics.png")
    save_state_machine_summary_plot(result_df, figure_dir / "state_machine_summary.png")
    metrics = compute_metrics(result_df)
    metrics["processed_rows"] = float(len(result_df))
    metrics["pipeline_runtime_s"] = time.perf_counter() - pipeline_start
    save_metrics(metrics, metrics_dir)
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
