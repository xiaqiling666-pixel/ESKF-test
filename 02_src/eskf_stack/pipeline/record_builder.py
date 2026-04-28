from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


MEASUREMENT_NAMES = ("gnss_pos", "gnss_vel", "baro", "mag")


@dataclass(frozen=True)
class MeasurementTraceSnapshot:
    available_flags: dict[str, bool]
    used_flags: dict[str, bool]
    innovation_outputs: dict[str, float]
    nis_values: dict[str, float]
    nis_adapt_thresholds: dict[str, float]
    nis_reject_thresholds: dict[str, float]
    rejected_flags: dict[str, bool]
    reject_bypassed_flags: dict[str, bool]
    management_modes: dict[str, str]
    adaptation_scales: dict[str, float]
    recovery_scales: dict[str, float]
    mode_scales: dict[str, float]
    applied_r_scales: dict[str, float]


@dataclass
class MeasurementTraceCollector:
    available_flags: dict[str, bool]
    used_flags: dict[str, bool]
    innovation_outputs: dict[str, float]
    nis_values: dict[str, float]
    nis_adapt_thresholds: dict[str, float]
    nis_reject_thresholds: dict[str, float]
    rejected_flags: dict[str, bool]
    reject_bypassed_flags: dict[str, bool]
    management_modes: dict[str, str]
    adaptation_scales: dict[str, float]
    recovery_scales: dict[str, float]
    mode_scales: dict[str, float]
    applied_r_scales: dict[str, float]

    @classmethod
    def create_empty(cls) -> "MeasurementTraceCollector":
        return cls(
            available_flags={name: False for name in MEASUREMENT_NAMES},
            used_flags={name: False for name in MEASUREMENT_NAMES},
            innovation_outputs={name: np.nan for name in MEASUREMENT_NAMES},
            nis_values={name: np.nan for name in MEASUREMENT_NAMES},
            nis_adapt_thresholds={name: np.nan for name in MEASUREMENT_NAMES},
            nis_reject_thresholds={name: np.nan for name in MEASUREMENT_NAMES},
            rejected_flags={name: False for name in MEASUREMENT_NAMES},
            reject_bypassed_flags={name: False for name in MEASUREMENT_NAMES},
            management_modes={name: "unavailable" for name in MEASUREMENT_NAMES},
            adaptation_scales={name: 1.0 for name in MEASUREMENT_NAMES},
            recovery_scales={name: 1.0 for name in MEASUREMENT_NAMES},
            mode_scales={name: 1.0 for name in MEASUREMENT_NAMES},
            applied_r_scales={name: 1.0 for name in MEASUREMENT_NAMES},
        )

    def record_policy(self, model_name: str, policy) -> None:
        self.nis_adapt_thresholds[model_name] = np.nan if policy.adapt_threshold is None else policy.adapt_threshold
        self.nis_reject_thresholds[model_name] = np.nan if policy.reject_threshold is None else policy.reject_threshold

    def record_result(self, result) -> None:
        self.available_flags[result.name] = result.available
        self.used_flags[result.name] = result.used
        if result.available:
            self.innovation_outputs[result.name] = result.innovation_value
        self.nis_values[result.name] = np.nan if result.nis is None else result.nis
        self.rejected_flags[result.name] = result.rejected
        self.reject_bypassed_flags[result.name] = result.reject_bypassed
        self.adaptation_scales[result.name] = result.adaptation_scale
        self.recovery_scales[result.name] = result.recovery_scale
        self.mode_scales[result.name] = result.mode_scale
        self.applied_r_scales[result.name] = result.applied_r_scale
        self.management_modes[result.name] = result.management_mode

    def innovation_norms(self) -> dict[str, float]:
        return {
            "gnss_pos": 0.0 if np.isnan(self.innovation_outputs["gnss_pos"]) else float(self.innovation_outputs["gnss_pos"]),
            "gnss_vel": 0.0 if np.isnan(self.innovation_outputs["gnss_vel"]) else float(self.innovation_outputs["gnss_vel"]),
            "baro": 0.0 if np.isnan(self.innovation_outputs["baro"]) else float(self.innovation_outputs["baro"]),
            "mag": 0.0 if np.isnan(self.innovation_outputs["mag"]) else float(self.innovation_outputs["mag"]),
        }

    def snapshot(self) -> MeasurementTraceSnapshot:
        return MeasurementTraceSnapshot(
            available_flags=self.available_flags,
            used_flags=self.used_flags,
            innovation_outputs=self.innovation_outputs,
            nis_values=self.nis_values,
            nis_adapt_thresholds=self.nis_adapt_thresholds,
            nis_reject_thresholds=self.nis_reject_thresholds,
            rejected_flags=self.rejected_flags,
            reject_bypassed_flags=self.reject_bypassed_flags,
            management_modes=self.management_modes,
            adaptation_scales=self.adaptation_scales,
            recovery_scales=self.recovery_scales,
            mode_scales=self.mode_scales,
            applied_r_scales=self.applied_r_scales,
        )


def truth_record(frame) -> dict[str, float]:
    return {
        "truth_x": np.nan if frame.truth_pos is None else frame.truth_pos[0],
        "truth_y": np.nan if frame.truth_pos is None else frame.truth_pos[1],
        "truth_z": np.nan if frame.truth_pos is None else frame.truth_pos[2],
        "truth_vx": np.nan if frame.truth_vel is None else frame.truth_vel[0],
        "truth_vy": np.nan if frame.truth_vel is None else frame.truth_vel[1],
        "truth_vz": np.nan if frame.truth_vel is None else frame.truth_vel[2],
        "truth_yaw": np.nan if frame.truth_yaw is None else frame.truth_yaw,
    }


def initialization_summary_record_fields(initialization_summary: dict[str, Any]) -> dict[str, Any]:
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


def bootstrap_anchor_diagnostics(frame, bootstrap_anchor_frame) -> tuple[float, float]:
    bootstrap_anchor_age_s = np.nan
    bootstrap_anchor_horizontal_displacement_m = np.nan
    if bootstrap_anchor_frame is not None:
        bootstrap_anchor_age_s = max(0.0, float(frame.time - bootstrap_anchor_frame.time))
        if frame.gnss_pos is not None and bootstrap_anchor_frame.gnss_pos is not None:
            displacement = frame.gnss_pos - bootstrap_anchor_frame.gnss_pos
            bootstrap_anchor_horizontal_displacement_m = float(np.linalg.norm(displacement[:2]))
    return bootstrap_anchor_age_s, bootstrap_anchor_horizontal_displacement_m


def build_preinit_record(frame, truth_frame, initialization_status, bootstrap_anchor_frame) -> dict[str, Any]:
    bootstrap_anchor_age_s, bootstrap_anchor_horizontal_displacement_m = bootstrap_anchor_diagnostics(
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
    record.update(truth_record(truth_frame))
    return record


def build_runtime_record(
    *,
    frame,
    truth_frame,
    predict_diag,
    filter_engine,
    nav_env,
    est_roll: float,
    est_pitch: float,
    est_yaw: float,
    sensor_status,
    covariance_health,
    quality_score: float,
    mode_state,
    measurement_trace: MeasurementTraceSnapshot,
    pos_sigma_norm_m: float,
    vel_sigma_norm_mps: float,
    att_sigma_norm_deg: float,
    covariance_diagonal: np.ndarray,
    gnss_lever_arm_diagnostics,
    initialization_summary: dict[str, Any],
    initialized_this_frame: bool,
    bootstrap_anchor_frame,
) -> dict[str, Any]:
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
        "gnss_lever_arm_body_x_m": gnss_lever_arm_diagnostics.body_vector_m[0],
        "gnss_lever_arm_body_y_m": gnss_lever_arm_diagnostics.body_vector_m[1],
        "gnss_lever_arm_body_z_m": gnss_lever_arm_diagnostics.body_vector_m[2],
        "gnss_lever_arm_nav_x_m": gnss_lever_arm_diagnostics.nav_vector_m[0],
        "gnss_lever_arm_nav_y_m": gnss_lever_arm_diagnostics.nav_vector_m[1],
        "gnss_lever_arm_nav_z_m": gnss_lever_arm_diagnostics.nav_vector_m[2],
        "gnss_lever_arm_nav_norm_m": gnss_lever_arm_diagnostics.nav_norm_m,
        "gnss_lever_arm_rotational_velocity_nav_x_mps": gnss_lever_arm_diagnostics.rotational_velocity_nav_mps[0],
        "gnss_lever_arm_rotational_velocity_nav_y_mps": gnss_lever_arm_diagnostics.rotational_velocity_nav_mps[1],
        "gnss_lever_arm_rotational_velocity_nav_z_mps": gnss_lever_arm_diagnostics.rotational_velocity_nav_mps[2],
        "gnss_lever_arm_rotational_speed_mps": gnss_lever_arm_diagnostics.rotational_speed_mps,
        "available_gnss_pos": measurement_trace.available_flags["gnss_pos"],
        "available_gnss_vel": measurement_trace.available_flags["gnss_vel"],
        "available_baro": measurement_trace.available_flags["baro"],
        "available_mag": measurement_trace.available_flags["mag"],
        "used_gnss_pos": measurement_trace.used_flags["gnss_pos"],
        "used_gnss_vel": measurement_trace.used_flags["gnss_vel"],
        "used_baro": measurement_trace.used_flags["baro"],
        "used_mag": measurement_trace.used_flags["mag"],
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
        "gnss_pos_innovation_norm": measurement_trace.innovation_outputs["gnss_pos"],
        "gnss_vel_innovation_norm": measurement_trace.innovation_outputs["gnss_vel"],
        "baro_innovation_abs": measurement_trace.innovation_outputs["baro"],
        "yaw_innovation_abs_deg": measurement_trace.innovation_outputs["mag"],
        "gnss_pos_nis": measurement_trace.nis_values["gnss_pos"],
        "gnss_vel_nis": measurement_trace.nis_values["gnss_vel"],
        "baro_nis": measurement_trace.nis_values["baro"],
        "mag_nis": measurement_trace.nis_values["mag"],
        "gnss_pos_nis_adapt_threshold": measurement_trace.nis_adapt_thresholds["gnss_pos"],
        "gnss_vel_nis_adapt_threshold": measurement_trace.nis_adapt_thresholds["gnss_vel"],
        "baro_nis_adapt_threshold": measurement_trace.nis_adapt_thresholds["baro"],
        "mag_nis_adapt_threshold": measurement_trace.nis_adapt_thresholds["mag"],
        "gnss_pos_nis_reject_threshold": measurement_trace.nis_reject_thresholds["gnss_pos"],
        "gnss_vel_nis_reject_threshold": measurement_trace.nis_reject_thresholds["gnss_vel"],
        "baro_nis_reject_threshold": measurement_trace.nis_reject_thresholds["baro"],
        "mag_nis_reject_threshold": measurement_trace.nis_reject_thresholds["mag"],
        "gnss_pos_rejected": measurement_trace.rejected_flags["gnss_pos"],
        "gnss_vel_rejected": measurement_trace.rejected_flags["gnss_vel"],
        "baro_rejected": measurement_trace.rejected_flags["baro"],
        "mag_rejected": measurement_trace.rejected_flags["mag"],
        "gnss_pos_reject_bypassed": measurement_trace.reject_bypassed_flags["gnss_pos"],
        "gnss_vel_reject_bypassed": measurement_trace.reject_bypassed_flags["gnss_vel"],
        "baro_reject_bypassed": measurement_trace.reject_bypassed_flags["baro"],
        "mag_reject_bypassed": measurement_trace.reject_bypassed_flags["mag"],
        "gnss_pos_management_mode": measurement_trace.management_modes["gnss_pos"],
        "gnss_vel_management_mode": measurement_trace.management_modes["gnss_vel"],
        "baro_management_mode": measurement_trace.management_modes["baro"],
        "mag_management_mode": measurement_trace.management_modes["mag"],
        "gnss_pos_r_scale": measurement_trace.applied_r_scales["gnss_pos"],
        "gnss_vel_r_scale": measurement_trace.applied_r_scales["gnss_vel"],
        "baro_r_scale": measurement_trace.applied_r_scales["baro"],
        "mag_r_scale": measurement_trace.applied_r_scales["mag"],
        "gnss_pos_adaptation_scale": measurement_trace.adaptation_scales["gnss_pos"],
        "gnss_vel_adaptation_scale": measurement_trace.adaptation_scales["gnss_vel"],
        "baro_adaptation_scale": measurement_trace.adaptation_scales["baro"],
        "mag_adaptation_scale": measurement_trace.adaptation_scales["mag"],
        "gnss_pos_recovery_scale": measurement_trace.recovery_scales["gnss_pos"],
        "gnss_vel_recovery_scale": measurement_trace.recovery_scales["gnss_vel"],
        "baro_recovery_scale": measurement_trace.recovery_scales["baro"],
        "mag_recovery_scale": measurement_trace.recovery_scales["mag"],
        "gnss_pos_mode_scale": measurement_trace.mode_scales["gnss_pos"],
        "gnss_vel_mode_scale": measurement_trace.mode_scales["gnss_vel"],
        "baro_mode_scale": measurement_trace.mode_scales["baro"],
        "mag_mode_scale": measurement_trace.mode_scales["mag"],
        "gnss_pos_applied_r_scale": measurement_trace.applied_r_scales["gnss_pos"],
        "gnss_vel_applied_r_scale": measurement_trace.applied_r_scales["gnss_vel"],
        "baro_applied_r_scale": measurement_trace.applied_r_scales["baro"],
        "mag_applied_r_scale": measurement_trace.applied_r_scales["mag"],
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
    record.update(initialization_summary_record_fields(initialization_summary))
    record["initialization_completed_this_frame"] = initialized_this_frame
    if initialized_this_frame:
        (
            record["bootstrap_anchor_age_s"],
            record["bootstrap_anchor_horizontal_displacement_m"],
        ) = bootstrap_anchor_diagnostics(frame, bootstrap_anchor_frame)
    record.update(truth_record(truth_frame))
    return record
