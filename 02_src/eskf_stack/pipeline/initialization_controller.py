from __future__ import annotations

from typing import Any

import numpy as np

from ..core import ImuInitializationSample, InitializationStatus, OfflineESKF, StaticAlignmentCheck, StaticCoarseInitializer


def resolve_initial_velocity(frame) -> np.ndarray:
    if frame.gnss_vel is not None:
        return frame.gnss_vel
    return np.zeros(3, dtype=float)


def resolve_initial_yaw(frame) -> float:
    if frame.mag_yaw is not None:
        return frame.mag_yaw
    if frame.gnss_vel is not None and float(np.linalg.norm(frame.gnss_vel[:2])) > 1e-6:
        return float(np.arctan2(frame.gnss_vel[1], frame.gnss_vel[0]))
    return 0.0


def resolve_heading_source(frame) -> tuple[float | None, str]:
    if frame.mag_yaw is not None:
        return float(frame.mag_yaw), "mag_yaw"
    if frame.gnss_vel is not None and float(np.linalg.norm(frame.gnss_vel[:2])) > 1e-6:
        return float(np.arctan2(frame.gnss_vel[1], frame.gnss_vel[0])), "gnss_velocity_course"
    return None, "none"


def preinit_sample_from_frame(frame) -> ImuInitializationSample:
    return ImuInitializationSample(time=float(frame.time), accel=frame.accel.copy(), gyro=frame.gyro.copy())


def assess_static_coarse_alignment(
    filter_engine: OfflineESKF,
    initialization_samples: list[ImuInitializationSample],
    alignment_position: np.ndarray,
    yaw: float,
) -> StaticAlignmentCheck:
    initializer = StaticCoarseInitializer(filter_engine.config.initialization)
    return initializer.assess(
        initialization_samples,
        position=alignment_position,
        base_environment=filter_engine.navigation_environment,
        use_wgs84_gravity=filter_engine.config.navigation_environment.use_wgs84_gravity,
        use_earth_rotation=filter_engine.config.navigation_environment.use_earth_rotation,
        yaw=yaw,
    )


def summarize_initialization_status(
    status: InitializationStatus,
    initialization_summary: dict[str, Any],
) -> None:
    initialization_summary.update(
        {
            "initialization_phase": status.phase,
            "initialization_reason": status.reason,
            "initialization_ready_mode": "" if status.ready_mode is None else status.ready_mode,
            "heading_source": status.heading_source,
            "static_alignment_ready": "true" if status.static_alignment_ready else "false",
            "static_alignment_reason": status.static_alignment_reason,
            "initialization_wait_s": f"{status.wait_time_s:.6f}",
        }
    )


def assess_initialization_status(
    filter_engine: OfflineESKF,
    frame,
    initialization_samples: list[ImuInitializationSample],
    bootstrap_anchor_frame,
) -> tuple[InitializationStatus, StaticAlignmentCheck | None]:
    init_config = filter_engine.config.initialization
    if filter_engine.initialized:
        return (
            InitializationStatus(
                phase="INITIALIZED",
                reason="already_initialized",
                ready_mode=None,
                heading_source="n/a",
                static_alignment_ready=False,
                static_alignment_reason="n/a",
                wait_time_s=0.0,
            ),
            None,
        )

    if frame.gnss_pos is None:
        return (
            InitializationStatus(
                phase="WAITING_GNSS",
                reason="no_gnss_position",
                ready_mode=None,
                heading_source="none",
                static_alignment_ready=False,
                static_alignment_reason="not_evaluated",
                wait_time_s=0.0,
            ),
            None,
        )

    yaw, heading_source = resolve_heading_source(frame)
    if yaw is not None:
        static_check = assess_static_coarse_alignment(
            filter_engine,
            initialization_samples,
            frame.gnss_pos,
            yaw,
        )
        return (
            InitializationStatus(
                phase="READY_DIRECT_INIT",
                reason="direct_init_prerequisites_met",
                ready_mode="direct",
                heading_source=heading_source,
                static_alignment_ready=static_check.ready,
                static_alignment_reason=static_check.reason,
                wait_time_s=0.0,
            ),
            static_check,
        )

    if bootstrap_anchor_frame is None or bootstrap_anchor_frame.gnss_pos is None:
        return (
            InitializationStatus(
                phase="WAITING_BOOTSTRAP_MOTION",
                reason="awaiting_bootstrap_anchor",
                ready_mode=None,
                heading_source="none",
                static_alignment_ready=False,
                static_alignment_reason="heading_not_available",
                wait_time_s=0.0,
            ),
            None,
        )

    dt = float(frame.time - bootstrap_anchor_frame.time)
    if init_config.zero_yaw_fallback_enabled and dt >= init_config.heading_wait_timeout_s:
        yaw = 0.0
        static_check = assess_static_coarse_alignment(
            filter_engine,
            initialization_samples,
            bootstrap_anchor_frame.gnss_pos,
            yaw,
        )
        return (
            InitializationStatus(
                phase="READY_DIRECT_INIT",
                reason="heading_wait_timeout_zero_yaw_fallback",
                ready_mode="direct",
                heading_source="zero_yaw_fallback",
                static_alignment_ready=static_check.ready,
                static_alignment_reason=static_check.reason,
                wait_time_s=dt,
            ),
            static_check,
        )

    if dt < init_config.bootstrap_min_dt_s:
        return (
            InitializationStatus(
                phase="WAITING_BOOTSTRAP_MOTION",
                reason="bootstrap_dt_too_short",
                ready_mode=None,
                heading_source="none",
                static_alignment_ready=False,
                static_alignment_reason="heading_not_available",
                wait_time_s=dt,
            ),
            None,
        )

    displacement = frame.gnss_pos - bootstrap_anchor_frame.gnss_pos
    horizontal_displacement = float(np.linalg.norm(displacement[:2]))
    if horizontal_displacement < init_config.bootstrap_min_horizontal_displacement_m:
        return (
            InitializationStatus(
                phase="WAITING_BOOTSTRAP_MOTION",
                reason="bootstrap_displacement_too_small",
                ready_mode=None,
                heading_source="none",
                static_alignment_ready=False,
                static_alignment_reason="heading_not_available",
                wait_time_s=dt,
            ),
            None,
        )

    yaw = float(np.arctan2(displacement[1], displacement[0]))
    alignment_position = bootstrap_anchor_frame.gnss_pos
    static_check = assess_static_coarse_alignment(
        filter_engine,
        initialization_samples,
        alignment_position,
        yaw,
    )
    return (
        InitializationStatus(
            phase="READY_BOOTSTRAP_INIT",
            reason="bootstrap_position_pair_ready",
            ready_mode="bootstrap_position_pair",
            heading_source="position_pair_course",
            static_alignment_ready=static_check.ready,
            static_alignment_reason=static_check.reason,
            wait_time_s=dt,
        ),
        static_check,
    )


def initialize_filter_state(
    filter_engine: OfflineESKF,
    position: np.ndarray,
    velocity: np.ndarray,
    yaw: float,
    initialization_samples: list[ImuInitializationSample],
    alignment_position: np.ndarray | None,
    initialization_summary: dict[str, Any],
    mode: str,
) -> None:
    static_alignment_check = None
    if alignment_position is not None:
        static_alignment_check = assess_static_coarse_alignment(
            filter_engine,
            initialization_samples,
            alignment_position,
            yaw,
        )
    static_alignment = None if static_alignment_check is None else static_alignment_check.estimate

    if static_alignment is None:
        filter_engine.initialize(position, velocity, yaw)
        initialization_summary.update(
            {
                "initialization_mode": mode,
                "static_coarse_alignment_used": "false",
            }
        )
        if static_alignment_check is not None:
            initialization_summary.update(
                {
                    "static_alignment_ready": "true" if static_alignment_check.ready else "false",
                    "static_alignment_reason": static_alignment_check.reason,
                }
            )
        return

    filter_engine.initialize(
        position,
        velocity,
        yaw,
        roll=static_alignment.roll,
        pitch=static_alignment.pitch,
        gyro_bias=static_alignment.gyro_bias,
        accel_bias=static_alignment.accel_bias,
    )
    initialization_summary.update(
        {
            "initialization_mode": f"{mode}_static_coarse_alignment",
            "static_coarse_alignment_used": "true",
            "static_alignment_ready": "true",
            "static_alignment_reason": static_alignment_check.reason if static_alignment_check is not None else "static_alignment_ready",
            "static_sample_count": str(static_alignment.sample_count),
            "static_duration_s": f"{static_alignment.duration_s:.6f}",
            "static_roll_deg": f"{np.rad2deg(static_alignment.roll):.6f}",
            "static_pitch_deg": f"{np.rad2deg(static_alignment.pitch):.6f}",
            "static_gyro_bias_x": f"{static_alignment.gyro_bias[0]:.9f}",
            "static_gyro_bias_y": f"{static_alignment.gyro_bias[1]:.9f}",
            "static_gyro_bias_z": f"{static_alignment.gyro_bias[2]:.9f}",
            "static_accel_bias_x": f"{static_alignment.accel_bias[0]:.9f}",
            "static_accel_bias_y": f"{static_alignment.accel_bias[1]:.9f}",
            "static_accel_bias_z": f"{static_alignment.accel_bias[2]:.9f}",
            "static_accel_std_norm": f"{static_alignment.accel_std_norm:.9f}",
            "static_gyro_std_norm": f"{static_alignment.gyro_std_norm:.9f}",
        }
    )


def initialize_filter(
    filter_engine: OfflineESKF,
    frame,
    initialization_samples: list[ImuInitializationSample],
    initialization_summary: dict[str, Any],
    bootstrap_anchor_frame=None,
) -> bool:
    status, static_check = assess_initialization_status(
        filter_engine,
        frame,
        initialization_samples,
        bootstrap_anchor_frame=bootstrap_anchor_frame,
    )
    if status.ready_mode != "direct":
        return False
    yaw = 0.0 if status.heading_source == "zero_yaw_fallback" else resolve_initial_yaw(frame)
    initialize_filter_state(
        filter_engine,
        position=frame.gnss_pos,
        velocity=resolve_initial_velocity(frame),
        yaw=yaw,
        initialization_samples=initialization_samples,
        alignment_position=frame.gnss_pos,
        initialization_summary=initialization_summary,
        mode="direct",
    )
    summarize_initialization_status(
        InitializationStatus(
            phase="INITIALIZED",
            reason="direct_init_completed",
            ready_mode="direct",
            heading_source=status.heading_source,
            static_alignment_ready=static_check.ready,
            static_alignment_reason=static_check.reason,
            wait_time_s=status.wait_time_s,
        ),
        initialization_summary,
    )
    return True


def bootstrap_initialize_from_position_pair(
    filter_engine: OfflineESKF,
    anchor_frame,
    frame,
    initialization_samples: list[ImuInitializationSample],
    initialization_summary: dict[str, Any],
) -> bool:
    if anchor_frame is None or anchor_frame.gnss_pos is None or frame.gnss_pos is None:
        return False

    init_config = filter_engine.config.initialization
    dt = float(frame.time - anchor_frame.time)
    if dt < init_config.bootstrap_min_dt_s:
        return False

    displacement = frame.gnss_pos - anchor_frame.gnss_pos
    horizontal_displacement = float(np.linalg.norm(displacement[:2]))
    if horizontal_displacement < init_config.bootstrap_min_horizontal_displacement_m:
        return False

    velocity = displacement / dt
    yaw = frame.mag_yaw if frame.mag_yaw is not None else float(np.arctan2(velocity[1], velocity[0]))
    alignment_position = anchor_frame.gnss_pos if anchor_frame.gnss_pos is not None else frame.gnss_pos
    static_check = assess_static_coarse_alignment(
        filter_engine,
        initialization_samples,
        alignment_position,
        yaw,
    )
    initialize_filter_state(
        filter_engine,
        position=frame.gnss_pos,
        velocity=velocity,
        yaw=yaw,
        initialization_samples=initialization_samples,
        alignment_position=alignment_position,
        initialization_summary=initialization_summary,
        mode="bootstrap_position_pair",
    )
    summarize_initialization_status(
        InitializationStatus(
            phase="INITIALIZED",
            reason="bootstrap_init_completed",
            ready_mode="bootstrap_position_pair",
            heading_source="position_pair_course" if frame.mag_yaw is None else "mag_yaw",
            static_alignment_ready=static_check.ready,
            static_alignment_reason=static_check.reason,
            wait_time_s=dt,
        ),
        initialization_summary,
    )
    return True

