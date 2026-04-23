from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import InitializationConfig
from .math_utils import euler_to_quat, quat_to_rotmat
from .navigation import LocalNavigationEnvironment, resolve_local_navigation_environment


@dataclass(frozen=True)
class ImuInitializationSample:
    time: float
    accel: np.ndarray
    gyro: np.ndarray


@dataclass(frozen=True)
class StaticAlignmentEstimate:
    roll: float
    pitch: float
    yaw: float
    gyro_bias: np.ndarray
    accel_bias: np.ndarray
    sample_count: int
    duration_s: float
    accel_std_norm: float
    gyro_std_norm: float


class StaticCoarseInitializer:
    def __init__(self, config: InitializationConfig) -> None:
        self.config = config

    def _select_stationary_window(
        self,
        samples: list[ImuInitializationSample],
    ) -> list[ImuInitializationSample]:
        if not samples:
            return []
        start_time = float(samples[0].time)
        end_time = start_time + self.config.static_window_duration_s
        window = [sample for sample in samples if float(sample.time) <= end_time]
        if len(window) < self.config.static_window_min_samples:
            return []
        return window

    def estimate(
        self,
        samples: list[ImuInitializationSample],
        position: np.ndarray,
        base_environment: LocalNavigationEnvironment,
        use_wgs84_gravity: bool,
        use_earth_rotation: bool,
        yaw: float,
    ) -> StaticAlignmentEstimate | None:
        if not self.config.static_coarse_alignment_enabled:
            return None

        stationary_window = self._select_stationary_window(samples)
        if not stationary_window:
            return None

        accel_samples = np.vstack([sample.accel for sample in stationary_window])
        gyro_samples = np.vstack([sample.gyro for sample in stationary_window])
        accel_mean = accel_samples.mean(axis=0)
        gyro_mean = gyro_samples.mean(axis=0)
        accel_std_norm = float(np.linalg.norm(accel_samples.std(axis=0)))
        gyro_std_norm = float(np.linalg.norm(gyro_samples.std(axis=0)))

        if accel_std_norm > self.config.static_max_accel_std_mps2:
            return None
        if gyro_std_norm > self.config.static_max_gyro_std_radps:
            return None

        static_environment = resolve_local_navigation_environment(
            base_environment,
            position,
            np.zeros(3, dtype=float),
            use_wgs84_gravity=use_wgs84_gravity,
            use_earth_rotation=use_earth_rotation,
        )
        expected_specific_force_nav = -static_environment.gravity_vector
        expected_specific_force_norm = float(np.linalg.norm(expected_specific_force_nav))
        if abs(float(np.linalg.norm(accel_mean)) - expected_specific_force_norm) > self.config.static_gravity_norm_tolerance_mps2:
            return None

        horizontal_force = float(np.hypot(accel_mean[1], accel_mean[2]))
        roll = float(np.arctan2(accel_mean[1], accel_mean[2]))
        pitch = float(np.arctan2(-accel_mean[0], horizontal_force))

        quaternion = euler_to_quat(roll, pitch, yaw)
        rotation = quat_to_rotmat(quaternion)
        expected_specific_force_body = rotation.T @ expected_specific_force_nav
        expected_nav_rate_body = rotation.T @ static_environment.omega_in_nav

        duration_s = float(stationary_window[-1].time - stationary_window[0].time)
        return StaticAlignmentEstimate(
            roll=roll,
            pitch=pitch,
            yaw=float(yaw),
            gyro_bias=gyro_mean - expected_nav_rate_body,
            accel_bias=accel_mean - expected_specific_force_body,
            sample_count=len(stationary_window),
            duration_s=duration_s,
            accel_std_norm=accel_std_norm,
            gyro_std_norm=gyro_std_norm,
        )
