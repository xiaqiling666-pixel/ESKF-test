from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.math_utils import quat_to_rotmat


@dataclass(frozen=True)
class GnssLeverArmDiagnostics:
    body_vector_m: np.ndarray
    nav_vector_m: np.ndarray
    rotational_velocity_nav_mps: np.ndarray

    @property
    def nav_norm_m(self) -> float:
        return float(np.linalg.norm(self.nav_vector_m))

    @property
    def rotational_speed_mps(self) -> float:
        return float(np.linalg.norm(self.rotational_velocity_nav_mps))


def evaluate_gnss_lever_arm_diagnostics(filter_engine, measurement_frame) -> GnssLeverArmDiagnostics:
    lever_arm_body = np.asarray(filter_engine.config.gnss_lever_arm_body_m, dtype=float)
    rotation_nav_from_body = quat_to_rotmat(filter_engine.state.quaternion)
    lever_arm_nav = rotation_nav_from_body @ lever_arm_body
    body_angular_rate = np.zeros(3, dtype=float)
    if getattr(measurement_frame, "gyro", None) is not None:
        body_angular_rate = np.asarray(measurement_frame.gyro, dtype=float) - np.asarray(
            filter_engine.state.gyro_bias,
            dtype=float,
        )
    rotational_velocity_body = np.cross(body_angular_rate, lever_arm_body)
    rotational_velocity_nav = rotation_nav_from_body @ rotational_velocity_body
    return GnssLeverArmDiagnostics(
        body_vector_m=lever_arm_body,
        nav_vector_m=lever_arm_nav,
        rotational_velocity_nav_mps=rotational_velocity_nav,
    )
