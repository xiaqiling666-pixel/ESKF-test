from __future__ import annotations

import numpy as np

from ..core.math_utils import quat_multiply, quat_normalize, quat_to_euler, rotvec_to_quat, wrap_angle
from ..core.state import ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementUpdate


def _yaw_error_jacobian(filter_engine) -> np.ndarray:
    base_yaw = quat_to_euler(filter_engine.state.quaternion)[2]
    jacobian = np.zeros(3)
    epsilon = 1e-6
    for axis in range(3):
        delta = np.zeros(3)
        delta[axis] = epsilon
        perturbed = quat_normalize(quat_multiply(filter_engine.state.quaternion, rotvec_to_quat(delta)))
        perturbed_yaw = quat_to_euler(perturbed)[2]
        jacobian[axis] = wrap_angle(perturbed_yaw - base_yaw) / epsilon
    return jacobian


class MagYawMeasurement(MeasurementModel):
    name = "mag"
    freshness_timeout_s = 0.6

    def is_available(self, frame) -> bool:
        return frame.mag_yaw is not None

    def build_update(self, filter_engine, frame) -> MeasurementUpdate | None:
        if frame.mag_yaw is None:
            return None

        current_yaw = quat_to_euler(filter_engine.state.quaternion)[2]
        residual = np.array([wrap_angle(frame.mag_yaw - current_yaw)])
        H = np.zeros((1, ERROR_STATE_DIM))
        H[0, ERROR_STATE.attitude] = _yaw_error_jacobian(filter_engine)
        std = np.deg2rad(filter_engine.config.measurement_noise.yaw_std_deg)
        return MeasurementUpdate(
            residual=residual,
            H=H,
            base_R=np.array([[std**2]]),
            innovation_value=float(abs(np.rad2deg(residual[0]))),
        )
