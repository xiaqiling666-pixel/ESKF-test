from __future__ import annotations

import numpy as np

from ..core.math_utils import quat_to_rotmat, skew
from ..core.state import AXIS_DIM, ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementPolicy, MeasurementUpdate
from .gnss_position import _gnss_lever_arm_body


class GnssVelocityMeasurement(MeasurementModel):
    name = "gnss_vel"
    freshness_timeout_s = 0.6

    def is_available(self, frame) -> bool:
        return frame.gnss_vel is not None

    def policy(self, filter_engine) -> MeasurementPolicy:
        thresholds = filter_engine.config.innovation_management
        return MeasurementPolicy(
            adapt_threshold=thresholds.gnss_vel_nis_adapt_threshold,
            reject_threshold=thresholds.gnss_vel_nis_reject_threshold,
        )

    def build_update(self, filter_engine, frame) -> MeasurementUpdate | None:
        if frame.gnss_vel is None:
            return None

        rotation_nav_from_body = quat_to_rotmat(filter_engine.state.quaternion)
        lever_arm_body = _gnss_lever_arm_body(filter_engine)
        body_angular_rate = np.zeros(AXIS_DIM, dtype=float)
        if frame.gyro is not None:
            body_angular_rate = np.asarray(frame.gyro, dtype=float) - filter_engine.state.gyro_bias
        lever_arm_velocity_body = np.cross(body_angular_rate, lever_arm_body)
        predicted_gnss_velocity = filter_engine.state.velocity + rotation_nav_from_body @ lever_arm_velocity_body
        residual = frame.gnss_vel - predicted_gnss_velocity
        H = np.zeros((AXIS_DIM, ERROR_STATE_DIM))
        H[:, ERROR_STATE.velocity] = np.eye(AXIS_DIM)
        H[:, ERROR_STATE.attitude] = -rotation_nav_from_body @ skew(lever_arm_velocity_body)
        H[:, ERROR_STATE.gyro_bias] = rotation_nav_from_body @ skew(lever_arm_body)
        measurement_noise = filter_engine.config.measurement_noise
        horizontal_std = measurement_noise.gnss_vel_std
        vertical_std = measurement_noise.gnss_vel_vertical_std
        return MeasurementUpdate(
            residual=residual,
            H=H,
            base_R=np.diag(
                [
                    horizontal_std**2,
                    horizontal_std**2,
                    vertical_std**2,
                ]
            ),
            innovation_value=float(np.linalg.norm(residual)),
        )
