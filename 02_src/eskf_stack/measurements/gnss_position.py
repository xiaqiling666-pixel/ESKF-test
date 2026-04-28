from __future__ import annotations

import numpy as np

from ..core.math_utils import quat_to_rotmat, skew
from ..core.state import AXIS_DIM, ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementPolicy, MeasurementUpdate


def _gnss_lever_arm_body(filter_engine) -> np.ndarray:
    return np.asarray(filter_engine.config.gnss_lever_arm_body_m, dtype=float)


class GnssPositionMeasurement(MeasurementModel):
    name = "gnss_pos"
    freshness_timeout_s = 0.6

    def is_available(self, frame) -> bool:
        return frame.gnss_pos is not None

    def policy(self, filter_engine) -> MeasurementPolicy:
        thresholds = filter_engine.config.innovation_management
        return MeasurementPolicy(
            adapt_threshold=thresholds.gnss_pos_nis_adapt_threshold,
            reject_threshold=thresholds.gnss_pos_nis_reject_threshold,
        )

    def build_update(self, filter_engine, frame) -> MeasurementUpdate | None:
        if frame.gnss_pos is None:
            return None

        rotation_nav_from_body = quat_to_rotmat(filter_engine.state.quaternion)
        lever_arm_body = _gnss_lever_arm_body(filter_engine)
        lever_arm_nav = rotation_nav_from_body @ lever_arm_body
        predicted_gnss_position = filter_engine.state.position + lever_arm_nav
        residual = frame.gnss_pos - predicted_gnss_position
        H = np.zeros((AXIS_DIM, ERROR_STATE_DIM))
        H[:, ERROR_STATE.position] = np.eye(AXIS_DIM)
        H[:, ERROR_STATE.attitude] = -rotation_nav_from_body @ skew(lever_arm_body)
        measurement_noise = filter_engine.config.measurement_noise
        horizontal_std = measurement_noise.gnss_pos_std
        vertical_std = measurement_noise.gnss_pos_vertical_std
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
