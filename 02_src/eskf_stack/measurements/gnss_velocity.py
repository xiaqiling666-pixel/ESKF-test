from __future__ import annotations

import numpy as np

from ..core.state import AXIS_DIM, ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementPolicy, MeasurementUpdate


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

        residual = frame.gnss_vel - filter_engine.state.velocity
        H = np.zeros((AXIS_DIM, ERROR_STATE_DIM))
        H[:, ERROR_STATE.velocity] = np.eye(AXIS_DIM)
        std = filter_engine.config.measurement_noise.gnss_vel_std
        return MeasurementUpdate(
            residual=residual,
            H=H,
            base_R=np.eye(AXIS_DIM) * (std**2),
            innovation_value=float(np.linalg.norm(residual)),
        )
