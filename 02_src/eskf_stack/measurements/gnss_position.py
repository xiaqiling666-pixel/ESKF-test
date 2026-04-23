from __future__ import annotations

import numpy as np

from ..core.state import AXIS_DIM, ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementPolicy, MeasurementUpdate


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

        residual = frame.gnss_pos - filter_engine.state.position
        H = np.zeros((AXIS_DIM, ERROR_STATE_DIM))
        H[:, ERROR_STATE.position] = np.eye(AXIS_DIM)
        std = filter_engine.config.measurement_noise.gnss_pos_std
        return MeasurementUpdate(
            residual=residual,
            H=H,
            base_R=np.eye(AXIS_DIM) * (std**2),
            innovation_value=float(np.linalg.norm(residual)),
        )
