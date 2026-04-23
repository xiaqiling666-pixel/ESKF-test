from __future__ import annotations

import numpy as np

from ..core.state import ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementPolicy, MeasurementUpdate


class BarometerMeasurement(MeasurementModel):
    name = "baro"
    freshness_timeout_s = 0.25

    def is_available(self, frame) -> bool:
        return frame.baro_h is not None

    def policy(self, filter_engine) -> MeasurementPolicy:
        thresholds = filter_engine.config.innovation_management
        return MeasurementPolicy(
            adapt_threshold=thresholds.baro_nis_adapt_threshold,
            reject_threshold=thresholds.baro_nis_reject_threshold,
        )

    def build_update(self, filter_engine, frame) -> MeasurementUpdate | None:
        if frame.baro_h is None:
            return None

        residual = np.array([frame.baro_h - filter_engine.state.position[2]])
        H = np.zeros((1, ERROR_STATE_DIM))
        H[0, ERROR_STATE.position.start + 2] = 1.0
        std = filter_engine.config.measurement_noise.baro_std
        return MeasurementUpdate(
            residual=residual,
            H=H,
            base_R=np.array([[std**2]]),
            innovation_value=float(abs(residual[0])),
        )
