from __future__ import annotations

import numpy as np

from ..core.state import ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementResult, innovation_covariance, mahalanobis_squared


class BarometerMeasurement(MeasurementModel):
    name = "baro"
    freshness_timeout_s = 0.25

    def is_available(self, frame) -> bool:
        return frame.baro_h is not None

    def apply(self, filter_engine, frame) -> MeasurementResult:
        if frame.baro_h is None:
            return MeasurementResult(name=self.name, available=False, used=False)

        residual = np.array([frame.baro_h - filter_engine.state.position[2]])
        H = np.zeros((1, ERROR_STATE_DIM))
        H[0, ERROR_STATE.position.start + 2] = 1.0
        std = filter_engine.config.measurement_noise.baro_std
        R = np.array([[std**2]])
        nis = mahalanobis_squared(residual, innovation_covariance(filter_engine, H, R))
        filter_engine.apply_linear_update(residual, H, R)
        return MeasurementResult(
            name=self.name,
            available=True,
            used=True,
            innovation_value=float(abs(residual[0])),
            nis=nis,
        )
