from __future__ import annotations

import numpy as np

from ..core.state import AXIS_DIM, ERROR_STATE, ERROR_STATE_DIM
from .base import MeasurementModel, MeasurementResult, innovation_covariance, mahalanobis_squared


class GnssVelocityMeasurement(MeasurementModel):
    name = "gnss_vel"
    freshness_timeout_s = 0.6

    def is_available(self, frame) -> bool:
        return frame.gnss_vel is not None

    def apply(self, filter_engine, frame) -> MeasurementResult:
        if frame.gnss_vel is None:
            return MeasurementResult(name=self.name, available=False, used=False)

        residual = frame.gnss_vel - filter_engine.state.velocity
        H = np.zeros((AXIS_DIM, ERROR_STATE_DIM))
        H[:, ERROR_STATE.velocity] = np.eye(AXIS_DIM)
        std = filter_engine.config.measurement_noise.gnss_vel_std
        base_R = np.eye(AXIS_DIM) * (std**2)
        nis = mahalanobis_squared(residual, innovation_covariance(filter_engine, H, base_R))
        thresholds = filter_engine.config.innovation_management

        if nis > thresholds.gnss_vel_nis_reject_threshold:
            return MeasurementResult(
                name=self.name,
                available=True,
                used=False,
                innovation_value=float(np.linalg.norm(residual)),
                nis=nis,
                rejected=True,
            )

        adaptation_scale = 1.0
        if nis > thresholds.gnss_vel_nis_adapt_threshold:
            adaptation_scale = nis / thresholds.gnss_vel_nis_adapt_threshold

        filter_engine.apply_linear_update(residual, H, base_R * adaptation_scale)
        return MeasurementResult(
            name=self.name,
            available=True,
            used=True,
            innovation_value=float(np.linalg.norm(residual)),
            nis=nis,
            adaptation_scale=adaptation_scale,
        )
