from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.csv_dataset import ObservationFrame
from ..core.filter import OfflineESKF
from .base import MeasurementModel, MeasurementResult, MeasurementUpdate, innovation_covariance, mahalanobis_squared


@dataclass
class MeasurementManager:
    reject_streaks: dict[str, int] = field(default_factory=dict)
    recovery_remaining: dict[str, int] = field(default_factory=dict)

    def _recovery_scale(self, remaining: int, window: int, max_scale: float) -> float:
        if window <= 1:
            return max(1.0, max_scale)
        ratio = float(remaining - 1) / float(window - 1)
        return 1.0 + max(0.0, ratio) * (max_scale - 1.0)

    def process(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        frame: ObservationFrame,
    ) -> MeasurementResult:
        if not model.is_available(frame):
            return MeasurementResult(name=model.name, available=False, used=False, management_mode="skip")

        update = model.build_update(filter_engine, frame)
        if update is None:
            return MeasurementResult(name=model.name, available=False, used=False, management_mode="skip")

        return self._apply_update(filter_engine, model, update)

    def _apply_update(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        update: MeasurementUpdate,
    ) -> MeasurementResult:
        policy = model.policy(filter_engine)
        nis = mahalanobis_squared(update.residual, innovation_covariance(filter_engine, update.H, update.base_R))

        if policy.reject_threshold is not None and nis > policy.reject_threshold:
            self.reject_streaks[model.name] = self.reject_streaks.get(model.name, 0) + 1
            self.recovery_remaining[model.name] = 0
            return MeasurementResult(
                name=model.name,
                available=True,
                used=False,
                innovation_value=update.innovation_value,
                nis=nis,
                rejected=True,
                management_mode="reject",
            )

        adaptation_scale = 1.0
        if policy.adapt_threshold is not None and policy.adapt_threshold > 0.0 and nis > policy.adapt_threshold:
            adaptation_scale = nis / policy.adapt_threshold

        prior_reject_streak = self.reject_streaks.get(model.name, 0)
        recovery_remaining = self.recovery_remaining.get(model.name, 0)
        management_mode = "update"
        recovery_scale = 1.0

        if policy.recovery_window > 0 and (
            recovery_remaining > 0 or prior_reject_streak >= policy.recovery_trigger_reject_streak
        ):
            if recovery_remaining <= 0:
                recovery_remaining = policy.recovery_window
            recovery_scale = self._recovery_scale(
                recovery_remaining,
                policy.recovery_window,
                policy.recovery_max_scale,
            )
            self.recovery_remaining[model.name] = recovery_remaining - 1
            management_mode = "recover"
        else:
            self.recovery_remaining[model.name] = 0

        applied_r_scale = adaptation_scale * recovery_scale
        filter_engine.apply_linear_update(update.residual, update.H, update.base_R * applied_r_scale)
        self.reject_streaks[model.name] = 0

        return MeasurementResult(
            name=model.name,
            available=True,
            used=True,
            innovation_value=update.innovation_value,
            nis=nis,
            adaptation_scale=adaptation_scale,
            recovery_scale=recovery_scale,
            applied_r_scale=applied_r_scale,
            management_mode=management_mode,
        )
