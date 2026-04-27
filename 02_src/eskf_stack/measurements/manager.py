from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.csv_dataset import ObservationFrame
from ..core.filter import OfflineESKF
from .base import MeasurementModel, MeasurementResult, MeasurementUpdate, innovation_covariance, mahalanobis_squared


def _policy_enabled(filter_engine: OfflineESKF, field_name: str) -> bool:
    config = getattr(filter_engine, "config", None)
    if config is None or not hasattr(config, "fusion_policy"):
        return True
    fusion_policy = config.fusion_policy
    if not hasattr(fusion_policy, field_name):
        raise AttributeError(f"FusionPolicy 缺少字段: {field_name}")
    return bool(getattr(fusion_policy, field_name))


def _mode_measurement_scale(filter_engine: OfflineESKF, sensor_name: str, current_mode: str | None) -> float:
    if not current_mode:
        return 1.0

    scaling = getattr(getattr(filter_engine, "config", None), "mode_measurement_scaling", None)
    if scaling is None or not getattr(scaling, "enabled", False):
        return 1.0

    scale_map = None
    if sensor_name == "gnss_pos":
        scale_map = scaling.gnss_pos
    elif sensor_name == "gnss_vel":
        scale_map = scaling.gnss_vel

    if not isinstance(scale_map, dict):
        return 1.0
    return max(1.0, float(scale_map.get(current_mode, 1.0)))


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
        current_mode: str | None = None,
    ) -> MeasurementResult:
        if not model.is_available(frame):
            return MeasurementResult(name=model.name, available=False, used=False, management_mode="skip")

        update = model.build_update(filter_engine, frame)
        if update is None:
            return MeasurementResult(name=model.name, available=False, used=False, management_mode="skip")

        return self._apply_update(filter_engine, model, update, current_mode=current_mode)

    def _apply_update(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        update: MeasurementUpdate,
        current_mode: str | None = None,
    ) -> MeasurementResult:
        policy = model.policy(filter_engine)
        mode_scale = _mode_measurement_scale(filter_engine, model.name, current_mode)
        effective_base_R = update.base_R * mode_scale
        nis = mahalanobis_squared(update.residual, innovation_covariance(filter_engine, update.H, effective_base_R))
        use_nis_rejection = _policy_enabled(filter_engine, "use_nis_rejection")
        reject_bypassed = bool(
            (not use_nis_rejection)
            and policy.reject_threshold is not None
            and nis > policy.reject_threshold
        )

        if use_nis_rejection and policy.reject_threshold is not None and nis > policy.reject_threshold:
            self.reject_streaks[model.name] = self.reject_streaks.get(model.name, 0) + 1
            self.recovery_remaining[model.name] = 0
            return MeasurementResult(
                name=model.name,
                available=True,
                used=False,
                innovation_value=update.innovation_value,
                nis=nis,
                rejected=True,
                reject_bypassed=False,
                mode_scale=mode_scale,
                management_mode="reject",
            )

        adaptation_scale = 1.0
        if (
            _policy_enabled(filter_engine, "use_adaptive_r")
            and policy.adapt_threshold is not None
            and policy.adapt_threshold > 0.0
            and nis > policy.adapt_threshold
        ):
            adaptation_scale = nis / policy.adapt_threshold

        prior_reject_streak = self.reject_streaks.get(model.name, 0)
        recovery_remaining = self.recovery_remaining.get(model.name, 0)
        management_mode = "update"
        recovery_scale = 1.0

        if _policy_enabled(filter_engine, "use_recovery_scale") and policy.recovery_window > 0 and (
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

        applied_r_scale = adaptation_scale * recovery_scale * mode_scale
        filter_engine.apply_linear_update(
            update.residual,
            update.H,
            effective_base_R * adaptation_scale * recovery_scale,
        )
        self.reject_streaks[model.name] = 0

        return MeasurementResult(
            name=model.name,
            available=True,
            used=True,
            innovation_value=update.innovation_value,
            nis=nis,
            reject_bypassed=reject_bypassed,
            adaptation_scale=adaptation_scale,
            recovery_scale=recovery_scale,
            mode_scale=mode_scale,
            applied_r_scale=applied_r_scale,
            management_mode=management_mode,
        )
