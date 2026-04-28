from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..adapters.csv_dataset import ObservationFrame
from ..core.filter import OfflineESKF
from .base import (
    MeasurementDecision,
    MeasurementModel,
    MeasurementResult,
    MeasurementUpdate,
    innovation_covariance,
    mahalanobis_squared,
)


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

@dataclass(frozen=True)
class _GatedMeasurement:
    update: MeasurementUpdate
    nis: float
    mode_scale: float
    reject_bypassed: bool
    rejected: bool = False


@dataclass(frozen=True)
class _ScaledMeasurement:
    effective_R: np.ndarray
    adaptation_scale: float
    recovery_scale: float
    applied_r_scale: float
    management_mode: str


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
        decision = self.decide(filter_engine, model, frame, current_mode=current_mode)
        return self._apply_decision(filter_engine, decision)

    def decide(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        frame: ObservationFrame,
        current_mode: str | None = None,
    ) -> MeasurementDecision:
        if not model.is_available(frame):
            return MeasurementDecision(name=model.name, available=False, management_mode="unavailable")

        update = model.build_update(filter_engine, frame)
        if update is None:
            return MeasurementDecision(name=model.name, available=True, management_mode="skip")

        return self._build_decision(filter_engine, model, update, current_mode=current_mode)

    def _build_decision(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        update: MeasurementUpdate,
        current_mode: str | None = None,
    ) -> MeasurementDecision:
        policy = model.policy(filter_engine)
        gated = self._gate_measurement(
            filter_engine,
            model,
            update,
            policy,
            current_mode=current_mode,
        )
        if gated.rejected:
            self.reject_streaks[model.name] = self.reject_streaks.get(model.name, 0) + 1
            self.recovery_remaining[model.name] = 0
            return MeasurementDecision(
                name=model.name,
                available=True,
                update=update,
                effective_R=None,
                innovation_value=update.innovation_value,
                nis=gated.nis,
                rejected=True,
                reject_bypassed=False,
                mode_scale=gated.mode_scale,
                management_mode="reject",
            )
        scaled = self._scale_measurement(filter_engine, model, gated, policy)
        self.reject_streaks[model.name] = 0

        return MeasurementDecision(
            name=model.name,
            available=True,
            update=update,
            effective_R=scaled.effective_R,
            innovation_value=update.innovation_value,
            nis=gated.nis,
            reject_bypassed=gated.reject_bypassed,
            adaptation_scale=scaled.adaptation_scale,
            recovery_scale=scaled.recovery_scale,
            mode_scale=gated.mode_scale,
            applied_r_scale=scaled.applied_r_scale,
            management_mode=scaled.management_mode,
        )

    def _gate_measurement(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        update: MeasurementUpdate,
        policy,
        current_mode: str | None = None,
    ) -> _GatedMeasurement:
        mode_scale = _mode_measurement_scale(filter_engine, model.name, current_mode)
        gated_base_R = update.base_R * mode_scale
        nis = mahalanobis_squared(update.residual, innovation_covariance(filter_engine, update.H, gated_base_R))
        use_nis_rejection = _policy_enabled(filter_engine, "use_nis_rejection")
        if use_nis_rejection and policy.reject_threshold is not None and nis > policy.reject_threshold:
            return _GatedMeasurement(
                update=update,
                nis=nis,
                mode_scale=mode_scale,
                reject_bypassed=False,
                rejected=True,
            )
        reject_bypassed = bool(
            (not use_nis_rejection)
            and policy.reject_threshold is not None
            and nis > policy.reject_threshold
        )
        return _GatedMeasurement(
            update=update,
            nis=nis,
            mode_scale=mode_scale,
            reject_bypassed=reject_bypassed,
        )

    def _scale_measurement(
        self,
        filter_engine: OfflineESKF,
        model: MeasurementModel,
        gated: _GatedMeasurement,
        policy,
    ) -> _ScaledMeasurement:
        adaptation_scale = 1.0
        if (
            _policy_enabled(filter_engine, "use_adaptive_r")
            and policy.adapt_threshold is not None
            and policy.adapt_threshold > 0.0
            and gated.nis > policy.adapt_threshold
        ):
            adaptation_scale = gated.nis / policy.adapt_threshold

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

        applied_r_scale = adaptation_scale * recovery_scale * gated.mode_scale
        effective_R = gated.update.base_R * gated.mode_scale * adaptation_scale * recovery_scale
        return _ScaledMeasurement(
            effective_R=effective_R,
            adaptation_scale=adaptation_scale,
            recovery_scale=recovery_scale,
            applied_r_scale=applied_r_scale,
            management_mode=management_mode,
        )

    def _apply_decision(
        self,
        filter_engine: OfflineESKF,
        decision: MeasurementDecision,
    ) -> MeasurementResult:
        used = decision.update is not None and decision.effective_R is not None and not decision.rejected
        if used and decision.update is not None and decision.effective_R is not None:
            filter_engine.apply_linear_update(
                decision.update.residual,
                decision.update.H,
                decision.effective_R,
            )
        innovation_value = 0.0 if decision.update is None else decision.update.innovation_value
        return MeasurementResult(
            name=decision.name,
            available=decision.available,
            used=used,
            innovation_value=innovation_value,
            nis=decision.nis,
            rejected=decision.rejected,
            reject_bypassed=decision.reject_bypassed,
            adaptation_scale=decision.adaptation_scale,
            recovery_scale=decision.recovery_scale,
            mode_scale=decision.mode_scale,
            applied_r_scale=decision.applied_r_scale,
            management_mode=decision.management_mode,
        )
