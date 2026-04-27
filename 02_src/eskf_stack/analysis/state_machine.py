from __future__ import annotations

from dataclasses import dataclass

from .quality import CovarianceHealthStatus, SensorStatus


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    reason: str


@dataclass(frozen=True)
class ModeThresholds:
    gnss_reject_streak_degraded: int = 3
    gnss_reject_bypass_streak_degraded: int = 2
    gnss_adaptive_streak_degraded: int = 3
    auxiliary_reject_streak_degraded: int = 2
    auxiliary_reject_bypass_streak_degraded: int = 2
    auxiliary_adaptive_streak_degraded: int = 3


@dataclass(frozen=True)
class ModeTrackerSnapshot:
    mode: str
    reason: str
    target_mode: str
    target_reason: str
    candidate_mode: str | None
    candidate_reason: str | None
    mode_hold_s: float
    candidate_hold_s: float
    transition_pending: bool


class ModeStateTracker:
    def __init__(
        self,
        min_mode_hold_s: float = 0.4,
        confirmation_time_s: dict[str, float] | None = None,
    ) -> None:
        self.min_mode_hold_s = min_mode_hold_s
        self.confirmation_time_s = confirmation_time_s or {
            "GNSS_STABLE": 0.3,
            "GNSS_AVAILABLE": 0.3,
            "GNSS_DEGRADED": 0.12,
            "RECOVERING": 0.2,
            "INERTIAL_HOLD": 0.5,
            "DEGRADED": 0.2,
        }
        self.current_decision: ModeDecision | None = None
        self.mode_enter_time: float | None = None
        self.candidate_decision: ModeDecision | None = None
        self.candidate_since_time: float | None = None

    def _confirmation_duration(self, target_mode: str) -> float:
        return self.confirmation_time_s.get(target_mode, 0.3)

    def _recovery_bridge_decision(self, target_decision: ModeDecision) -> ModeDecision:
        if self.current_decision is None:
            return target_decision
        degraded_modes = {"GNSS_DEGRADED", "INERTIAL_HOLD", "DEGRADED"}
        recovering_targets = {"GNSS_AVAILABLE", "GNSS_STABLE"}
        if self.current_decision.mode in degraded_modes and target_decision.mode in recovering_targets:
            if target_decision.mode == "GNSS_STABLE":
                return ModeDecision("RECOVERING", "restoring_gnss_stability")
            return ModeDecision("RECOVERING", "reacquiring_gnss")
        return target_decision

    def step(self, current_time: float, target_decision: ModeDecision) -> ModeTrackerSnapshot:
        if self.current_decision is None or self.mode_enter_time is None:
            self.current_decision = target_decision
            self.mode_enter_time = current_time
            return ModeTrackerSnapshot(
                mode=target_decision.mode,
                reason=target_decision.reason,
                target_mode=target_decision.mode,
                target_reason=target_decision.reason,
                candidate_mode=None,
                candidate_reason=None,
                mode_hold_s=0.0,
                candidate_hold_s=0.0,
                transition_pending=False,
            )

        transition_target = self._recovery_bridge_decision(target_decision)
        mode_hold_s = max(0.0, current_time - self.mode_enter_time)

        if transition_target.mode == self.current_decision.mode:
            self.current_decision = transition_target
            self.candidate_decision = None
            self.candidate_since_time = None
            return ModeTrackerSnapshot(
                mode=self.current_decision.mode,
                reason=self.current_decision.reason,
                target_mode=target_decision.mode,
                target_reason=target_decision.reason,
                candidate_mode=None,
                candidate_reason=None,
                mode_hold_s=mode_hold_s,
                candidate_hold_s=0.0,
                transition_pending=False,
            )

        if self.candidate_decision != transition_target:
            self.candidate_decision = transition_target
            self.candidate_since_time = current_time

        candidate_hold_s = 0.0 if self.candidate_since_time is None else max(0.0, current_time - self.candidate_since_time)
        can_transition = (
            mode_hold_s >= self.min_mode_hold_s
            and candidate_hold_s >= self._confirmation_duration(transition_target.mode)
        )

        if can_transition and self.candidate_decision is not None:
            self.current_decision = self.candidate_decision
            self.mode_enter_time = current_time
            self.candidate_decision = None
            self.candidate_since_time = None
            return ModeTrackerSnapshot(
                mode=self.current_decision.mode,
                reason=self.current_decision.reason,
                target_mode=target_decision.mode,
                target_reason=target_decision.reason,
                candidate_mode=None,
                candidate_reason=None,
                mode_hold_s=0.0,
                candidate_hold_s=candidate_hold_s,
                transition_pending=False,
            )

        return ModeTrackerSnapshot(
            mode=self.current_decision.mode,
            reason=self.current_decision.reason,
            target_mode=target_decision.mode,
            target_reason=target_decision.reason,
            candidate_mode=None if self.candidate_decision is None else self.candidate_decision.mode,
            candidate_reason=None if self.candidate_decision is None else self.candidate_decision.reason,
            mode_hold_s=mode_hold_s,
            candidate_hold_s=candidate_hold_s,
            transition_pending=self.candidate_decision is not None,
        )


def determine_mode(
    sensor_status: SensorStatus,
    quality_score: float,
    covariance_health: CovarianceHealthStatus,
    thresholds: ModeThresholds | None = None,
) -> ModeDecision:
    thresholds = thresholds or ModeThresholds()
    gnss_reject_streak = max(sensor_status.gnss_pos_reject_streak, sensor_status.gnss_vel_reject_streak)
    gnss_reject_bypass_streak = max(
        sensor_status.gnss_pos_reject_bypass_streak,
        sensor_status.gnss_vel_reject_bypass_streak,
    )
    auxiliary_reject_streak = max(sensor_status.baro_reject_streak, sensor_status.mag_reject_streak)
    auxiliary_reject_bypass_streak = max(
        sensor_status.baro_reject_bypass_streak,
        sensor_status.mag_reject_bypass_streak,
    )
    gnss_adaptive_streak = max(sensor_status.gnss_pos_adaptive_streak, sensor_status.gnss_vel_adaptive_streak)
    auxiliary_adaptive_streak = max(sensor_status.baro_adaptive_streak, sensor_status.mag_adaptive_streak)
    recent_any_gnss = sensor_status.recent_gnss_pos or sensor_status.recent_gnss_vel
    recent_any_auxiliary = sensor_status.recent_baro or sensor_status.recent_mag
    covariance_caution = covariance_health.caution
    covariance_unhealthy = covariance_health.unhealthy
    persistent_caution = covariance_health.caution_duration_s >= 0.4
    persistent_unhealthy = covariance_health.unhealthy_duration_s >= 0.2

    if sensor_status.recent_gnss_pos and sensor_status.recent_gnss_vel:
        if gnss_reject_streak >= thresholds.gnss_reject_streak_degraded:
            return ModeDecision("GNSS_DEGRADED", "gnss_rejection_streak")
        if gnss_reject_bypass_streak >= thresholds.gnss_reject_bypass_streak_degraded:
            return ModeDecision("GNSS_DEGRADED", "gnss_reject_bypass_streak")
        if gnss_adaptive_streak >= thresholds.gnss_adaptive_streak_degraded and quality_score < 75.0:
            return ModeDecision("GNSS_DEGRADED", "gnss_adaptive_scaling_streak")
        if quality_score < 65.0:
            return ModeDecision("GNSS_DEGRADED", "quality_drop_under_gnss")
        if covariance_caution and persistent_caution and quality_score < 65.0:
            return ModeDecision("GNSS_DEGRADED", covariance_health.reason)
        return ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance")

    if recent_any_gnss:
        if gnss_reject_streak >= thresholds.gnss_reject_streak_degraded:
            return ModeDecision("GNSS_DEGRADED", "partial_gnss_with_rejections")
        if gnss_reject_bypass_streak >= thresholds.gnss_reject_bypass_streak_degraded and quality_score < 75.0:
            return ModeDecision("GNSS_DEGRADED", "gnss_reject_bypass_streak")
        if gnss_adaptive_streak >= max(2, thresholds.gnss_adaptive_streak_degraded - 1) and quality_score < 65.0:
            return ModeDecision("GNSS_DEGRADED", "gnss_adaptive_scaling_streak")
        if not recent_any_auxiliary and sensor_status.auxiliary_outage_s > 1.0:
            return ModeDecision("GNSS_DEGRADED", "partial_gnss_without_aux_support")
        if (
            auxiliary_reject_streak >= thresholds.auxiliary_reject_streak_degraded
            or auxiliary_reject_bypass_streak >= thresholds.auxiliary_reject_bypass_streak_degraded
            or auxiliary_adaptive_streak >= thresholds.auxiliary_adaptive_streak_degraded
        ) and quality_score < 55.0:
            return ModeDecision("GNSS_DEGRADED", "auxiliary_sensor_instability")
        if covariance_unhealthy and persistent_unhealthy:
            return ModeDecision("GNSS_DEGRADED", covariance_health.reason)
        return ModeDecision("GNSS_AVAILABLE", "partial_or_recovering_gnss")

    if sensor_status.gnss_outage_s <= 2.0 and not recent_any_auxiliary:
        return ModeDecision("DEGRADED", "gnss_outage_without_aux_support")
    if sensor_status.gnss_outage_s <= 2.0 and (
        auxiliary_reject_streak >= thresholds.auxiliary_reject_streak_degraded
        or auxiliary_reject_bypass_streak >= thresholds.auxiliary_reject_bypass_streak_degraded
        or auxiliary_adaptive_streak >= thresholds.auxiliary_adaptive_streak_degraded
    ) and quality_score < 45.0:
        return ModeDecision("DEGRADED", "auxiliary_sensor_instability")
    if sensor_status.gnss_outage_s <= 2.0 and recent_any_auxiliary and not persistent_unhealthy:
        return ModeDecision("INERTIAL_HOLD", "short_gnss_outage")
    if sensor_status.gnss_outage_s <= 5.0 and not recent_any_auxiliary and sensor_status.auxiliary_outage_s > 1.0:
        return ModeDecision("DEGRADED", "extended_outage_without_aux_support")
    if sensor_status.gnss_outage_s <= 5.0 and (
        auxiliary_reject_streak >= thresholds.auxiliary_reject_streak_degraded
        or auxiliary_reject_bypass_streak >= thresholds.auxiliary_reject_bypass_streak_degraded
        or auxiliary_adaptive_streak >= thresholds.auxiliary_adaptive_streak_degraded
    ) and quality_score < 35.0:
        return ModeDecision("DEGRADED", "auxiliary_sensor_instability")
    if sensor_status.gnss_outage_s <= 5.0 and recent_any_auxiliary and not persistent_unhealthy:
        return ModeDecision("INERTIAL_HOLD", "extended_gnss_outage")
    if covariance_unhealthy and persistent_unhealthy:
        return ModeDecision("DEGRADED", covariance_health.reason)
    return ModeDecision("DEGRADED", "long_gnss_outage")
