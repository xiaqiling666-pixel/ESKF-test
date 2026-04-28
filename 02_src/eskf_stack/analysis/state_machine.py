from __future__ import annotations

from dataclasses import dataclass

from .quality import CovarianceHealthStatus, MeasurementSupportSummary, SensorStatus, summarize_measurement_support


@dataclass(frozen=True)
class ModeDecision:
    mode: str
    reason: str


@dataclass(frozen=True)
class ModeThresholds:
    gnss_reject_streak_degraded: int = 3
    gnss_reject_bypass_streak_degraded: int = 2
    gnss_adaptive_streak_degraded: int = 3
    gnss_skip_streak_degraded: int = 2
    auxiliary_reject_streak_degraded: int = 2
    auxiliary_reject_bypass_streak_degraded: int = 2
    auxiliary_adaptive_streak_degraded: int = 3
    auxiliary_skip_streak_degraded: int = 2
    gnss_reject_bypass_quality_floor: float = 75.0
    gnss_adaptive_quality_floor: float = 75.0
    full_gnss_quality_floor: float = 65.0
    partial_gnss_adaptive_quality_floor: float = 65.0
    partial_gnss_aux_instability_quality_floor: float = 55.0
    short_outage_aux_instability_quality_floor: float = 45.0
    extended_outage_aux_instability_quality_floor: float = 35.0
    gnss_short_outage_threshold_s: float = 2.0
    gnss_extended_outage_threshold_s: float = 5.0
    auxiliary_available_outage_threshold_s: float = 1.0
    covariance_caution_duration_threshold_s: float = 0.4
    covariance_unhealthy_duration_threshold_s: float = 0.2


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
    support_summary: MeasurementSupportSummary | None = None,
) -> ModeDecision:
    thresholds = thresholds or ModeThresholds()
    support = support_summary or summarize_measurement_support(
        sensor_status,
        gnss_skip_streak_degraded=thresholds.gnss_skip_streak_degraded,
        auxiliary_reject_streak_degraded=thresholds.auxiliary_reject_streak_degraded,
        auxiliary_reject_bypass_streak_degraded=thresholds.auxiliary_reject_bypass_streak_degraded,
        auxiliary_adaptive_streak_degraded=thresholds.auxiliary_adaptive_streak_degraded,
        auxiliary_skip_streak_degraded=thresholds.auxiliary_skip_streak_degraded,
    )
    gnss_reject_streak = max(sensor_status.gnss_pos_reject_streak, sensor_status.gnss_vel_reject_streak)
    gnss_reject_bypass_streak = max(
        sensor_status.gnss_pos_reject_bypass_streak,
        sensor_status.gnss_vel_reject_bypass_streak,
    )
    gnss_adaptive_streak = max(sensor_status.gnss_pos_adaptive_streak, sensor_status.gnss_vel_adaptive_streak)
    covariance_caution = covariance_health.caution
    covariance_unhealthy = covariance_health.unhealthy
    persistent_caution = covariance_health.caution_duration_s >= thresholds.covariance_caution_duration_threshold_s
    persistent_unhealthy = covariance_health.unhealthy_duration_s >= thresholds.covariance_unhealthy_duration_threshold_s

    if support.full_gnss_support:
        if gnss_reject_streak >= thresholds.gnss_reject_streak_degraded:
            return ModeDecision("GNSS_DEGRADED", "gnss_rejection_streak")
        if gnss_reject_bypass_streak >= thresholds.gnss_reject_bypass_streak_degraded:
            return ModeDecision("GNSS_DEGRADED", "gnss_reject_bypass_streak")
        if (
            gnss_adaptive_streak >= thresholds.gnss_adaptive_streak_degraded
            and quality_score < thresholds.gnss_adaptive_quality_floor
        ):
            return ModeDecision("GNSS_DEGRADED", "gnss_adaptive_scaling_streak")
        if quality_score < thresholds.full_gnss_quality_floor:
            return ModeDecision("GNSS_DEGRADED", "quality_drop_under_gnss")
        if covariance_caution and persistent_caution:
            if covariance_health.reason != "multiple_caution_sigmas" or quality_score < thresholds.full_gnss_quality_floor:
                return ModeDecision("GNSS_DEGRADED", covariance_health.reason)
        return ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance")

    if support.recent_any_gnss:
        if gnss_reject_streak >= thresholds.gnss_reject_streak_degraded:
            return ModeDecision("GNSS_DEGRADED", "partial_gnss_with_rejections")
        if (
            gnss_reject_bypass_streak >= thresholds.gnss_reject_bypass_streak_degraded
            and quality_score < thresholds.gnss_reject_bypass_quality_floor
        ):
            return ModeDecision("GNSS_DEGRADED", "gnss_reject_bypass_streak")
        if (
            gnss_adaptive_streak >= max(2, thresholds.gnss_adaptive_streak_degraded - 1)
            and quality_score < thresholds.partial_gnss_adaptive_quality_floor
        ):
            return ModeDecision("GNSS_DEGRADED", "gnss_adaptive_scaling_streak")
        if not support.recent_any_auxiliary and support.recent_any_auxiliary_available:
            return ModeDecision("GNSS_DEGRADED", "auxiliary_available_without_successful_updates")
        if not support.recent_any_auxiliary and sensor_status.auxiliary_available_outage_s > thresholds.auxiliary_available_outage_threshold_s:
            return ModeDecision("GNSS_DEGRADED", "partial_gnss_without_aux_support")
        if (
            support.auxiliary_instability_without_backup
            and quality_score < thresholds.partial_gnss_aux_instability_quality_floor
        ):
            return ModeDecision("GNSS_DEGRADED", "auxiliary_sensor_instability")
        if covariance_unhealthy and persistent_unhealthy:
            return ModeDecision("GNSS_DEGRADED", covariance_health.reason)
        return ModeDecision("GNSS_AVAILABLE", "partial_or_recovering_gnss")

    if support.gnss_available_without_successful_updates:
        return ModeDecision("DEGRADED", "gnss_available_without_successful_updates")

    if sensor_status.gnss_outage_s <= thresholds.gnss_short_outage_threshold_s and not support.recent_any_auxiliary:
        if support.recent_any_auxiliary_available:
            return ModeDecision("DEGRADED", "auxiliary_available_without_successful_updates")
        return ModeDecision("DEGRADED", "gnss_outage_without_aux_support")
    if (
        sensor_status.gnss_outage_s <= thresholds.gnss_short_outage_threshold_s
        and support.auxiliary_instability_without_backup
        and quality_score < thresholds.short_outage_aux_instability_quality_floor
    ):
        return ModeDecision("DEGRADED", "auxiliary_sensor_instability")
    if sensor_status.gnss_outage_s <= thresholds.gnss_short_outage_threshold_s and support.recent_any_auxiliary and not persistent_unhealthy:
        return ModeDecision("INERTIAL_HOLD", "short_gnss_outage")
    if sensor_status.gnss_outage_s <= thresholds.gnss_extended_outage_threshold_s and not support.recent_any_auxiliary and support.recent_any_auxiliary_available:
        return ModeDecision("DEGRADED", "auxiliary_available_without_successful_updates")
    if (
        sensor_status.gnss_outage_s <= thresholds.gnss_extended_outage_threshold_s
        and not support.recent_any_auxiliary
        and sensor_status.auxiliary_available_outage_s > thresholds.auxiliary_available_outage_threshold_s
    ):
        return ModeDecision("DEGRADED", "extended_outage_without_aux_support")
    if (
        sensor_status.gnss_outage_s <= thresholds.gnss_extended_outage_threshold_s
        and support.auxiliary_instability_without_backup
        and quality_score < thresholds.extended_outage_aux_instability_quality_floor
    ):
        return ModeDecision("DEGRADED", "auxiliary_sensor_instability")
    if sensor_status.gnss_outage_s <= thresholds.gnss_extended_outage_threshold_s and support.recent_any_auxiliary and not persistent_unhealthy:
        return ModeDecision("INERTIAL_HOLD", "extended_gnss_outage")
    if covariance_unhealthy and persistent_unhealthy:
        return ModeDecision("DEGRADED", covariance_health.reason)
    return ModeDecision("DEGRADED", "long_gnss_outage")
