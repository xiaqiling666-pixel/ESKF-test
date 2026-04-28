from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class SensorStatus:
    recent_gnss_pos: bool
    recent_gnss_vel: bool
    recent_baro: bool
    recent_mag: bool
    recent_available_gnss_pos: bool
    recent_available_gnss_vel: bool
    recent_available_baro: bool
    recent_available_mag: bool
    gnss_pos_reject_streak: int
    gnss_vel_reject_streak: int
    baro_reject_streak: int
    mag_reject_streak: int
    gnss_pos_skip_streak: int
    gnss_vel_skip_streak: int
    baro_skip_streak: int
    mag_skip_streak: int
    gnss_pos_reject_bypass_streak: int
    gnss_vel_reject_bypass_streak: int
    baro_reject_bypass_streak: int
    mag_reject_bypass_streak: int
    gnss_pos_adaptive_streak: int
    gnss_vel_adaptive_streak: int
    baro_adaptive_streak: int
    mag_adaptive_streak: int
    gnss_pos_outage_s: float
    gnss_vel_outage_s: float
    gnss_outage_s: float
    baro_outage_s: float
    mag_outage_s: float
    auxiliary_outage_s: float
    gnss_pos_available_outage_s: float
    gnss_vel_available_outage_s: float
    baro_available_outage_s: float
    mag_available_outage_s: float
    auxiliary_available_outage_s: float


@dataclass(frozen=True)
class CovarianceHealthStatus:
    level: str
    reason: str
    caution: bool
    unhealthy: bool
    pos_excess_m: float
    vel_excess_mps: float
    att_excess_deg: float
    caution_duration_s: float = 0.0
    unhealthy_duration_s: float = 0.0


@dataclass(frozen=True)
class MeasurementSupportSummary:
    full_gnss_support: bool
    recent_any_gnss: bool
    recent_any_gnss_available: bool
    gnss_available_without_successful_updates: bool
    recent_any_auxiliary: bool
    recent_any_auxiliary_available: bool
    baro_hard_unstable: bool
    mag_hard_unstable: bool
    baro_soft_unstable: bool
    mag_soft_unstable: bool
    stable_auxiliary_support: bool
    auxiliary_instability_without_backup: bool
    auxiliary_hard_issue_count: int
    auxiliary_soft_issue_count: int


class SensorFreshnessTracker:
    def __init__(self) -> None:
        self.last_update_time: dict[str, float | None] = {
            "gnss_pos": None,
            "gnss_vel": None,
            "baro": None,
            "mag": None,
        }
        self.last_available_time: dict[str, float | None] = {
            "gnss_pos": None,
            "gnss_vel": None,
            "baro": None,
            "mag": None,
        }
        self.reject_streak: dict[str, int] = {
            "gnss_pos": 0,
            "gnss_vel": 0,
            "baro": 0,
            "mag": 0,
        }
        self.skip_streak: dict[str, int] = {
            "gnss_pos": 0,
            "gnss_vel": 0,
            "baro": 0,
            "mag": 0,
        }
        self.reject_bypass_streak: dict[str, int] = {
            "gnss_pos": 0,
            "gnss_vel": 0,
            "baro": 0,
            "mag": 0,
        }
        self.adaptive_streak: dict[str, int] = {
            "gnss_pos": 0,
            "gnss_vel": 0,
            "baro": 0,
            "mag": 0,
        }

    def note_result(
        self,
        sensor_name: str,
        current_time: float,
        available: bool,
        used: bool,
        rejected: bool,
        reject_bypassed: bool = False,
        management_mode: str = "skip",
        adaptation_scale: float = 1.0,
    ) -> None:
        if available:
            self.last_available_time[sensor_name] = current_time
        if used:
            self.last_update_time[sensor_name] = current_time
            self.reject_streak[sensor_name] = 0
            self.skip_streak[sensor_name] = 0
            if reject_bypassed:
                self.reject_bypass_streak[sensor_name] += 1
            else:
                self.reject_bypass_streak[sensor_name] = 0
            if adaptation_scale > 1.0 or management_mode == "recover":
                self.adaptive_streak[sensor_name] += 1
            else:
                self.adaptive_streak[sensor_name] = 0
            return
        if rejected and available:
            self.reject_streak[sensor_name] += 1
            self.skip_streak[sensor_name] = 0
            self.reject_bypass_streak[sensor_name] = 0
            self.adaptive_streak[sensor_name] = 0
            return
        if available:
            self.reject_streak[sensor_name] = 0
            self.skip_streak[sensor_name] += 1
            self.reject_bypass_streak[sensor_name] = 0
            self.adaptive_streak[sensor_name] = 0
            return
        self.skip_streak[sensor_name] = 0

    def is_recent(self, sensor_name: str, current_time: float, timeout_s: float) -> bool:
        last_time = self.last_update_time[sensor_name]
        if last_time is None:
            return False
        return (current_time - last_time) <= timeout_s

    def is_recent_available(self, sensor_name: str, current_time: float, timeout_s: float) -> bool:
        last_time = self.last_available_time[sensor_name]
        if last_time is None:
            return False
        return (current_time - last_time) <= timeout_s

    def outage_duration(self, sensor_name: str, current_time: float) -> float:
        last_time = self.last_update_time[sensor_name]
        if last_time is None:
            return float("inf")
        return max(0.0, current_time - last_time)

    def available_outage_duration(self, sensor_name: str, current_time: float) -> float:
        last_time = self.last_available_time[sensor_name]
        if last_time is None:
            return float("inf")
        return max(0.0, current_time - last_time)

    def gnss_outage_duration(self, current_time: float) -> float:
        last_pos = self.last_update_time["gnss_pos"]
        last_vel = self.last_update_time["gnss_vel"]
        last_any = max(
            (time_value for time_value in (last_pos, last_vel) if time_value is not None),
            default=None,
        )
        if last_any is None:
            return float("inf")
        return max(0.0, current_time - last_any)

    def auxiliary_outage_duration(self, current_time: float) -> float:
        last_baro = self.last_update_time["baro"]
        last_mag = self.last_update_time["mag"]
        last_any = max(
            (time_value for time_value in (last_baro, last_mag) if time_value is not None),
            default=None,
        )
        if last_any is None:
            return float("inf")
        return max(0.0, current_time - last_any)

    def auxiliary_available_outage_duration(self, current_time: float) -> float:
        last_baro = self.last_available_time["baro"]
        last_mag = self.last_available_time["mag"]
        last_any = max(
            (time_value for time_value in (last_baro, last_mag) if time_value is not None),
            default=None,
        )
        if last_any is None:
            return float("inf")
        return max(0.0, current_time - last_any)

    def snapshot(self, current_time: float) -> SensorStatus:
        return SensorStatus(
            recent_gnss_pos=self.is_recent("gnss_pos", current_time, timeout_s=0.6),
            recent_gnss_vel=self.is_recent("gnss_vel", current_time, timeout_s=0.6),
            recent_baro=self.is_recent("baro", current_time, timeout_s=0.25),
            recent_mag=self.is_recent("mag", current_time, timeout_s=0.6),
            recent_available_gnss_pos=self.is_recent_available("gnss_pos", current_time, timeout_s=0.6),
            recent_available_gnss_vel=self.is_recent_available("gnss_vel", current_time, timeout_s=0.6),
            recent_available_baro=self.is_recent_available("baro", current_time, timeout_s=0.25),
            recent_available_mag=self.is_recent_available("mag", current_time, timeout_s=0.6),
            gnss_pos_reject_streak=self.reject_streak["gnss_pos"],
            gnss_vel_reject_streak=self.reject_streak["gnss_vel"],
            baro_reject_streak=self.reject_streak["baro"],
            mag_reject_streak=self.reject_streak["mag"],
            gnss_pos_skip_streak=self.skip_streak["gnss_pos"],
            gnss_vel_skip_streak=self.skip_streak["gnss_vel"],
            baro_skip_streak=self.skip_streak["baro"],
            mag_skip_streak=self.skip_streak["mag"],
            gnss_pos_reject_bypass_streak=self.reject_bypass_streak["gnss_pos"],
            gnss_vel_reject_bypass_streak=self.reject_bypass_streak["gnss_vel"],
            baro_reject_bypass_streak=self.reject_bypass_streak["baro"],
            mag_reject_bypass_streak=self.reject_bypass_streak["mag"],
            gnss_pos_adaptive_streak=self.adaptive_streak["gnss_pos"],
            gnss_vel_adaptive_streak=self.adaptive_streak["gnss_vel"],
            baro_adaptive_streak=self.adaptive_streak["baro"],
            mag_adaptive_streak=self.adaptive_streak["mag"],
            gnss_pos_outage_s=self.outage_duration("gnss_pos", current_time),
            gnss_vel_outage_s=self.outage_duration("gnss_vel", current_time),
            gnss_outage_s=self.gnss_outage_duration(current_time),
            baro_outage_s=self.outage_duration("baro", current_time),
            mag_outage_s=self.outage_duration("mag", current_time),
            auxiliary_outage_s=self.auxiliary_outage_duration(current_time),
            gnss_pos_available_outage_s=self.available_outage_duration("gnss_pos", current_time),
            gnss_vel_available_outage_s=self.available_outage_duration("gnss_vel", current_time),
            baro_available_outage_s=self.available_outage_duration("baro", current_time),
            mag_available_outage_s=self.available_outage_duration("mag", current_time),
            auxiliary_available_outage_s=self.auxiliary_available_outage_duration(current_time),
        )


class CovarianceHealthTracker:
    def __init__(self) -> None:
        self.caution_since_time: float | None = None
        self.unhealthy_since_time: float | None = None

    def step(self, current_time: float, health: CovarianceHealthStatus) -> CovarianceHealthStatus:
        if health.caution:
            if self.caution_since_time is None:
                self.caution_since_time = current_time
        else:
            self.caution_since_time = None

        if health.unhealthy:
            if self.unhealthy_since_time is None:
                self.unhealthy_since_time = current_time
        else:
            self.unhealthy_since_time = None

        caution_duration_s = (
            0.0 if self.caution_since_time is None else max(0.0, current_time - self.caution_since_time)
        )
        unhealthy_duration_s = (
            0.0 if self.unhealthy_since_time is None else max(0.0, current_time - self.unhealthy_since_time)
        )
        return replace(
            health,
            caution_duration_s=caution_duration_s,
            unhealthy_duration_s=unhealthy_duration_s,
        )


def summarize_measurement_support(
    sensor_status: SensorStatus,
    *,
    gnss_skip_streak_degraded: int = 2,
    auxiliary_reject_streak_degraded: int = 2,
    auxiliary_reject_bypass_streak_degraded: int = 2,
    auxiliary_adaptive_streak_degraded: int = 3,
    auxiliary_skip_streak_degraded: int = 2,
) -> MeasurementSupportSummary:
    full_gnss_support = sensor_status.recent_gnss_pos and sensor_status.recent_gnss_vel
    recent_any_gnss = sensor_status.recent_gnss_pos or sensor_status.recent_gnss_vel
    recent_any_gnss_available = sensor_status.recent_available_gnss_pos or sensor_status.recent_available_gnss_vel
    gnss_skip_streak = max(sensor_status.gnss_pos_skip_streak, sensor_status.gnss_vel_skip_streak)
    gnss_available_without_successful_updates = (
        recent_any_gnss_available and not recent_any_gnss and gnss_skip_streak >= gnss_skip_streak_degraded
    )
    recent_any_auxiliary = sensor_status.recent_baro or sensor_status.recent_mag
    recent_any_auxiliary_available = sensor_status.recent_available_baro or sensor_status.recent_available_mag
    baro_hard_unstable = (
        sensor_status.baro_reject_streak >= auxiliary_reject_streak_degraded
        or sensor_status.baro_reject_bypass_streak >= auxiliary_reject_bypass_streak_degraded
        or sensor_status.baro_skip_streak >= auxiliary_skip_streak_degraded
    )
    mag_hard_unstable = (
        sensor_status.mag_reject_streak >= auxiliary_reject_streak_degraded
        or sensor_status.mag_reject_bypass_streak >= auxiliary_reject_bypass_streak_degraded
        or sensor_status.mag_skip_streak >= auxiliary_skip_streak_degraded
    )
    baro_soft_unstable = sensor_status.baro_adaptive_streak >= auxiliary_adaptive_streak_degraded
    mag_soft_unstable = sensor_status.mag_adaptive_streak >= auxiliary_adaptive_streak_degraded
    stable_auxiliary_support = (
        (sensor_status.recent_baro and not baro_hard_unstable)
        or (sensor_status.recent_mag and not mag_hard_unstable)
    )
    auxiliary_instability_without_backup = (
        not stable_auxiliary_support
        and (
            (sensor_status.recent_available_baro and (baro_hard_unstable or baro_soft_unstable))
            or (sensor_status.recent_available_mag and (mag_hard_unstable or mag_soft_unstable))
        )
    )
    auxiliary_hard_issue_count = int(sensor_status.recent_available_baro and baro_hard_unstable) + int(
        sensor_status.recent_available_mag and mag_hard_unstable
    )
    auxiliary_soft_issue_count = int(sensor_status.recent_available_baro and baro_soft_unstable) + int(
        sensor_status.recent_available_mag and mag_soft_unstable
    )
    return MeasurementSupportSummary(
        full_gnss_support=full_gnss_support,
        recent_any_gnss=recent_any_gnss,
        recent_any_gnss_available=recent_any_gnss_available,
        gnss_available_without_successful_updates=gnss_available_without_successful_updates,
        recent_any_auxiliary=recent_any_auxiliary,
        recent_any_auxiliary_available=recent_any_auxiliary_available,
        baro_hard_unstable=baro_hard_unstable,
        mag_hard_unstable=mag_hard_unstable,
        baro_soft_unstable=baro_soft_unstable,
        mag_soft_unstable=mag_soft_unstable,
        stable_auxiliary_support=stable_auxiliary_support,
        auxiliary_instability_without_backup=auxiliary_instability_without_backup,
        auxiliary_hard_issue_count=auxiliary_hard_issue_count,
        auxiliary_soft_issue_count=auxiliary_soft_issue_count,
    )


def classify_covariance_health(
    pos_sigma_norm_m: float,
    vel_sigma_norm_mps: float,
    att_sigma_norm_deg: float,
) -> CovarianceHealthStatus:
    caution_limits = {
        "position": 1.5,
        "velocity": 0.7,
        "attitude": 20.0,
    }
    unhealthy_limits = {
        "position": 3.0,
        "velocity": 1.5,
        "attitude": 30.0,
    }
    unhealthy_sources = [
        name
        for name, value, limit in (
            ("position", pos_sigma_norm_m, unhealthy_limits["position"]),
            ("velocity", vel_sigma_norm_mps, unhealthy_limits["velocity"]),
            ("attitude", att_sigma_norm_deg, unhealthy_limits["attitude"]),
        )
        if value > limit
    ]
    caution_sources = [
        name
        for name, value, limit in (
            ("position", pos_sigma_norm_m, caution_limits["position"]),
            ("velocity", vel_sigma_norm_mps, caution_limits["velocity"]),
            ("attitude", att_sigma_norm_deg, caution_limits["attitude"]),
        )
        if value > limit
    ]

    if unhealthy_sources:
        reason = (
            "multiple_unhealthy_sigmas"
            if len(unhealthy_sources) > 1
            else f"{unhealthy_sources[0]}_sigma_unhealthy"
        )
        return CovarianceHealthStatus(
            level="UNHEALTHY",
            reason=reason,
            caution=True,
            unhealthy=True,
            pos_excess_m=max(0.0, pos_sigma_norm_m - unhealthy_limits["position"]),
            vel_excess_mps=max(0.0, vel_sigma_norm_mps - unhealthy_limits["velocity"]),
            att_excess_deg=max(0.0, att_sigma_norm_deg - unhealthy_limits["attitude"]),
        )

    if caution_sources:
        reason = "multiple_caution_sigmas" if len(caution_sources) > 1 else f"{caution_sources[0]}_sigma_caution"
        return CovarianceHealthStatus(
            level="CAUTION",
            reason=reason,
            caution=True,
            unhealthy=False,
            pos_excess_m=max(0.0, pos_sigma_norm_m - caution_limits["position"]),
            vel_excess_mps=max(0.0, vel_sigma_norm_mps - caution_limits["velocity"]),
            att_excess_deg=max(0.0, att_sigma_norm_deg - caution_limits["attitude"]),
        )

    return CovarianceHealthStatus(
        level="HEALTHY",
        reason="nominal",
        caution=False,
        unhealthy=False,
        pos_excess_m=0.0,
        vel_excess_mps=0.0,
        att_excess_deg=0.0,
    )


def compute_quality_score(
    sensor_status: SensorStatus,
    pos_innovation_norm: float,
    vel_innovation_norm: float,
    baro_innovation_abs: float,
    yaw_innovation_abs_deg: float,
    pos_sigma_norm_m: float,
    vel_sigma_norm_mps: float,
    att_sigma_norm_deg: float,
    support_summary: MeasurementSupportSummary | None = None,
) -> float:
    score = 35.0
    support = support_summary or summarize_measurement_support(sensor_status)
    gnss_bypass_streak = max(sensor_status.gnss_pos_reject_bypass_streak, sensor_status.gnss_vel_reject_bypass_streak)
    auxiliary_bypass_streak = max(sensor_status.baro_reject_bypass_streak, sensor_status.mag_reject_bypass_streak)
    gnss_adaptive_streak = max(sensor_status.gnss_pos_adaptive_streak, sensor_status.gnss_vel_adaptive_streak)
    auxiliary_adaptive_streak = max(sensor_status.baro_adaptive_streak, sensor_status.mag_adaptive_streak)
    gnss_skip_streak = max(sensor_status.gnss_pos_skip_streak, sensor_status.gnss_vel_skip_streak)
    auxiliary_skip_streak = max(sensor_status.baro_skip_streak, sensor_status.mag_skip_streak)
    if support.full_gnss_support:
        score += 12.0
    if sensor_status.recent_gnss_pos:
        score += 25.0
    if sensor_status.recent_gnss_vel:
        score += 20.0
    if sensor_status.recent_baro:
        score += 10.0
    if sensor_status.recent_mag:
        score += 10.0

    score -= min(pos_innovation_norm * 3.0, 18.0)
    score -= min(vel_innovation_norm * 6.0, 12.0)
    score -= min(baro_innovation_abs * 4.0, 8.0)
    score -= min(yaw_innovation_abs_deg * 0.8, 10.0)
    score -= min(max(pos_sigma_norm_m - 1.5, 0.0) * 6.0, 10.0)
    score -= min(max(vel_sigma_norm_mps - 0.7, 0.0) * 12.0, 10.0)
    score -= min(max(att_sigma_norm_deg - 8.0, 0.0) * 1.5, 10.0)
    score -= min(gnss_bypass_streak * 4.0, 12.0)
    score -= min(gnss_adaptive_streak * 3.0, 9.0)
    if support.gnss_available_without_successful_updates:
        score -= min(gnss_skip_streak * 3.0, 9.0)
    if support.stable_auxiliary_support:
        score -= min(support.auxiliary_hard_issue_count * 1.5, 4.0)
        score -= min(support.auxiliary_soft_issue_count * 1.0, 3.0)
    else:
        score -= min((sensor_status.baro_reject_streak + sensor_status.mag_reject_streak) * 2.0, 8.0)
        score -= min(auxiliary_bypass_streak * 2.5, 8.0)
        score -= min(auxiliary_adaptive_streak * 2.0, 6.0)
        if not sensor_status.recent_baro and not sensor_status.recent_mag and not support.full_gnss_support:
            if sensor_status.recent_available_baro or sensor_status.recent_available_mag:
                score -= min(auxiliary_skip_streak * 2.0, 6.0)
            else:
                score -= min(max(sensor_status.auxiliary_outage_s - 0.5, 0.0) * 6.0, 10.0)
    return max(0.0, min(100.0, score))
