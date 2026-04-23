from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass
class SensorStatus:
    recent_gnss_pos: bool
    recent_gnss_vel: bool
    recent_baro: bool
    recent_mag: bool
    gnss_pos_reject_streak: int
    gnss_vel_reject_streak: int
    gnss_pos_outage_s: float
    gnss_vel_outage_s: float
    gnss_outage_s: float


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


class SensorFreshnessTracker:
    def __init__(self) -> None:
        self.last_update_time: dict[str, float | None] = {
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

    def note_result(self, sensor_name: str, current_time: float, available: bool, used: bool, rejected: bool) -> None:
        if used:
            self.last_update_time[sensor_name] = current_time
            self.reject_streak[sensor_name] = 0
            return
        if rejected and available:
            self.reject_streak[sensor_name] += 1
            return
        if available:
            self.reject_streak[sensor_name] = 0

    def is_recent(self, sensor_name: str, current_time: float, timeout_s: float) -> bool:
        last_time = self.last_update_time[sensor_name]
        if last_time is None:
            return False
        return (current_time - last_time) <= timeout_s

    def outage_duration(self, sensor_name: str, current_time: float) -> float:
        last_time = self.last_update_time[sensor_name]
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

    def snapshot(self, current_time: float) -> SensorStatus:
        return SensorStatus(
            recent_gnss_pos=self.is_recent("gnss_pos", current_time, timeout_s=0.6),
            recent_gnss_vel=self.is_recent("gnss_vel", current_time, timeout_s=0.6),
            recent_baro=self.is_recent("baro", current_time, timeout_s=0.25),
            recent_mag=self.is_recent("mag", current_time, timeout_s=0.6),
            gnss_pos_reject_streak=self.reject_streak["gnss_pos"],
            gnss_vel_reject_streak=self.reject_streak["gnss_vel"],
            gnss_pos_outage_s=self.outage_duration("gnss_pos", current_time),
            gnss_vel_outage_s=self.outage_duration("gnss_vel", current_time),
            gnss_outage_s=self.gnss_outage_duration(current_time),
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
    yaw_innovation_abs_deg: float,
    pos_sigma_norm_m: float,
    vel_sigma_norm_mps: float,
    att_sigma_norm_deg: float,
) -> float:
    score = 35.0
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
    score -= min(yaw_innovation_abs_deg * 0.8, 10.0)
    score -= min(max(pos_sigma_norm_m - 1.5, 0.0) * 6.0, 10.0)
    score -= min(max(vel_sigma_norm_mps - 0.7, 0.0) * 12.0, 10.0)
    score -= min(max(att_sigma_norm_deg - 8.0, 0.0) * 1.5, 10.0)
    return max(0.0, min(100.0, score))
