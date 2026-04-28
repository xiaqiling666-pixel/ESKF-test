from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..analysis.quality import (
    SensorFreshnessTracker,
    CovarianceHealthTracker,
    SensorStatus,
    CovarianceHealthStatus,
    MeasurementSupportSummary,
    classify_covariance_health,
    compute_quality_score,
    summarize_measurement_support,
)
from ..analysis.state_machine import (
    ModeStateTracker,
    ModeThresholds,
    ModeTrackerSnapshot,
    determine_mode,
)
from ..core.filter import OfflineESKF
from ..core.state import ERROR_STATE


@dataclass(frozen=True)
class ModeEvaluationContext:
    sensor_status: SensorStatus
    covariance_health: CovarianceHealthStatus
    support_summary: MeasurementSupportSummary
    quality_score: float
    mode_state: ModeTrackerSnapshot
    pos_sigma_norm_m: float
    vel_sigma_norm_mps: float
    att_sigma_norm_deg: float
    covariance_diagonal: np.ndarray


def evaluate_mode_context(
    *,
    filter_engine: OfflineESKF,
    freshness_tracker: SensorFreshnessTracker,
    covariance_tracker: CovarianceHealthTracker,
    mode_tracker: ModeStateTracker,
    mode_thresholds: ModeThresholds,
    current_time: float,
    pos_innovation_norm: float,
    vel_innovation_norm: float,
    baro_innovation_abs: float,
    yaw_innovation_abs_deg: float,
) -> ModeEvaluationContext:
    pos_sigma_norm_m = float(np.sqrt(np.trace(filter_engine.P[ERROR_STATE.position, ERROR_STATE.position])))
    vel_sigma_norm_mps = float(np.sqrt(np.trace(filter_engine.P[ERROR_STATE.velocity, ERROR_STATE.velocity])))
    att_sigma_norm_deg = float(
        np.rad2deg(np.sqrt(np.trace(filter_engine.P[ERROR_STATE.attitude, ERROR_STATE.attitude])))
    )
    covariance_diagonal = np.diag(filter_engine.P)
    sensor_status = freshness_tracker.snapshot(current_time)
    covariance_health = classify_covariance_health(
        pos_sigma_norm_m=pos_sigma_norm_m,
        vel_sigma_norm_mps=vel_sigma_norm_mps,
        att_sigma_norm_deg=att_sigma_norm_deg,
    )
    covariance_health = covariance_tracker.step(current_time, covariance_health)
    support_summary = summarize_measurement_support(
        sensor_status,
        gnss_skip_streak_degraded=mode_thresholds.gnss_skip_streak_degraded,
        auxiliary_reject_streak_degraded=mode_thresholds.auxiliary_reject_streak_degraded,
        auxiliary_reject_bypass_streak_degraded=mode_thresholds.auxiliary_reject_bypass_streak_degraded,
        auxiliary_adaptive_streak_degraded=mode_thresholds.auxiliary_adaptive_streak_degraded,
        auxiliary_skip_streak_degraded=mode_thresholds.auxiliary_skip_streak_degraded,
    )
    quality_score = compute_quality_score(
        sensor_status=sensor_status,
        pos_innovation_norm=pos_innovation_norm,
        vel_innovation_norm=vel_innovation_norm,
        baro_innovation_abs=baro_innovation_abs,
        yaw_innovation_abs_deg=yaw_innovation_abs_deg,
        pos_sigma_norm_m=pos_sigma_norm_m,
        vel_sigma_norm_mps=vel_sigma_norm_mps,
        att_sigma_norm_deg=att_sigma_norm_deg,
        support_summary=support_summary,
    )
    target_mode_decision = determine_mode(
        sensor_status,
        quality_score,
        covariance_health,
        thresholds=mode_thresholds,
        support_summary=support_summary,
    )
    mode_state = mode_tracker.step(current_time, target_mode_decision)
    return ModeEvaluationContext(
        sensor_status=sensor_status,
        covariance_health=covariance_health,
        support_summary=support_summary,
        quality_score=quality_score,
        mode_state=mode_state,
        pos_sigma_norm_m=pos_sigma_norm_m,
        vel_sigma_norm_mps=vel_sigma_norm_mps,
        att_sigma_norm_deg=att_sigma_norm_deg,
        covariance_diagonal=covariance_diagonal,
    )
