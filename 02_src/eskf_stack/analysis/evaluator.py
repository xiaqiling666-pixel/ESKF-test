from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.math_utils import wrap_angle
from .truth_access import has_diagnostic_truth, truth_position_columns, truth_velocity_columns, truth_yaw_column


def metric_category(metric_name: str) -> str:
    if metric_name.startswith("input_quality_"):
        return "input_quality"
    if metric_name.startswith("initialization_"):
        return "initialization"
    if metric_name in {
        "position_rmse_m",
        "velocity_rmse_mps",
        "yaw_rmse_deg",
        "final_position_error_m",
    }:
        return "estimation_error"
    if metric_name == "mean_quality_score":
        return "measurement_management"
    for sensor_name in ("gnss_pos", "gnss_vel", "baro", "mag"):
        if (
            metric_name.startswith(f"{sensor_name}_")
            or metric_name == f"mean_{sensor_name}_nis"
            or metric_name == f"max_{sensor_name}_nis"
            or metric_name == f"max_{sensor_name}_reject_streak"
            or metric_name == f"max_{sensor_name}_skip_streak"
            or metric_name == f"max_{sensor_name}_reject_bypass_streak"
            or metric_name == f"max_{sensor_name}_adaptive_streak"
            or metric_name == f"max_{sensor_name}_outage_s"
            or metric_name == f"max_{sensor_name}_available_outage_s"
        ):
            return "measurement_management"
    if metric_name.startswith("auxiliary_") or metric_name in {"max_auxiliary_outage_s", "max_auxiliary_available_outage_s"}:
        return "measurement_management"
    if metric_name.startswith(("gnss_", "baro_", "mag_")):
        return "measurement_management"
    if metric_name.startswith("predict_") or metric_name in {"max_dt_raw_s", "max_dt_applied_s"}:
        return "prediction_diagnostics"
    if metric_name.startswith("covariance_") or metric_name in {
        "final_pos_sigma_norm_m",
        "final_vel_sigma_norm_mps",
        "final_att_sigma_norm_deg",
        "min_cov_diag",
    }:
        return "covariance_health"
    if metric_name.startswith("mode_") or metric_name in {"pending_row_count", "pending_duration_s"}:
        return "mode_state"
    if "gravity" in metric_name or "coriolis" in metric_name:
        return "navigation_environment"
    if metric_name in {"processed_rows", "pipeline_runtime_s"}:
        return "runtime"
    return "other"


def metric_supports_experiment_delta(metric_name: str) -> bool:
    return metric_category(metric_name) != "other"


def metric_experiment_comparison_direction(metric_name: str) -> str | None:
    lower_better_metrics = {
        "position_rmse_m",
        "velocity_rmse_mps",
        "yaw_rmse_deg",
        "final_position_error_m",
        "gnss_pos_rejections",
        "gnss_vel_rejections",
        "baro_rejections",
        "mag_rejections",
        "covariance_unhealthy_row_count",
        "pipeline_runtime_s",
    }
    higher_better_metrics = {
        "mean_quality_score",
        "initialization_completed_flag",
    }
    if metric_name in lower_better_metrics:
        return "lower_better"
    if metric_name in higher_better_metrics:
        return "higher_better"
    return None


def _parse_bool_flag(value: object) -> float | None:
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return 1.0
    if text in {"false", "0", "no", ""}:
        return 0.0
    return None


def _add_initialization_metrics(metrics: dict[str, float], initialization_summary: dict[str, object] | None) -> None:
    if not initialization_summary:
        return

    ready_mode = str(initialization_summary.get("initialization_ready_mode", "")).strip()
    if not ready_mode:
        initialization_mode = str(initialization_summary.get("initialization_mode", "")).strip()
        if initialization_mode.startswith("direct"):
            ready_mode = "direct"
        elif initialization_mode.startswith("bootstrap_position_pair"):
            ready_mode = "bootstrap_position_pair"

    metrics["initialization_completed_flag"] = float(
        str(initialization_summary.get("initialization_phase", "")).strip() == "INITIALIZED"
    )
    metrics["initialization_mode_direct_flag"] = float(ready_mode == "direct")
    metrics["initialization_mode_bootstrap_position_pair_flag"] = float(ready_mode == "bootstrap_position_pair")

    static_alignment_used = _parse_bool_flag(initialization_summary.get("static_coarse_alignment_used", ""))
    if static_alignment_used is not None:
        metrics["initialization_static_coarse_alignment_used_flag"] = static_alignment_used

    static_alignment_ready = _parse_bool_flag(initialization_summary.get("static_alignment_ready", ""))
    if static_alignment_ready is not None:
        metrics["initialization_static_alignment_ready_flag"] = static_alignment_ready

    metrics["initialization_zero_yaw_fallback_used_flag"] = float(
        str(initialization_summary.get("heading_source", "")).strip() == "zero_yaw_fallback"
    )

    wait_time = initialization_summary.get("initialization_wait_s")
    if wait_time is not None:
        metrics["initialization_wait_s"] = float(wait_time)


def _normalise_initialization_mode(initialization_summary: dict[str, object]) -> str:
    ready_mode = str(initialization_summary.get("initialization_ready_mode", "")).strip()
    if ready_mode:
        return ready_mode

    initialization_mode = str(initialization_summary.get("initialization_mode", "")).strip()
    if initialization_mode.startswith("direct"):
        return "direct"
    if initialization_mode.startswith("bootstrap_position_pair"):
        return "bootstrap_position_pair"
    return initialization_mode


def _format_initialization_summary(initialization_summary: dict[str, object] | None) -> list[str]:
    if not initialization_summary:
        return []

    phase = str(initialization_summary.get("initialization_phase", "")).strip() or "unknown"
    mode = _normalise_initialization_mode(initialization_summary) or "unknown"
    heading_source = str(initialization_summary.get("heading_source", "")).strip() or "unknown"
    reason = str(initialization_summary.get("initialization_reason", "")).strip() or "unknown"
    static_alignment_reason = str(initialization_summary.get("static_alignment_reason", "")).strip() or "unknown"

    static_alignment_used = _parse_bool_flag(initialization_summary.get("static_coarse_alignment_used", ""))
    static_alignment_ready = _parse_bool_flag(initialization_summary.get("static_alignment_ready", ""))
    zero_yaw_fallback_used = heading_source == "zero_yaw_fallback"

    static_alignment_used_text = "unknown"
    if static_alignment_used == 1.0:
        static_alignment_used_text = "yes"
    elif static_alignment_used == 0.0:
        static_alignment_used_text = "no"

    static_alignment_ready_text = "unknown"
    if static_alignment_ready == 1.0:
        static_alignment_ready_text = "yes"
    elif static_alignment_ready == 0.0:
        static_alignment_ready_text = "no"

    lines = ["Initialization Summary", ""]
    lines.append(f"phase: {phase}")
    lines.append(f"mode: {mode}")
    lines.append(f"reason: {reason}")
    lines.append(f"heading_source: {heading_source}")
    lines.append(f"static_coarse_alignment_used: {static_alignment_used_text}")
    lines.append(f"static_alignment_ready: {static_alignment_ready_text}")
    lines.append(f"static_alignment_reason: {static_alignment_reason}")
    lines.append(f"zero_yaw_fallback_used: {'yes' if zero_yaw_fallback_used else 'no'}")

    wait_time = initialization_summary.get("initialization_wait_s")
    if wait_time is not None:
        lines.append(f"initialization_wait_s: {float(wait_time):.6f}")

    return lines


def _metric_section_name(metric_name: str) -> str:
    if metric_name.startswith("initialization_"):
        return "Initialization Metrics"
    if metric_name.startswith("gnss_") or metric_name.startswith("mean_gnss_") or metric_name.startswith("max_gnss_"):
        return "GNSS Metrics"
    if metric_name.startswith("baro_") or metric_name.startswith("mean_baro_") or metric_name.startswith("max_baro_"):
        return "Barometer Metrics"
    if metric_name.startswith("mag_") or metric_name.startswith("mean_mag_") or metric_name.startswith("max_mag_"):
        return "Magnetometer Metrics"
    if metric_name == "mean_quality_score" or metric_name.startswith("auxiliary_"):
        return "Quality Metrics"
    category = metric_category(metric_name)
    if category == "covariance_health":
        return "Covariance Metrics"
    if category == "mode_state":
        return "Mode Metrics"
    if category == "prediction_diagnostics":
        return "Prediction Metrics"
    if category == "estimation_error":
        return "Estimation Error Metrics"
    if category == "input_quality":
        return "Input Quality Metrics"
    if category == "navigation_environment":
        return "Navigation Environment Metrics"
    if category == "runtime":
        return "Runtime Metrics"
    return "Other Metrics"


def _metric_section_sort_key(metric_name: str) -> tuple[int, str]:
    section_order = {
        "Initialization Metrics": 0,
        "GNSS Metrics": 1,
        "Barometer Metrics": 2,
        "Magnetometer Metrics": 3,
        "Quality Metrics": 4,
        "Covariance Metrics": 5,
        "Mode Metrics": 6,
        "Prediction Metrics": 7,
        "Estimation Error Metrics": 8,
        "Input Quality Metrics": 9,
        "Navigation Environment Metrics": 10,
        "Runtime Metrics": 11,
        "Other Metrics": 12,
    }
    section_name = _metric_section_name(metric_name)
    return (section_order.get(section_name, 99), metric_name)


def _format_metric_sections(metrics: dict[str, float]) -> list[str]:
    sectioned_metrics: dict[str, list[tuple[str, float]]] = {}
    for metric_name, metric_value in sorted(metrics.items(), key=lambda item: _metric_section_sort_key(item[0])):
        section_name = _metric_section_name(metric_name)
        sectioned_metrics.setdefault(section_name, []).append((metric_name, metric_value))

    lines: list[str] = ["Metric Values", ""]
    for index, (section_name, section_metrics) in enumerate(sectioned_metrics.items()):
        lines.append(section_name)
        lines.append("")
        for metric_name, metric_value in section_metrics:
            lines.append(f"{metric_name}: {metric_value:.6f}")
        if index != len(sectioned_metrics) - 1:
            lines.append("")
    return lines


def _sample_durations(time_series: pd.Series) -> pd.Series:
    if len(time_series) <= 1:
        return pd.Series([0.0] * len(time_series), index=time_series.index, dtype=float)
    durations = time_series.shift(-1) - time_series
    positive_steps = durations[durations > 0.0]
    fallback_step = float(positive_steps.median()) if not positive_steps.empty else 0.0
    durations = durations.fillna(fallback_step).clip(lower=0.0)
    return durations.astype(float)


def _add_categorical_duration_metrics(
    metrics: dict[str, float],
    result_df: pd.DataFrame,
    category_column: str,
    prefix: str,
) -> None:
    if category_column not in result_df.columns or "time" not in result_df.columns or result_df.empty:
        return

    category_series = result_df[category_column].fillna("").astype(str)
    category_series = category_series.where(category_series != "", other="EMPTY")
    sample_durations = _sample_durations(result_df["time"])
    total_duration = float(sample_durations.sum())
    enter_flags = category_series.ne(category_series.shift(1))

    changes = category_series.ne(category_series.shift(1)).sum()
    metrics[f"{prefix}_transition_count"] = float(max(int(changes) - 1, 0))

    for category_name, _ in result_df.groupby(category_series, sort=False):
        category_mask = category_series == category_name
        category_duration = float(sample_durations[category_mask].sum())
        category_entries = int((category_mask & enter_flags).sum())
        metrics[f"{prefix}_duration_{category_name}_s"] = category_duration
        metrics[f"{prefix}_entry_count_{category_name}"] = float(category_entries)
        if total_duration > 0.0:
            metrics[f"{prefix}_share_{category_name}_pct"] = 100.0 * category_duration / total_duration


def _add_mode_scale_metrics(
    metrics: dict[str, float],
    result_df: pd.DataFrame,
    sensor_name: str,
    used_column: str,
    rejected_column: str,
) -> None:
    mode_scale_column = f"{sensor_name}_mode_scale"
    if mode_scale_column not in result_df.columns:
        return

    scaled_mask = result_df[mode_scale_column] > 1.0
    metrics[f"{sensor_name}_mode_scaled_measurements"] = float(scaled_mask.sum())

    if used_column in result_df.columns:
        metrics[f"{sensor_name}_mode_scaled_updates"] = float((scaled_mask & result_df[used_column].astype(bool)).sum())
    else:
        metrics[f"{sensor_name}_mode_scaled_updates"] = float(scaled_mask.sum())

    if rejected_column in result_df.columns:
        metrics[f"{sensor_name}_mode_scaled_rejections"] = float(
            (scaled_mask & result_df[rejected_column].astype(bool)).sum()
        )


def _add_adaptive_r_metrics(
    metrics: dict[str, float],
    result_df: pd.DataFrame,
    sensor_name: str,
    used_column: str | None = None,
) -> None:
    adaptation_column = f"{sensor_name}_adaptation_scale"
    legacy_r_scale_column = f"{sensor_name}_r_scale"
    used_mask = pd.Series([True] * len(result_df), index=result_df.index)
    if used_column is not None and used_column in result_df.columns:
        used_mask = result_df[used_column].astype(bool)

    if adaptation_column in result_df.columns:
        metrics[f"{sensor_name}_adapted_updates"] = float(((result_df[adaptation_column] > 1.0) & used_mask).sum())
    elif legacy_r_scale_column in result_df.columns:
        metrics[f"{sensor_name}_adapted_updates"] = float(((result_df[legacy_r_scale_column] > 1.0) & used_mask).sum())


def _add_rejection_metrics(metrics: dict[str, float], result_df: pd.DataFrame, sensor_name: str) -> None:
    rejected_column = f"{sensor_name}_rejected"
    if rejected_column in result_df.columns:
        metrics[f"{sensor_name}_rejections"] = float(result_df[rejected_column].sum())


def _add_reject_bypass_metrics(metrics: dict[str, float], result_df: pd.DataFrame, sensor_name: str) -> None:
    bypassed_column = f"{sensor_name}_reject_bypassed"
    if bypassed_column not in result_df.columns:
        return

    bypassed_count = float(result_df[bypassed_column].astype(bool).sum())
    metrics[f"{sensor_name}_reject_bypassed_updates"] = bypassed_count

    reject_exceed_key = f"{sensor_name}_nis_reject_exceed_count"
    metrics[f"{sensor_name}_nis_reject_policy_bypass_count"] = bypassed_count
    if reject_exceed_key in metrics and metrics[reject_exceed_key] > 0.0:
        metrics[f"{sensor_name}_nis_reject_policy_bypass_pct"] = 100.0 * bypassed_count / metrics[reject_exceed_key]


def _add_management_mode_metrics(metrics: dict[str, float], result_df: pd.DataFrame, sensor_name: str) -> None:
    management_mode_column = f"{sensor_name}_management_mode"
    if management_mode_column not in result_df.columns:
        return

    management_modes = result_df[management_mode_column].fillna("").astype(str)
    for mode_name in ("update", "reject", "recover", "skip", "unavailable", "pending_init"):
        metrics[f"{sensor_name}_management_count_{mode_name}"] = float((management_modes == mode_name).sum())


def _add_recovery_scale_metrics(
    metrics: dict[str, float],
    result_df: pd.DataFrame,
    sensor_name: str,
    used_column: str | None = None,
) -> None:
    recovery_scale_column = f"{sensor_name}_recovery_scale"
    if recovery_scale_column not in result_df.columns:
        return

    used_mask = pd.Series([True] * len(result_df), index=result_df.index)
    if used_column is not None and used_column in result_df.columns:
        used_mask = result_df[used_column].astype(bool)

    recovery_scale_values = pd.to_numeric(result_df[recovery_scale_column], errors="coerce")
    recovery_scaled_mask = recovery_scale_values > 1.0
    metrics[f"{sensor_name}_recovery_scaled_updates"] = float((recovery_scaled_mask & used_mask).sum())
    if recovery_scale_values.notna().any():
        metrics[f"max_{sensor_name}_recovery_scale"] = float(recovery_scale_values.max())


def _bool_column_sum(result_df: pd.DataFrame, column_name: str) -> float:
    if column_name not in result_df.columns:
        return 0.0
    return float(result_df[column_name].sum())


def _add_availability_metrics(metrics: dict[str, float], result_df: pd.DataFrame, sensor_name: str) -> None:
    available_column = f"available_{sensor_name}"
    if available_column not in result_df.columns:
        return
    metrics[f"{sensor_name}_available_measurements"] = _bool_column_sum(result_df, available_column)


def _add_nis_metrics(metrics: dict[str, float], result_df: pd.DataFrame, sensor_name: str) -> None:
    nis_column = f"{sensor_name}_nis"
    if nis_column not in result_df.columns:
        return

    valid_nis = pd.to_numeric(result_df[nis_column], errors="coerce").dropna()
    if not valid_nis.empty:
        metrics[f"{sensor_name}_nis_valid_count"] = float(len(valid_nis))
        metrics[f"mean_{sensor_name}_nis"] = float(valid_nis.mean())
        metrics[f"max_{sensor_name}_nis"] = float(valid_nis.max())


def _add_innovation_metrics(
    metrics: dict[str, float],
    result_df: pd.DataFrame,
    sensor_name: str,
    innovation_column: str,
) -> None:
    if innovation_column not in result_df.columns:
        return

    innovation_values = pd.to_numeric(result_df[innovation_column], errors="coerce").dropna()
    if not innovation_values.empty:
        metrics[f"{sensor_name}_innovation_valid_count"] = float(len(innovation_values))
        metrics[f"mean_{sensor_name}_innovation"] = float(innovation_values.mean())
        metrics[f"max_{sensor_name}_innovation"] = float(innovation_values.max())


def _add_nis_threshold_metrics(metrics: dict[str, float], result_df: pd.DataFrame, sensor_name: str) -> None:
    nis_column = f"{sensor_name}_nis"
    if nis_column not in result_df.columns:
        return

    nis_values = pd.to_numeric(result_df[nis_column], errors="coerce")
    used_column = f"used_{sensor_name}"
    rejected_column = f"{sensor_name}_rejected"
    used_values = result_df[used_column].astype(bool) if used_column in result_df.columns else None
    rejected_values = result_df[rejected_column].astype(bool) if rejected_column in result_df.columns else None

    for threshold_kind in ("adapt", "reject"):
        threshold_column = f"{sensor_name}_nis_{threshold_kind}_threshold"
        if threshold_column not in result_df.columns:
            continue
        thresholds = pd.to_numeric(result_df[threshold_column], errors="coerce")
        valid_mask = nis_values.notna() & thresholds.notna()
        if valid_mask.any():
            exceed_mask = valid_mask & (nis_values > thresholds)
            metrics[f"{sensor_name}_nis_{threshold_kind}_exceed_count"] = float(exceed_mask.sum())
            if used_values is not None:
                metrics[f"{sensor_name}_nis_{threshold_kind}_exceed_used_count"] = float(
                    (exceed_mask & used_values).sum()
                )
            if rejected_values is not None:
                metrics[f"{sensor_name}_nis_{threshold_kind}_exceed_rejected_count"] = float(
                    (exceed_mask & rejected_values).sum()
                )
                metrics[f"{sensor_name}_nis_{threshold_kind}_exceed_not_rejected_count"] = float(
                    (exceed_mask & ~rejected_values).sum()
                )

    reject_exceed_key = f"{sensor_name}_nis_reject_exceed_count"
    reject_bypass_key = f"{sensor_name}_nis_reject_exceed_not_rejected_count"
    explicit_bypass_key = f"{sensor_name}_reject_bypassed_updates"
    if explicit_bypass_key not in metrics and reject_exceed_key in metrics and reject_bypass_key in metrics:
        reject_exceed_count = metrics[reject_exceed_key]
        reject_bypass_count = metrics[reject_bypass_key]
        metrics[f"{sensor_name}_nis_reject_policy_bypass_count"] = reject_bypass_count
        if reject_exceed_count > 0.0:
            metrics[f"{sensor_name}_nis_reject_policy_bypass_pct"] = 100.0 * reject_bypass_count / reject_exceed_count


def compute_metrics(
    result_df: pd.DataFrame,
    initialization_summary: dict[str, object] | None = None,
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if has_diagnostic_truth(result_df):
        position_error = result_df[["est_x", "est_y", "est_z"]].to_numpy() - result_df[list(truth_position_columns())].to_numpy()
        velocity_error = result_df[["est_vx", "est_vy", "est_vz"]].to_numpy() - result_df[list(truth_velocity_columns())].to_numpy()
        yaw_error = np.array(
            [
                wrap_angle(est - truth)
                for est, truth in zip(result_df["est_yaw"], result_df[truth_yaw_column()], strict=False)
            ]
        )
        metrics["position_rmse_m"] = float(np.sqrt(np.mean(np.sum(position_error**2, axis=1))))
        metrics["velocity_rmse_mps"] = float(np.sqrt(np.mean(np.sum(velocity_error**2, axis=1))))
        metrics["yaw_rmse_deg"] = float(np.rad2deg(np.sqrt(np.mean(yaw_error**2))))
        metrics["final_position_error_m"] = float(np.linalg.norm(position_error[-1]))
    metrics["mean_quality_score"] = float(result_df["quality_score"].mean())
    for sensor_name in ("gnss_pos", "gnss_vel", "baro", "mag"):
        _add_availability_metrics(metrics, result_df, sensor_name)
    metrics["gnss_pos_updates"] = _bool_column_sum(result_df, "used_gnss_pos")
    metrics["gnss_vel_updates"] = _bool_column_sum(result_df, "used_gnss_vel")
    metrics["baro_updates"] = _bool_column_sum(result_df, "used_baro")
    metrics["mag_updates"] = _bool_column_sum(result_df, "used_mag")
    for sensor_name in ("gnss_pos", "gnss_vel", "baro", "mag"):
        _add_rejection_metrics(metrics, result_df, sensor_name)
        _add_management_mode_metrics(metrics, result_df, sensor_name)
    _add_adaptive_r_metrics(metrics, result_df, "gnss_pos", "used_gnss_pos")
    _add_adaptive_r_metrics(metrics, result_df, "gnss_vel", "used_gnss_vel")
    _add_adaptive_r_metrics(metrics, result_df, "baro", "used_baro")
    _add_adaptive_r_metrics(metrics, result_df, "mag", "used_mag")
    _add_recovery_scale_metrics(metrics, result_df, "gnss_pos", "used_gnss_pos")
    _add_recovery_scale_metrics(metrics, result_df, "gnss_vel", "used_gnss_vel")
    _add_recovery_scale_metrics(metrics, result_df, "baro", "used_baro")
    _add_recovery_scale_metrics(metrics, result_df, "mag", "used_mag")
    _add_mode_scale_metrics(metrics, result_df, "gnss_pos", "used_gnss_pos", "gnss_pos_rejected")
    _add_mode_scale_metrics(metrics, result_df, "gnss_vel", "used_gnss_vel", "gnss_vel_rejected")
    for sensor_name in ("gnss_pos", "gnss_vel", "baro", "mag"):
        _add_nis_metrics(metrics, result_df, sensor_name)
        _add_nis_threshold_metrics(metrics, result_df, sensor_name)
        _add_reject_bypass_metrics(metrics, result_df, sensor_name)
    _add_innovation_metrics(metrics, result_df, "gnss_pos", "gnss_pos_innovation_norm")
    _add_innovation_metrics(metrics, result_df, "gnss_vel", "gnss_vel_innovation_norm")
    _add_innovation_metrics(metrics, result_df, "baro", "baro_innovation_abs")
    _add_innovation_metrics(metrics, result_df, "mag", "yaw_innovation_abs_deg")
    for sensor_name in ("gnss_pos", "gnss_vel", "baro", "mag"):
        reject_streak_column = f"{sensor_name}_reject_streak"
        skip_streak_column = f"{sensor_name}_skip_streak"
        reject_bypass_streak_column = f"{sensor_name}_reject_bypass_streak"
        adaptive_streak_column = f"{sensor_name}_adaptive_streak"
        outage_column = f"{sensor_name}_outage_s"
        available_outage_column = f"{sensor_name}_available_outage_s"
        if reject_streak_column in result_df.columns:
            metrics[f"max_{sensor_name}_reject_streak"] = float(pd.to_numeric(result_df[reject_streak_column], errors="coerce").max())
        if skip_streak_column in result_df.columns:
            metrics[f"max_{sensor_name}_skip_streak"] = float(pd.to_numeric(result_df[skip_streak_column], errors="coerce").max())
        if reject_bypass_streak_column in result_df.columns:
            metrics[f"max_{sensor_name}_reject_bypass_streak"] = float(
                pd.to_numeric(result_df[reject_bypass_streak_column], errors="coerce").max()
            )
        if adaptive_streak_column in result_df.columns:
            metrics[f"max_{sensor_name}_adaptive_streak"] = float(pd.to_numeric(result_df[adaptive_streak_column], errors="coerce").max())
        if outage_column in result_df.columns:
            metrics[f"max_{sensor_name}_outage_s"] = float(pd.to_numeric(result_df[outage_column], errors="coerce").max())
        if available_outage_column in result_df.columns:
            available_outage_values = pd.to_numeric(result_df[available_outage_column], errors="coerce").replace([np.inf, -np.inf], np.nan)
            metrics[f"max_{sensor_name}_available_outage_s"] = float(
                available_outage_values.max()
            )
    if "auxiliary_outage_s" in result_df.columns:
        metrics["max_auxiliary_outage_s"] = float(pd.to_numeric(result_df["auxiliary_outage_s"], errors="coerce").max())
    if "auxiliary_available_outage_s" in result_df.columns:
        auxiliary_available_outage_values = pd.to_numeric(
            result_df["auxiliary_available_outage_s"], errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
        metrics["max_auxiliary_available_outage_s"] = float(
            auxiliary_available_outage_values.max()
        )
    if "gravity_gradient_norm" in result_df.columns:
        metrics["mean_gravity_gradient_norm"] = float(result_df["gravity_gradient_norm"].mean())
        metrics["max_gravity_gradient_norm"] = float(result_df["gravity_gradient_norm"].max())
    if "coriolis_position_gradient_norm" in result_df.columns:
        metrics["mean_coriolis_position_gradient_norm"] = float(result_df["coriolis_position_gradient_norm"].mean())
        metrics["max_coriolis_position_gradient_norm"] = float(result_df["coriolis_position_gradient_norm"].max())
    if "coriolis_velocity_gradient_norm" in result_df.columns:
        metrics["mean_coriolis_velocity_gradient_norm"] = float(result_df["coriolis_velocity_gradient_norm"].mean())
        metrics["max_coriolis_velocity_gradient_norm"] = float(result_df["coriolis_velocity_gradient_norm"].max())
    if "pos_sigma_norm_m" in result_df.columns:
        metrics["final_pos_sigma_norm_m"] = float(result_df["pos_sigma_norm_m"].iloc[-1])
    if "vel_sigma_norm_mps" in result_df.columns:
        metrics["final_vel_sigma_norm_mps"] = float(result_df["vel_sigma_norm_mps"].iloc[-1])
    if "att_sigma_norm_deg" in result_df.columns:
        metrics["final_att_sigma_norm_deg"] = float(result_df["att_sigma_norm_deg"].iloc[-1])
    if "cov_min_diag" in result_df.columns:
        metrics["min_cov_diag"] = float(result_df["cov_min_diag"].min())
    if "predict_skipped" in result_df.columns:
        metrics["predict_skipped_count"] = float(result_df["predict_skipped"].astype(bool).sum())
    if "predict_warning" in result_df.columns:
        metrics["predict_warning_count"] = float(result_df["predict_warning"].astype(bool).sum())
    if "dt_raw_s" in result_df.columns:
        metrics["max_dt_raw_s"] = float(result_df["dt_raw_s"].max())
    if "dt_applied_s" in result_df.columns:
        metrics["max_dt_applied_s"] = float(result_df["dt_applied_s"].max())
    if "predict_reason" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "predict_reason", "predict_reason")
    if "mode" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "mode", "mode")
    if "mode_reason" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "mode_reason", "mode_reason")
    if "covariance_health" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "covariance_health", "covariance_health")
    if "covariance_health_reason" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "covariance_health_reason", "covariance_health_reason")
    if "covariance_caution" in result_df.columns:
        metrics["covariance_caution_row_count"] = float(result_df["covariance_caution"].astype(bool).sum())
    if "covariance_unhealthy" in result_df.columns:
        metrics["covariance_unhealthy_row_count"] = float(result_df["covariance_unhealthy"].astype(bool).sum())
    if "covariance_caution_duration_s" in result_df.columns:
        metrics["max_covariance_caution_duration_s"] = float(result_df["covariance_caution_duration_s"].max())
    if "covariance_unhealthy_duration_s" in result_df.columns:
        metrics["max_covariance_unhealthy_duration_s"] = float(result_df["covariance_unhealthy_duration_s"].max())
    if "mode_transition_pending" in result_df.columns:
        pending_series = result_df["mode_transition_pending"].astype(bool)
        metrics["pending_row_count"] = float(pending_series.sum())
        if "time" in result_df.columns and len(result_df) > 1:
            time_step = result_df["time"].diff().fillna(0.0)
            metrics["pending_duration_s"] = float(time_step[pending_series].sum())
    if "initialization_phase" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "initialization_phase", "initialization_phase")
        phase_series = result_df["initialization_phase"].fillna("").astype(str)
        metrics["initialization_row_count"] = float((phase_series != "").sum())
        metrics["initialization_pending_row_count"] = float((phase_series != "INITIALIZED").sum())
    if "initialization_completed_this_frame" in result_df.columns:
        metrics["initialization_completed_row_count"] = float(
            result_df["initialization_completed_this_frame"].astype(bool).sum()
        )
    if "initialization_reason" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "initialization_reason", "initialization_reason")
    if "initialization_heading_source" in result_df.columns:
        _add_categorical_duration_metrics(metrics, result_df, "initialization_heading_source", "initialization_heading_source")
    if "initialization_wait_s" in result_df.columns:
        wait_values = pd.to_numeric(result_df["initialization_wait_s"], errors="coerce")
        if wait_values.notna().any():
            metrics["max_initialization_wait_s"] = float(wait_values.max())
    _add_initialization_metrics(metrics, initialization_summary)
    return metrics


def save_metrics(
    metrics: dict[str, float],
    output_dir: Path,
    initialization_summary: dict[str, object] | None = None,
) -> None:
    metrics_frame = pd.DataFrame(
        [
            {
                "metric": metric_name,
                "value": metric_value,
                "category": metric_category(metric_name),
            }
            for metric_name, metric_value in metrics.items()
        ]
    )
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    lines = ["ESKF demo metrics", ""]
    initialization_lines = _format_initialization_summary(initialization_summary)
    if initialization_lines:
        lines.extend(initialization_lines)
        lines.append("")
    lines.extend(_format_metric_sections(metrics))
    (output_dir / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")
