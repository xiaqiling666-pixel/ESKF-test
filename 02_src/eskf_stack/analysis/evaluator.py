from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.math_utils import wrap_angle
from .truth_access import has_diagnostic_truth, truth_position_columns, truth_velocity_columns, truth_yaw_column


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


def compute_metrics(result_df: pd.DataFrame) -> dict[str, float]:
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
    metrics["gnss_pos_updates"] = float(result_df["used_gnss_pos"].sum())
    metrics["baro_updates"] = float(result_df["used_baro"].sum())
    metrics["mag_updates"] = float(result_df["used_mag"].sum())
    if "gnss_pos_rejected" in result_df.columns:
        metrics["gnss_pos_rejections"] = float(result_df["gnss_pos_rejected"].sum())
    if "gnss_vel_rejected" in result_df.columns:
        metrics["gnss_vel_rejections"] = float(result_df["gnss_vel_rejected"].sum())
    if "gnss_pos_r_scale" in result_df.columns:
        metrics["gnss_pos_adapted_updates"] = float((result_df["gnss_pos_r_scale"] > 1.0).sum())
    if "gnss_vel_r_scale" in result_df.columns:
        metrics["gnss_vel_adapted_updates"] = float((result_df["gnss_vel_r_scale"] > 1.0).sum())
    if "gnss_pos_nis" in result_df.columns:
        valid_pos_nis = result_df["gnss_pos_nis"].dropna()
        if not valid_pos_nis.empty:
            metrics["mean_gnss_pos_nis"] = float(valid_pos_nis.mean())
    if "gnss_vel_nis" in result_df.columns:
        valid_vel_nis = result_df["gnss_vel_nis"].dropna()
        if not valid_vel_nis.empty:
            metrics["mean_gnss_vel_nis"] = float(valid_vel_nis.mean())
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
    return metrics


def save_metrics(metrics: dict[str, float], output_dir: Path) -> None:
    metrics_frame = pd.DataFrame(
        [{"metric": metric_name, "value": metric_value} for metric_name, metric_value in metrics.items()]
    )
    metrics_frame.to_csv(output_dir / "metrics.csv", index=False)

    lines = ["ESKF demo metrics", ""]
    for metric_name, metric_value in metrics.items():
        lines.append(f"{metric_name}: {metric_value:.6f}")
    (output_dir / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")
