from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ..core.math_utils import ensure_dir
from .evaluator import compute_metrics, save_metrics
from .plotter import (
    save_covariance_plot,
    save_error_plot,
    save_navigation_plot,
    save_quality_plot,
    save_state_machine_summary_plot,
    save_trajectory_plot,
)


def _has_columns(result_df: pd.DataFrame, required_columns: tuple[str, ...]) -> bool:
    return all(column in result_df.columns for column in required_columns)


def save_dataset_source_summary(
    metrics_dir: Path,
    config,
    source_summary: dict[str, str] | None,
    navigation_reference_override,
    initialization_summary: dict[str, str] | None,
) -> None:
    lines = [
        "Dataset Source Summary",
        "",
        f"config_path: {config.config_path}",
        f"config_profile: {config.config_metadata.profile}",
        f"config_name: {config.config_metadata.name}",
        f"config_purpose: {config.config_metadata.purpose}",
        f"fusion_policy_use_nis_rejection: {config.fusion_policy.use_nis_rejection}",
        f"fusion_policy_use_adaptive_r: {config.fusion_policy.use_adaptive_r}",
        f"fusion_policy_use_recovery_scale: {config.fusion_policy.use_recovery_scale}",
        "",
    ]
    if source_summary:
        for key, value in source_summary.items():
            lines.append(f"{key}: {value}")
    else:
        lines.append("adapter_kind: unknown")
    if initialization_summary:
        lines.extend(["", "initialization_summary:"])
        for key, value in initialization_summary.items():
            lines.append(f"{key}: {value}")
    if navigation_reference_override is not None:
        lines.extend(
            [
                "",
                "navigation_reference_override:",
                f"reference_lat_deg: {navigation_reference_override.reference_lat_deg:.9f}",
                f"reference_lon_deg: {navigation_reference_override.reference_lon_deg:.9f}",
                f"reference_height_m: {navigation_reference_override.reference_height_m:.6f}",
            ]
        )
    (metrics_dir / "dataset_source_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_pipeline_results(
    result_df: pd.DataFrame,
    results_root: Path,
    config,
    source_summary: dict[str, str] | None,
    navigation_reference_override,
    initialization_summary: dict[str, str],
    extra_metrics: dict[str, float] | None = None,
) -> dict[str, float]:
    figure_dir = ensure_dir(results_root / "figures")
    metrics_dir = ensure_dir(results_root / "metrics")

    result_df.to_csv(metrics_dir / "fusion_output.csv", index=False)
    save_dataset_source_summary(
        metrics_dir,
        config,
        source_summary,
        navigation_reference_override,
        initialization_summary,
    )

    if _has_columns(result_df, ("time", "est_x", "est_y", "est_z")):
        save_trajectory_plot(result_df, figure_dir / "trajectory.png")
    if _has_columns(result_df, ("time", "est_x", "est_y", "est_z", "est_vx", "est_vy", "est_vz", "est_yaw")):
        save_error_plot(result_df, figure_dir / "error_summary.png")
    if _has_columns(result_df, ("time", "quality_score", "mode")):
        save_quality_plot(result_df, figure_dir / "quality_mode.png")
    if _has_columns(result_df, ("time",)):
        save_navigation_plot(result_df, figure_dir / "navigation_diagnostics.png")
        save_covariance_plot(result_df, figure_dir / "covariance_diagnostics.png")
        save_state_machine_summary_plot(result_df, figure_dir / "state_machine_summary.png")

    metrics = compute_metrics(result_df, initialization_summary=initialization_summary)
    if extra_metrics:
        metrics.update(extra_metrics)
    save_metrics(metrics, metrics_dir, initialization_summary=initialization_summary)
    return metrics
