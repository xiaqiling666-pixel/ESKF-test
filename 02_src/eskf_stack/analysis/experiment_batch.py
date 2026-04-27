from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import load_config, project_path


DEFAULT_EXPERIMENT_CONFIGS = [
    "01_data/config_experiment_baseline_eskf.json",
    "01_data/config_experiment_nis_reject.json",
    "01_data/config_experiment_adaptive_r.json",
    "01_data/config_experiment_adaptive_r_recovery.json",
    "01_data/config_experiment_full_method.json",
]


@dataclass(frozen=True)
class ExperimentBatchResult:
    summary_path: Path
    key_summary_path: Path
    run_count: int


KEY_SUMMARY_COLUMNS = [
    "experiment_name",
    "use_nis_rejection",
    "use_adaptive_r",
    "use_recovery_scale",
    "position_rmse_m",
    "velocity_rmse_mps",
    "yaw_rmse_deg",
    "final_position_error_m",
    "mean_quality_score",
    "gnss_pos_updates",
    "gnss_pos_rejections",
    "gnss_pos_adapted_updates",
    "gnss_pos_mode_scaled_updates",
    "gnss_vel_mode_scaled_updates",
    "mean_gnss_pos_nis",
    "final_pos_sigma_norm_m",
    "final_vel_sigma_norm_mps",
    "covariance_caution_row_count",
    "covariance_unhealthy_row_count",
    "mode_transition_count",
    "initialization_completed_flag",
    "initialization_mode_direct_flag",
    "initialization_mode_bootstrap_position_pair_flag",
    "processed_rows",
    "pipeline_runtime_s",
]


def _read_metrics_csv(metrics_path: Path) -> dict[str, float]:
    metrics_frame = pd.read_csv(metrics_path)
    if not {"metric", "value"}.issubset(metrics_frame.columns):
        raise ValueError(f"{metrics_path} must contain metric and value columns")
    return {
        str(row.metric): float(row.value)
        for row in metrics_frame.itertuples(index=False)
    }


def _experiment_row(config_path: str | Path, metrics: dict[str, float]) -> dict[str, object]:
    config = load_config(config_path)
    row: dict[str, object] = {
        "experiment_name": config.config_metadata.name,
        "config_path": str(config_path),
        "results_dir": config.results_dir,
        "use_nis_rejection": config.fusion_policy.use_nis_rejection,
        "use_adaptive_r": config.fusion_policy.use_adaptive_r,
        "use_recovery_scale": config.fusion_policy.use_recovery_scale,
    }
    row.update(metrics)
    return row


def run_experiment_batch(
    config_paths: list[str | Path] | None = None,
    summary_dir: str | Path = "03_results_experiment_batch",
    run_pipeline_fn: Callable[[str | Path], object] | None = None,
) -> ExperimentBatchResult:
    selected_config_paths = DEFAULT_EXPERIMENT_CONFIGS if config_paths is None else config_paths
    if run_pipeline_fn is None:
        from ..app import run_pipeline

        run_pipeline_fn = run_pipeline

    rows: list[dict[str, object]] = []
    for config_path in selected_config_paths:
        run_pipeline_fn(config_path)
        config = load_config(config_path)
        metrics_path = project_path(config.results_dir) / "metrics" / "metrics.csv"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing metrics output for {config_path}: {metrics_path}")
        rows.append(_experiment_row(config_path, _read_metrics_csv(metrics_path)))

    resolved_summary_dir = project_path(str(summary_dir))
    resolved_summary_dir.mkdir(parents=True, exist_ok=True)
    summary_frame = pd.DataFrame(rows)
    summary_path = resolved_summary_dir / "experiment_metrics_summary.csv"
    key_summary_path = resolved_summary_dir / "experiment_metrics_key_summary.csv"
    summary_frame.to_csv(summary_path, index=False)
    summary_frame[[column for column in KEY_SUMMARY_COLUMNS if column in summary_frame.columns]].to_csv(
        key_summary_path,
        index=False,
    )
    return ExperimentBatchResult(summary_path=summary_path, key_summary_path=key_summary_path, run_count=len(rows))
