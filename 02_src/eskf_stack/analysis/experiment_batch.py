from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from ..config import load_config, project_path
from .evaluator import (
    metric_category,
    metric_experiment_comparison_direction,
    metric_supports_experiment_delta,
)


DEFAULT_EXPERIMENT_CONFIGS = [
    "01_data/config_experiment_baseline_eskf.json",
    "01_data/config_experiment_nis_reject.json",
    "01_data/config_experiment_adaptive_r.json",
    "01_data/config_experiment_adaptive_r_recovery.json",
    "01_data/config_experiment_full_method.json",
]

KEY_SUMMARY_PREVIEW_LIMIT = 6
CATEGORY_SUMMARY_PREVIEW_LIMIT = 4
DELTA_EQUALITY_TOLERANCE = 1e-12


@dataclass(frozen=True)
class ExperimentBatchResult:
    summary_path: Path
    key_summary_path: Path
    delta_summary_path: Path
    delta_key_summary_path: Path
    core_compare_path: Path
    manifest_path: Path
    baseline_experiment_name: str
    key_summary_columns: list[str]
    key_summary_preview_columns: list[str]
    core_compare_columns: list[str]
    category_summary_names: list[str]
    category_summary_preview_names: list[str]
    category_summary_paths: dict[str, Path]
    category_metric_columns: dict[str, list[str]]
    delta_category_summary_paths: dict[str, Path]
    delta_category_metric_columns: dict[str, list[str]]
    delta_category_source_metrics: dict[str, list[str]]
    run_count: int


SUMMARY_METADATA_COLUMNS = [
    "experiment_name",
    "config_path",
    "results_dir",
    "use_nis_rejection",
    "use_adaptive_r",
    "use_recovery_scale",
]

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
    "baro_updates",
    "baro_rejections",
    "mag_updates",
    "mag_rejections",
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


CATEGORY_SUMMARY_FILES = {
    "estimation_error": "experiment_metrics_estimation_error_summary.csv",
    "measurement_management": "experiment_metrics_measurement_management_summary.csv",
    "initialization": "experiment_metrics_initialization_summary.csv",
    "prediction_diagnostics": "experiment_metrics_prediction_diagnostics_summary.csv",
    "covariance_health": "experiment_metrics_covariance_health_summary.csv",
    "mode_state": "experiment_metrics_mode_state_summary.csv",
    "input_quality": "experiment_metrics_input_quality_summary.csv",
    "navigation_environment": "experiment_metrics_navigation_environment_summary.csv",
    "runtime": "experiment_metrics_runtime_summary.csv",
}

CORE_BASELINE_COMPARE_METRICS = [
    "position_rmse_m",
    "velocity_rmse_mps",
    "yaw_rmse_deg",
    "final_position_error_m",
    "mean_quality_score",
    "gnss_pos_rejections",
    "gnss_vel_rejections",
    "baro_rejections",
    "mag_rejections",
    "covariance_unhealthy_row_count",
    "initialization_completed_flag",
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


def _summary_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column in frame.columns]


def _write_summary(frame: pd.DataFrame, output_path: Path, columns: list[str]) -> None:
    selected_columns = _summary_columns(frame, columns)
    if selected_columns:
        frame[selected_columns].to_csv(output_path, index=False)


def _category_metric_columns(frame: pd.DataFrame, category_name: str) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in SUMMARY_METADATA_COLUMNS and metric_category(column) == category_name
    ]


def _select_baseline_experiment_name(summary_frame: pd.DataFrame) -> str:
    experiment_names = [str(name) for name in summary_frame.get("experiment_name", pd.Series(dtype=str)).tolist()]
    for expected_name in ("baseline_eskf",):
        if expected_name in experiment_names:
            return expected_name
    for experiment_name in experiment_names:
        if "baseline" in experiment_name.lower():
            return experiment_name
    if experiment_names:
        return experiment_names[0]
    raise ValueError("Experiment summary is empty; cannot select baseline experiment")


def _delta_metric_columns(frame: pd.DataFrame) -> list[str]:
    columns: list[str] = []
    for column in frame.columns:
        if column in SUMMARY_METADATA_COLUMNS:
            continue
        if column == "experiment_name":
            continue
        if not metric_supports_experiment_delta(column):
            continue
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        columns.append(column)
    return columns


def _delta_column_name(metric_name: str) -> str:
    return f"delta_{metric_name}"


def _delta_metric_names_for_category(
    metric_columns: list[str],
    delta_metric_columns: list[str],
) -> tuple[list[str], list[str]]:
    delta_metric_set = set(delta_metric_columns)
    source_metric_names = [metric_name for metric_name in metric_columns if metric_name in delta_metric_set]
    return source_metric_names, [_delta_column_name(metric_name) for metric_name in source_metric_names]


def _build_delta_summary(
    summary_frame: pd.DataFrame,
    baseline_experiment_name: str,
    delta_metric_columns: list[str],
) -> pd.DataFrame:
    baseline_rows = summary_frame[summary_frame["experiment_name"] == baseline_experiment_name]
    if baseline_rows.empty:
        raise ValueError(f"Baseline experiment not found in summary frame: {baseline_experiment_name}")

    baseline_row = baseline_rows.iloc[0]
    metadata_columns = _summary_columns(summary_frame, list(dict.fromkeys(["experiment_name"] + SUMMARY_METADATA_COLUMNS)))
    delta_frame = summary_frame[metadata_columns].copy()
    delta_frame["baseline_experiment_name"] = baseline_experiment_name
    delta_frame["is_baseline"] = summary_frame["experiment_name"].astype(str) == baseline_experiment_name

    for metric_name in delta_metric_columns:
        delta_series = pd.to_numeric(summary_frame[metric_name], errors="coerce") - float(baseline_row[metric_name])
        delta_frame[_delta_column_name(metric_name)] = delta_series.round(12)
    return delta_frame


def _comparison_status(delta_value: float, direction: str | None) -> str:
    if direction is None or pd.isna(delta_value):
        return "unsupported"
    if abs(float(delta_value)) <= DELTA_EQUALITY_TOLERANCE:
        return "unchanged"
    if direction == "lower_better":
        return "improved" if float(delta_value) < 0.0 else "regressed"
    if direction == "higher_better":
        return "improved" if float(delta_value) > 0.0 else "regressed"
    return "unsupported"


def _build_core_baseline_compare_summary(
    delta_summary_frame: pd.DataFrame,
    delta_metric_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    selected_metrics = [metric_name for metric_name in CORE_BASELINE_COMPARE_METRICS if metric_name in delta_metric_columns]
    compare_frame = delta_summary_frame[["experiment_name", "baseline_experiment_name", "is_baseline"]].copy()
    core_compare_columns = ["experiment_name", "baseline_experiment_name", "is_baseline"]

    if not selected_metrics:
        return compare_frame, core_compare_columns

    for metric_name in selected_metrics:
        delta_column = _delta_column_name(metric_name)
        status_column = f"{metric_name}_vs_baseline"
        direction = metric_experiment_comparison_direction(metric_name)
        compare_frame[delta_column] = delta_summary_frame[delta_column]
        compare_frame[status_column] = delta_summary_frame[delta_column].apply(lambda value: _comparison_status(value, direction))
        core_compare_columns.extend([delta_column, status_column])

    status_columns = [f"{metric_name}_vs_baseline" for metric_name in selected_metrics]
    compare_frame["improved_metric_count"] = compare_frame[status_columns].eq("improved").sum(axis=1).astype(float)
    compare_frame["regressed_metric_count"] = compare_frame[status_columns].eq("regressed").sum(axis=1).astype(float)
    core_compare_columns.extend(["improved_metric_count", "regressed_metric_count"])
    return compare_frame, core_compare_columns


def _key_summary_preview_columns(columns: list[str], limit: int = KEY_SUMMARY_PREVIEW_LIMIT) -> list[str]:
    return columns[:limit]


def _category_summary_preview_names(
    category_names: list[str],
    limit: int = CATEGORY_SUMMARY_PREVIEW_LIMIT,
) -> list[str]:
    return category_names[:limit]


def _write_manifest(
    output_path: Path,
    *,
    summary_path: Path,
    key_summary_path: Path,
    delta_summary_path: Path,
    delta_key_summary_path: Path,
    core_compare_path: Path,
    key_summary_columns: list[str],
    key_summary_preview_columns: list[str],
    core_compare_columns: list[str],
    baseline_experiment_name: str,
    category_summary_names: list[str],
    category_summary_preview_names: list[str],
    category_summary_paths: dict[str, Path],
    category_metric_columns: dict[str, list[str]],
    delta_category_summary_paths: dict[str, Path],
    delta_category_metric_columns: dict[str, list[str]],
    delta_category_source_metrics: dict[str, list[str]],
    run_count: int,
) -> None:
    payload = {
        "run_count": run_count,
        "summary_path": str(summary_path),
        "key_summary_path": str(key_summary_path),
        "delta_summary_path": str(delta_summary_path),
        "delta_key_summary_path": str(delta_key_summary_path),
        "core_compare_path": str(core_compare_path),
        "baseline_experiment_name": baseline_experiment_name,
        "key_summary_columns": key_summary_columns,
        "key_summary_preview_columns": key_summary_preview_columns,
        "core_compare_columns": core_compare_columns,
        "key_summary_preview_limit": KEY_SUMMARY_PREVIEW_LIMIT,
        "category_summary_names": category_summary_names,
        "category_summary_preview_names": category_summary_preview_names,
        "category_summary_preview_limit": CATEGORY_SUMMARY_PREVIEW_LIMIT,
        "category_summary_paths": {name: str(path) for name, path in category_summary_paths.items()},
        "category_metric_columns": category_metric_columns,
        "delta_category_summary_paths": {name: str(path) for name, path in delta_category_summary_paths.items()},
        "delta_category_metric_columns": delta_category_metric_columns,
        "delta_category_source_metrics": delta_category_source_metrics,
    }
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
    delta_summary_path = resolved_summary_dir / "experiment_metrics_delta_vs_baseline.csv"
    delta_key_summary_path = resolved_summary_dir / "experiment_metrics_delta_key_summary.csv"
    core_compare_path = resolved_summary_dir / "experiment_metrics_core_baseline_compare.csv"
    manifest_path = resolved_summary_dir / "experiment_metrics_manifest.json"
    summary_frame.to_csv(summary_path, index=False)
    key_summary_columns = _summary_columns(summary_frame, KEY_SUMMARY_COLUMNS)
    key_summary_preview_columns = _key_summary_preview_columns(key_summary_columns)
    _write_summary(summary_frame, key_summary_path, key_summary_columns)
    baseline_experiment_name = _select_baseline_experiment_name(summary_frame)
    delta_metric_columns = _delta_metric_columns(summary_frame)
    delta_summary_frame = _build_delta_summary(summary_frame, baseline_experiment_name, delta_metric_columns)
    delta_summary_frame.to_csv(delta_summary_path, index=False)
    delta_key_summary_columns = _summary_columns(
        delta_summary_frame,
        ["experiment_name", "baseline_experiment_name", "is_baseline"]
        + [_delta_column_name(column) for column in KEY_SUMMARY_COLUMNS if column in delta_metric_columns],
    )
    _write_summary(delta_summary_frame, delta_key_summary_path, delta_key_summary_columns)
    core_compare_frame, core_compare_columns = _build_core_baseline_compare_summary(
        delta_summary_frame,
        delta_metric_columns,
    )
    _write_summary(core_compare_frame, core_compare_path, core_compare_columns)

    category_summary_paths: dict[str, Path] = {}
    category_metric_columns: dict[str, list[str]] = {}
    delta_category_summary_paths: dict[str, Path] = {}
    delta_category_metric_columns: dict[str, list[str]] = {}
    delta_category_source_metrics: dict[str, list[str]] = {}
    for category_name, filename in CATEGORY_SUMMARY_FILES.items():
        metric_columns = _category_metric_columns(summary_frame, category_name)
        if not metric_columns:
            continue
        category_path = resolved_summary_dir / filename
        _write_summary(
            summary_frame,
            category_path,
            SUMMARY_METADATA_COLUMNS + metric_columns,
        )
        category_summary_paths[category_name] = category_path
        category_metric_columns[category_name] = metric_columns
        source_metric_names, delta_metric_names = _delta_metric_names_for_category(metric_columns, delta_metric_columns)
        if delta_metric_names:
            delta_category_path = resolved_summary_dir / filename.replace("_summary.csv", "_delta_summary.csv")
            _write_summary(
                delta_summary_frame,
                delta_category_path,
                ["experiment_name", "baseline_experiment_name", "is_baseline"] + delta_metric_names,
            )
            delta_category_summary_paths[category_name] = delta_category_path
            delta_category_metric_columns[category_name] = delta_metric_names
            delta_category_source_metrics[category_name] = source_metric_names
    category_summary_names = sorted(category_summary_paths)
    category_summary_preview_names = _category_summary_preview_names(category_summary_names)
    _write_manifest(
        manifest_path,
        summary_path=summary_path,
        key_summary_path=key_summary_path,
        delta_summary_path=delta_summary_path,
        delta_key_summary_path=delta_key_summary_path,
        core_compare_path=core_compare_path,
        key_summary_columns=key_summary_columns,
        key_summary_preview_columns=key_summary_preview_columns,
        core_compare_columns=core_compare_columns,
        baseline_experiment_name=baseline_experiment_name,
        category_summary_names=category_summary_names,
        category_summary_preview_names=category_summary_preview_names,
        category_summary_paths=category_summary_paths,
        category_metric_columns=category_metric_columns,
        delta_category_summary_paths=delta_category_summary_paths,
        delta_category_metric_columns=delta_category_metric_columns,
        delta_category_source_metrics=delta_category_source_metrics,
        run_count=len(rows),
    )

    return ExperimentBatchResult(
        summary_path=summary_path,
        key_summary_path=key_summary_path,
        delta_summary_path=delta_summary_path,
        delta_key_summary_path=delta_key_summary_path,
        core_compare_path=core_compare_path,
        manifest_path=manifest_path,
        baseline_experiment_name=baseline_experiment_name,
        key_summary_columns=key_summary_columns,
        key_summary_preview_columns=key_summary_preview_columns,
        core_compare_columns=core_compare_columns,
        category_summary_names=category_summary_names,
        category_summary_preview_names=category_summary_preview_names,
        category_summary_paths=category_summary_paths,
        category_metric_columns=category_metric_columns,
        delta_category_summary_paths=delta_category_summary_paths,
        delta_category_metric_columns=delta_category_metric_columns,
        delta_category_source_metrics=delta_category_source_metrics,
        run_count=len(rows),
    )
