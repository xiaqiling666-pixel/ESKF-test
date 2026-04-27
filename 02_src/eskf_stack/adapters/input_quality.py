from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contract import CORE_INPUT_COLUMNS, DIAGNOSTIC_TRUTH_COLUMNS, OPTIONAL_MEASUREMENT_COLUMNS


@dataclass(frozen=True)
class InputQualityReport:
    summary: dict[str, str]
    metrics: dict[str, float]


def _bool_text(value: bool) -> str:
    return "true" if value else "false"


def _complete_vector_mask(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    return dataframe.loc[:, list(columns)].notna().all(axis=1)


def _complete_scalar_mask(dataframe: pd.DataFrame, column: str) -> pd.Series:
    return dataframe[column].notna()


def _availability(
    name: str,
    mask: pd.Series,
    row_count: int,
    summary: dict[str, str],
    metrics: dict[str, float],
) -> int:
    available_rows = int(mask.sum())
    ratio = 0.0 if row_count == 0 else available_rows / row_count
    summary[f"input_quality_{name}_available_rows"] = str(available_rows)
    summary[f"input_quality_{name}_coverage_ratio"] = f"{ratio:.6f}"
    metrics[f"input_quality_{name}_available_rows"] = float(available_rows)
    metrics[f"input_quality_{name}_coverage_ratio"] = float(ratio)
    return available_rows


def _time_quality(raw_dataframe: pd.DataFrame, summary: dict[str, str], metrics: dict[str, float]) -> None:
    row_count = int(len(raw_dataframe))
    if "time" not in raw_dataframe.columns or row_count == 0:
        summary.update(
            {
                "input_quality_time_monotonic_input_order": "false",
                "input_quality_time_duplicate_count": "0",
                "input_quality_time_non_positive_dt_count": "0",
                "input_quality_time_large_gap_count": "0",
            }
        )
        metrics.update(
            {
                "input_quality_time_duplicate_count": 0.0,
                "input_quality_time_non_positive_dt_count": 0.0,
                "input_quality_time_large_gap_count": 0.0,
                "input_quality_time_max_gap_s": 0.0,
            }
        )
        return

    time_values = pd.to_numeric(raw_dataframe["time"], errors="coerce").to_numpy(dtype=float)
    finite_time_values = time_values[np.isfinite(time_values)]
    if finite_time_values.size == 0:
        raise ValueError("adapter 输入质量检查失败: time 字段没有可用数值")

    dt_values = np.diff(time_values)
    finite_dt_values = dt_values[np.isfinite(dt_values)]
    positive_dt_values = finite_dt_values[finite_dt_values > 0.0]
    duplicate_count = int(pd.Series(finite_time_values).duplicated().sum())
    non_positive_dt_count = int(np.sum(finite_dt_values <= 0.0))
    monotonic_input_order = bool(non_positive_dt_count == 0 and len(finite_time_values) == row_count)
    max_gap_s = 0.0 if positive_dt_values.size == 0 else float(np.max(positive_dt_values))
    median_gap_s = 0.0 if positive_dt_values.size == 0 else float(np.median(positive_dt_values))
    large_gap_threshold_s = 0.0 if median_gap_s <= 0.0 else max(1.0, 10.0 * median_gap_s)
    large_gap_count = 0 if large_gap_threshold_s <= 0.0 else int(np.sum(positive_dt_values > large_gap_threshold_s))

    summary.update(
        {
            "input_quality_time_start_s": f"{float(np.min(finite_time_values)):.6f}",
            "input_quality_time_end_s": f"{float(np.max(finite_time_values)):.6f}",
            "input_quality_time_duration_s": f"{float(np.max(finite_time_values) - np.min(finite_time_values)):.6f}",
            "input_quality_time_monotonic_input_order": _bool_text(monotonic_input_order),
            "input_quality_time_duplicate_count": str(duplicate_count),
            "input_quality_time_non_positive_dt_count": str(non_positive_dt_count),
            "input_quality_time_median_gap_s": f"{median_gap_s:.6f}",
            "input_quality_time_max_gap_s": f"{max_gap_s:.6f}",
            "input_quality_time_large_gap_threshold_s": f"{large_gap_threshold_s:.6f}",
            "input_quality_time_large_gap_count": str(large_gap_count),
        }
    )
    metrics.update(
        {
            "input_quality_time_duration_s": float(np.max(finite_time_values) - np.min(finite_time_values)),
            "input_quality_time_duplicate_count": float(duplicate_count),
            "input_quality_time_non_positive_dt_count": float(non_positive_dt_count),
            "input_quality_time_monotonic_input_order_flag": 1.0 if monotonic_input_order else 0.0,
            "input_quality_time_median_gap_s": median_gap_s,
            "input_quality_time_max_gap_s": max_gap_s,
            "input_quality_time_large_gap_threshold_s": large_gap_threshold_s,
            "input_quality_time_large_gap_count": float(large_gap_count),
        }
    )


def build_input_quality_report(raw_dataframe: pd.DataFrame, standardized_dataframe: pd.DataFrame) -> InputQualityReport:
    row_count = int(len(standardized_dataframe))
    raw_columns = set(raw_dataframe.columns)
    missing_required = sorted(set(CORE_INPUT_COLUMNS).difference(raw_columns))
    missing_optional_measurements = sorted(set(OPTIONAL_MEASUREMENT_COLUMNS).difference(raw_columns))
    missing_diagnostic_truth = sorted(set(DIAGNOSTIC_TRUTH_COLUMNS).difference(raw_columns))

    summary: dict[str, str] = {
        "input_quality_row_count": str(row_count),
        "input_quality_required_missing_columns": ",".join(missing_required),
        "input_quality_optional_measurement_missing_columns": ",".join(missing_optional_measurements),
        "input_quality_diagnostic_truth_missing_columns": ",".join(missing_diagnostic_truth),
        "input_quality_diagnostic_truth_role": "evaluation_only_not_filter_input",
    }
    metrics: dict[str, float] = {
        "input_quality_row_count": float(row_count),
        "input_quality_required_missing_column_count": float(len(missing_required)),
        "input_quality_optional_measurement_missing_column_count": float(len(missing_optional_measurements)),
        "input_quality_diagnostic_truth_missing_column_count": float(len(missing_diagnostic_truth)),
    }

    _time_quality(raw_dataframe, summary, metrics)

    core_complete_mask = _complete_vector_mask(standardized_dataframe, CORE_INPUT_COLUMNS)
    core_complete_rows = int(core_complete_mask.sum())
    core_complete_ratio = 0.0 if row_count == 0 else core_complete_rows / row_count
    summary["input_quality_core_complete_rows"] = str(core_complete_rows)
    summary["input_quality_core_complete_ratio"] = f"{core_complete_ratio:.6f}"
    metrics["input_quality_core_complete_rows"] = float(core_complete_rows)
    metrics["input_quality_core_complete_ratio"] = float(core_complete_ratio)

    gnss_pos_rows = _availability(
        "gnss_pos",
        _complete_vector_mask(standardized_dataframe, ("gnss_x", "gnss_y", "gnss_z")),
        row_count,
        summary,
        metrics,
    )
    gnss_vel_rows = _availability(
        "gnss_vel",
        _complete_vector_mask(standardized_dataframe, ("gnss_vx", "gnss_vy", "gnss_vz")),
        row_count,
        summary,
        metrics,
    )
    baro_rows = _availability(
        "baro",
        _complete_scalar_mask(standardized_dataframe, "baro_h"),
        row_count,
        summary,
        metrics,
    )
    mag_rows = _availability(
        "mag_yaw",
        _complete_scalar_mask(standardized_dataframe, "mag_yaw"),
        row_count,
        summary,
        metrics,
    )
    truth_pos_rows = _availability(
        "truth_pos",
        _complete_vector_mask(standardized_dataframe, ("truth_x", "truth_y", "truth_z")),
        row_count,
        summary,
        metrics,
    )
    truth_vel_rows = _availability(
        "truth_vel",
        _complete_vector_mask(standardized_dataframe, ("truth_vx", "truth_vy", "truth_vz")),
        row_count,
        summary,
        metrics,
    )
    truth_yaw_rows = _availability(
        "truth_yaw",
        _complete_scalar_mask(standardized_dataframe, "truth_yaw"),
        row_count,
        summary,
        metrics,
    )

    has_any_measurement = any(value > 0 for value in (gnss_pos_rows, gnss_vel_rows, baro_rows, mag_rows))
    has_any_truth = any(value > 0 for value in (truth_pos_rows, truth_vel_rows, truth_yaw_rows))
    summary["input_quality_optional_measurement_present"] = _bool_text(has_any_measurement)
    summary["input_quality_diagnostic_truth_present"] = _bool_text(has_any_truth)
    metrics["input_quality_optional_measurement_present_flag"] = 1.0 if has_any_measurement else 0.0
    metrics["input_quality_diagnostic_truth_present_flag"] = 1.0 if has_any_truth else 0.0

    return InputQualityReport(summary=summary, metrics=metrics)
