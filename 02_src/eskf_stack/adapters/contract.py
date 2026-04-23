from __future__ import annotations

import pandas as pd


CORE_INPUT_COLUMNS: tuple[str, ...] = (
    "time",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
)

OPTIONAL_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "gnss_x",
    "gnss_y",
    "gnss_z",
    "gnss_vx",
    "gnss_vy",
    "gnss_vz",
    "baro_h",
    "mag_yaw",
)

DIAGNOSTIC_TRUTH_COLUMNS: tuple[str, ...] = (
    "truth_x",
    "truth_y",
    "truth_z",
    "truth_vx",
    "truth_vy",
    "truth_vz",
    "truth_yaw",
)

REQUIRED_SENSOR_COLUMNS: tuple[str, ...] = CORE_INPUT_COLUMNS
OPTIONAL_SENSOR_COLUMNS: tuple[str, ...] = OPTIONAL_MEASUREMENT_COLUMNS + DIAGNOSTIC_TRUTH_COLUMNS


def standard_sensor_columns() -> tuple[str, ...]:
    return CORE_INPUT_COLUMNS + OPTIONAL_MEASUREMENT_COLUMNS + DIAGNOSTIC_TRUTH_COLUMNS


def contract_column_groups() -> dict[str, tuple[str, ...]]:
    return {
        "core_input": CORE_INPUT_COLUMNS,
        "optional_measurement": OPTIONAL_MEASUREMENT_COLUMNS,
        "diagnostic_truth": DIAGNOSTIC_TRUTH_COLUMNS,
    }


def ensure_standard_sensor_dataframe(dataframe: pd.DataFrame, source_name: str = "dataset") -> pd.DataFrame:
    missing_required = sorted(set(CORE_INPUT_COLUMNS).difference(dataframe.columns))
    if missing_required:
        raise ValueError(f"{source_name} 缺少统一输入契约必需字段: {', '.join(missing_required)}")

    standardized = dataframe.copy()
    for column_name in standard_sensor_columns():
        if column_name in standardized.columns:
            standardized[column_name] = pd.to_numeric(standardized[column_name], errors="raise")
        else:
            standardized[column_name] = float("nan")

    if standardized["time"].isna().any():
        raise ValueError(f"{source_name} 的 time 字段存在空值，无法作为统一时间轴")

    return standardized.sort_values("time").reset_index(drop=True)
