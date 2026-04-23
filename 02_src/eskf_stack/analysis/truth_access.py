from __future__ import annotations

import pandas as pd

from ..adapters import DIAGNOSTIC_TRUTH_COLUMNS


def has_diagnostic_truth(result_df: pd.DataFrame) -> bool:
    return all(column in result_df.columns for column in DIAGNOSTIC_TRUTH_COLUMNS)


def truth_column_names() -> tuple[str, ...]:
    return DIAGNOSTIC_TRUTH_COLUMNS


def truth_position_columns() -> tuple[str, str, str]:
    return ("truth_x", "truth_y", "truth_z")


def truth_velocity_columns() -> tuple[str, str, str]:
    return ("truth_vx", "truth_vy", "truth_vz")


def truth_yaw_column() -> str:
    return "truth_yaw"
