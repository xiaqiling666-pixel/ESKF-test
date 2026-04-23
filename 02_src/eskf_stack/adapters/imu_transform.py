from __future__ import annotations

import pandas as pd


SUPPORTED_IMU_TRANSFORM_MODES = {
    "raw",
    "ardupilot_frd_to_flu",
    "flip_z_accel",
    "negate_all_accel",
}


def apply_imu_transform(
    dataframe: pd.DataFrame,
    accel_columns: tuple[str, str, str],
    gyro_columns: tuple[str, str, str],
    transform_mode: str,
) -> pd.DataFrame:
    if transform_mode == "raw":
        return dataframe
    if transform_mode == "flip_z_accel":
        dataframe.loc[:, accel_columns[2]] = -dataframe[accel_columns[2]].to_numpy(dtype=float)
        return dataframe
    if transform_mode == "negate_all_accel":
        for column_name in accel_columns:
            dataframe.loc[:, column_name] = -dataframe[column_name].to_numpy(dtype=float)
        return dataframe
    if transform_mode == "ardupilot_frd_to_flu":
        dataframe.loc[:, accel_columns[1]] = -dataframe[accel_columns[1]].to_numpy(dtype=float)
        dataframe.loc[:, accel_columns[2]] = -dataframe[accel_columns[2]].to_numpy(dtype=float)
        dataframe.loc[:, gyro_columns[1]] = -dataframe[gyro_columns[1]].to_numpy(dtype=float)
        dataframe.loc[:, gyro_columns[2]] = -dataframe[gyro_columns[2]].to_numpy(dtype=float)
        return dataframe
    raise ValueError(f"不支持的 imu_transform_mode: {transform_mode}")
