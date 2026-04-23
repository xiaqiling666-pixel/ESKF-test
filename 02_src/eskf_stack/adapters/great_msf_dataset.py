from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, project_path
from .loader import DatasetLoadResult, NavigationReferenceOverride


WGS84_A = 6378137.0
WGS84_E2 = 0.0066943799901413156


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_path(path_like)


def _read_great_msf_imu_txt(imu_path: Path) -> pd.DataFrame:
    dataframe = pd.read_csv(
        imu_path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=["time", "ax", "ay", "az", "gx", "gy", "gz"],
    )
    if dataframe.empty:
        raise ValueError(f"GREAT-MSF IMU 文件为空: {imu_path}")
    return dataframe.sort_values("time").reset_index(drop=True)


def _read_great_msf_ins(ins_path: Path) -> pd.DataFrame:
    column_names = [
        "time",
        "ecef_x",
        "ecef_y",
        "ecef_z",
        "ecef_vx",
        "ecef_vy",
        "ecef_vz",
        "pitch_deg",
        "roll_deg",
        "yaw_deg",
        "gyro_bias_x",
        "gyro_bias_y",
        "gyro_bias_z",
        "acce_bias_x",
        "acce_bias_y",
        "acce_bias_z",
        "meas_type",
        "nsat",
        "pdop",
        "amb_status",
        "ratio",
    ]
    dataframe = pd.read_csv(
        ins_path,
        sep=r"\s+",
        header=None,
        comment="#",
        names=column_names,
        usecols=list(range(len(column_names))),
    )
    if dataframe.empty:
        raise ValueError(f"GREAT-MSF INS 文件为空: {ins_path}")
    return dataframe.sort_values("time").reset_index(drop=True)


def _ecef_to_geodetic(ecef_xyz: np.ndarray) -> tuple[float, float, float]:
    x, y, z = ecef_xyz
    longitude = np.arctan2(y, x)
    p = float(np.hypot(x, y))
    latitude = np.arctan2(z, p * (1.0 - WGS84_E2))

    for _ in range(8):
        sin_lat = np.sin(latitude)
        prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        height = p / max(np.cos(latitude), 1e-12) - prime_vertical
        latitude = np.arctan2(z, p * (1.0 - WGS84_E2 * prime_vertical / max(prime_vertical + height, 1e-12)))

    sin_lat = np.sin(latitude)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    height = p / max(np.cos(latitude), 1e-12) - prime_vertical
    return float(latitude), float(longitude), float(height)


def _geodetic_to_ecef(latitude_rad: float, longitude_rad: float, height_m: float) -> np.ndarray:
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_lon = np.sin(longitude_rad)
    cos_lon = np.cos(longitude_rad)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    return np.array(
        [
            (prime_vertical + height_m) * cos_lat * cos_lon,
            (prime_vertical + height_m) * cos_lat * sin_lon,
            (prime_vertical * (1.0 - WGS84_E2) + height_m) * sin_lat,
        ],
        dtype=float,
    )


def _ecef_to_enu_rotation(reference_lat_rad: float, reference_lon_rad: float) -> np.ndarray:
    sin_lat = np.sin(reference_lat_rad)
    cos_lat = np.cos(reference_lat_rad)
    sin_lon = np.sin(reference_lon_rad)
    cos_lon = np.cos(reference_lon_rad)
    return np.array(
        [
            [-sin_lon, cos_lon, 0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [cos_lat * cos_lon, cos_lat * sin_lon, sin_lat],
        ],
        dtype=float,
    )


def _choose_reference(
    config: AppConfig,
    ins_df: pd.DataFrame,
    reference_mode: str,
) -> tuple[float, float, float, NavigationReferenceOverride | None]:
    if reference_mode == "config":
        return (
            float(np.deg2rad(config.navigation_environment.reference_lat_deg)),
            float(np.deg2rad(config.navigation_environment.reference_lon_deg)),
            float(config.navigation_environment.reference_height_m),
            None,
        )

    if reference_mode == "first_ins":
        first_ecef = ins_df.loc[0, ["ecef_x", "ecef_y", "ecef_z"]].to_numpy(dtype=float)
        reference_lat_rad, reference_lon_rad, reference_height_m = _ecef_to_geodetic(first_ecef)
        return (
            reference_lat_rad,
            reference_lon_rad,
            reference_height_m,
            NavigationReferenceOverride(
                reference_lat_deg=float(np.rad2deg(reference_lat_rad)),
                reference_lon_deg=float(np.rad2deg(reference_lon_rad)),
                reference_height_m=reference_height_m,
            ),
        )

    raise ValueError(f"不支持的 GREAT-MSF 参考系模式: {reference_mode}")


def _assign_nearest_indices(imu_times: np.ndarray, measurement_times: np.ndarray, tolerance_s: float) -> dict[int, int]:
    assignments: dict[int, tuple[float, int]] = {}
    for measurement_index, measurement_time in enumerate(measurement_times):
        insert_index = int(np.searchsorted(imu_times, measurement_time))
        candidate_indices = []
        if insert_index < len(imu_times):
            candidate_indices.append(insert_index)
        if insert_index > 0:
            candidate_indices.append(insert_index - 1)
        if not candidate_indices:
            continue
        nearest_index = min(candidate_indices, key=lambda index: abs(float(imu_times[index] - measurement_time)))
        time_diff = abs(float(imu_times[nearest_index] - measurement_time))
        if time_diff > tolerance_s:
            continue
        current = assignments.get(nearest_index)
        if current is None or time_diff < current[0]:
            assignments[nearest_index] = (time_diff, measurement_index)
    return {imu_index: measurement_index for imu_index, (_, measurement_index) in assignments.items()}


def load_great_msf_imu_ins_dataset(config: AppConfig) -> DatasetLoadResult:
    options = config.dataset_adapter.options
    imu_path_like = options.get("imu_txt_path")
    ins_path_like = options.get("ins_path")
    if not isinstance(imu_path_like, str) or not isinstance(ins_path_like, str):
        raise ValueError("great_msf_imu_ins 适配器需要 dataset_adapter.options.imu_txt_path 和 ins_path")

    imu_path = _resolve_path(imu_path_like)
    ins_path = _resolve_path(ins_path_like)
    normalize_time_to_zero = bool(options.get("normalize_time_to_zero", True))
    reference_mode = str(options.get("reference_mode", "first_ins"))
    alignment_tolerance_s = float(options.get("alignment_tolerance_s", 0.03))

    imu_df = _read_great_msf_imu_txt(imu_path)
    ins_df = _read_great_msf_ins(ins_path)

    reference_lat_rad, reference_lon_rad, reference_height_m, reference_override = _choose_reference(
        config,
        ins_df,
        reference_mode=reference_mode,
    )
    reference_ecef = _geodetic_to_ecef(reference_lat_rad, reference_lon_rad, reference_height_m)
    rot_ecef_to_enu = _ecef_to_enu_rotation(reference_lat_rad, reference_lon_rad)

    ecef_positions = ins_df[["ecef_x", "ecef_y", "ecef_z"]].to_numpy(dtype=float)
    ecef_velocities = ins_df[["ecef_vx", "ecef_vy", "ecef_vz"]].to_numpy(dtype=float)
    enu_positions = (rot_ecef_to_enu @ (ecef_positions - reference_ecef).T).T
    enu_velocities = (rot_ecef_to_enu @ ecef_velocities.T).T

    standard_df = imu_df.copy()
    standard_df["source_time_sow"] = standard_df["time"]
    for column_name in ["gnss_x", "gnss_y", "gnss_z", "gnss_vx", "gnss_vy", "gnss_vz"]:
        standard_df[column_name] = np.nan

    assignments = _assign_nearest_indices(
        imu_df["time"].to_numpy(dtype=float),
        ins_df["time"].to_numpy(dtype=float),
        tolerance_s=alignment_tolerance_s,
    )
    for imu_index, ins_index in assignments.items():
        standard_df.loc[imu_index, ["gnss_x", "gnss_y", "gnss_z"]] = enu_positions[ins_index]
        standard_df.loc[imu_index, ["gnss_vx", "gnss_vy", "gnss_vz"]] = enu_velocities[ins_index]

    if normalize_time_to_zero:
        standard_df["time"] = standard_df["time"] - float(standard_df["time"].iloc[0])

    return DatasetLoadResult(
        dataframe=standard_df.reset_index(drop=True),
        navigation_reference_override=reference_override,
        source_summary={
            "adapter_kind": "great_msf_imu_ins",
            "imu_txt_file": str(imu_path),
            "ins_file": str(ins_path),
            "primary_input_role": "great_msf_imu_plus_solution",
            "solution_source_semantics": "external_navigation_solution",
            "reference_mode": reference_mode,
            "normalize_time_to_zero": str(normalize_time_to_zero),
            "alignment_tolerance_s": f"{alignment_tolerance_s:.6f}",
            "imu_row_count": str(len(imu_df)),
            "ins_row_count": str(len(ins_df)),
        },
    )
