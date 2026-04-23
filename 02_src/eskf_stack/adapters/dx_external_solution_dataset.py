from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..config import AppConfig, project_path
from ..core.math_utils import wrap_angle
from .imu_transform import SUPPORTED_IMU_TRANSFORM_MODES, apply_imu_transform
from .loader import DatasetLoadResult, NavigationReferenceOverride


WGS84_A = 6378137.0
WGS84_E2 = 0.0066943799901413156


def _resolve_path(path_like: str) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else project_path(path_like)


def _read_csv_required(path: Path, required_columns: list[str]) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = sorted(set(required_columns).difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path.name} 缺少必需字段: {', '.join(missing)}")
    return dataframe


def _read_solution_ins(ins_path: Path) -> pd.DataFrame:
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
        raise ValueError(f"外部解算结果文件为空: {ins_path}")
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


def _geodetic_series_to_ecef(latitude_rad: np.ndarray, longitude_rad: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_lon = np.sin(longitude_rad)
    cos_lon = np.cos(longitude_rad)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (prime_vertical + height_m) * cos_lat * cos_lon
    y = (prime_vertical + height_m) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - WGS84_E2) + height_m) * sin_lat
    return np.column_stack((x, y, z))


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


def _choose_reference(
    config: AppConfig,
    solution_df: pd.DataFrame,
    reference_mode: str,
) -> tuple[float, float, float, NavigationReferenceOverride | None]:
    if reference_mode == "config":
        return (
            float(np.deg2rad(config.navigation_environment.reference_lat_deg)),
            float(np.deg2rad(config.navigation_environment.reference_lon_deg)),
            float(config.navigation_environment.reference_height_m),
            None,
        )

    if reference_mode == "first_solution":
        first_ecef = solution_df.loc[0, ["ecef_x", "ecef_y", "ecef_z"]].to_numpy(dtype=float)
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

    raise ValueError(f"不支持的外部解算参考系模式: {reference_mode}")


def _estimate_time_offset_s(
    imu_time_s: np.ndarray,
    solution_time_sow: np.ndarray,
    time_offset_mode: str,
    options: dict[str, object],
) -> float:
    if time_offset_mode == "manual":
        manual_offset_s = options.get("imu_solution_time_offset_s")
        if not isinstance(manual_offset_s, (int, float)):
            raise ValueError("time_offset_mode=manual 时需要 dataset_adapter.options.imu_solution_time_offset_s")
        return float(manual_offset_s)

    if time_offset_mode == "align_first_sample":
        return float(solution_time_sow[0] - imu_time_s[0])

    raise ValueError(f"不支持的时间偏移模式: {time_offset_mode}")


def _optional_truth_frame(decoded_dir: Path | None, filename: str, required_columns: list[str]) -> pd.DataFrame | None:
    if decoded_dir is None:
        return None
    path = decoded_dir / filename
    if not path.exists():
        return None
    return _read_csv_required(path, required_columns)


def load_dx_imu_external_solution_dataset(config: AppConfig) -> DatasetLoadResult:
    options = config.dataset_adapter.options
    imu_csv_path_like = options.get("imu_csv_path")
    ins_path_like = options.get("ins_path")
    if not isinstance(imu_csv_path_like, str) or not isinstance(ins_path_like, str):
        raise ValueError("dx_imu_external_solution 适配器需要 dataset_adapter.options.imu_csv_path 和 ins_path")

    imu_csv_path = _resolve_path(imu_csv_path_like)
    ins_path = _resolve_path(ins_path_like)
    decoded_dir_like = options.get("decoded_dir_for_truth")
    decoded_dir = None if not isinstance(decoded_dir_like, str) else _resolve_path(decoded_dir_like)
    normalize_time_to_zero = bool(options.get("normalize_time_to_zero", True))
    reference_mode = str(options.get("reference_mode", "first_solution"))
    alignment_tolerance_s = float(options.get("alignment_tolerance_s", 0.05))
    time_offset_mode = str(options.get("time_offset_mode", "align_first_sample"))
    include_solution_yaw = bool(options.get("include_solution_yaw", False))
    imu_transform_mode = str(options.get("imu_transform_mode", "raw"))
    if imu_transform_mode not in SUPPORTED_IMU_TRANSFORM_MODES:
        raise ValueError(f"不支持的 imu_transform_mode: {imu_transform_mode}")

    imu_df = _read_csv_required(
        imu_csv_path,
        ["TimeUS", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
    ).sort_values("TimeUS").reset_index(drop=True)
    imu_df = apply_imu_transform(
        imu_df,
        accel_columns=("AccX", "AccY", "AccZ"),
        gyro_columns=("GyrX", "GyrY", "GyrZ"),
        transform_mode=imu_transform_mode,
    )
    solution_df = _read_solution_ins(ins_path)
    pos_truth_df = _optional_truth_frame(decoded_dir, "POS_Global_Truth.csv", ["TimeUS", "Lat", "Lng", "Alt"])
    xkf_truth_df = _optional_truth_frame(
        decoded_dir,
        "XKF_Local_Truth.csv",
        ["TimeUS", "Roll", "Pitch", "Yaw", "VN", "VE", "VD", "PN", "PE", "PD"],
    )

    reference_lat_rad, reference_lon_rad, reference_height_m, reference_override = _choose_reference(
        config,
        solution_df,
        reference_mode=reference_mode,
    )
    reference_ecef = _geodetic_to_ecef(reference_lat_rad, reference_lon_rad, reference_height_m)
    rot_ecef_to_enu = _ecef_to_enu_rotation(reference_lat_rad, reference_lon_rad)

    imu_time_s = imu_df["TimeUS"].to_numpy(dtype=float) / 1e6
    solution_time_sow = solution_df["time"].to_numpy(dtype=float)
    imu_solution_time_offset_s = _estimate_time_offset_s(
        imu_time_s,
        solution_time_sow,
        time_offset_mode=time_offset_mode,
        options=options,
    )
    aligned_solution_time_s = solution_time_sow - imu_solution_time_offset_s

    standard_df = pd.DataFrame(
        {
            "time": imu_time_s.copy(),
            "source_time_us": imu_df["TimeUS"].to_numpy(dtype=np.int64),
            "ax": imu_df["AccX"].to_numpy(dtype=float),
            "ay": imu_df["AccY"].to_numpy(dtype=float),
            "az": imu_df["AccZ"].to_numpy(dtype=float),
            "gx": imu_df["GyrX"].to_numpy(dtype=float),
            "gy": imu_df["GyrY"].to_numpy(dtype=float),
            "gz": imu_df["GyrZ"].to_numpy(dtype=float),
        }
    )
    for column_name in [
        "gnss_x",
        "gnss_y",
        "gnss_z",
        "gnss_vx",
        "gnss_vy",
        "gnss_vz",
        "mag_yaw",
        "truth_x",
        "truth_y",
        "truth_z",
        "truth_vx",
        "truth_vy",
        "truth_vz",
        "truth_yaw",
    ]:
        standard_df[column_name] = np.nan

    ecef_positions = solution_df[["ecef_x", "ecef_y", "ecef_z"]].to_numpy(dtype=float)
    ecef_velocities = solution_df[["ecef_vx", "ecef_vy", "ecef_vz"]].to_numpy(dtype=float)
    enu_positions = (rot_ecef_to_enu @ (ecef_positions - reference_ecef).T).T
    enu_velocities = (rot_ecef_to_enu @ ecef_velocities.T).T
    solution_yaw_enu = np.array(
        [wrap_angle(np.pi / 2.0 - np.deg2rad(yaw_deg)) for yaw_deg in solution_df["yaw_deg"].to_numpy(dtype=float)]
    )

    assignments = _assign_nearest_indices(imu_time_s, aligned_solution_time_s, tolerance_s=alignment_tolerance_s)
    for imu_index, solution_index in assignments.items():
        standard_df.loc[imu_index, ["gnss_x", "gnss_y", "gnss_z"]] = enu_positions[solution_index]
        standard_df.loc[imu_index, ["gnss_vx", "gnss_vy", "gnss_vz"]] = enu_velocities[solution_index]
        if include_solution_yaw:
            standard_df.loc[imu_index, "mag_yaw"] = solution_yaw_enu[solution_index]

    if pos_truth_df is not None:
        pos_truth_time_s = pos_truth_df["TimeUS"].to_numpy(dtype=float) / 1e6
        geodetic_positions = pos_truth_df[["Lat", "Lng", "Alt"]].to_numpy(dtype=float)
        lat_rad = np.deg2rad(geodetic_positions[:, 0])
        lon_rad = np.deg2rad(geodetic_positions[:, 1])
        alt_m = geodetic_positions[:, 2]
        truth_ecef = _geodetic_series_to_ecef(lat_rad, lon_rad, alt_m)
        truth_positions_enu = (rot_ecef_to_enu @ (truth_ecef - reference_ecef).T).T
        truth_assignments = _assign_nearest_indices(imu_time_s, pos_truth_time_s, tolerance_s=alignment_tolerance_s)
        for imu_index, truth_index in truth_assignments.items():
            standard_df.loc[imu_index, ["truth_x", "truth_y", "truth_z"]] = truth_positions_enu[truth_index]

    if xkf_truth_df is not None:
        xkf_time_s = xkf_truth_df["TimeUS"].to_numpy(dtype=float) / 1e6
        truth_assignments = _assign_nearest_indices(imu_time_s, xkf_time_s, tolerance_s=alignment_tolerance_s)
        truth_velocities_enu = np.column_stack(
            (
                xkf_truth_df["VE"].to_numpy(dtype=float),
                xkf_truth_df["VN"].to_numpy(dtype=float),
                -xkf_truth_df["VD"].to_numpy(dtype=float),
            )
        )
        truth_yaw_enu = np.array(
            [wrap_angle(np.pi / 2.0 - np.deg2rad(yaw_deg)) for yaw_deg in xkf_truth_df["Yaw"].to_numpy(dtype=float)]
        )
        for imu_index, truth_index in truth_assignments.items():
            standard_df.loc[imu_index, ["truth_vx", "truth_vy", "truth_vz"]] = truth_velocities_enu[truth_index]
            standard_df.loc[imu_index, "truth_yaw"] = truth_yaw_enu[truth_index]

    if normalize_time_to_zero:
        standard_df["time"] = standard_df["time"] - float(standard_df["time"].iloc[0])

    return DatasetLoadResult(
        dataframe=standard_df.reset_index(drop=True),
        navigation_reference_override=reference_override,
        source_summary={
            "adapter_kind": "dx_imu_external_solution",
            "imu_csv_file": str(imu_csv_path),
            "ins_file": str(ins_path),
            "decoded_dir_for_truth": "" if decoded_dir is None else str(decoded_dir),
            "primary_input_role": "decoded_imu_plus_external_navigation_solution",
            "solution_source_semantics": "external_position_velocity_solution",
            "imu_transform_mode": imu_transform_mode,
            "reference_mode": reference_mode,
            "time_offset_mode": time_offset_mode,
            "imu_solution_time_offset_s": f"{imu_solution_time_offset_s:.6f}",
            "normalize_time_to_zero": str(normalize_time_to_zero),
            "alignment_tolerance_s": f"{alignment_tolerance_s:.6f}",
            "include_solution_yaw": str(include_solution_yaw),
            "imu_row_count": str(len(imu_df)),
            "solution_row_count": str(len(solution_df)),
            "assigned_solution_rows": str(len(assignments)),
        },
    )
