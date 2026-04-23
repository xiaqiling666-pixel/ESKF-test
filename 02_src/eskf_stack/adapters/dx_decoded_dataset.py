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


def _geodetic_to_ecef(latitude_rad: np.ndarray, longitude_rad: np.ndarray, height_m: np.ndarray) -> np.ndarray:
    sin_lat = np.sin(latitude_rad)
    cos_lat = np.cos(latitude_rad)
    sin_lon = np.sin(longitude_rad)
    cos_lon = np.cos(longitude_rad)
    prime_vertical = WGS84_A / np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (prime_vertical + height_m) * cos_lat * cos_lon
    y = (prime_vertical + height_m) * cos_lat * sin_lon
    z = (prime_vertical * (1.0 - WGS84_E2) + height_m) * sin_lat
    return np.column_stack((x, y, z))


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
    gps_df: pd.DataFrame,
    reference_mode: str,
) -> tuple[float, float, float, NavigationReferenceOverride | None]:
    if reference_mode == "config":
        return (
            float(np.deg2rad(config.navigation_environment.reference_lat_deg)),
            float(np.deg2rad(config.navigation_environment.reference_lon_deg)),
            float(config.navigation_environment.reference_height_m),
            None,
        )

    if reference_mode == "first_gps":
        first_row = gps_df.iloc[0]
        reference_lat_rad = float(np.deg2rad(first_row["Lat"]))
        reference_lon_rad = float(np.deg2rad(first_row["Lng"]))
        reference_height_m = float(first_row["Alt"])
        return (
            reference_lat_rad,
            reference_lon_rad,
            reference_height_m,
            NavigationReferenceOverride(
                reference_lat_deg=float(first_row["Lat"]),
                reference_lon_deg=float(first_row["Lng"]),
                reference_height_m=reference_height_m,
            ),
        )

    raise ValueError(f"不支持的 decoded flight 参考系模式: {reference_mode}")


def _global_geodetic_to_enu(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    reference_lat_rad: float,
    reference_lon_rad: float,
    reference_height_m: float,
) -> np.ndarray:
    ecef_positions = _geodetic_to_ecef(np.deg2rad(lat_deg), np.deg2rad(lon_deg), alt_m)
    reference_ecef = _geodetic_to_ecef(
        np.array([reference_lat_rad]),
        np.array([reference_lon_rad]),
        np.array([reference_height_m]),
    )[0]
    rotation = _ecef_to_enu_rotation(reference_lat_rad, reference_lon_rad)
    return (rotation @ (ecef_positions - reference_ecef).T).T


def _derive_velocity(time_s: np.ndarray, positions_enu: np.ndarray) -> np.ndarray:
    if len(time_s) < 2:
        return np.zeros_like(positions_enu)
    return np.column_stack(
        [np.gradient(positions_enu[:, axis], time_s, edge_order=1) for axis in range(positions_enu.shape[1])]
    )


def _resolve_gps_velocity_mode(options: dict[str, object]) -> str:
    velocity_mode = str(options.get("gps_velocity_mode", "derived_from_pos"))
    supported_modes = {"derived_from_pos", "none", "from_xkf"}
    if velocity_mode not in supported_modes:
        raise ValueError(f"不支持的 gps_velocity_mode: {velocity_mode}")
    return velocity_mode


def _map_xkf_velocity_to_gps_rows(
    gps_time_s: np.ndarray,
    xkf_truth_df: pd.DataFrame | None,
    tolerance_s: float,
) -> np.ndarray:
    gps_velocities_enu = np.full((len(gps_time_s), 3), np.nan, dtype=float)
    if xkf_truth_df is None:
        raise ValueError("gps_velocity_mode=from_xkf 需要 XKF_Local_Truth.csv")

    xkf_time_s = xkf_truth_df["TimeUS"].to_numpy(dtype=float) / 1e6
    xkf_velocities_enu = np.column_stack(
        (
            xkf_truth_df["VE"].to_numpy(dtype=float),
            xkf_truth_df["VN"].to_numpy(dtype=float),
            -xkf_truth_df["VD"].to_numpy(dtype=float),
        )
    )
    gps_to_xkf = _assign_nearest_indices(gps_time_s, xkf_time_s, tolerance_s=tolerance_s)
    for gps_index, xkf_index in gps_to_xkf.items():
        gps_velocities_enu[gps_index] = xkf_velocities_enu[xkf_index]
    return gps_velocities_enu


def _optional_truth_frame(decoded_dir: Path, filename: str, required_columns: list[str]) -> pd.DataFrame | None:
    path = decoded_dir / filename
    if not path.exists():
        return None
    return _read_csv_required(path, required_columns)


def load_dx_decoded_flight_dataset(config: AppConfig) -> DatasetLoadResult:
    options = config.dataset_adapter.options
    decoded_dir_like = options.get("decoded_dir")
    if not isinstance(decoded_dir_like, str):
        raise ValueError("dx_decoded_flight_csv 适配器需要 dataset_adapter.options.decoded_dir")

    decoded_dir = _resolve_path(decoded_dir_like)
    imu_df = _read_csv_required(
        decoded_dir / "IMU_Full.csv",
        ["TimeUS", "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ"],
    ).sort_values("TimeUS").reset_index(drop=True)
    gps_df = _read_csv_required(
        decoded_dir / "GPS_Raw.csv",
        ["TimeUS", "Status", "NSats", "Lat", "Lng", "Alt"],
    ).sort_values("TimeUS").reset_index(drop=True)
    pos_truth_df = _optional_truth_frame(decoded_dir, "POS_Global_Truth.csv", ["TimeUS", "Lat", "Lng", "Alt"])
    xkf_truth_df = _optional_truth_frame(
        decoded_dir,
        "XKF_Local_Truth.csv",
        ["TimeUS", "Roll", "Pitch", "Yaw", "VN", "VE", "VD", "PN", "PE", "PD"],
    )

    reference_mode = str(options.get("reference_mode", "first_gps"))
    normalize_time_to_zero = bool(options.get("normalize_time_to_zero", True))
    gps_alignment_tolerance_s = float(options.get("gps_alignment_tolerance_s", 0.12))
    truth_alignment_tolerance_s = float(options.get("truth_alignment_tolerance_s", 0.06))
    gps_velocity_mode = _resolve_gps_velocity_mode(options)
    gps_velocity_alignment_tolerance_s = float(options.get("gps_velocity_alignment_tolerance_s", truth_alignment_tolerance_s))
    imu_transform_mode = str(options.get("imu_transform_mode", "raw"))
    if imu_transform_mode not in SUPPORTED_IMU_TRANSFORM_MODES:
        raise ValueError(f"不支持的 imu_transform_mode: {imu_transform_mode}")

    imu_df = apply_imu_transform(
        imu_df,
        accel_columns=("AccX", "AccY", "AccZ"),
        gyro_columns=("GyrX", "GyrY", "GyrZ"),
        transform_mode=imu_transform_mode,
    )

    reference_lat_rad, reference_lon_rad, reference_height_m, reference_override = _choose_reference(
        config,
        gps_df,
        reference_mode=reference_mode,
    )

    imu_time_s = imu_df["TimeUS"].to_numpy(dtype=float) / 1e6
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
        "truth_x",
        "truth_y",
        "truth_z",
        "truth_vx",
        "truth_vy",
        "truth_vz",
        "truth_yaw",
    ]:
        standard_df[column_name] = np.nan

    gps_time_s = gps_df["TimeUS"].to_numpy(dtype=float) / 1e6
    gps_positions_enu = _global_geodetic_to_enu(
        gps_df["Lat"].to_numpy(dtype=float),
        gps_df["Lng"].to_numpy(dtype=float),
        gps_df["Alt"].to_numpy(dtype=float),
        reference_lat_rad,
        reference_lon_rad,
        reference_height_m,
    )
    if gps_velocity_mode == "derived_from_pos":
        gps_velocities_enu = _derive_velocity(gps_time_s, gps_positions_enu)
        gps_velocity_semantics = "derived_from_decoded_positions"
    elif gps_velocity_mode == "from_xkf":
        gps_velocities_enu = _map_xkf_velocity_to_gps_rows(
            gps_time_s,
            xkf_truth_df,
            tolerance_s=gps_velocity_alignment_tolerance_s,
        )
        gps_velocity_semantics = "mapped_from_xkf_reference_velocity"
    else:
        gps_velocities_enu = np.full_like(gps_positions_enu, np.nan)
        gps_velocity_semantics = "disabled_position_only_updates"

    gps_assignments = _assign_nearest_indices(imu_time_s, gps_time_s, tolerance_s=gps_alignment_tolerance_s)
    for imu_index, gps_index in gps_assignments.items():
        standard_df.loc[imu_index, ["gnss_x", "gnss_y", "gnss_z"]] = gps_positions_enu[gps_index]
        if np.all(np.isfinite(gps_velocities_enu[gps_index])):
            standard_df.loc[imu_index, ["gnss_vx", "gnss_vy", "gnss_vz"]] = gps_velocities_enu[gps_index]

    if pos_truth_df is not None:
        pos_truth_time_s = pos_truth_df["TimeUS"].to_numpy(dtype=float) / 1e6
        truth_positions_enu = _global_geodetic_to_enu(
            pos_truth_df["Lat"].to_numpy(dtype=float),
            pos_truth_df["Lng"].to_numpy(dtype=float),
            pos_truth_df["Alt"].to_numpy(dtype=float),
            reference_lat_rad,
            reference_lon_rad,
            reference_height_m,
        )
        truth_assignments = _assign_nearest_indices(
            imu_time_s,
            pos_truth_time_s,
            tolerance_s=truth_alignment_tolerance_s,
        )
        for imu_index, truth_index in truth_assignments.items():
            standard_df.loc[imu_index, ["truth_x", "truth_y", "truth_z"]] = truth_positions_enu[truth_index]

    if xkf_truth_df is not None:
        xkf_time_s = xkf_truth_df["TimeUS"].to_numpy(dtype=float) / 1e6
        truth_assignments = _assign_nearest_indices(
            imu_time_s,
            xkf_time_s,
            tolerance_s=truth_alignment_tolerance_s,
        )
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
            "adapter_kind": "dx_decoded_flight_csv",
            "decoded_dir": str(decoded_dir),
            "imu_file": str(decoded_dir / "IMU_Full.csv"),
            "gps_file": str(decoded_dir / "GPS_Raw.csv"),
            "pos_truth_file": "" if pos_truth_df is None else str(decoded_dir / "POS_Global_Truth.csv"),
            "xkf_truth_file": "" if xkf_truth_df is None else str(decoded_dir / "XKF_Local_Truth.csv"),
            "primary_input_role": "extracted_imu_plus_decoded_flight_csv",
            "gps_source_semantics": "decoded_flight_position_only",
            "gps_velocity_mode": gps_velocity_mode,
            "gps_velocity_semantics": gps_velocity_semantics,
            "imu_transform_mode": imu_transform_mode,
            "reference_mode": reference_mode,
            "normalize_time_to_zero": str(normalize_time_to_zero),
            "gps_alignment_tolerance_s": f"{gps_alignment_tolerance_s:.6f}",
            "gps_velocity_alignment_tolerance_s": f"{gps_velocity_alignment_tolerance_s:.6f}",
            "truth_alignment_tolerance_s": f"{truth_alignment_tolerance_s:.6f}",
            "imu_row_count": str(len(imu_df)),
            "gps_row_count": str(len(gps_df)),
            "pos_truth_row_count": "0" if pos_truth_df is None else str(len(pos_truth_df)),
            "xkf_truth_row_count": "0" if xkf_truth_df is None else str(len(xkf_truth_df)),
        },
    )
