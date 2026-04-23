from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..config import project_path
from .contract import ensure_standard_sensor_dataframe


@dataclass
class SensorFrame:
    time: float
    accel: np.ndarray
    gyro: np.ndarray
    gnss_pos: np.ndarray | None
    gnss_vel: np.ndarray | None
    baro_h: float | None
    mag_yaw: float | None
    truth_pos: np.ndarray | None
    truth_vel: np.ndarray | None
    truth_yaw: float | None


@dataclass(frozen=True)
class ObservationFrame:
    time: float
    gnss_pos: np.ndarray | None
    gnss_vel: np.ndarray | None
    baro_h: float | None
    mag_yaw: float | None


@dataclass(frozen=True)
class DiagnosticTruthFrame:
    truth_pos: np.ndarray | None
    truth_vel: np.ndarray | None
    truth_yaw: float | None


def _optional_vector(row: pd.Series, columns: list[str]) -> np.ndarray | None:
    values = [row.get(column) for column in columns]
    if any(pd.isna(value) for value in values):
        return None
    return np.array(values, dtype=float)


def _optional_scalar(row: pd.Series, column: str) -> float | None:
    value = row.get(column)
    if pd.isna(value):
        return None
    return float(value)


def load_sensor_dataframe(dataset_path: str) -> pd.DataFrame:
    path = project_path(dataset_path)
    dataframe = pd.read_csv(path)
    return ensure_standard_sensor_dataframe(dataframe, source_name=str(path))


def row_to_frame(row: pd.Series) -> SensorFrame:
    return SensorFrame(
        time=float(row["time"]),
        accel=np.array([row["ax"], row["ay"], row["az"]], dtype=float),
        gyro=np.array([row["gx"], row["gy"], row["gz"]], dtype=float),
        gnss_pos=_optional_vector(row, ["gnss_x", "gnss_y", "gnss_z"]),
        gnss_vel=_optional_vector(row, ["gnss_vx", "gnss_vy", "gnss_vz"]),
        baro_h=_optional_scalar(row, "baro_h"),
        mag_yaw=_optional_scalar(row, "mag_yaw"),
        truth_pos=_optional_vector(row, ["truth_x", "truth_y", "truth_z"]),
        truth_vel=_optional_vector(row, ["truth_vx", "truth_vy", "truth_vz"]),
        truth_yaw=_optional_scalar(row, "truth_yaw"),
    )


def observation_view(frame: SensorFrame) -> ObservationFrame:
    return ObservationFrame(
        time=frame.time,
        gnss_pos=frame.gnss_pos,
        gnss_vel=frame.gnss_vel,
        baro_h=frame.baro_h,
        mag_yaw=frame.mag_yaw,
    )


def diagnostic_truth_view(frame: SensorFrame) -> DiagnosticTruthFrame:
    return DiagnosticTruthFrame(
        truth_pos=frame.truth_pos,
        truth_vel=frame.truth_vel,
        truth_yaw=frame.truth_yaw,
    )
