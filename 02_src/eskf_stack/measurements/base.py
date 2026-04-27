from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from ..adapters.csv_dataset import ObservationFrame
from ..core.filter import OfflineESKF


@dataclass
class MeasurementResult:
    name: str
    available: bool
    used: bool
    innovation_value: float = 0.0
    nis: float | None = None
    rejected: bool = False
    adaptation_scale: float = 1.0
    recovery_scale: float = 1.0
    mode_scale: float = 1.0
    applied_r_scale: float = 1.0
    management_mode: str = "skip"


@dataclass(frozen=True)
class MeasurementUpdate:
    residual: np.ndarray
    H: np.ndarray
    base_R: np.ndarray
    innovation_value: float


@dataclass(frozen=True)
class MeasurementPolicy:
    adapt_threshold: float | None = None
    reject_threshold: float | None = None
    recovery_trigger_reject_streak: int = 3
    recovery_window: int = 3
    recovery_max_scale: float = 3.0


def innovation_covariance(filter_engine: OfflineESKF, H: np.ndarray, R: np.ndarray) -> np.ndarray:
    return H @ filter_engine.P @ H.T + R


def mahalanobis_squared(residual: np.ndarray, covariance: np.ndarray) -> float:
    try:
        inverse = np.linalg.inv(covariance)
    except np.linalg.LinAlgError:
        inverse = np.linalg.pinv(covariance)
    return float(residual.T @ inverse @ residual)


class MeasurementModel(ABC):
    name: str
    freshness_timeout_s: float

    def policy(self, filter_engine: OfflineESKF) -> MeasurementPolicy:
        return MeasurementPolicy()

    @abstractmethod
    def is_available(self, frame: ObservationFrame) -> bool:
        raise NotImplementedError

    @abstractmethod
    def build_update(self, filter_engine: OfflineESKF, frame: ObservationFrame) -> MeasurementUpdate | None:
        raise NotImplementedError
