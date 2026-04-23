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

    @abstractmethod
    def is_available(self, frame: ObservationFrame) -> bool:
        raise NotImplementedError

    @abstractmethod
    def apply(self, filter_engine: OfflineESKF, frame: ObservationFrame) -> MeasurementResult:
        raise NotImplementedError
