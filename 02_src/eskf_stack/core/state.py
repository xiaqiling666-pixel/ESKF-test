from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .math_utils import quat_to_euler, quat_to_rotmat


AXIS_DIM = 3
ERROR_STATE_DIM = 15
PROCESS_NOISE_DIM = 12


@dataclass(frozen=True)
class ErrorStateLayout:
    position: slice = slice(0, 3)
    velocity: slice = slice(3, 6)
    attitude: slice = slice(6, 9)
    gyro_bias: slice = slice(9, 12)
    accel_bias: slice = slice(12, 15)


@dataclass(frozen=True)
class ProcessNoiseLayout:
    accel: slice = slice(0, 3)
    gyro: slice = slice(3, 6)
    gyro_bias: slice = slice(6, 9)
    accel_bias: slice = slice(9, 12)


ERROR_STATE = ErrorStateLayout()
PROCESS_NOISE = ProcessNoiseLayout()


@dataclass
class NavState:
    position: np.ndarray
    velocity: np.ndarray
    quaternion: np.ndarray
    gyro_bias: np.ndarray
    accel_bias: np.ndarray

    @classmethod
    def zero(cls) -> "NavState":
        return cls(
            position=np.zeros(3),
            velocity=np.zeros(3),
            quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
            gyro_bias=np.zeros(3),
            accel_bias=np.zeros(3),
        )

    @property
    def yaw(self) -> float:
        return quat_to_euler(self.quaternion)[2]

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rotmat(self.quaternion)
