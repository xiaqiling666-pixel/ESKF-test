from __future__ import annotations

from pathlib import Path

import numpy as np


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def skew(vec: np.ndarray) -> np.ndarray:
    x, y, z = vec
    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def quat_normalize(quat: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(quat)
    if norm == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return quat / norm


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def rotvec_to_quat(rotvec: np.ndarray) -> np.ndarray:
    angle = np.linalg.norm(rotvec)
    if angle < 1e-12:
        half = 0.5 * rotvec
        return quat_normalize(np.array([1.0, half[0], half[1], half[2]]))
    axis = rotvec / angle
    half_angle = 0.5 * angle
    return np.hstack([np.cos(half_angle), axis * np.sin(half_angle)])


def quat_to_rotmat(quat: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(quat)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ]
    )


def yaw_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    half_yaw = 0.5 * yaw
    cr = np.cos(half_roll)
    sr = np.sin(half_roll)
    cp = np.cos(half_pitch)
    sp = np.sin(half_pitch)
    cy = np.cos(half_yaw)
    sy = np.sin(half_yaw)
    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ]
        )
    )


def quat_to_euler(quat: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quat_normalize(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = np.sign(sinp) * np.pi / 2.0
    else:
        pitch = np.arcsin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw
