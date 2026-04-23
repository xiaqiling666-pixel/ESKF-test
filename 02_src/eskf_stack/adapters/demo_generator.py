from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..core.math_utils import quat_to_rotmat, wrap_angle, yaw_to_quat


def generate_demo_dataset(output_csv: Path, duration_s: float = 120.0, imu_dt: float = 0.02) -> Path:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    times = np.arange(0.0, duration_s + imu_dt, imu_dt)
    radius = 25.0
    omega = 0.08
    z_base = 5.0
    z_amp = 1.2
    z_freq = 0.05

    x = radius * np.cos(omega * times)
    y = radius * np.sin(omega * times)
    z = z_base + z_amp * np.sin(z_freq * times)

    vx = -radius * omega * np.sin(omega * times)
    vy = radius * omega * np.cos(omega * times)
    vz = z_amp * z_freq * np.cos(z_freq * times)

    ax = -radius * omega * omega * np.cos(omega * times)
    ay = -radius * omega * omega * np.sin(omega * times)
    az = -z_amp * z_freq * z_freq * np.sin(z_freq * times)

    yaw = np.array([wrap_angle(omega * t + np.pi / 2.0) for t in times])
    gravity = np.array([0.0, 0.0, -9.81])

    gyro_bias = np.array([0.002, -0.0015, 0.0012])
    accel_bias = np.array([0.08, -0.05, 0.06])

    imu_rows = []
    for index in range(len(times)):
        quat = yaw_to_quat(yaw[index])
        rotation = quat_to_rotmat(quat)
        world_acc = np.array([ax[index], ay[index], az[index]])
        specific_force = rotation.T @ (world_acc - gravity)
        accel_meas = specific_force + accel_bias + rng.normal(0.0, 0.08, size=3)
        gyro_meas = np.array([0.0, 0.0, omega]) + gyro_bias + rng.normal(0.0, 0.01, size=3)
        imu_rows.append((accel_meas, gyro_meas))

    gnss_x = np.full(times.shape, np.nan)
    gnss_y = np.full(times.shape, np.nan)
    gnss_z = np.full(times.shape, np.nan)
    gnss_vx = np.full(times.shape, np.nan)
    gnss_vy = np.full(times.shape, np.nan)
    gnss_vz = np.full(times.shape, np.nan)
    baro_h = np.full(times.shape, np.nan)
    mag_yaw = np.full(times.shape, np.nan)

    for index, current_time in enumerate(times):
        if index % 10 == 0 and not (45.0 <= current_time <= 60.0):
            gnss_x[index] = x[index] + rng.normal(0.0, 0.8)
            gnss_y[index] = y[index] + rng.normal(0.0, 0.8)
            gnss_z[index] = z[index] + rng.normal(0.0, 0.8)
            gnss_vx[index] = vx[index] + rng.normal(0.0, 0.2)
            gnss_vy[index] = vy[index] + rng.normal(0.0, 0.2)
            gnss_vz[index] = vz[index] + rng.normal(0.0, 0.2)
        if index % 5 == 0:
            baro_h[index] = z[index] + rng.normal(0.0, 0.35)
        if index % 10 == 0:
            mag_yaw[index] = wrap_angle(yaw[index] + rng.normal(0.0, np.deg2rad(3.0)))

    dataframe = pd.DataFrame(
        {
            "time": times,
            "ax": [row[0][0] for row in imu_rows],
            "ay": [row[0][1] for row in imu_rows],
            "az": [row[0][2] for row in imu_rows],
            "gx": [row[1][0] for row in imu_rows],
            "gy": [row[1][1] for row in imu_rows],
            "gz": [row[1][2] for row in imu_rows],
            "gnss_x": gnss_x,
            "gnss_y": gnss_y,
            "gnss_z": gnss_z,
            "gnss_vx": gnss_vx,
            "gnss_vy": gnss_vy,
            "gnss_vz": gnss_vz,
            "baro_h": baro_h,
            "mag_yaw": mag_yaw,
            "truth_x": x,
            "truth_y": y,
            "truth_z": z,
            "truth_vx": vx,
            "truth_vy": vy,
            "truth_vz": vz,
            "truth_yaw": yaw,
        }
    )
    dataframe.to_csv(output_csv, index=False)
    return output_csv

