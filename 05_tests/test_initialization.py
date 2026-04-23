from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.config import load_config
from eskf_stack.core import ImuInitializationSample, OfflineESKF, StaticCoarseInitializer
from eskf_stack.core.math_utils import euler_to_quat, quat_to_euler, quat_to_rotmat
from eskf_stack.core.navigation import build_local_navigation_environment, resolve_local_navigation_environment


class InitializationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "01_data" / "config.json")

    def test_static_coarse_initializer_estimates_roll_pitch_and_biases(self) -> None:
        position = np.array([0.0, 0.0, 0.0], dtype=float)
        velocity = np.zeros(3, dtype=float)
        yaw = np.deg2rad(30.0)
        roll = np.deg2rad(8.0)
        pitch = np.deg2rad(-4.0)
        gyro_bias = np.array([0.003, -0.002, 0.001], dtype=float)
        accel_bias = np.array([0.05, -0.04, 0.03], dtype=float)

        base_environment = build_local_navigation_environment(self.config)
        static_environment = resolve_local_navigation_environment(
            base_environment,
            position,
            velocity,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )
        rotation = quat_to_rotmat(euler_to_quat(roll, pitch, yaw))
        expected_specific_force_body = rotation.T @ (-static_environment.gravity_vector)
        expected_nav_rate_body = rotation.T @ static_environment.omega_in_nav

        rng = np.random.default_rng(42)
        samples: list[ImuInitializationSample] = []
        for index in range(120):
            samples.append(
                ImuInitializationSample(
                    time=0.02 * index,
                    accel=expected_specific_force_body + accel_bias + rng.normal(0.0, 0.01, size=3),
                    gyro=expected_nav_rate_body + gyro_bias + rng.normal(0.0, 0.001, size=3),
                )
            )

        initializer = StaticCoarseInitializer(self.config.initialization)
        estimate = initializer.estimate(
            samples,
            position=position,
            base_environment=base_environment,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            yaw=yaw,
        )

        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertAlmostEqual(estimate.roll, roll, places=2)
        self.assertAlmostEqual(estimate.pitch, pitch, places=2)
        self.assertTrue(np.allclose(estimate.gyro_bias, gyro_bias, atol=3e-3))
        self.assertAlmostEqual(float(estimate.accel_bias[2]), float(accel_bias[2]), places=2)
        self.assertLess(float(np.linalg.norm(estimate.accel_bias[:2])), 0.2)

    def test_filter_initialize_accepts_roll_pitch_and_biases(self) -> None:
        filter_engine = OfflineESKF(self.config)
        roll = np.deg2rad(6.0)
        pitch = np.deg2rad(-3.0)
        yaw = np.deg2rad(25.0)
        gyro_bias = np.array([0.01, -0.02, 0.03], dtype=float)
        accel_bias = np.array([0.1, -0.2, 0.3], dtype=float)

        filter_engine.initialize(
            np.array([1.0, 2.0, 3.0], dtype=float),
            np.array([0.4, 0.5, 0.6], dtype=float),
            yaw=yaw,
            roll=roll,
            pitch=pitch,
            gyro_bias=gyro_bias,
            accel_bias=accel_bias,
        )

        est_roll, est_pitch, est_yaw = quat_to_euler(filter_engine.state.quaternion)
        self.assertAlmostEqual(est_roll, roll, places=6)
        self.assertAlmostEqual(est_pitch, pitch, places=6)
        self.assertAlmostEqual(est_yaw, yaw, places=6)
        self.assertTrue(np.allclose(filter_engine.state.gyro_bias, gyro_bias))
        self.assertTrue(np.allclose(filter_engine.state.accel_bias, accel_bias))


if __name__ == "__main__":
    unittest.main()
