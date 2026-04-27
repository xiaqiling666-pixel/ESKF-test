from __future__ import annotations

from dataclasses import replace
import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.config import load_config
from eskf_stack.core import OfflineESKF
from eskf_stack.core.math_utils import quat_to_rotmat, rotvec_to_quat
from eskf_stack.core.navigation import (
    build_local_navigation_environment,
    coriolis_position_jacobian,
    coriolis_velocity_jacobian,
    resolve_local_navigation_environment,
)
from eskf_stack.core.state import ERROR_STATE, ERROR_STATE_DIM


class NavigationCoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "01_data" / "config.json")

    def test_coriolis_velocity_jacobian_matches_numeric_difference(self) -> None:
        base_environment = build_local_navigation_environment(self.config)
        position_nav = np.array([120.0, 80.0, 15.0])
        velocity_nav = np.array([2.0, 3.0, -0.2])
        environment = resolve_local_navigation_environment(
            base_environment,
            position_nav,
            velocity_nav,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )

        analytic = coriolis_velocity_jacobian(environment, velocity_nav)

        step = 1e-3
        numeric = np.zeros((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = step
            velocity_plus = velocity_nav + delta
            velocity_minus = velocity_nav - delta
            resolved_plus = resolve_local_navigation_environment(
                base_environment,
                position_nav,
                velocity_plus,
                use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
                use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            )
            resolved_minus = resolve_local_navigation_environment(
                base_environment,
                position_nav,
                velocity_minus,
                use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
                use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            )
            coriolis_plus = -np.cross(resolved_plus.omega_coriolis_nav, velocity_plus)
            coriolis_minus = -np.cross(resolved_minus.omega_coriolis_nav, velocity_minus)
            numeric[:, axis] = (coriolis_plus - coriolis_minus) / (2.0 * step)

        self.assertTrue(np.allclose(analytic, numeric, rtol=1e-6, atol=1e-10))

    def test_coriolis_position_jacobian_matches_numeric_difference(self) -> None:
        base_environment = build_local_navigation_environment(self.config)
        position_nav = np.array([120.0, 80.0, 15.0])
        velocity_nav = np.array([2.0, 3.0, -0.2])

        analytic = coriolis_position_jacobian(
            base_environment,
            position_nav,
            velocity_nav,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )

        step = 1.0
        numeric = np.zeros((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = step
            environment_plus = resolve_local_navigation_environment(
                base_environment,
                position_nav + delta,
                velocity_nav,
                use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
                use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            )
            environment_minus = resolve_local_navigation_environment(
                base_environment,
                position_nav - delta,
                velocity_nav,
                use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
                use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            )
            coriolis_plus = -np.cross(environment_plus.omega_coriolis_nav, velocity_nav)
            coriolis_minus = -np.cross(environment_minus.omega_coriolis_nav, velocity_nav)
            numeric[:, axis] = (coriolis_plus - coriolis_minus) / (2.0 * step)

        self.assertTrue(np.allclose(analytic, numeric, rtol=1e-4, atol=1e-11))

    def test_gravity_gradient_matches_numeric_difference(self) -> None:
        base_environment = build_local_navigation_environment(self.config)
        position_nav = np.array([120.0, 80.0, 15.0])
        velocity_nav = np.array([2.0, 3.0, -0.2])
        environment = resolve_local_navigation_environment(
            base_environment,
            position_nav,
            velocity_nav,
            use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
            use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
        )

        analytic = environment.gravity_gradient_nav

        step = 1.0
        numeric = np.zeros((3, 3))
        for axis in range(3):
            delta = np.zeros(3)
            delta[axis] = step
            gravity_plus = resolve_local_navigation_environment(
                base_environment,
                position_nav + delta,
                velocity_nav,
                use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
                use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            ).gravity_vector
            gravity_minus = resolve_local_navigation_environment(
                base_environment,
                position_nav - delta,
                velocity_nav,
                use_wgs84_gravity=self.config.navigation_environment.use_wgs84_gravity,
                use_earth_rotation=self.config.navigation_environment.use_earth_rotation,
            ).gravity_vector
            numeric[:, axis] = (gravity_plus - gravity_minus) / (2.0 * step)

        self.assertTrue(np.allclose(analytic, numeric, rtol=1e-4, atol=1e-11))

    def test_navigation_environment_syncs_after_linear_update(self) -> None:
        filter_engine = OfflineESKF(self.config)
        initial_position = np.array([10.0, 20.0, 5.0])
        initial_velocity = np.array([1.0, 2.0, 0.1])
        filter_engine.initialize(initial_position, initial_velocity, yaw=0.0)

        residual = np.array([1.5])
        H = np.zeros((1, ERROR_STATE_DIM))
        H[0, ERROR_STATE.position.start + 2] = 1.0
        R = np.array([[0.01]])
        filter_engine.apply_linear_update(residual, H, R)

        self.assertAlmostEqual(
            filter_engine.current_navigation_environment.current_height_m,
            float(filter_engine.state.position[2]),
            places=9,
        )

    def test_covariance_remains_symmetric_after_attitude_injection(self) -> None:
        filter_engine = OfflineESKF(self.config)
        filter_engine.initialize(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), yaw=0.0)

        residual = np.array([0.05])
        H = np.zeros((1, ERROR_STATE_DIM))
        H[0, ERROR_STATE.attitude.start + 2] = 1.0
        R = np.array([[1e-3]])
        filter_engine.apply_linear_update(residual, H, R)

        self.assertTrue(np.allclose(filter_engine.P, filter_engine.P.T, atol=1e-12))
        self.assertTrue(np.all(np.isfinite(filter_engine.P)))

    def test_predict_pure_rotation_tracks_expected_yaw_without_spurious_translation(self) -> None:
        navigation_config = replace(
            self.config.navigation_environment,
            use_wgs84_gravity=False,
            use_earth_rotation=False,
        )
        config = replace(self.config, navigation_environment=navigation_config)
        filter_engine = OfflineESKF(config)
        filter_engine.initialize(np.zeros(3), np.zeros(3), yaw=0.0)

        dt = 0.01
        yaw_rate = 0.2
        accel_meas = np.array([0.0, 0.0, 9.81], dtype=float)
        gyro_meas = np.array([0.0, 0.0, yaw_rate], dtype=float)

        for _ in range(200):
            filter_engine.predict(accel_meas, gyro_meas, dt)

        self.assertAlmostEqual(filter_engine.state.yaw, yaw_rate * dt * 200, places=6)
        self.assertTrue(np.allclose(filter_engine.state.velocity, np.zeros(3), atol=1e-12))
        self.assertTrue(np.allclose(filter_engine.state.position, np.zeros(3), atol=1e-12))

    def test_attitude_error_injection_matches_right_multiplication_convention(self) -> None:
        filter_engine = OfflineESKF(self.config)
        filter_engine.initialize(
            np.zeros(3),
            np.zeros(3),
            yaw=0.4,
            roll=0.2,
            pitch=-0.1,
        )
        rotation_before = filter_engine.state.rotation.copy()
        delta_theta = np.array([1.0e-4, -2.0e-4, 3.0e-4], dtype=float)

        delta_x = np.zeros(ERROR_STATE_DIM)
        delta_x[ERROR_STATE.attitude] = delta_theta
        filter_engine._inject_error_state(delta_x)

        rotation_after = filter_engine.state.rotation
        rotation_expected_right = rotation_before @ quat_to_rotmat(rotvec_to_quat(delta_theta))
        rotation_expected_left = quat_to_rotmat(rotvec_to_quat(delta_theta)) @ rotation_before
        right_error = float(np.linalg.norm(rotation_after - rotation_expected_right))
        left_error = float(np.linalg.norm(rotation_after - rotation_expected_left))

        self.assertTrue(np.allclose(rotation_after, rotation_expected_right, atol=1e-10))
        self.assertLess(right_error, left_error)

    def test_predict_skips_dt_below_min_positive_threshold(self) -> None:
        filter_engine = OfflineESKF(self.config)
        filter_engine.initialize(np.zeros(3), np.zeros(3), yaw=0.0)
        tiny_dt = 0.5 * self.config.time_step_management.min_positive_dt_s

        filter_engine.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3), tiny_dt)

        self.assertTrue(filter_engine.last_predict_diagnostics.skipped)
        self.assertTrue(filter_engine.last_predict_diagnostics.warning)
        self.assertEqual(filter_engine.last_predict_diagnostics.reason, "below_min_positive_dt")
        self.assertAlmostEqual(filter_engine.last_predict_diagnostics.applied_dt, 0.0, places=12)
        self.assertTrue(np.allclose(filter_engine.state.position, np.zeros(3)))

    def test_predict_skips_large_dt_when_policy_requests_skip(self) -> None:
        filter_engine = OfflineESKF(self.config)
        filter_engine.initialize(np.zeros(3), np.zeros(3), yaw=0.0)
        large_dt = self.config.time_step_management.max_dt_s + 0.1

        filter_engine.predict(np.array([0.0, 0.0, 9.81]), np.zeros(3), large_dt)

        self.assertTrue(filter_engine.last_predict_diagnostics.skipped)
        self.assertTrue(filter_engine.last_predict_diagnostics.warning)
        self.assertEqual(filter_engine.last_predict_diagnostics.reason, "above_max_dt_skipped")
        self.assertAlmostEqual(filter_engine.last_predict_diagnostics.applied_dt, 0.0, places=12)

    def test_predict_can_clamp_large_dt_when_skip_disabled(self) -> None:
        navigation_config = replace(
            self.config.navigation_environment,
            use_wgs84_gravity=False,
            use_earth_rotation=False,
        )
        time_step_config = replace(self.config.time_step_management, skip_large_dt=False, max_dt_s=0.05)
        config = replace(self.config, navigation_environment=navigation_config, time_step_management=time_step_config)
        filter_engine = OfflineESKF(config)
        filter_engine.initialize(np.zeros(3), np.zeros(3), yaw=0.0)

        filter_engine.predict(np.array([0.0, 0.0, 9.81]), np.array([0.0, 0.0, 1.0]), 0.2)

        self.assertFalse(filter_engine.last_predict_diagnostics.skipped)
        self.assertTrue(filter_engine.last_predict_diagnostics.warning)
        self.assertEqual(filter_engine.last_predict_diagnostics.reason, "above_max_dt_clamped")
        self.assertAlmostEqual(filter_engine.last_predict_diagnostics.applied_dt, 0.05, places=12)
        self.assertAlmostEqual(filter_engine.state.yaw, 0.05, places=6)


if __name__ == "__main__":
    unittest.main()
