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

from eskf_stack.adapters.csv_dataset import ObservationFrame
from eskf_stack.config import load_config
from eskf_stack.core.state import ERROR_STATE_DIM
from eskf_stack.measurements import (
    BarometerMeasurement,
    GnssPositionMeasurement,
    GnssVelocityMeasurement,
    MagYawMeasurement,
)
from eskf_stack.measurements.base import MeasurementModel, MeasurementPolicy, MeasurementUpdate
from eskf_stack.measurements.manager import MeasurementManager


class DummyFilter:
    def __init__(self) -> None:
        self.P = np.zeros((ERROR_STATE_DIM, ERROR_STATE_DIM))
        self.last_residual = None
        self.last_H = None
        self.last_R = None

    def apply_linear_update(self, residual, H, R) -> None:
        self.last_residual = residual
        self.last_H = H
        self.last_R = R


class PhysicalDummyFilter(DummyFilter):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.state = type(
            "State",
            (),
            {
                "position": np.zeros(3, dtype=float),
                "velocity": np.zeros(3, dtype=float),
                "quaternion": np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            },
        )()


class DummyMeasurement(MeasurementModel):
    freshness_timeout_s = 1.0

    def __init__(self, residual_value: float, policy: MeasurementPolicy, name: str = "dummy") -> None:
        self._residual_value = residual_value
        self._policy = policy
        self.name = name

    def is_available(self, frame: ObservationFrame) -> bool:
        return True

    def policy(self, filter_engine) -> MeasurementPolicy:
        return self._policy

    def build_update(self, filter_engine, frame: ObservationFrame) -> MeasurementUpdate | None:
        residual = np.array([self._residual_value], dtype=float)
        H = np.zeros((1, ERROR_STATE_DIM))
        return MeasurementUpdate(
            residual=residual,
            H=H,
            base_R=np.array([[1.0]], dtype=float),
            innovation_value=float(abs(self._residual_value)),
        )


class MeasurementManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        self.frame = ObservationFrame(
            time=0.0,
            gnss_pos=None,
            gnss_vel=None,
            baro_h=None,
            mag_yaw=None,
        )
        self.filter_engine = DummyFilter()
        self.manager = MeasurementManager()

    def test_manager_rejects_measurement_above_reject_threshold(self) -> None:
        model = DummyMeasurement(
            residual_value=5.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
        )

        result = self.manager.process(self.filter_engine, model, self.frame)

        self.assertTrue(result.available)
        self.assertFalse(result.used)
        self.assertTrue(result.rejected)
        self.assertEqual(result.management_mode, "reject")
        self.assertIsNone(self.filter_engine.last_R)

    def test_manager_can_disable_nis_rejection_for_baseline_runs(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_nis_rejection=False, use_adaptive_r=False),
        )
        model = DummyMeasurement(
            residual_value=5.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
        )

        result = self.manager.process(self.filter_engine, model, self.frame)

        self.assertTrue(result.used)
        self.assertFalse(result.rejected)
        self.assertTrue(result.reject_bypassed)
        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.nis, 25.0, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 1.0, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 1.0, places=6)

    def test_manager_marks_reject_bypass_when_rejection_disabled_and_adaptive_enabled(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_nis_rejection=False, use_adaptive_r=True, use_recovery_scale=False),
        )
        model = DummyMeasurement(
            residual_value=5.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
            name="gnss_pos",
        )

        result = self.manager.process(self.filter_engine, model, self.frame)

        self.assertTrue(result.available)
        self.assertTrue(result.used)
        self.assertFalse(result.rejected)
        self.assertTrue(result.reject_bypassed)
        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.nis, 25.0, places=6)
        self.assertAlmostEqual(result.adaptation_scale, 6.25, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 6.25, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 6.25, places=6)

    def test_manager_applies_adaptive_r_scaling(self) -> None:
        model = DummyMeasurement(
            residual_value=3.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
        )

        result = self.manager.process(self.filter_engine, model, self.frame)

        self.assertTrue(result.used)
        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.nis, 9.0, places=6)
        self.assertAlmostEqual(result.adaptation_scale, 2.25, places=6)
        self.assertAlmostEqual(result.recovery_scale, 1.0, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 2.25, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 2.25, places=6)

    def test_manager_can_disable_adaptive_r_for_ablation(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False),
        )
        model = DummyMeasurement(
            residual_value=3.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
        )

        result = self.manager.process(self.filter_engine, model, self.frame)

        self.assertTrue(result.used)
        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.nis, 9.0, places=6)
        self.assertAlmostEqual(result.adaptation_scale, 1.0, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 1.0, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 1.0, places=6)

    def test_manager_enters_gradual_recovery_after_repeated_rejections(self) -> None:
        reject_model = DummyMeasurement(
            residual_value=5.0,
            policy=MeasurementPolicy(
                adapt_threshold=4.0,
                reject_threshold=16.0,
                recovery_trigger_reject_streak=2,
                recovery_window=3,
                recovery_max_scale=3.0,
            ),
        )
        accept_model = DummyMeasurement(
            residual_value=1.0,
            policy=MeasurementPolicy(
                adapt_threshold=4.0,
                reject_threshold=16.0,
                recovery_trigger_reject_streak=2,
                recovery_window=3,
                recovery_max_scale=3.0,
            ),
        )

        self.manager.process(self.filter_engine, reject_model, self.frame)
        self.manager.process(self.filter_engine, reject_model, self.frame)
        first = self.manager.process(self.filter_engine, accept_model, self.frame)
        second = self.manager.process(self.filter_engine, accept_model, self.frame)
        third = self.manager.process(self.filter_engine, accept_model, self.frame)

        self.assertEqual(first.management_mode, "recover")
        self.assertAlmostEqual(first.recovery_scale, 3.0, places=6)
        self.assertAlmostEqual(second.recovery_scale, 2.0, places=6)
        self.assertAlmostEqual(third.recovery_scale, 1.0, places=6)

    def test_manager_can_disable_recovery_scale_for_ablation(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_recovery_scale=False),
        )
        reject_model = DummyMeasurement(
            residual_value=5.0,
            policy=MeasurementPolicy(
                adapt_threshold=4.0,
                reject_threshold=16.0,
                recovery_trigger_reject_streak=2,
                recovery_window=3,
                recovery_max_scale=3.0,
            ),
        )
        accept_model = DummyMeasurement(
            residual_value=1.0,
            policy=MeasurementPolicy(
                adapt_threshold=4.0,
                reject_threshold=16.0,
                recovery_trigger_reject_streak=2,
                recovery_window=3,
                recovery_max_scale=3.0,
            ),
        )

        self.manager.process(self.filter_engine, reject_model, self.frame)
        self.manager.process(self.filter_engine, reject_model, self.frame)
        result = self.manager.process(self.filter_engine, accept_model, self.frame)

        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.recovery_scale, 1.0, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 1.0, places=6)

    def test_manager_applies_mode_based_gnss_scaling(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )
        model = DummyMeasurement(
            residual_value=1.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
            name="gnss_pos",
        )

        result = self.manager.process(self.filter_engine, model, self.frame, current_mode="GNSS_DEGRADED")

        self.assertTrue(result.used)
        self.assertAlmostEqual(result.mode_scale, 2.0, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 2.0, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 2.0, places=6)

    def test_manager_applies_recovering_mode_scales_for_gnss_measurements(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )

        for sensor_name, expected_scale in (("gnss_pos", 1.25), ("gnss_vel", 1.15)):
            with self.subTest(sensor_name=sensor_name):
                self.filter_engine.last_R = None
                model = DummyMeasurement(
                    residual_value=1.0,
                    policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
                    name=sensor_name,
                )

                result = self.manager.process(self.filter_engine, model, self.frame, current_mode="RECOVERING")

                self.assertTrue(result.used)
                self.assertAlmostEqual(result.mode_scale, expected_scale, places=6)
                self.assertAlmostEqual(result.applied_r_scale, expected_scale, places=6)
                self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), expected_scale, places=6)

    def test_mode_scaling_is_limited_to_gnss_measurements(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )

        for sensor_name in ("baro", "mag_yaw"):
            with self.subTest(sensor_name=sensor_name):
                self.filter_engine.last_R = None
                model = DummyMeasurement(
                    residual_value=1.0,
                    policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
                    name=sensor_name,
                )

                result = self.manager.process(self.filter_engine, model, self.frame, current_mode="GNSS_DEGRADED")

                self.assertTrue(result.used)
                self.assertAlmostEqual(result.mode_scale, 1.0, places=6)
                self.assertAlmostEqual(result.applied_r_scale, 1.0, places=6)
                self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 1.0, places=6)

    def test_mode_scaling_changes_gnss_rejection_decision(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )
        model = DummyMeasurement(
            residual_value=5.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
            name="gnss_pos",
        )

        result = self.manager.process(self.filter_engine, model, self.frame, current_mode="GNSS_DEGRADED")

        self.assertTrue(result.available)
        self.assertTrue(result.used)
        self.assertFalse(result.rejected)
        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.mode_scale, 2.0, places=6)
        self.assertAlmostEqual(result.nis, 12.5, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 2.0, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 2.0, places=6)

    def test_mode_scale_is_recorded_when_gnss_measurement_is_rejected(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )
        model = DummyMeasurement(
            residual_value=6.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
            name="gnss_pos",
        )

        result = self.manager.process(self.filter_engine, model, self.frame, current_mode="GNSS_DEGRADED")

        self.assertTrue(result.available)
        self.assertFalse(result.used)
        self.assertTrue(result.rejected)
        self.assertEqual(result.management_mode, "reject")
        self.assertAlmostEqual(result.mode_scale, 2.0, places=6)
        self.assertAlmostEqual(result.nis, 18.0, places=6)
        self.assertIsNone(self.filter_engine.last_R)

    def test_mode_scaling_changes_gnss_velocity_rejection_decision(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )
        model = DummyMeasurement(
            residual_value=4.5,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
            name="gnss_vel",
        )

        result = self.manager.process(self.filter_engine, model, self.frame, current_mode="GNSS_DEGRADED")

        self.assertTrue(result.available)
        self.assertTrue(result.used)
        self.assertFalse(result.rejected)
        self.assertEqual(result.management_mode, "update")
        self.assertAlmostEqual(result.mode_scale, 1.5, places=6)
        self.assertAlmostEqual(result.nis, 13.5, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 1.5, places=6)
        self.assertAlmostEqual(float(self.filter_engine.last_R[0, 0]), 1.5, places=6)

    def test_manager_skips_mode_scaling_when_disabled(self) -> None:
        self.filter_engine.config = replace(
            self.config,
            mode_measurement_scaling=replace(self.config.mode_measurement_scaling, enabled=False),
            fusion_policy=replace(self.config.fusion_policy, use_adaptive_r=False, use_recovery_scale=False),
        )
        model = DummyMeasurement(
            residual_value=1.0,
            policy=MeasurementPolicy(adapt_threshold=4.0, reject_threshold=16.0),
            name="gnss_vel",
        )

        result = self.manager.process(self.filter_engine, model, self.frame, current_mode="RECOVERING")

        self.assertTrue(result.used)
        self.assertAlmostEqual(result.mode_scale, 1.0, places=6)
        self.assertAlmostEqual(result.applied_r_scale, 1.0, places=6)

    def test_barometer_measurement_uses_manager_rejection_policy(self) -> None:
        filter_engine = PhysicalDummyFilter(self.config)
        frame = ObservationFrame(
            time=0.0,
            gnss_pos=None,
            gnss_vel=None,
            baro_h=20.0,
            mag_yaw=None,
        )

        result = self.manager.process(filter_engine, BarometerMeasurement(), frame)

        self.assertTrue(result.available)
        self.assertFalse(result.used)
        self.assertTrue(result.rejected)
        self.assertEqual(result.management_mode, "reject")

    def test_mag_yaw_measurement_uses_manager_adaptive_policy(self) -> None:
        filter_engine = PhysicalDummyFilter(self.config)
        frame = ObservationFrame(
            time=0.0,
            gnss_pos=None,
            gnss_vel=None,
            baro_h=None,
            mag_yaw=np.deg2rad(7.0),
        )

        result = self.manager.process(filter_engine, MagYawMeasurement(), frame)

        self.assertTrue(result.available)
        self.assertTrue(result.used)
        self.assertFalse(result.rejected)
        self.assertGreater(result.nis, self.config.innovation_management.mag_yaw_nis_adapt_threshold)
        self.assertGreater(result.applied_r_scale, 1.0)

    def test_gnss_position_measurement_uses_anisotropic_noise(self) -> None:
        filter_engine = PhysicalDummyFilter(self.config)
        frame = ObservationFrame(
            time=0.0,
            gnss_pos=np.array([3.0, -1.0, 5.0], dtype=float),
            gnss_vel=None,
            baro_h=None,
            mag_yaw=None,
        )

        update = GnssPositionMeasurement().build_update(filter_engine, frame)

        self.assertIsNotNone(update)
        self.assertTrue(
            np.allclose(
                np.diag(update.base_R),
                np.array([0.8**2, 0.8**2, 1.6**2], dtype=float),
            )
        )

    def test_gnss_velocity_measurement_uses_anisotropic_noise(self) -> None:
        filter_engine = PhysicalDummyFilter(self.config)
        frame = ObservationFrame(
            time=0.0,
            gnss_pos=None,
            gnss_vel=np.array([0.5, -0.2, 0.8], dtype=float),
            baro_h=None,
            mag_yaw=None,
        )

        update = GnssVelocityMeasurement().build_update(filter_engine, frame)

        self.assertIsNotNone(update)
        self.assertTrue(
            np.allclose(
                np.diag(update.base_R),
                np.array([0.2**2, 0.2**2, 0.4**2], dtype=float),
            )
        )


if __name__ == "__main__":
    unittest.main()
