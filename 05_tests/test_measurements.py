from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.adapters.csv_dataset import ObservationFrame
from eskf_stack.core.state import ERROR_STATE_DIM
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


class DummyMeasurement(MeasurementModel):
    name = "dummy"
    freshness_timeout_s = 1.0

    def __init__(self, residual_value: float, policy: MeasurementPolicy) -> None:
        self._residual_value = residual_value
        self._policy = policy

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


if __name__ == "__main__":
    unittest.main()
