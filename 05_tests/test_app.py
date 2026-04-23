from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.adapters.csv_dataset import SensorFrame, diagnostic_truth_view, observation_view
from eskf_stack.core import ImuInitializationSample
from eskf_stack.app import _bootstrap_initialize_from_position_pair, _initialize_filter
from eskf_stack.config import load_config
from eskf_stack.core import OfflineESKF


class AppInitializationTests(unittest.TestCase):
    def test_observation_view_excludes_truth_fields(self) -> None:
        frame = SensorFrame(
            time=0.0,
            accel=np.array([0.0, 0.0, -9.81], dtype=float),
            gyro=np.zeros(3, dtype=float),
            gnss_pos=np.array([1.0, 2.0, 3.0], dtype=float),
            gnss_vel=np.array([0.1, 0.2, 0.3], dtype=float),
            baro_h=5.0,
            mag_yaw=0.2,
            truth_pos=np.array([10.0, 20.0, 30.0], dtype=float),
            truth_vel=np.array([1.0, 2.0, 3.0], dtype=float),
            truth_yaw=0.6,
        )

        measurement_frame = observation_view(frame)
        truth_frame = diagnostic_truth_view(frame)

        self.assertEqual(measurement_frame.time, 0.0)
        self.assertTrue(np.allclose(measurement_frame.gnss_pos, [1.0, 2.0, 3.0]))
        self.assertFalse(hasattr(measurement_frame, "truth_pos"))
        self.assertTrue(np.allclose(truth_frame.truth_pos, [10.0, 20.0, 30.0]))
        self.assertFalse(hasattr(truth_frame, "gnss_pos"))

    def test_initialize_filter_requires_velocity_or_heading_for_direct_init(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        filter_engine = OfflineESKF(config)
        frame = SensorFrame(
            time=0.0,
            accel=np.array([0.0, 0.0, -9.81], dtype=float),
            gyro=np.zeros(3, dtype=float),
            gnss_pos=np.array([10.0, 20.0, 5.0], dtype=float),
            gnss_vel=None,
            baro_h=None,
            mag_yaw=None,
            truth_pos=None,
            truth_vel=None,
            truth_yaw=None,
        )

        initialized = _initialize_filter(filter_engine, frame, [], {})

        self.assertFalse(initialized)
        self.assertFalse(filter_engine.initialized)

    def test_bootstrap_initialize_from_position_pair_estimates_velocity_and_yaw(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        filter_engine = OfflineESKF(config)
        anchor_frame = SensorFrame(
            time=0.0,
            accel=np.array([0.0, 0.0, -9.81], dtype=float),
            gyro=np.zeros(3, dtype=float),
            gnss_pos=np.array([10.0, 20.0, 5.0], dtype=float),
            gnss_vel=None,
            baro_h=None,
            mag_yaw=None,
            truth_pos=None,
            truth_vel=None,
            truth_yaw=None,
        )
        frame = SensorFrame(
            time=0.2,
            accel=np.array([0.0, 0.0, -9.81], dtype=float),
            gyro=np.zeros(3, dtype=float),
            gnss_pos=np.array([10.4, 20.8, 5.2], dtype=float),
            gnss_vel=None,
            baro_h=None,
            mag_yaw=None,
            truth_pos=None,
            truth_vel=None,
            truth_yaw=None,
        )

        initialization_samples = [
            ImuInitializationSample(time=0.0, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3)),
            ImuInitializationSample(time=0.2, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3)),
        ]
        initialized = _bootstrap_initialize_from_position_pair(
            filter_engine,
            anchor_frame,
            frame,
            initialization_samples,
            {},
        )

        self.assertTrue(initialized)
        self.assertTrue(filter_engine.initialized)
        self.assertTrue(np.allclose(filter_engine.state.position, [10.4, 20.8, 5.2]))
        self.assertTrue(np.allclose(filter_engine.state.velocity, [2.0, 4.0, 1.0]))
        self.assertAlmostEqual(filter_engine.state.quaternion[0], np.cos(np.arctan2(4.0, 2.0) / 2.0), places=6)


if __name__ == "__main__":
    unittest.main()
