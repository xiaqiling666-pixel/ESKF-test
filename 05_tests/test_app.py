from __future__ import annotations

from dataclasses import replace
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.adapters.csv_dataset import SensorFrame, diagnostic_truth_view, observation_view
from eskf_stack.adapters.loader import DatasetLoadResult
from eskf_stack.analysis.evaluator import compute_metrics
from eskf_stack.core import ImuInitializationSample
from eskf_stack.app import _assess_initialization_status, _bootstrap_initialize_from_position_pair, _initialize_filter, run_pipeline
from eskf_stack.config import load_config
from eskf_stack.core import OfflineESKF
from eskf_stack.measurements.base import MeasurementResult


def _minimal_metrics_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time": [0.0, 1.0],
            "quality_score": [90.0, 92.0],
            "used_gnss_pos": [1, 1],
            "used_baro": [0, 0],
            "used_mag": [0, 0],
        }
    )


def _static_initialization_samples(sample_count: int = 50, dt_s: float = 0.02) -> list[ImuInitializationSample]:
    return [
        ImuInitializationSample(
            time=index * dt_s,
            accel=np.array([0.0, 0.0, 9.81], dtype=float),
            gyro=np.zeros(3, dtype=float),
        )
        for index in range(sample_count)
    ]


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

    def test_initialization_status_waits_for_gnss_when_position_missing(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        filter_engine = OfflineESKF(config)
        frame = observation_view(
            SensorFrame(
                time=0.0,
                accel=np.array([0.0, 0.0, -9.81], dtype=float),
                gyro=np.zeros(3, dtype=float),
                gnss_pos=None,
                gnss_vel=np.array([1.0, 0.0, 0.0], dtype=float),
                baro_h=None,
                mag_yaw=0.2,
                truth_pos=None,
                truth_vel=None,
                truth_yaw=None,
            )
        )

        status, static_check = _assess_initialization_status(filter_engine, frame, [], None)

        self.assertEqual(status.phase, "WAITING_GNSS")
        self.assertEqual(status.reason, "no_gnss_position")
        self.assertIsNone(static_check)

    def test_initialization_status_reports_ready_direct_init(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        filter_engine = OfflineESKF(config)
        samples = [
            ImuInitializationSample(time=0.0, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3)),
            ImuInitializationSample(time=0.1, accel=np.array([0.0, 0.0, 9.81]), gyro=np.zeros(3)),
        ]
        frame = observation_view(
            SensorFrame(
                time=0.1,
                accel=np.array([0.0, 0.0, -9.81], dtype=float),
                gyro=np.zeros(3, dtype=float),
                gnss_pos=np.array([1.0, 2.0, 3.0], dtype=float),
                gnss_vel=np.array([0.3, 0.0, 0.0], dtype=float),
                baro_h=None,
                mag_yaw=None,
                truth_pos=None,
                truth_vel=None,
                truth_yaw=None,
            )
        )

        status, static_check = _assess_initialization_status(filter_engine, frame, samples, None)

        self.assertEqual(status.phase, "READY_DIRECT_INIT")
        self.assertEqual(status.ready_mode, "direct")
        self.assertEqual(status.heading_source, "gnss_velocity_course")
        self.assertIsNotNone(static_check)

    def test_initialization_status_waits_for_bootstrap_motion_without_heading_source(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        filter_engine = OfflineESKF(config)
        anchor_frame = observation_view(
            SensorFrame(
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
        )
        frame = observation_view(
            SensorFrame(
                time=0.05,
                accel=np.array([0.0, 0.0, -9.81], dtype=float),
                gyro=np.zeros(3, dtype=float),
                gnss_pos=np.array([10.1, 20.1, 5.0], dtype=float),
                gnss_vel=None,
                baro_h=None,
                mag_yaw=None,
                truth_pos=None,
                truth_vel=None,
                truth_yaw=None,
            )
        )

        status, static_check = _assess_initialization_status(filter_engine, frame, [], anchor_frame)

        self.assertEqual(status.phase, "WAITING_BOOTSTRAP_MOTION")
        self.assertEqual(status.reason, "bootstrap_dt_too_short")
        self.assertIsNone(static_check)

    def test_initialization_status_can_enable_zero_yaw_timeout_fallback(self) -> None:
        base_config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        init_config = replace(
            base_config.initialization,
            heading_wait_timeout_s=0.5,
            zero_yaw_fallback_enabled=True,
        )
        config = replace(base_config, initialization=init_config)
        filter_engine = OfflineESKF(config)
        anchor_frame = observation_view(
            SensorFrame(
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
        )
        frame = observation_view(
            SensorFrame(
                time=0.6,
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
        )

        status, static_check = _assess_initialization_status(filter_engine, frame, [], anchor_frame)

        self.assertEqual(status.phase, "READY_DIRECT_INIT")
        self.assertEqual(status.reason, "heading_wait_timeout_zero_yaw_fallback")
        self.assertEqual(status.heading_source, "zero_yaw_fallback")
        self.assertAlmostEqual(status.wait_time_s, 0.6, places=9)
        self.assertIsNotNone(static_check)

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

    def test_initialize_filter_can_fallback_to_zero_yaw_after_timeout(self) -> None:
        base_config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        init_config = replace(
            base_config.initialization,
            heading_wait_timeout_s=0.5,
            zero_yaw_fallback_enabled=True,
        )
        config = replace(base_config, initialization=init_config)
        filter_engine = OfflineESKF(config)
        frame = SensorFrame(
            time=0.6,
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
        anchor_frame = observation_view(
            SensorFrame(
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
        )

        initialized = _initialize_filter(filter_engine, frame, [], {}, anchor_frame)

        self.assertTrue(initialized)
        self.assertTrue(filter_engine.initialized)
        self.assertAlmostEqual(filter_engine.state.yaw, 0.0, places=9)

    def test_direct_initialization_summary_matches_metrics_flags(self) -> None:
        base_config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        init_config = replace(
            base_config.initialization,
            heading_wait_timeout_s=0.5,
            zero_yaw_fallback_enabled=True,
        )
        config = replace(base_config, initialization=init_config)
        filter_engine = OfflineESKF(config)
        initialization_summary: dict[str, str] = {}
        frame = SensorFrame(
            time=0.6,
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
        anchor_frame = observation_view(
            SensorFrame(
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
        )

        initialized = _initialize_filter(filter_engine, frame, [], initialization_summary, anchor_frame)

        self.assertTrue(initialized)
        self.assertEqual(initialization_summary["initialization_phase"], "INITIALIZED")
        self.assertEqual(initialization_summary["initialization_reason"], "direct_init_completed")
        self.assertEqual(initialization_summary["initialization_ready_mode"], "direct")
        self.assertEqual(initialization_summary["heading_source"], "zero_yaw_fallback")
        self.assertEqual(initialization_summary["static_coarse_alignment_used"], "false")
        self.assertAlmostEqual(float(initialization_summary["initialization_wait_s"]), 0.6, places=9)

        metrics = compute_metrics(_minimal_metrics_frame(), initialization_summary=initialization_summary)

        self.assertEqual(metrics["initialization_completed_flag"], 1.0)
        self.assertEqual(metrics["initialization_mode_direct_flag"], 1.0)
        self.assertEqual(metrics["initialization_mode_bootstrap_position_pair_flag"], 0.0)
        self.assertEqual(metrics["initialization_static_coarse_alignment_used_flag"], 0.0)
        self.assertEqual(metrics["initialization_zero_yaw_fallback_used_flag"], 1.0)
        self.assertAlmostEqual(metrics["initialization_wait_s"], 0.6, places=9)

    def test_bootstrap_initialization_summary_matches_metrics_flags(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        filter_engine = OfflineESKF(config)
        initialization_summary: dict[str, str] = {}
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
            time=1.0,
            accel=np.array([0.0, 0.0, -9.81], dtype=float),
            gyro=np.zeros(3, dtype=float),
            gnss_pos=np.array([12.0, 24.0, 6.0], dtype=float),
            gnss_vel=None,
            baro_h=None,
            mag_yaw=None,
            truth_pos=None,
            truth_vel=None,
            truth_yaw=None,
        )

        initialized = _bootstrap_initialize_from_position_pair(
            filter_engine,
            anchor_frame,
            frame,
            _static_initialization_samples(),
            initialization_summary,
        )

        self.assertTrue(initialized)
        self.assertEqual(initialization_summary["initialization_phase"], "INITIALIZED")
        self.assertEqual(initialization_summary["initialization_reason"], "bootstrap_init_completed")
        self.assertEqual(initialization_summary["initialization_ready_mode"], "bootstrap_position_pair")
        self.assertEqual(initialization_summary["heading_source"], "position_pair_course")
        self.assertEqual(initialization_summary["static_coarse_alignment_used"], "true")
        self.assertAlmostEqual(float(initialization_summary["initialization_wait_s"]), 1.0, places=9)

        metrics = compute_metrics(_minimal_metrics_frame(), initialization_summary=initialization_summary)

        self.assertEqual(metrics["initialization_completed_flag"], 1.0)
        self.assertEqual(metrics["initialization_mode_direct_flag"], 0.0)
        self.assertEqual(metrics["initialization_mode_bootstrap_position_pair_flag"], 1.0)
        self.assertEqual(metrics["initialization_static_coarse_alignment_used_flag"], 1.0)
        self.assertEqual(metrics["initialization_zero_yaw_fallback_used_flag"], 0.0)
        self.assertAlmostEqual(metrics["initialization_wait_s"], 1.0, places=9)

    def test_pipeline_records_mode_scale_for_rejected_gnss_measurement(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0],
                "ax": [0.0],
                "ay": [0.0],
                "az": [-9.81],
                "gx": [0.0],
                "gy": [0.0],
                "gz": [0.0],
                "gnss_x": [10.0],
                "gnss_y": [20.0],
                "gnss_z": [5.0],
                "gnss_vx": [np.nan],
                "gnss_vy": [np.nan],
                "gnss_vz": [np.nan],
                "baro_h": [np.nan],
                "mag_yaw": [0.0],
                "truth_x": [np.nan],
                "truth_y": [np.nan],
                "truth_z": [np.nan],
                "truth_vx": [np.nan],
                "truth_vy": [np.nan],
                "truth_vz": [np.nan],
                "truth_yaw": [np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos":
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=False,
                        innovation_value=6.0,
                        nis=18.0,
                        rejected=True,
                        mode_scale=2.0,
                        management_mode="reject",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="skip")

        with tempfile.TemporaryDirectory() as temp_dir:
            captured: dict[str, pd.DataFrame] = {}

            def fake_export_pipeline_results(**kwargs):
                captured["result_df"] = kwargs["result_df"].copy()
                return {}

            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app.MeasurementManager", return_value=FakeMeasurementManager()),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", side_effect=fake_export_pipeline_results),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 1)
        self.assertIn("result_df", captured)
        self.assertFalse(bool(result_df.loc[0, "used_gnss_pos"]))
        self.assertTrue(bool(result_df.loc[0, "gnss_pos_rejected"]))
        self.assertEqual(result_df.loc[0, "gnss_pos_management_mode"], "reject")
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_mode_scale"]), 2.0, places=6)


if __name__ == "__main__":
    unittest.main()
