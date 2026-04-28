from __future__ import annotations

from dataclasses import replace
import shutil
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
from eskf_stack.measurements.base import MeasurementPolicy, MeasurementResult
from eskf_stack.measurements.barometer import BarometerMeasurement
from eskf_stack.measurements.gnss_position import GnssPositionMeasurement
from eskf_stack.measurements.gnss_velocity import GnssVelocityMeasurement
from eskf_stack.analysis.state_machine import ModeThresholds


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
    def _workspace_temp_dir(self, name: str) -> Path:
        temp_dir = PROJECT_ROOT / "_tmp_test_app" / name
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.addCleanup(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        return temp_dir

    def _run_pipeline_with_manager(
        self,
        config,
        sensor_df: pd.DataFrame,
        measurement_manager,
        temp_name: str,
    ) -> pd.DataFrame:
        temp_dir = self._workspace_temp_dir(temp_name)
        with (
            patch("eskf_stack.app.load_config", return_value=config),
            patch(
                "eskf_stack.app.load_dataset_from_config",
                return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
            ),
            patch("eskf_stack.app.MeasurementManager", return_value=measurement_manager),
            patch("eskf_stack.app.generate_demo_dataset"),
            patch("eskf_stack.app.ensure_dir", return_value=temp_dir),
            patch("eskf_stack.app.export_pipeline_results", return_value={}),
        ):
            return run_pipeline()

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

    def test_pipeline_records_waiting_gnss_rows_before_initialization(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "ax": [0.0, 0.0],
                "ay": [0.0, 0.0],
                "az": [-9.81, -9.81],
                "gx": [0.0, 0.0],
                "gy": [0.0, 0.0],
                "gz": [0.0, 0.0],
                "gnss_x": [np.nan, np.nan],
                "gnss_y": [np.nan, np.nan],
                "gnss_z": [np.nan, np.nan],
                "gnss_vx": [np.nan, np.nan],
                "gnss_vy": [np.nan, np.nan],
                "gnss_vz": [np.nan, np.nan],
                "baro_h": [np.nan, np.nan],
                "mag_yaw": [np.nan, np.nan],
                "truth_x": [np.nan, np.nan],
                "truth_y": [np.nan, np.nan],
                "truth_z": [np.nan, np.nan],
                "truth_vx": [np.nan, np.nan],
                "truth_vy": [np.nan, np.nan],
                "truth_vz": [np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            captured: dict[str, object] = {}

            def fake_export_pipeline_results(**kwargs):
                captured["result_df"] = kwargs["result_df"].copy()
                captured["initialization_summary"] = dict(kwargs["initialization_summary"])
                return {}

            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", side_effect=fake_export_pipeline_results),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 2)
        self.assertTrue(result_df["predict_skipped"].all())
        self.assertTrue((result_df["predict_reason"] == "not_initialized").all())
        self.assertTrue((result_df["initialization_phase"] == "WAITING_GNSS").all())
        self.assertTrue((result_df["initialization_reason"] == "no_gnss_position").all())
        self.assertTrue(result_df["initialization_pending"].all())
        self.assertTrue(result_df["est_x"].isna().all())
        self.assertTrue((result_df["gnss_pos_management_mode"] == "unavailable").all())
        self.assertTrue((result_df["gnss_vel_management_mode"] == "unavailable").all())
        self.assertIn("result_df", captured)
        self.assertEqual(captured["initialization_summary"]["initialization_phase"], "WAITING_GNSS")

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["initialization_row_count"], 2.0)
        self.assertEqual(metrics["initialization_pending_row_count"], 2.0)
        self.assertEqual(metrics["gnss_pos_management_count_unavailable"], 2.0)
        self.assertAlmostEqual(metrics["initialization_phase_duration_WAITING_GNSS_s"], 2.0, places=6)

    def test_pipeline_records_bootstrap_wait_diagnostics_before_initialization(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.05],
                "ax": [0.0, 0.0],
                "ay": [0.0, 0.0],
                "az": [-9.81, -9.81],
                "gx": [0.0, 0.0],
                "gy": [0.0, 0.0],
                "gz": [0.0, 0.0],
                "gnss_x": [10.0, 10.1],
                "gnss_y": [20.0, 20.1],
                "gnss_z": [5.0, 5.0],
                "gnss_vx": [np.nan, np.nan],
                "gnss_vy": [np.nan, np.nan],
                "gnss_vz": [np.nan, np.nan],
                "baro_h": [np.nan, np.nan],
                "mag_yaw": [np.nan, np.nan],
                "truth_x": [np.nan, np.nan],
                "truth_y": [np.nan, np.nan],
                "truth_z": [np.nan, np.nan],
                "truth_vx": [np.nan, np.nan],
                "truth_vy": [np.nan, np.nan],
                "truth_vz": [np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", return_value={}),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 2)
        self.assertEqual(result_df.loc[0, "initialization_phase"], "WAITING_BOOTSTRAP_MOTION")
        self.assertEqual(result_df.loc[0, "initialization_reason"], "awaiting_bootstrap_anchor")
        self.assertEqual(result_df.loc[0, "gnss_pos_management_mode"], "pending_init")
        self.assertAlmostEqual(float(result_df.loc[0, "bootstrap_anchor_age_s"]), 0.0, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "bootstrap_anchor_horizontal_displacement_m"]), 0.0, places=6)
        self.assertEqual(result_df.loc[1, "initialization_phase"], "WAITING_BOOTSTRAP_MOTION")
        self.assertEqual(result_df.loc[1, "initialization_reason"], "bootstrap_dt_too_short")
        self.assertEqual(result_df.loc[1, "gnss_pos_management_mode"], "pending_init")
        self.assertAlmostEqual(float(result_df.loc[1, "bootstrap_anchor_age_s"]), 0.05, places=6)
        self.assertAlmostEqual(float(result_df.loc[1, "bootstrap_anchor_horizontal_displacement_m"]), np.sqrt(0.02), places=6)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["gnss_pos_management_count_pending_init"], 2.0)

    def test_pipeline_marks_direct_initialization_transition_row(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.02],
                "ax": [0.0, 0.0],
                "ay": [0.0, 0.0],
                "az": [-9.81, -9.81],
                "gx": [0.0, 0.0],
                "gy": [0.0, 0.0],
                "gz": [0.0, 0.0],
                "gnss_x": [10.0, 10.04],
                "gnss_y": [20.0, 20.0],
                "gnss_z": [5.0, 5.0],
                "gnss_vx": [2.0, 2.0],
                "gnss_vy": [0.0, 0.0],
                "gnss_vz": [0.0, 0.0],
                "baro_h": [5.0, 5.0],
                "mag_yaw": [np.nan, np.nan],
                "truth_x": [np.nan, np.nan],
                "truth_y": [np.nan, np.nan],
                "truth_z": [np.nan, np.nan],
                "truth_vx": [np.nan, np.nan],
                "truth_vy": [np.nan, np.nan],
                "truth_vz": [np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", return_value={}),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 2)
        self.assertTrue(bool(result_df.loc[0, "initialization_completed_this_frame"]))
        self.assertFalse(bool(result_df.loc[0, "initialization_pending"]))
        self.assertEqual(result_df.loc[0, "initialization_phase"], "INITIALIZED")
        self.assertEqual(result_df.loc[0, "initialization_heading_source"], "gnss_velocity_course")
        self.assertTrue(bool(result_df.loc[0, "predict_skipped"]))
        self.assertEqual(result_df.loc[0, "predict_reason"], "initialization_completed_this_frame")
        self.assertTrue(bool(result_df.loc[0, "used_gnss_pos"]))
        self.assertTrue(bool(result_df.loc[0, "used_gnss_vel"]))
        self.assertTrue(np.isnan(result_df.loc[0, "bootstrap_anchor_age_s"]))
        self.assertFalse(bool(result_df.loc[1, "initialization_completed_this_frame"]))
        self.assertEqual(result_df.loc[1, "predict_reason"], "applied")

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["initialization_completed_row_count"], 1.0)
        self.assertEqual(metrics["initialization_pending_row_count"], 0.0)

    def test_pipeline_marks_bootstrap_initialization_transition_row(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.2, 0.22],
                "ax": [0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0],
                "gnss_x": [10.0, 12.0, 12.04],
                "gnss_y": [20.0, 24.0, 24.08],
                "gnss_z": [5.0, 6.0, 6.02],
                "gnss_vx": [np.nan, np.nan, np.nan],
                "gnss_vy": [np.nan, np.nan, np.nan],
                "gnss_vz": [np.nan, np.nan, np.nan],
                "baro_h": [np.nan, np.nan, np.nan],
                "mag_yaw": [np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", return_value={}),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 3)
        self.assertEqual(result_df.loc[0, "initialization_phase"], "WAITING_BOOTSTRAP_MOTION")
        self.assertTrue(bool(result_df.loc[1, "initialization_completed_this_frame"]))
        self.assertEqual(result_df.loc[1, "initialization_ready_mode"], "bootstrap_position_pair")
        self.assertEqual(result_df.loc[1, "initialization_heading_source"], "position_pair_course")
        self.assertEqual(result_df.loc[1, "predict_reason"], "initialization_completed_this_frame")
        self.assertAlmostEqual(float(result_df.loc[1, "bootstrap_anchor_age_s"]), 0.2, places=6)
        self.assertAlmostEqual(float(result_df.loc[1, "bootstrap_anchor_horizontal_displacement_m"]), np.sqrt(20.0), places=6)
        self.assertFalse(bool(result_df.loc[2, "initialization_completed_this_frame"]))
        self.assertEqual(result_df.loc[2, "predict_reason"], "applied")

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["initialization_completed_row_count"], 1.0)
        self.assertEqual(metrics["initialization_pending_row_count"], 1.0)

    def test_pipeline_uses_policy_aligned_gnss_rejection_threshold_for_mode_degradation(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")

        class StrictGnssPositionMeasurement(GnssPositionMeasurement):
            def policy(self, filter_engine) -> MeasurementPolicy:
                base_policy = super().policy(filter_engine)
                return MeasurementPolicy(
                    adapt_threshold=base_policy.adapt_threshold,
                    reject_threshold=base_policy.reject_threshold,
                    recovery_trigger_reject_streak=2,
                    recovery_window=base_policy.recovery_window,
                    recovery_max_scale=base_policy.recovery_max_scale,
                )

        class StrictGnssVelocityMeasurement(GnssVelocityMeasurement):
            def policy(self, filter_engine) -> MeasurementPolicy:
                base_policy = super().policy(filter_engine)
                return MeasurementPolicy(
                    adapt_threshold=base_policy.adapt_threshold,
                    reject_threshold=base_policy.reject_threshold,
                    recovery_trigger_reject_streak=2,
                    recovery_window=base_policy.recovery_window,
                    recovery_max_scale=base_policy.recovery_max_scale,
                )

        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.7, 1.4],
                "ax": [0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0],
                "gnss_x": [10.0, 100.0, 120.0],
                "gnss_y": [20.0, 20.0, 20.0],
                "gnss_z": [5.0, 5.0, 5.0],
                "gnss_vx": [2.0, 2.0, 2.0],
                "gnss_vy": [0.0, 0.0, 0.0],
                "gnss_vz": [0.0, 0.0, 0.0],
                "baro_h": [np.nan, np.nan, np.nan],
                "mag_yaw": [np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan],
            }
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app._build_measurements", return_value=[StrictGnssPositionMeasurement(), StrictGnssVelocityMeasurement()]),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", return_value={}),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 3)
        self.assertTrue(bool(result_df.loc[1, "gnss_pos_rejected"]))
        self.assertTrue(bool(result_df.loc[2, "gnss_pos_rejected"]))
        self.assertEqual(result_df.loc[2, "mode_target"], "GNSS_DEGRADED")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "partial_gnss_with_rejections")

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
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=2.0,
                        applied_r_scale=2.0,
                        management_mode="reject",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

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
        self.assertAlmostEqual(
            float(result_df.loc[0, "gnss_pos_nis_adapt_threshold"]),
            config.innovation_management.gnss_pos_nis_adapt_threshold,
            places=6,
        )
        self.assertAlmostEqual(
            float(result_df.loc[0, "gnss_pos_nis_reject_threshold"]),
            config.innovation_management.gnss_pos_nis_reject_threshold,
            places=6,
        )
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_adaptation_scale"]), 1.0, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_mode_scale"]), 2.0, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_applied_r_scale"]), 2.0, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_r_scale"]), 2.0, places=6)

    def test_pipeline_records_gnss_position_and_velocity_outputs_for_metrics(self) -> None:
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
                "gnss_vx": [1.0],
                "gnss_vy": [2.0],
                "gnss_vz": [0.5],
                "baro_h": [np.nan],
                "mag_yaw": [np.nan],
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
                        used=True,
                        innovation_value=3.5,
                        nis=2.5,
                        rejected=False,
                        adaptation_scale=1.2,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.2,
                        management_mode="update",
                    )
                if model.name == "gnss_vel":
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=1.5,
                        nis=1.25,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            captured: dict[str, object] = {}

            def fake_export_pipeline_results(**kwargs):
                captured["result_df"] = kwargs["result_df"].copy()
                captured["extra_metrics"] = dict(kwargs["extra_metrics"])
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
        self.assertTrue(bool(result_df.loc[0, "used_gnss_pos"]))
        self.assertTrue(bool(result_df.loc[0, "used_gnss_vel"]))
        self.assertFalse(bool(result_df.loc[0, "gnss_pos_rejected"]))
        self.assertFalse(bool(result_df.loc[0, "gnss_vel_rejected"]))
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_innovation_norm"]), 3.5, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_vel_innovation_norm"]), 1.5, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_nis"]), 2.5, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_vel_nis"]), 1.25, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_adaptation_scale"]), 1.2, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_vel_adaptation_scale"]), 1.0, places=6)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["gnss_pos_updates"], 1.0)
        self.assertEqual(metrics["gnss_vel_updates"], 1.0)
        self.assertAlmostEqual(metrics["mean_gnss_pos_innovation"], 3.5, places=6)
        self.assertAlmostEqual(metrics["mean_gnss_vel_innovation"], 1.5, places=6)
        self.assertAlmostEqual(metrics["mean_gnss_pos_nis"], 2.5, places=6)
        self.assertAlmostEqual(metrics["mean_gnss_vel_nis"], 1.25, places=6)

    def test_pipeline_records_reject_bypass_for_gnss_measurements(self) -> None:
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
                "gnss_vx": [1.0],
                "gnss_vy": [0.0],
                "gnss_vz": [0.0],
                "baro_h": [np.nan],
                "mag_yaw": [np.nan],
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
                        used=True,
                        innovation_value=4.0,
                        nis=18.0,
                        rejected=False,
                        reject_bypassed=True,
                        adaptation_scale=2.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=2.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "reject_bypass_single",
        )

        self.assertEqual(len(result_df), 1)
        self.assertTrue(bool(result_df.loc[0, "used_gnss_pos"]))
        self.assertFalse(bool(result_df.loc[0, "gnss_pos_rejected"]))
        self.assertTrue(bool(result_df.loc[0, "gnss_pos_reject_bypassed"]))
        self.assertEqual(int(result_df.loc[0, "gnss_pos_reject_bypass_streak"]), 1)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["gnss_pos_reject_bypassed_updates"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_policy_bypass_count"], 1.0)
        self.assertAlmostEqual(metrics["gnss_pos_nis_reject_policy_bypass_pct"], 100.0, places=6)
        self.assertEqual(metrics["max_gnss_pos_reject_bypass_streak"], 1.0)

    def test_pipeline_escalates_repeated_gnss_reject_bypass_to_degraded_mode(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.4, 0.7, 0.9],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, 10.5, 11.0, 11.5],
                "gnss_y": [20.0, 20.0, 20.0, 20.0],
                "gnss_z": [5.0, 5.0, 5.0, 5.0],
                "gnss_vx": [1.0, 1.0, 1.0, 1.0],
                "gnss_vy": [0.0, 0.0, 0.0, 0.0],
                "gnss_vz": [0.0, 0.0, 0.0, 0.0],
                "baro_h": [np.nan, np.nan, np.nan, np.nan],
                "mag_yaw": [np.nan, np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def __init__(self) -> None:
                self._frame_index = -1

            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos":
                    self._frame_index += 1
                    bypassed = self._frame_index >= 1
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=4.0 if bypassed else 0.5,
                        nis=18.0 if bypassed else 1.0,
                        rejected=False,
                        reject_bypassed=bypassed,
                        adaptation_scale=2.0 if bypassed else 1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=2.0 if bypassed else 1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel":
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.3,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "reject_bypass_mode",
        )

        self.assertEqual(len(result_df), 4)
        self.assertEqual(int(result_df.loc[1, "gnss_pos_reject_bypass_streak"]), 1)
        self.assertEqual(int(result_df.loc[2, "gnss_pos_reject_bypass_streak"]), 2)
        self.assertEqual(result_df.loc[0, "mode_reason"], "quality_drop_under_gnss")
        self.assertEqual(result_df.loc[1, "mode_reason"], "quality_drop_under_gnss")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "gnss_reject_bypass_streak")
        self.assertEqual(result_df.loc[2, "mode_reason"], "gnss_reject_bypass_streak")
        self.assertEqual(result_df.loc[3, "mode"], "GNSS_DEGRADED")
        self.assertEqual(result_df.loc[3, "mode_reason"], "gnss_reject_bypass_streak")

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["max_gnss_pos_reject_bypass_streak"], 3.0)

    def test_pipeline_enters_inertial_hold_on_short_gnss_outage_with_aux_support(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan],
                "baro_h": [12.0, 12.1, 12.2, 12.3],
                "mag_yaw": [np.nan, np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.5,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "short_gnss_outage_with_aux",
        )

        self.assertEqual(result_df.loc[0, "mode"], "GNSS_STABLE")
        self.assertEqual(result_df.loc[2, "mode_target"], "INERTIAL_HOLD")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "short_gnss_outage")
        self.assertEqual(result_df.loc[3, "mode"], "INERTIAL_HOLD")
        self.assertEqual(result_df.loc[3, "mode_reason"], "short_gnss_outage")

        metrics = compute_metrics(result_df)
        self.assertGreaterEqual(metrics["mode_duration_INERTIAL_HOLD_s"], 0.5)
        self.assertEqual(metrics["mode_entry_count_INERTIAL_HOLD"], 1.0)
        self.assertAlmostEqual(metrics["max_auxiliary_outage_s"], 0.0, places=6)

    def test_pipeline_enters_degraded_on_short_gnss_outage_without_aux_support(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan],
                "baro_h": [np.nan, np.nan, np.nan, np.nan],
                "mag_yaw": [np.nan, np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "short_gnss_outage_without_aux",
        )

        self.assertEqual(result_df.loc[0, "mode"], "GNSS_STABLE")
        self.assertEqual(result_df.loc[2, "mode_target"], "DEGRADED")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "gnss_outage_without_aux_support")
        self.assertEqual(result_df.loc[3, "mode"], "DEGRADED")
        self.assertEqual(result_df.loc[3, "mode_reason"], "gnss_outage_without_aux_support")

        metrics = compute_metrics(result_df)
        self.assertGreaterEqual(metrics["mode_duration_DEGRADED_s"], 0.5)
        self.assertEqual(metrics["mode_entry_count_DEGRADED"], 1.0)
        self.assertTrue(metrics["max_auxiliary_outage_s"] > 1e6)

    def test_pipeline_distinguishes_available_but_unsuccessful_auxiliary_support(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan],
                "baro_h": [np.nan, 12.1, 12.2, 12.3],
                "mag_yaw": [np.nan, np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=False,
                        rejected=False,
                        management_mode="skip",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "available_but_unsuccessful_aux",
        )

        self.assertEqual(result_df.loc[2, "mode_target"], "DEGRADED")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "auxiliary_available_without_successful_updates")
        self.assertTrue(bool(result_df.loc[2, "recent_available_baro"]))
        self.assertFalse(bool(result_df.loc[2, "recent_baro"]))
        self.assertEqual(int(result_df.loc[2, "baro_skip_streak"]), 2)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["baro_management_count_skip"], 3.0)
        self.assertEqual(metrics["max_baro_skip_streak"], 3.0)
        self.assertAlmostEqual(metrics["max_auxiliary_available_outage_s"], 0.0, places=6)

    def test_pipeline_keeps_inertial_hold_when_one_aux_source_remains_stable(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan],
                "baro_h": [np.nan, 12.1, 12.2, 12.3],
                "mag_yaw": [0.1, 0.2, 0.3, 0.4],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=False,
                        rejected=False,
                        management_mode="skip",
                    )
                if model.name == "mag" and frame.mag_yaw is not None:
                    if frame.time <= 0.5:
                        return MeasurementResult(
                            name="mag",
                            available=True,
                            used=True,
                            innovation_value=1.0,
                            nis=0.5,
                            rejected=False,
                            adaptation_scale=1.0,
                            recovery_scale=1.0,
                            mode_scale=1.0,
                            applied_r_scale=1.0,
                            management_mode="update",
                        )
                    return MeasurementResult(
                        name="mag",
                        available=True,
                        used=True,
                        innovation_value=2.0,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "stable_auxiliary_backup",
        )

        self.assertEqual(result_df.loc[2, "mode_target"], "INERTIAL_HOLD")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "short_gnss_outage")
        self.assertEqual(result_df.loc[3, "mode"], "INERTIAL_HOLD")
        self.assertTrue(bool(result_df.loc[2, "recent_baro"]) is False)
        self.assertTrue(bool(result_df.loc[2, "recent_mag"]))
        self.assertEqual(int(result_df.loc[2, "baro_skip_streak"]), 2)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["baro_management_count_skip"], 3.0)
        self.assertEqual(metrics["mag_management_count_update"], 4.0)
        self.assertGreaterEqual(metrics["mode_duration_INERTIAL_HOLD_s"], 0.5)

    def test_pipeline_marks_gnss_available_without_successful_updates_as_degraded(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")

        class FastSkipGnssPositionMeasurement(GnssPositionMeasurement):
            def policy(self, filter_engine) -> MeasurementPolicy:
                base_policy = super().policy(filter_engine)
                return MeasurementPolicy(
                    adapt_threshold=base_policy.adapt_threshold,
                    reject_threshold=base_policy.reject_threshold,
                    recovery_trigger_reject_streak=2,
                    recovery_window=base_policy.recovery_window,
                    recovery_max_scale=base_policy.recovery_max_scale,
                )

        class FastSkipGnssVelocityMeasurement(GnssVelocityMeasurement):
            def policy(self, filter_engine) -> MeasurementPolicy:
                base_policy = super().policy(filter_engine)
                return MeasurementPolicy(
                    adapt_threshold=base_policy.adapt_threshold,
                    reject_threshold=base_policy.reject_threshold,
                    recovery_trigger_reject_streak=2,
                    recovery_window=base_policy.recovery_window,
                    recovery_max_scale=base_policy.recovery_max_scale,
                )

        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, 10.5, 11.0, 11.5],
                "gnss_y": [20.0, 20.0, 20.0, 20.0],
                "gnss_z": [5.0, 5.0, 5.0, 5.0],
                "gnss_vx": [1.0, 1.0, 1.0, 1.0],
                "gnss_vy": [0.0, 0.0, 0.0, 0.0],
                "gnss_vz": [0.0, 0.0, 0.0, 0.0],
                "baro_h": [np.nan, 12.1, 12.2, 12.3],
                "mag_yaw": [np.nan, np.nan, np.nan, np.nan],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    if frame.time <= 0.5:
                        return MeasurementResult(
                            name="gnss_pos",
                            available=True,
                            used=True,
                            innovation_value=0.2,
                            nis=0.8,
                            rejected=False,
                            adaptation_scale=1.0,
                            recovery_scale=1.0,
                            mode_scale=1.0,
                            applied_r_scale=1.0,
                            management_mode="update",
                        )
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=False,
                        rejected=False,
                        management_mode="skip",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    if frame.time <= 0.5:
                        return MeasurementResult(
                            name="gnss_vel",
                            available=True,
                            used=True,
                            innovation_value=0.1,
                            nis=0.4,
                            rejected=False,
                            adaptation_scale=1.0,
                            recovery_scale=1.0,
                            mode_scale=1.0,
                            applied_r_scale=1.0,
                            management_mode="update",
                        )
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=False,
                        rejected=False,
                        management_mode="skip",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.6,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        with patch(
            "eskf_stack.app._build_measurements",
            return_value=[FastSkipGnssPositionMeasurement(), FastSkipGnssVelocityMeasurement(), BarometerMeasurement()],
        ):
            result_df = self._run_pipeline_with_manager(
                config,
                sensor_df,
                FakeMeasurementManager(),
                "gnss_available_without_successful_updates",
            )

        self.assertEqual(result_df.loc[3, "mode_target"], "DEGRADED")
        self.assertEqual(result_df.loc[3, "mode_target_reason"], "gnss_available_without_successful_updates")
        self.assertEqual(result_df.loc[3, "gnss_pos_management_mode"], "skip")
        self.assertEqual(result_df.loc[3, "gnss_vel_management_mode"], "skip")
        self.assertTrue(bool(result_df.loc[3, "recent_available_gnss_pos"]))
        self.assertFalse(bool(result_df.loc[3, "recent_gnss_pos"]))
        self.assertEqual(int(result_df.loc[3, "gnss_pos_skip_streak"]), 2)
        self.assertEqual(int(result_df.loc[3, "gnss_vel_skip_streak"]), 2)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["gnss_pos_management_count_skip"], 2.0)
        self.assertEqual(metrics["gnss_vel_management_count_skip"], 2.0)
        self.assertLess(float(result_df.loc[3, "quality_score"]), float(result_df.loc[1, "quality_score"]))

    def test_pipeline_keeps_inertial_hold_with_baro_reject_and_mag_recovering(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan],
                "baro_h": [np.nan, 12.1, 12.2, 12.3],
                "mag_yaw": [0.1, 0.2, 0.3, 0.4],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=False,
                        rejected=True,
                        management_mode="reject",
                    )
                if model.name == "mag" and frame.mag_yaw is not None:
                    return MeasurementResult(
                        name="mag",
                        available=True,
                        used=True,
                        innovation_value=2.0,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=2.0,
                        mode_scale=1.0,
                        applied_r_scale=2.0,
                        management_mode="recover",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "baro_reject_mag_recover",
        )

        self.assertEqual(result_df.loc[2, "mode_target"], "INERTIAL_HOLD")
        self.assertEqual(result_df.loc[2, "mode_target_reason"], "short_gnss_outage")
        self.assertEqual(result_df.loc[3, "mode"], "INERTIAL_HOLD")
        self.assertEqual(result_df.loc[2, "baro_management_mode"], "reject")
        self.assertEqual(result_df.loc[2, "mag_management_mode"], "recover")
        self.assertEqual(int(result_df.loc[2, "baro_reject_streak"]), 2)
        self.assertEqual(int(result_df.loc[2, "mag_adaptive_streak"]), 3)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["baro_management_count_reject"], 3.0)
        self.assertEqual(metrics["mag_management_count_recover"], 4.0)
        self.assertGreaterEqual(metrics["mode_duration_INERTIAL_HOLD_s"], 0.5)

        class NoBackupMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=False,
                        rejected=True,
                        management_mode="reject",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        no_backup_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            NoBackupMeasurementManager(),
            "baro_reject_no_mag_backup",
        )

        self.assertGreater(float(result_df.loc[2, "quality_score"]), float(no_backup_df.loc[2, "quality_score"]))

    def test_pipeline_keeps_inertial_hold_with_baro_skip_and_mag_recovering(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5],
                "ax": [0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan],
                "baro_h": [np.nan, 12.1, 12.2, 12.3],
                "mag_yaw": [0.1, 0.2, 0.3, 0.4],
                "truth_x": [np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos", available=True, used=True, management_mode="update"
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel", available=True, used=True, management_mode="update"
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro", available=True, used=False, rejected=False, management_mode="skip"
                    )
                if model.name == "mag" and frame.mag_yaw is not None:
                    # Alternating reject and recover to simulate instability but staying within limits
                    if frame.time == 0.5:
                        return MeasurementResult(
                            name="mag", available=True, used=False, rejected=True, management_mode="reject"
                        )
                    return MeasurementResult(
                        name="mag", available=True, used=True, innovation_value=1.5, management_mode="recover", applied_r_scale=1.5
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "baro_skip_mag_recover",
        )

        # Frame 2 (time=1.0): GNSS Outage (1.0s), Baro skip (streak 2), Mag updated/recovered (streak 0)
        # Should stay in INERTIAL_HOLD because Mag is a stable backup
        self.assertEqual(result_df.loc[2, "mode_target"], "INERTIAL_HOLD")
        self.assertEqual(result_df.loc[2, "baro_management_mode"], "skip")
        self.assertEqual(result_df.loc[2, "mag_management_mode"], "recover")
        self.assertEqual(int(result_df.loc[2, "baro_skip_streak"]), 2)
        self.assertEqual(int(result_df.loc[2, "mag_reject_streak"]), 0)

        # Frame 3 (time=1.5): Baro skip streak 3, but Mag still update/recover
        self.assertEqual(result_df.loc[3, "mode"], "INERTIAL_HOLD")
        self.assertEqual(int(result_df.loc[3, "baro_skip_streak"]), 3)

    def test_pipeline_respects_custom_auxiliary_instability_quality_floor(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.4, 0.7, 0.9, 1.1],
                "ax": [0.0, 0.0, 0.0, 0.0, 0.0],
                "ay": [0.0, 0.0, 0.0, 0.0, 0.0],
                "az": [-9.81, -9.81, -9.81, -9.81, -9.81],
                "gx": [0.0, 0.0, 0.0, 0.0, 0.0],
                "gy": [0.0, 0.0, 0.0, 0.0, 0.0],
                "gz": [0.0, 0.0, 0.0, 0.0, 0.0],
                "gnss_x": [10.0, np.nan, np.nan, np.nan, np.nan],
                "gnss_y": [20.0, np.nan, np.nan, np.nan, np.nan],
                "gnss_z": [5.0, np.nan, np.nan, np.nan, np.nan],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan, np.nan],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan, np.nan],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan, np.nan],
                "baro_h": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "mag_yaw": [0.1, 0.2, 0.3, 0.4, 0.5],
                "truth_x": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "truth_y": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "truth_z": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "truth_vx": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "truth_vy": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "truth_vz": [np.nan, np.nan, np.nan, np.nan, np.nan],
                "truth_yaw": [np.nan, np.nan, np.nan, np.nan, np.nan],
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "mag" and frame.mag_yaw is not None:
                    if frame.time <= 0.7:
                        return MeasurementResult(
                            name="mag",
                            available=True,
                            used=True,
                            innovation_value=1.0,
                            nis=0.5,
                            rejected=False,
                            adaptation_scale=1.0,
                            recovery_scale=1.0,
                            mode_scale=1.0,
                            applied_r_scale=1.0,
                            management_mode="update",
                        )
                    return MeasurementResult(
                        name="mag",
                        available=True,
                        used=False,
                        rejected=True,
                        management_mode="reject",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        custom_thresholds = replace(ModeThresholds(), short_outage_aux_instability_quality_floor=60.0)

        with patch("eskf_stack.app._mode_thresholds_for_measurements", return_value=custom_thresholds):
            result_df = self._run_pipeline_with_manager(
                config,
                sensor_df,
                FakeMeasurementManager(),
                "custom_aux_instability_quality_floor",
            )

        self.assertEqual(result_df.loc[4, "mode_target"], "DEGRADED")
        self.assertEqual(result_df.loc[4, "mode_target_reason"], "auxiliary_sensor_instability")

    def test_pipeline_transitions_from_degraded_to_recovering_and_back_to_stable(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = pd.DataFrame(
            {
                "time": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
                "ax": [0.0] * 8,
                "ay": [0.0] * 8,
                "az": [-9.81] * 8,
                "gx": [0.0] * 8,
                "gy": [0.0] * 8,
                "gz": [0.0] * 8,
                "gnss_x": [10.0, np.nan, np.nan, np.nan, 11.0, 11.5, 12.0, 12.5],
                "gnss_y": [20.0, np.nan, np.nan, np.nan, 20.0, 20.0, 20.0, 20.0],
                "gnss_z": [5.0, np.nan, np.nan, np.nan, 5.0, 5.0, 5.0, 5.0],
                "gnss_vx": [1.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0, 1.0],
                "gnss_vy": [0.0, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0],
                "gnss_vz": [0.0, np.nan, np.nan, np.nan, 0.0, 0.0, 0.0, 0.0],
                "baro_h": [np.nan] * 8,
                "mag_yaw": [np.nan] * 8,
                "truth_x": [np.nan] * 8,
                "truth_y": [np.nan] * 8,
                "truth_z": [np.nan] * 8,
                "truth_vx": [np.nan] * 8,
                "truth_vy": [np.nan] * 8,
                "truth_vz": [np.nan] * 8,
                "truth_yaw": [np.nan] * 8,
            }
        )

        class FakeMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "degraded_to_recovering_to_stable",
        )

        self.assertEqual(result_df.loc[3, "mode"], "DEGRADED")
        self.assertEqual(result_df.loc[3, "mode_reason"], "gnss_outage_without_aux_support")
        self.assertEqual(result_df.loc[4, "mode_target"], "GNSS_STABLE")
        self.assertEqual(result_df.loc[5, "mode"], "RECOVERING")
        self.assertEqual(result_df.loc[5, "mode_reason"], "restoring_gnss_stability")
        self.assertEqual(result_df.loc[7, "mode"], "GNSS_STABLE")
        self.assertEqual(result_df.loc[7, "mode_reason"], "fresh_gnss_and_healthy_covariance")

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["mode_entry_count_RECOVERING"], 1.0)
        self.assertEqual(metrics["mode_entry_count_GNSS_STABLE"], 2.0)
        self.assertEqual(metrics["mode_entry_count_DEGRADED"], 1.0)

    def test_pipeline_records_baro_and_mag_outputs_for_metrics(self) -> None:
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
                "baro_h": [12.3],
                "mag_yaw": [0.4],
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
                if model.name == "baro":
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=True,
                        innovation_value=0.8,
                        nis=1.6,
                        rejected=False,
                        adaptation_scale=1.1,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.1,
                        management_mode="update",
                    )
                if model.name == "mag":
                    return MeasurementResult(
                        name="mag",
                        available=True,
                        used=False,
                        innovation_value=12.0,
                        nis=9.0,
                        rejected=True,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="reject",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            captured: dict[str, object] = {}

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
        self.assertTrue(bool(result_df.loc[0, "available_baro"]))
        self.assertTrue(bool(result_df.loc[0, "available_mag"]))
        self.assertTrue(bool(result_df.loc[0, "used_baro"]))
        self.assertFalse(bool(result_df.loc[0, "used_mag"]))
        self.assertFalse(bool(result_df.loc[0, "baro_rejected"]))
        self.assertTrue(bool(result_df.loc[0, "mag_rejected"]))
        self.assertEqual(int(result_df.loc[0, "baro_reject_streak"]), 0)
        self.assertEqual(int(result_df.loc[0, "mag_reject_streak"]), 1)
        self.assertEqual(int(result_df.loc[0, "gnss_pos_adaptive_streak"]), 0)
        self.assertEqual(int(result_df.loc[0, "gnss_vel_adaptive_streak"]), 0)
        self.assertEqual(int(result_df.loc[0, "baro_adaptive_streak"]), 1)
        self.assertEqual(int(result_df.loc[0, "mag_adaptive_streak"]), 0)
        self.assertAlmostEqual(float(result_df.loc[0, "baro_outage_s"]), 0.0, places=6)
        self.assertTrue(np.isinf(result_df.loc[0, "mag_outage_s"]))
        self.assertAlmostEqual(float(result_df.loc[0, "auxiliary_outage_s"]), 0.0, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "baro_innovation_abs"]), 0.8, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "yaw_innovation_abs_deg"]), 12.0, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "baro_nis"]), 1.6, places=6)
        self.assertAlmostEqual(float(result_df.loc[0, "mag_nis"]), 9.0, places=6)

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["baro_available_measurements"], 1.0)
        self.assertEqual(metrics["mag_available_measurements"], 1.0)
        self.assertEqual(metrics["baro_updates"], 1.0)
        self.assertEqual(metrics["mag_updates"], 0.0)
        self.assertEqual(metrics["mag_rejections"], 1.0)
        self.assertEqual(metrics["max_baro_reject_streak"], 0.0)
        self.assertEqual(metrics["max_mag_reject_streak"], 1.0)
        self.assertEqual(metrics["max_gnss_pos_adaptive_streak"], 0.0)
        self.assertEqual(metrics["max_gnss_vel_adaptive_streak"], 0.0)
        self.assertEqual(metrics["max_baro_adaptive_streak"], 1.0)
        self.assertEqual(metrics["max_mag_adaptive_streak"], 0.0)
        self.assertAlmostEqual(metrics["max_baro_outage_s"], 0.0, places=6)
        self.assertTrue(np.isinf(metrics["max_mag_outage_s"]))
        self.assertAlmostEqual(metrics["max_auxiliary_outage_s"], 0.0, places=6)
        self.assertAlmostEqual(metrics["mean_baro_innovation"], 0.8, places=6)
        self.assertAlmostEqual(metrics["mean_mag_innovation"], 12.0, places=6)
        self.assertAlmostEqual(metrics["mean_baro_nis"], 1.6, places=6)
        self.assertAlmostEqual(metrics["mean_mag_nis"], 9.0, places=6)

    def test_pipeline_distinguishes_skip_from_unavailable_management_mode(self) -> None:
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
                "gnss_vx": [1.0],
                "gnss_vy": [0.0],
                "gnss_vz": [0.0],
                "baro_h": [12.3],
                "mag_yaw": [np.nan],
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
                        used=True,
                        innovation_value=0.2,
                        nis=0.8,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "gnss_vel":
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.4,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                if model.name == "baro":
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=False,
                        rejected=False,
                        management_mode="skip",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "skip_vs_unavailable_management_mode",
        )

        self.assertEqual(result_df.loc[0, "baro_management_mode"], "skip")
        self.assertEqual(result_df.loc[0, "mag_management_mode"], "unavailable")
        self.assertTrue(bool(result_df.loc[0, "available_baro"]))
        self.assertFalse(bool(result_df.loc[0, "available_mag"]))

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["baro_management_count_skip"], 1.0)
        self.assertEqual(metrics["baro_management_count_unavailable"], 0.0)
        self.assertEqual(metrics["mag_management_count_skip"], 0.0)
        self.assertEqual(metrics["mag_management_count_unavailable"], 1.0)
        self.assertEqual(metrics["baro_available_measurements"], 1.0)
        self.assertEqual(metrics["mag_available_measurements"], 0.0)

    def test_pipeline_records_recover_management_mode_and_recovery_metrics(self) -> None:
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
                "gnss_vx": [1.0],
                "gnss_vy": [0.0],
                "gnss_vz": [0.0],
                "baro_h": [np.nan],
                "mag_yaw": [np.nan],
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
                        used=True,
                        innovation_value=1.0,
                        nis=2.0,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=2.5,
                        mode_scale=1.0,
                        applied_r_scale=2.5,
                        management_mode="recover",
                    )
                if model.name == "gnss_vel":
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.2,
                        nis=0.5,
                        rejected=False,
                        adaptation_scale=1.0,
                        recovery_scale=1.0,
                        mode_scale=1.0,
                        applied_r_scale=1.0,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        result_df = self._run_pipeline_with_manager(
            config,
            sensor_df,
            FakeMeasurementManager(),
            "recover_management_mode",
        )

        self.assertEqual(result_df.loc[0, "gnss_pos_management_mode"], "recover")
        self.assertAlmostEqual(float(result_df.loc[0, "gnss_pos_recovery_scale"]), 2.5, places=6)
        self.assertEqual(result_df.loc[0, "baro_management_mode"], "unavailable")
        self.assertEqual(result_df.loc[0, "mag_management_mode"], "unavailable")

        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["gnss_pos_management_count_recover"], 1.0)
        self.assertEqual(metrics["gnss_pos_management_count_update"], 0.0)
        self.assertEqual(metrics["gnss_pos_recovery_scaled_updates"], 1.0)
        self.assertAlmostEqual(metrics["max_gnss_pos_recovery_scale"], 2.5, places=6)
        self.assertEqual(metrics["baro_management_count_unavailable"], 1.0)
        self.assertEqual(metrics["mag_management_count_unavailable"], 1.0)

    def test_pipeline_keeps_missing_baro_and_mag_innovations_as_nan(self) -> None:
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
                "gnss_vx": [1.0],
                "gnss_vy": [0.0],
                "gnss_vz": [0.0],
                "baro_h": [np.nan],
                "mag_yaw": [np.nan],
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
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "test"}),
                ),
                patch("eskf_stack.app.MeasurementManager", return_value=FakeMeasurementManager()),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", return_value={}),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 1)
        self.assertTrue(np.isnan(result_df.loc[0, "baro_innovation_abs"]))
        self.assertTrue(np.isnan(result_df.loc[0, "yaw_innovation_abs_deg"]))
        self.assertEqual(result_df.loc[0, "baro_management_mode"], "unavailable")
        self.assertEqual(result_df.loc[0, "mag_management_mode"], "unavailable")
        metrics = compute_metrics(result_df)
        self.assertEqual(metrics["baro_available_measurements"], 0.0)
        self.assertEqual(metrics["mag_available_measurements"], 0.0)
        self.assertEqual(metrics["baro_management_count_unavailable"], 1.0)
        self.assertEqual(metrics["mag_management_count_unavailable"], 1.0)
        self.assertNotIn("mean_baro_innovation", metrics)
        self.assertNotIn("mean_mag_innovation", metrics)


if __name__ == "__main__":
    unittest.main()
