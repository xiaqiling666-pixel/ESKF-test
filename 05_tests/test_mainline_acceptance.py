from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.adapters.loader import DatasetLoadResult
from eskf_stack.analysis.evaluator import compute_metrics
from eskf_stack.app import run_pipeline
from eskf_stack.config import load_config
from eskf_stack.measurements.base import MeasurementResult


def _mainline_sensor_df(
    *,
    times: list[float],
    gnss_available: list[bool] | None = None,
    aux_available: list[bool] | None = None,
    yaw_rate_radps: float = 0.0,
) -> pd.DataFrame:
    if gnss_available is None:
        gnss_available = [True for _ in times]
    if aux_available is None:
        aux_available = [True for _ in times]

    gnss_x: list[float] = []
    gnss_y: list[float] = []
    gnss_z: list[float] = []
    gnss_vx: list[float] = []
    gnss_vy: list[float] = []
    gnss_vz: list[float] = []
    baro_h: list[float] = []
    mag_yaw: list[float] = []

    for time_s, has_gnss, has_aux in zip(times, gnss_available, aux_available):
        if has_gnss:
            gnss_x.append(10.0 + time_s)
            gnss_y.append(20.0)
            gnss_z.append(5.0)
            gnss_vx.append(1.0)
            gnss_vy.append(0.0)
            gnss_vz.append(0.0)
        else:
            gnss_x.append(np.nan)
            gnss_y.append(np.nan)
            gnss_z.append(np.nan)
            gnss_vx.append(np.nan)
            gnss_vy.append(np.nan)
            gnss_vz.append(np.nan)

        if has_aux:
            baro_h.append(5.0)
            mag_yaw.append(0.0)
        else:
            baro_h.append(np.nan)
            mag_yaw.append(np.nan)

    row_count = len(times)
    return pd.DataFrame(
        {
            "time": times,
            "ax": [0.0] * row_count,
            "ay": [0.0] * row_count,
            "az": [-9.81] * row_count,
            "gx": [0.0] * row_count,
            "gy": [0.0] * row_count,
            "gz": [yaw_rate_radps] * row_count,
            "gnss_x": gnss_x,
            "gnss_y": gnss_y,
            "gnss_z": gnss_z,
            "gnss_vx": gnss_vx,
            "gnss_vy": gnss_vy,
            "gnss_vz": gnss_vz,
            "baro_h": baro_h,
            "mag_yaw": mag_yaw,
            "truth_x": [np.nan] * row_count,
            "truth_y": [np.nan] * row_count,
            "truth_z": [np.nan] * row_count,
            "truth_vx": [np.nan] * row_count,
            "truth_vy": [np.nan] * row_count,
            "truth_vz": [np.nan] * row_count,
            "truth_yaw": [np.nan] * row_count,
        }
    )


class MainlineAcceptanceTests(unittest.TestCase):
    def tearDown(self) -> None:
        shutil.rmtree(PROJECT_ROOT / "_tmp_mainline_acceptance", ignore_errors=True)

    def test_minimum_pipeline_exports_fusion_output_metrics_and_lever_arm_diagnostics(self) -> None:
        base_config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        config = replace(base_config, gnss_lever_arm_body_m=[0.2, 0.0, 0.0])
        sensor_df = _mainline_sensor_df(
            times=[0.0, 0.1, 0.2, 0.3],
            yaw_rate_radps=2.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            results_root = Path(temp_dir)
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "acceptance"}),
                ),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=results_root),
            ):
                result_df = run_pipeline()

            metrics_dir = results_root / "metrics"
            fusion_output_path = metrics_dir / "fusion_output.csv"
            metrics_csv_path = metrics_dir / "metrics.csv"
            metrics_summary_path = metrics_dir / "metrics_summary.txt"

            self.assertTrue(fusion_output_path.exists())
            self.assertTrue(metrics_csv_path.exists())
            self.assertTrue(metrics_summary_path.exists())

            exported_df = pd.read_csv(fusion_output_path)
            exported_metrics = pd.read_csv(metrics_csv_path).set_index("metric")["value"]

            self.assertEqual(len(result_df), 4)
            self.assertEqual(len(exported_df), 4)
            self.assertTrue(result_df["used_gnss_pos"].any())
            self.assertTrue(result_df["used_gnss_vel"].any())
            self.assertTrue(result_df["used_baro"].any())
            self.assertTrue(result_df["used_mag"].any())
            self.assertFalse(result_df["initialization_pending"].any())
            self.assertAlmostEqual(float(result_df["gnss_lever_arm_nav_norm_m"].max()), 0.2, places=6)
            self.assertAlmostEqual(float(result_df["gnss_lever_arm_rotational_speed_mps"].max()), 0.4, places=5)

            self.assertIn("gnss_pos_updates", exported_metrics.index)
            self.assertIn("gnss_lever_arm_body_norm_m", exported_metrics.index)
            self.assertGreater(float(exported_metrics["gnss_pos_updates"]), 0.0)
            self.assertAlmostEqual(float(exported_metrics["gnss_lever_arm_body_norm_m"]), 0.2, places=6)
            self.assertIn("initialization_phase", metrics_summary_path.read_text(encoding="utf-8"))

    def test_pipeline_records_gnss_outage_hold_and_recovery_in_acceptance_scenario(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        sensor_df = _mainline_sensor_df(
            times=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            gnss_available=[True, True, False, False, False, True, True, True],
            aux_available=[True, True, True, True, True, True, True, True],
        )

        class ScenarioMeasurementManager:
            def process(self, filter_engine, model, frame, current_mode=None):
                if model.name == "gnss_pos" and frame.gnss_pos is not None:
                    return MeasurementResult(
                        name="gnss_pos",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.2,
                        management_mode="update",
                    )
                if model.name == "gnss_vel" and frame.gnss_vel is not None:
                    return MeasurementResult(
                        name="gnss_vel",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.2,
                        management_mode="update",
                    )
                if model.name == "baro" and frame.baro_h is not None:
                    return MeasurementResult(
                        name="baro",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.2,
                        management_mode="update",
                    )
                if model.name == "mag" and frame.mag_yaw is not None:
                    return MeasurementResult(
                        name="mag",
                        available=True,
                        used=True,
                        innovation_value=0.1,
                        nis=0.2,
                        management_mode="update",
                    )
                return MeasurementResult(name=model.name, available=False, used=False, management_mode="unavailable")

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch("eskf_stack.app.load_config", return_value=config),
                patch(
                    "eskf_stack.app.load_dataset_from_config",
                    return_value=DatasetLoadResult(dataframe=sensor_df, source_summary={"adapter_kind": "acceptance"}),
                ),
                patch("eskf_stack.app.MeasurementManager", return_value=ScenarioMeasurementManager()),
                patch("eskf_stack.app.generate_demo_dataset"),
                patch("eskf_stack.app.ensure_dir", return_value=Path(temp_dir)),
                patch("eskf_stack.app.export_pipeline_results", return_value={}),
            ):
                result_df = run_pipeline()

        self.assertEqual(len(result_df), 8)
        self.assertEqual(result_df.loc[0, "mode"], "GNSS_STABLE")
        self.assertIn("INERTIAL_HOLD", set(result_df["mode_target"]))
        self.assertEqual(result_df.loc[2, "gnss_pos_management_mode"], "unavailable")
        self.assertEqual(result_df.loc[2, "baro_management_mode"], "update")
        self.assertGreater(float(result_df.loc[3, "gnss_outage_s"]), 0.0)
        self.assertIn("INERTIAL_HOLD", set(result_df["mode"]))
        self.assertIn(result_df.loc[7, "mode_target"], {"GNSS_STABLE", "GNSS_AVAILABLE"})

        metrics = compute_metrics(result_df)
        self.assertGreater(metrics["mode_duration_INERTIAL_HOLD_s"], 0.0)
        self.assertEqual(metrics["gnss_pos_management_count_unavailable"], 3.0)
        self.assertEqual(metrics["baro_updates"], 8.0)


if __name__ == "__main__":
    unittest.main()
