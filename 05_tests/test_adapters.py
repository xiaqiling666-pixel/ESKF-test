from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.adapters import (
    CORE_INPUT_COLUMNS,
    DIAGNOSTIC_TRUTH_COLUMNS,
    OPTIONAL_MEASUREMENT_COLUMNS,
    contract_column_groups,
    load_dataframe_from_config,
    load_dataset_from_config,
)
from eskf_stack.config import DatasetAdapterConfig, load_config


class AdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config(PROJECT_ROOT / "01_data" / "config.json")

    def test_standard_csv_adapter_loads_and_sorts_dataframe(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "time,ax,ay,az,gx,gy,gz",
                        "1.0,0,0,-9.8,0,0,0",
                        "0.5,0,0,-9.8,0,0,0",
                    ]
                ),
                encoding="utf-8",
            )
            config = replace(self.config, dataset_path=str(csv_path))

            result = load_dataset_from_config(config)
            dataframe = result.dataframe

            self.assertEqual(list(dataframe["time"]), [0.5, 1.0])
            self.assertIsNotNone(result.source_summary)
            self.assertEqual(result.source_summary["adapter_kind"], "standard_csv")
            self.assertEqual(result.source_summary["input_quality_row_count"], "2")
            self.assertEqual(result.source_summary["input_quality_time_monotonic_input_order"], "false")
            self.assertEqual(result.source_summary["input_quality_time_non_positive_dt_count"], "1")
            self.assertEqual(result.source_summary["input_quality_core_complete_ratio"], "1.000000")
            self.assertEqual(result.source_summary["input_quality_gnss_pos_coverage_ratio"], "0.000000")
            self.assertEqual(result.source_summary["input_quality_diagnostic_truth_role"], "evaluation_only_not_filter_input")
            self.assertIsNotNone(result.input_quality_metrics)
            self.assertEqual(result.input_quality_metrics["input_quality_row_count"], 2.0)
            self.assertEqual(result.input_quality_metrics["input_quality_time_monotonic_input_order_flag"], 0.0)
            self.assertIn("gnss_x", dataframe.columns)
            self.assertIn("truth_yaw", dataframe.columns)
            self.assertTrue(np.isnan(float(dataframe.loc[0, "gnss_x"])))

    def test_contract_column_groups_are_split_by_role(self) -> None:
        groups = contract_column_groups()

        self.assertEqual(groups["core_input"], CORE_INPUT_COLUMNS)
        self.assertEqual(groups["optional_measurement"], OPTIONAL_MEASUREMENT_COLUMNS)
        self.assertEqual(groups["diagnostic_truth"], DIAGNOSTIC_TRUTH_COLUMNS)
        self.assertIn("time", CORE_INPUT_COLUMNS)
        self.assertIn("gnss_x", OPTIONAL_MEASUREMENT_COLUMNS)
        self.assertIn("truth_x", DIAGNOSTIC_TRUTH_COLUMNS)
        self.assertNotIn("truth_x", OPTIONAL_MEASUREMENT_COLUMNS)
        self.assertNotIn("gnss_x", DIAGNOSTIC_TRUTH_COLUMNS)

    def test_unknown_adapter_kind_raises_clear_error(self) -> None:
        config = replace(
            self.config,
            dataset_adapter=DatasetAdapterConfig(kind="unknown_adapter"),
        )

        with self.assertRaisesRegex(ValueError, "不支持的数据适配器类型"):
            load_dataframe_from_config(config)

    def test_standard_csv_adapter_rejects_missing_required_contract_field(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "bad_sample.csv"
            csv_path.write_text(
                "\n".join(
                    [
                        "time,ax,ay,az,gx,gy",
                        "1.0,0,0,-9.8,0,0",
                    ]
                ),
                encoding="utf-8",
            )
            config = replace(self.config, dataset_path=str(csv_path))

            with self.assertRaisesRegex(ValueError, "统一输入契约必需字段: gz"):
                load_dataset_from_config(config)

    def test_great_msf_imu_ins_adapter_maps_to_enu_and_overrides_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            imu_path = Path(temp_dir) / "imu.txt"
            ins_path = Path(temp_dir) / "solution.ins"
            imu_path.write_text(
                "\n".join(
                    [
                        "100.00 0.10 0.20 -9.80 0.01 0.02 0.03",
                        "100.04 0.11 0.21 -9.81 0.01 0.02 0.03",
                        "100.08 0.12 0.22 -9.82 0.01 0.02 0.03",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ins_path.write_text(
                "\n".join(
                    [
                        "# header",
                        "100.04 6378137.0 0.0 0.0 0.0 0.0 0.0 0 0 0 0 0 0 0 0 0 GNSS 10 1.0 Float 0.0",
                        "100.08 6378139.0 10.0 5.0 2.0 1.0 3.0 0 0 0 0 0 0 0 0 0 GNSS 10 1.0 Float 0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = replace(
                self.config,
                dataset_adapter=DatasetAdapterConfig(
                    kind="great_msf_imu_ins",
                    options={
                        "imu_txt_path": str(imu_path),
                        "ins_path": str(ins_path),
                        "reference_mode": "first_ins",
                        "normalize_time_to_zero": True,
                        "alignment_tolerance_s": 0.01,
                    },
                ),
            )

            result = load_dataset_from_config(config)
            dataframe = result.dataframe

            self.assertIsNotNone(result.navigation_reference_override)
            self.assertIsNotNone(result.source_summary)
            self.assertEqual(result.source_summary["adapter_kind"], "great_msf_imu_ins")
            self.assertEqual(result.source_summary["primary_input_role"], "great_msf_imu_plus_solution")
            self.assertAlmostEqual(result.navigation_reference_override.reference_lat_deg, 0.0, places=6)
            self.assertAlmostEqual(result.navigation_reference_override.reference_lon_deg, 0.0, places=6)
            self.assertAlmostEqual(result.navigation_reference_override.reference_height_m, 0.0, places=4)
            self.assertTrue(np.allclose(dataframe["time"].to_numpy(), [0.0, 0.04, 0.08], atol=1e-9))
            self.assertAlmostEqual(float(dataframe.loc[1, "gnss_x"]), 0.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[1, "gnss_y"]), 0.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[1, "gnss_z"]), 0.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_x"]), 10.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_y"]), 5.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_z"]), 2.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_vx"]), 1.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_vy"]), 3.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_vz"]), 2.0, places=6)

    def test_dx_decoded_flight_adapter_loads_decoded_csv_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            decoded_dir = Path(temp_dir) / "decoded_case"
            decoded_dir.mkdir(parents=True, exist_ok=True)
            (decoded_dir / "IMU_Full.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,AccX,AccY,AccZ,GyrX,GyrY,GyrZ",
                        "1000000,0.1,0.2,-9.8,0.01,0.02,0.03",
                        "1100000,0.1,0.2,-9.8,0.01,0.02,0.03",
                        "1200000,0.1,0.2,-9.8,0.01,0.02,0.03",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (decoded_dir / "GPS_Raw.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,Status,NSats,Lat,Lng,Alt",
                        "1000000,4,20,0.0,0.0,10.0",
                        "1200000,4,20,0.00001,0.0,12.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (decoded_dir / "POS_Global_Truth.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,Lat,Lng,Alt",
                        "1000000,0.0,0.0,10.0",
                        "1100000,0.000005,0.0,11.0",
                        "1200000,0.00001,0.0,12.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (decoded_dir / "XKF_Local_Truth.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,Roll,Pitch,Yaw,VN,VE,VD,PN,PE,PD",
                        "1000000,0.0,0.0,90.0,1.0,2.0,-0.5,0.0,0.0,0.0",
                        "1100000,0.0,0.0,90.0,1.5,2.5,-0.6,0.5,0.2,-0.1",
                        "1200000,0.0,0.0,90.0,2.0,3.0,-0.7,1.0,0.4,-0.2",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = replace(
                self.config,
                dataset_adapter=DatasetAdapterConfig(
                    kind="dx_decoded_flight_csv",
                    options={
                        "decoded_dir": str(decoded_dir),
                        "reference_mode": "first_gps",
                        "normalize_time_to_zero": True,
                        "gps_alignment_tolerance_s": 0.15,
                        "truth_alignment_tolerance_s": 0.15,
                    },
                ),
            )

            result = load_dataset_from_config(config)
            dataframe = result.dataframe

            self.assertIsNotNone(result.navigation_reference_override)
            self.assertIsNotNone(result.source_summary)
            self.assertEqual(result.source_summary["adapter_kind"], "dx_decoded_flight_csv")
            self.assertEqual(result.source_summary["primary_input_role"], "extracted_imu_plus_decoded_flight_csv")
            self.assertAlmostEqual(result.navigation_reference_override.reference_lat_deg, 0.0, places=9)
            self.assertAlmostEqual(result.navigation_reference_override.reference_lon_deg, 0.0, places=9)
            self.assertAlmostEqual(result.navigation_reference_override.reference_height_m, 10.0, places=9)
            self.assertTrue(np.allclose(dataframe["time"].to_numpy(), [0.0, 0.1, 0.2], atol=1e-9))
            self.assertAlmostEqual(float(dataframe.loc[0, "gnss_z"]), 0.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_y"]), 1.105742758, places=3)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_z"]), 2.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[1, "truth_z"]), 1.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "truth_vx"]), 3.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "truth_vy"]), 2.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "truth_vz"]), 0.7, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "truth_yaw"]), 0.0, places=9)

    def test_dx_decoded_flight_adapter_supports_position_only_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            decoded_dir = Path(temp_dir) / "decoded_case"
            decoded_dir.mkdir(parents=True, exist_ok=True)
            (decoded_dir / "IMU_Full.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,AccX,AccY,AccZ,GyrX,GyrY,GyrZ",
                        "1000000,0.1,0.2,-9.8,0.01,0.02,0.03",
                        "1100000,0.1,0.2,-9.8,0.01,0.02,0.03",
                        "1200000,0.1,0.2,-9.8,0.01,0.02,0.03",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (decoded_dir / "GPS_Raw.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,Status,NSats,Lat,Lng,Alt",
                        "1000000,4,20,0.0,0.0,10.0",
                        "1200000,4,20,0.00001,0.0,12.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            config = replace(
                self.config,
                dataset_adapter=DatasetAdapterConfig(
                    kind="dx_decoded_flight_csv",
                    options={
                        "decoded_dir": str(decoded_dir),
                        "reference_mode": "first_gps",
                        "normalize_time_to_zero": True,
                        "gps_alignment_tolerance_s": 0.15,
                        "gps_velocity_mode": "none",
                    },
                ),
            )

            result = load_dataset_from_config(config)
            dataframe = result.dataframe

            self.assertEqual(result.source_summary["gps_velocity_mode"], "none")
            self.assertEqual(result.source_summary["gps_velocity_semantics"], "disabled_position_only_updates")
            self.assertTrue(np.isnan(float(dataframe.loc[0, "gnss_vx"])))
            self.assertTrue(np.isnan(float(dataframe.loc[0, "gnss_vy"])))
            self.assertTrue(np.isnan(float(dataframe.loc[0, "gnss_vz"])))
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_z"]), 2.0, places=6)

    def test_dx_imu_external_solution_adapter_aligns_solution_timebase(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            imu_path = temp_root / "IMU_Full.csv"
            ins_path = temp_root / "solution.ins"
            decoded_dir = temp_root / "decoded_truth"
            decoded_dir.mkdir(parents=True, exist_ok=True)

            imu_path.write_text(
                "\n".join(
                    [
                        "TimeUS,AccX,AccY,AccZ,GyrX,GyrY,GyrZ",
                        "500000000,0.1,0.2,-9.8,0.01,0.02,0.03",
                        "500100000,0.1,0.2,-9.8,0.01,0.02,0.03",
                        "500200000,0.1,0.2,-9.8,0.01,0.02,0.03",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ins_path.write_text(
                "\n".join(
                    [
                        "# header",
                        "466000.000 6378137.0 0.0 0.0 1.0 2.0 3.0 0 0 90 0 0 0 0 0 0 GNSS 12 1.0 Float 0.0",
                        "466000.100 6378138.0 10.0 5.0 4.0 5.0 6.0 0 0 80 0 0 0 0 0 0 GNSS 12 1.0 Float 0.0",
                        "466000.200 6378140.0 20.0 8.0 7.0 8.0 9.0 0 0 70 0 0 0 0 0 0 GNSS 12 1.0 Float 0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (decoded_dir / "POS_Global_Truth.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,Lat,Lng,Alt",
                        "500000000,0.0,0.0,0.0",
                        "500100000,0.0,0.00001,1.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (decoded_dir / "XKF_Local_Truth.csv").write_text(
                "\n".join(
                    [
                        "TimeUS,Roll,Pitch,Yaw,VN,VE,VD,PN,PE,PD",
                        "500000000,0.0,0.0,90.0,1.0,2.0,-0.5,0.0,0.0,0.0",
                        "500100000,0.0,0.0,80.0,1.5,2.5,-0.6,0.0,0.0,0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = replace(
                self.config,
                dataset_adapter=DatasetAdapterConfig(
                    kind="dx_imu_external_solution",
                    options={
                        "imu_csv_path": str(imu_path),
                        "ins_path": str(ins_path),
                        "decoded_dir_for_truth": str(decoded_dir),
                        "reference_mode": "first_solution",
                        "time_offset_mode": "align_first_sample",
                        "alignment_tolerance_s": 0.02,
                        "normalize_time_to_zero": True,
                    },
                ),
            )

            result = load_dataset_from_config(config)
            dataframe = result.dataframe

            self.assertIsNotNone(result.navigation_reference_override)
            self.assertEqual(result.source_summary["adapter_kind"], "dx_imu_external_solution")
            self.assertEqual(result.source_summary["assigned_solution_rows"], "3")
            self.assertAlmostEqual(float(dataframe.loc[0, "gnss_x"]), 0.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "gnss_vx"]), 2.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "gnss_vy"]), 3.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "gnss_vz"]), 1.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[1, "gnss_x"]), 10.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[1, "gnss_y"]), 5.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[1, "gnss_z"]), 1.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_x"]), 20.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_y"]), 8.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[2, "gnss_z"]), 3.0, places=6)

    def test_dx_imu_external_solution_adapter_supports_imu_transform_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_root = Path(temp_dir)
            imu_path = temp_root / "IMU_Full.csv"
            ins_path = temp_root / "solution.ins"

            imu_path.write_text(
                "\n".join(
                    [
                        "TimeUS,AccX,AccY,AccZ,GyrX,GyrY,GyrZ",
                        "500000000,1.0,2.0,-9.0,0.1,0.2,0.3",
                        "500100000,1.0,2.0,-9.0,0.1,0.2,0.3",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            ins_path.write_text(
                "\n".join(
                    [
                        "# header",
                        "466000.000 6378137.0 0.0 0.0 1.0 2.0 3.0 0 0 90 0 0 0 0 0 0 GNSS 12 1.0 Float 0.0",
                        "466000.100 6378138.0 10.0 5.0 4.0 5.0 6.0 0 0 80 0 0 0 0 0 0 GNSS 12 1.0 Float 0.0",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )

            config = replace(
                self.config,
                dataset_adapter=DatasetAdapterConfig(
                    kind="dx_imu_external_solution",
                    options={
                        "imu_csv_path": str(imu_path),
                        "ins_path": str(ins_path),
                        "reference_mode": "first_solution",
                        "time_offset_mode": "align_first_sample",
                        "alignment_tolerance_s": 0.02,
                        "normalize_time_to_zero": True,
                        "imu_transform_mode": "ardupilot_frd_to_flu",
                    },
                ),
            )

            result = load_dataset_from_config(config)
            dataframe = result.dataframe

            self.assertEqual(result.source_summary["imu_transform_mode"], "ardupilot_frd_to_flu")
            self.assertAlmostEqual(float(dataframe.loc[0, "ax"]), 1.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "ay"]), -2.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "az"]), 9.0, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "gx"]), 0.1, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "gy"]), -0.2, places=6)
            self.assertAlmostEqual(float(dataframe.loc[0, "gz"]), -0.3, places=6)


if __name__ == "__main__":
    unittest.main()
