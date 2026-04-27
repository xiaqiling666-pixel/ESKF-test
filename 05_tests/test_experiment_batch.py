from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.analysis.experiment_batch import (
    CATEGORY_SUMMARY_PREVIEW_LIMIT,
    KEY_SUMMARY_PREVIEW_LIMIT,
    ExperimentBatchResult,
    run_experiment_batch,
)
from eskf_stack.config import load_config, project_path
from run_experiment_batch import _result_output_lines, main


class ExperimentBatchTests(unittest.TestCase):
    def test_result_output_lines_include_category_summaries(self) -> None:
        result = ExperimentBatchResult(
            summary_path=Path("summary.csv"),
            key_summary_path=Path("key.csv"),
            delta_summary_path=Path("delta.csv"),
            delta_key_summary_path=Path("delta_key.csv"),
            core_compare_path=Path("core_compare.csv"),
            manifest_path=Path("manifest.json"),
            baseline_experiment_name="baseline_eskf",
            key_summary_columns=["experiment_name", "position_rmse_m"],
            key_summary_preview_columns=["experiment_name", "position_rmse_m"],
            core_compare_columns=["experiment_name", "delta_position_rmse_m", "position_rmse_m_vs_baseline"],
            category_summary_names=["measurement_management", "runtime"],
            category_summary_preview_names=["measurement_management", "runtime"],
            category_summary_paths={
                "measurement_management": Path("measurement.csv"),
                "runtime": Path("runtime.csv"),
            },
            category_metric_columns={
                "measurement_management": ["gnss_pos_rejections", "gnss_pos_nis_reject_policy_bypass_count"],
                "runtime": ["pipeline_runtime_s"],
            },
            delta_category_summary_paths={
                "measurement_management": Path("measurement_delta.csv"),
                "runtime": Path("runtime_delta.csv"),
            },
            delta_category_metric_columns={
                "measurement_management": ["delta_gnss_pos_rejections"],
                "runtime": ["delta_pipeline_runtime_s"],
            },
            delta_category_source_metrics={
                "measurement_management": ["gnss_pos_rejections"],
                "runtime": ["pipeline_runtime_s"],
            },
            run_count=2,
        )

        lines = _result_output_lines(result)

        self.assertEqual(lines[0], "Experiment batch finished. Runs: 2")
        self.assertEqual(lines[1], "Summary: summary.csv")
        self.assertEqual(lines[2], "Key summary: key.csv")
        self.assertEqual(lines[3], "Manifest: manifest.json")
        self.assertEqual(lines[4], "Key summary columns: 2")
        self.assertEqual(lines[5], "Key summary preview: experiment_name, position_rmse_m")
        self.assertEqual(lines[6], "Category summary names: measurement_management, runtime")
        self.assertEqual(lines[7], "Category summaries:")
        self.assertEqual(lines[8], "- measurement_management (2 metrics): measurement.csv")
        self.assertEqual(lines[9], "- runtime (1 metrics): runtime.csv")

    def test_result_output_lines_add_ellipsis_when_preview_lists_are_truncated(self) -> None:
        result = ExperimentBatchResult(
            summary_path=Path("summary.csv"),
            key_summary_path=Path("key.csv"),
            delta_summary_path=Path("delta.csv"),
            delta_key_summary_path=Path("delta_key.csv"),
            core_compare_path=Path("core_compare.csv"),
            manifest_path=Path("manifest.json"),
            baseline_experiment_name="baseline_eskf",
            key_summary_columns=[
                "experiment_name",
                "use_nis_rejection",
                "use_adaptive_r",
                "position_rmse_m",
                "velocity_rmse_mps",
                "yaw_rmse_deg",
                "final_position_error_m",
            ],
            key_summary_preview_columns=[
                "experiment_name",
                "use_nis_rejection",
                "use_adaptive_r",
                "position_rmse_m",
                "velocity_rmse_mps",
                "yaw_rmse_deg",
            ],
            core_compare_columns=["experiment_name", "delta_position_rmse_m", "position_rmse_m_vs_baseline"],
            category_summary_names=[
                "covariance_health",
                "estimation_error",
                "initialization",
                "measurement_management",
                "mode_state",
            ],
            category_summary_preview_names=[
                "covariance_health",
                "estimation_error",
                "initialization",
                "measurement_management",
            ],
            category_summary_paths={"measurement_management": Path("measurement.csv")},
            category_metric_columns={"measurement_management": ["gnss_pos_rejections"]},
            delta_category_summary_paths={"measurement_management": Path("measurement_delta.csv")},
            delta_category_metric_columns={"measurement_management": ["delta_gnss_pos_rejections"]},
            delta_category_source_metrics={"measurement_management": ["gnss_pos_rejections"]},
            run_count=3,
        )

        lines = _result_output_lines(result)

        self.assertEqual(
            lines[5],
            "Key summary preview: experiment_name, use_nis_rejection, use_adaptive_r, position_rmse_m, velocity_rmse_mps, yaw_rmse_deg, ...",
        )
        self.assertEqual(
            lines[6],
            "Category summary names: covariance_health, estimation_error, initialization, measurement_management, ...",
        )

    def test_result_output_lines_report_none_when_no_category_summaries(self) -> None:
        result = ExperimentBatchResult(
            summary_path=Path("summary.csv"),
            key_summary_path=Path("key.csv"),
            delta_summary_path=Path("delta.csv"),
            delta_key_summary_path=Path("delta_key.csv"),
            core_compare_path=Path("core_compare.csv"),
            manifest_path=Path("manifest.json"),
            baseline_experiment_name="baseline_eskf",
            key_summary_columns=["experiment_name"],
            key_summary_preview_columns=["experiment_name"],
            core_compare_columns=["experiment_name"],
            category_summary_names=[],
            category_summary_preview_names=[],
            category_summary_paths={},
            category_metric_columns={},
            delta_category_summary_paths={},
            delta_category_metric_columns={},
            delta_category_source_metrics={},
            run_count=1,
        )

        lines = _result_output_lines(result)

        self.assertEqual(lines[0], "Experiment batch finished. Runs: 1")
        self.assertEqual(lines[1], "Summary: summary.csv")
        self.assertEqual(lines[2], "Key summary: key.csv")
        self.assertEqual(lines[3], "Manifest: manifest.json")
        self.assertEqual(lines[4], "Key summary columns: 1")
        self.assertEqual(lines[5], "Key summary preview: experiment_name")
        self.assertEqual(lines[6], "Category summaries: none")

    def test_main_prints_category_summaries(self) -> None:
        result = ExperimentBatchResult(
            summary_path=Path("summary.csv"),
            key_summary_path=Path("key.csv"),
            delta_summary_path=Path("delta.csv"),
            delta_key_summary_path=Path("delta_key.csv"),
            core_compare_path=Path("core_compare.csv"),
            manifest_path=Path("manifest.json"),
            baseline_experiment_name="baseline_eskf",
            key_summary_columns=["experiment_name", "gnss_pos_rejections"],
            key_summary_preview_columns=["experiment_name", "gnss_pos_rejections"],
            core_compare_columns=["experiment_name", "delta_gnss_pos_rejections", "gnss_pos_rejections_vs_baseline"],
            category_summary_names=["measurement_management"],
            category_summary_preview_names=["measurement_management"],
            category_summary_paths={"measurement_management": Path("measurement.csv")},
            category_metric_columns={"measurement_management": ["gnss_pos_rejections"]},
            delta_category_summary_paths={"measurement_management": Path("measurement_delta.csv")},
            delta_category_metric_columns={"measurement_management": ["delta_gnss_pos_rejections"]},
            delta_category_source_metrics={"measurement_management": ["gnss_pos_rejections"]},
            run_count=1,
        )

        output = io.StringIO()
        with (
            patch("run_experiment_batch.run_experiment_batch", return_value=result) as mock_run,
            contextlib.redirect_stdout(output),
        ):
            exit_code = main(["config_a.json"])

        self.assertEqual(exit_code, 0)
        mock_run.assert_called_once_with(config_paths=["config_a.json"])
        printed = output.getvalue()
        self.assertIn("Experiment batch finished. Runs: 1", printed)
        self.assertIn("Summary: summary.csv", printed)
        self.assertIn("Key summary: key.csv", printed)
        self.assertIn("Manifest: manifest.json", printed)
        self.assertIn("Key summary columns: 2", printed)
        self.assertIn("Key summary preview: experiment_name, gnss_pos_rejections", printed)
        self.assertIn("Category summary names: measurement_management", printed)
        self.assertIn("Category summaries:", printed)
        self.assertIn("- measurement_management (1 metrics): measurement.csv", printed)

    def _write_experiment_config(
        self,
        temp_dir: Path,
        name: str,
        results_dir: Path,
        fusion_policy: dict[str, bool],
    ) -> Path:
        payload = json.loads((PROJECT_ROOT / "01_data" / "config.json").read_text(encoding="utf-8"))
        payload["config_metadata"] = {
            "profile": "experiment",
            "name": name,
            "purpose": "temporary experiment batch test config",
        }
        payload["results_dir"] = str(results_dir)
        payload["fusion_policy"] = fusion_policy
        config_path = temp_dir / f"{name}.json"
        config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return config_path

    def test_run_experiment_batch_collects_metrics_and_delta_summaries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_dir = Path(temp_root)
            config_a = self._write_experiment_config(
                temp_dir,
                "baseline_eskf",
                temp_dir / "results_a",
                {
                    "use_nis_rejection": False,
                    "use_adaptive_r": False,
                    "use_recovery_scale": False,
                },
            )
            config_b = self._write_experiment_config(
                temp_dir,
                "eskf_nis_reject",
                temp_dir / "results_b",
                {
                    "use_nis_rejection": True,
                    "use_adaptive_r": False,
                    "use_recovery_scale": False,
                },
            )

            def fake_run_pipeline(config_path: str | Path) -> None:
                config = load_config(config_path)
                metrics_dir = project_path(config.results_dir) / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [
                        {"metric": "position_rmse_m", "value": 1.0 if config.config_metadata.name == "baseline_eskf" else 0.8},
                        {"metric": "gnss_pos_rejections", "value": 3.0},
                        {"metric": "gnss_pos_nis_reject_exceed_not_rejected_count", "value": 1.0},
                        {"metric": "gnss_pos_nis_reject_policy_bypass_count", "value": 1.0},
                        {"metric": "gnss_pos_nis_reject_policy_bypass_pct", "value": 100.0},
                        {"metric": "initialization_completed_flag", "value": 1.0},
                    ]
                ).to_csv(metrics_dir / "metrics.csv", index=False)

            result = run_experiment_batch(
                config_paths=[config_a, config_b],
                summary_dir=temp_dir / "summary",
                run_pipeline_fn=fake_run_pipeline,
            )

            summary = pd.read_csv(result.summary_path)
            key_summary = pd.read_csv(result.key_summary_path)
            delta_summary = pd.read_csv(result.delta_summary_path)
            delta_key_summary = pd.read_csv(result.delta_key_summary_path)
            core_compare = pd.read_csv(result.core_compare_path)
            manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
            measurement_summary = pd.read_csv(result.category_summary_paths["measurement_management"])
            initialization_summary = pd.read_csv(result.category_summary_paths["initialization"])
            measurement_delta_summary = pd.read_csv(result.delta_category_summary_paths["measurement_management"])
            initialization_delta_summary = pd.read_csv(result.delta_category_summary_paths["initialization"])

        self.assertEqual(result.run_count, 2)
        self.assertEqual(result.baseline_experiment_name, "baseline_eskf")
        self.assertEqual(list(summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertEqual(list(summary["use_nis_rejection"]), [False, True])
        self.assertEqual(list(summary["use_adaptive_r"]), [False, False])
        self.assertEqual(list(summary["use_recovery_scale"]), [False, False])
        self.assertEqual(list(summary["position_rmse_m"]), [1.0, 0.8])
        self.assertEqual(list(summary["gnss_pos_rejections"]), [3.0, 3.0])
        self.assertIn("position_rmse_m", result.key_summary_columns)
        self.assertIn("gnss_pos_rejections", result.key_summary_columns)
        self.assertNotIn("gnss_pos_nis_reject_exceed_not_rejected_count", result.key_summary_columns)
        self.assertEqual(result.key_summary_columns, list(key_summary.columns))
        self.assertEqual(result.manifest_path.name, "experiment_metrics_manifest.json")
        self.assertEqual(manifest["run_count"], 2)
        self.assertEqual(Path(manifest["summary_path"]).name, result.summary_path.name)
        self.assertEqual(Path(manifest["key_summary_path"]).name, result.key_summary_path.name)
        self.assertEqual(Path(manifest["delta_summary_path"]).name, result.delta_summary_path.name)
        self.assertEqual(Path(manifest["delta_key_summary_path"]).name, result.delta_key_summary_path.name)
        self.assertEqual(Path(manifest["core_compare_path"]).name, result.core_compare_path.name)
        self.assertEqual(manifest["baseline_experiment_name"], result.baseline_experiment_name)
        self.assertEqual(manifest["key_summary_columns"], result.key_summary_columns)
        self.assertEqual(manifest["key_summary_preview_columns"], result.key_summary_preview_columns)
        self.assertEqual(manifest["core_compare_columns"], result.core_compare_columns)
        self.assertEqual(manifest["key_summary_preview_limit"], KEY_SUMMARY_PREVIEW_LIMIT)
        self.assertEqual(
            result.key_summary_preview_columns,
            result.key_summary_columns[: len(result.key_summary_preview_columns)],
        )
        self.assertLessEqual(len(result.key_summary_preview_columns), KEY_SUMMARY_PREVIEW_LIMIT)
        self.assertEqual(result.category_summary_names, sorted(result.category_summary_paths))
        self.assertEqual(manifest["category_summary_names"], result.category_summary_names)
        self.assertEqual(manifest["category_summary_preview_names"], result.category_summary_preview_names)
        self.assertEqual(manifest["category_summary_preview_limit"], CATEGORY_SUMMARY_PREVIEW_LIMIT)
        self.assertEqual(
            result.category_summary_preview_names,
            result.category_summary_names[: len(result.category_summary_preview_names)],
        )
        self.assertLessEqual(len(result.category_summary_preview_names), CATEGORY_SUMMARY_PREVIEW_LIMIT)
        self.assertEqual(sorted(manifest["category_summary_paths"]), result.category_summary_names)
        self.assertEqual(list(key_summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertIn("position_rmse_m", key_summary.columns)
        self.assertIn("gnss_pos_rejections", key_summary.columns)
        self.assertNotIn("gnss_pos_nis_reject_exceed_not_rejected_count", key_summary.columns)
        self.assertNotIn("config_path", key_summary.columns)
        self.assertEqual(list(delta_summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertEqual(list(delta_summary["baseline_experiment_name"]), ["baseline_eskf", "baseline_eskf"])
        self.assertEqual(list(delta_summary["is_baseline"]), [True, False])
        self.assertEqual(list(delta_summary["delta_position_rmse_m"]), [0.0, -0.2])
        self.assertEqual(list(delta_summary["delta_gnss_pos_rejections"]), [0.0, 0.0])
        self.assertEqual(list(delta_key_summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertIn("delta_position_rmse_m", delta_key_summary.columns)
        self.assertIn("delta_gnss_pos_rejections", delta_key_summary.columns)
        self.assertNotIn("position_rmse_m", delta_key_summary.columns)
        self.assertEqual(list(core_compare["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertIn("delta_position_rmse_m", core_compare.columns)
        self.assertIn("position_rmse_m_vs_baseline", core_compare.columns)
        self.assertIn("delta_gnss_pos_rejections", core_compare.columns)
        self.assertIn("gnss_pos_rejections_vs_baseline", core_compare.columns)
        self.assertEqual(list(core_compare["position_rmse_m_vs_baseline"]), ["unchanged", "improved"])
        self.assertEqual(list(core_compare["gnss_pos_rejections_vs_baseline"]), ["unchanged", "unchanged"])
        self.assertEqual(list(core_compare["improved_metric_count"]), [0.0, 1.0])
        self.assertEqual(list(core_compare["regressed_metric_count"]), [0.0, 0.0])
        self.assertIn("gnss_pos_nis_reject_exceed_not_rejected_count", measurement_summary.columns)
        self.assertIn("gnss_pos_nis_reject_policy_bypass_count", measurement_summary.columns)
        self.assertIn("gnss_pos_nis_reject_policy_bypass_pct", measurement_summary.columns)
        self.assertIn("measurement_management", result.category_metric_columns)
        self.assertIn("measurement_management", result.delta_category_metric_columns)
        self.assertIn("measurement_management", result.delta_category_source_metrics)
        self.assertIn(
            "gnss_pos_nis_reject_policy_bypass_count",
            result.category_metric_columns["measurement_management"],
        )
        self.assertIn(
            "delta_gnss_pos_nis_reject_policy_bypass_count",
            result.delta_category_metric_columns["measurement_management"],
        )
        self.assertEqual(
            result.delta_category_source_metrics["measurement_management"],
            [
                "gnss_pos_rejections",
                "gnss_pos_nis_reject_exceed_not_rejected_count",
                "gnss_pos_nis_reject_policy_bypass_count",
                "gnss_pos_nis_reject_policy_bypass_pct",
            ],
        )
        self.assertEqual(list(measurement_summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertEqual(
            list(measurement_summary["gnss_pos_nis_reject_exceed_not_rejected_count"]),
            [1.0, 1.0],
        )
        self.assertEqual(list(measurement_summary["gnss_pos_nis_reject_policy_bypass_count"]), [1.0, 1.0])
        self.assertEqual(list(measurement_summary["gnss_pos_nis_reject_policy_bypass_pct"]), [100.0, 100.0])
        self.assertIn("delta_gnss_pos_rejections", measurement_delta_summary.columns)
        self.assertIn("delta_gnss_pos_nis_reject_policy_bypass_count", measurement_delta_summary.columns)
        self.assertEqual(
            list(measurement_delta_summary.columns),
            [
                "experiment_name",
                "baseline_experiment_name",
                "is_baseline",
                "delta_gnss_pos_rejections",
                "delta_gnss_pos_nis_reject_exceed_not_rejected_count",
                "delta_gnss_pos_nis_reject_policy_bypass_count",
                "delta_gnss_pos_nis_reject_policy_bypass_pct",
            ],
        )
        self.assertEqual(list(measurement_delta_summary["delta_gnss_pos_rejections"]), [0.0, 0.0])
        self.assertIn("initialization_completed_flag", initialization_summary.columns)
        self.assertEqual(list(initialization_summary["initialization_completed_flag"]), [1.0, 1.0])
        self.assertIn("delta_initialization_completed_flag", initialization_delta_summary.columns)
        self.assertEqual(list(initialization_delta_summary["delta_initialization_completed_flag"]), [0.0, 0.0])
        self.assertIn("delta_category_summary_paths", manifest)
        self.assertIn("measurement_management", manifest["delta_category_summary_paths"])
        self.assertIn("delta_category_metric_columns", manifest)
        self.assertIn("measurement_management", manifest["delta_category_metric_columns"])
        self.assertIn("delta_category_source_metrics", manifest)
        self.assertEqual(
            manifest["delta_category_source_metrics"]["measurement_management"],
            result.delta_category_source_metrics["measurement_management"],
        )

    def test_run_experiment_batch_falls_back_to_first_experiment_when_no_baseline_name_present(self) -> None:
        with tempfile.TemporaryDirectory() as temp_root:
            temp_dir = Path(temp_root)
            config_a = self._write_experiment_config(
                temp_dir,
                "variant_a",
                temp_dir / "results_a",
                {
                    "use_nis_rejection": False,
                    "use_adaptive_r": False,
                    "use_recovery_scale": False,
                },
            )
            config_b = self._write_experiment_config(
                temp_dir,
                "variant_b",
                temp_dir / "results_b",
                {
                    "use_nis_rejection": True,
                    "use_adaptive_r": True,
                    "use_recovery_scale": False,
                },
            )

            metrics_by_name = {
                "variant_a": 1.2,
                "variant_b": 0.9,
            }

            def fake_run_pipeline(config_path: str | Path) -> None:
                config = load_config(config_path)
                metrics_dir = project_path(config.results_dir) / "metrics"
                metrics_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(
                    [
                        {"metric": "position_rmse_m", "value": metrics_by_name[config.config_metadata.name]},
                        {"metric": "initialization_completed_flag", "value": 1.0},
                    ]
                ).to_csv(metrics_dir / "metrics.csv", index=False)

            result = run_experiment_batch(
                config_paths=[config_a, config_b],
                summary_dir=temp_dir / "summary",
                run_pipeline_fn=fake_run_pipeline,
            )

            delta_summary = pd.read_csv(result.delta_summary_path)
            core_compare = pd.read_csv(result.core_compare_path)

        self.assertEqual(result.baseline_experiment_name, "variant_a")
        self.assertEqual(list(delta_summary["experiment_name"]), ["variant_a", "variant_b"])
        self.assertEqual(list(delta_summary["is_baseline"]), [True, False])
        self.assertEqual(list(delta_summary["delta_position_rmse_m"]), [0.0, -0.3])
        self.assertEqual(list(core_compare["position_rmse_m_vs_baseline"]), ["unchanged", "improved"])


if __name__ == "__main__":
    unittest.main()
