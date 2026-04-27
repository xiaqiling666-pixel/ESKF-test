from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.analysis.experiment_batch import run_experiment_batch
from eskf_stack.config import load_config, project_path


class ExperimentBatchTests(unittest.TestCase):
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

    def test_run_experiment_batch_collects_metrics_from_each_config(self) -> None:
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
                    ]
                ).to_csv(metrics_dir / "metrics.csv", index=False)

            result = run_experiment_batch(
                config_paths=[config_a, config_b],
                summary_dir=temp_dir / "summary",
                run_pipeline_fn=fake_run_pipeline,
            )

            summary = pd.read_csv(result.summary_path)
            key_summary = pd.read_csv(result.key_summary_path)

        self.assertEqual(result.run_count, 2)
        self.assertEqual(list(summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertEqual(list(summary["use_nis_rejection"]), [False, True])
        self.assertEqual(list(summary["use_adaptive_r"]), [False, False])
        self.assertEqual(list(summary["use_recovery_scale"]), [False, False])
        self.assertEqual(list(summary["position_rmse_m"]), [1.0, 0.8])
        self.assertEqual(list(summary["gnss_pos_rejections"]), [3.0, 3.0])
        self.assertEqual(list(key_summary["experiment_name"]), ["baseline_eskf", "eskf_nis_reject"])
        self.assertIn("position_rmse_m", key_summary.columns)
        self.assertIn("gnss_pos_rejections", key_summary.columns)
        self.assertNotIn("config_path", key_summary.columns)


if __name__ == "__main__":
    unittest.main()
