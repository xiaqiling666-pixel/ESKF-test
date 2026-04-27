from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.analysis.exporter import save_dataset_source_summary
from eskf_stack.config import load_config


class ExporterTests(unittest.TestCase):
    def test_save_dataset_source_summary_includes_config_policy_and_initialization(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")
        initialization_summary = {
            "initialization_phase": "INITIALIZED",
            "initialization_reason": "direct_init_completed",
            "heading_source": "mag_yaw",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            metrics_dir = Path(temp_dir)
            save_dataset_source_summary(
                metrics_dir,
                config,
                source_summary={"adapter_kind": "standard_csv"},
                navigation_reference_override=None,
                initialization_summary=initialization_summary,
            )
            summary_text = (metrics_dir / "dataset_source_summary.txt").read_text(encoding="utf-8")

        self.assertIn("config_profile: default_general", summary_text)
        self.assertIn("fusion_policy_use_nis_rejection: True", summary_text)
        self.assertIn("fusion_policy_use_adaptive_r: True", summary_text)
        self.assertIn("fusion_policy_use_recovery_scale: True", summary_text)
        self.assertIn("adapter_kind: standard_csv", summary_text)
        self.assertIn("initialization_phase: INITIALIZED", summary_text)
        self.assertIn("heading_source: mag_yaw", summary_text)


if __name__ == "__main__":
    unittest.main()
