from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.config import DEFAULT_CONFIG_PATH, load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_uses_default_general_profile(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config.json")

        self.assertEqual(config.config_metadata.profile, "default_general")
        self.assertEqual(Path(config.config_path).resolve(), DEFAULT_CONFIG_PATH.resolve())
        self.assertTrue(config.initialization.static_coarse_alignment_enabled)
        self.assertGreater(config.time_step_management.max_dt_s, config.time_step_management.min_positive_dt_s)
        self.assertGreater(config.initialization.heading_wait_timeout_s, 0.0)
        self.assertFalse(config.initialization.zero_yaw_fallback_enabled)
        self.assertTrue(config.fusion_policy.use_nis_rejection)
        self.assertTrue(config.fusion_policy.use_adaptive_r)
        self.assertTrue(config.fusion_policy.use_recovery_scale)
        self.assertGreater(config.innovation_management.baro_nis_reject_threshold, 0.0)
        self.assertGreater(config.innovation_management.mag_yaw_nis_reject_threshold, 0.0)

    def test_sample_config_uses_non_default_profile(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config_00000422_decoded.json")

        self.assertEqual(config.config_metadata.profile, "sample_validation")
        self.assertNotEqual(Path(config.config_path).resolve(), DEFAULT_CONFIG_PATH.resolve())
        self.assertGreater(config.initialization.static_window_min_samples, 0)
        self.assertGreater(config.initialization.bootstrap_min_dt_s, 0.0)
        self.assertTrue(config.time_step_management.skip_large_dt)
        self.assertGreater(config.innovation_management.baro_nis_adapt_threshold, 0.0)
        self.assertGreater(config.innovation_management.mag_yaw_nis_adapt_threshold, 0.0)

    def test_experiment_configs_define_ablation_fusion_policies(self) -> None:
        expected_policies = {
            "config_experiment_baseline_eskf.json": (False, False, False),
            "config_experiment_nis_reject.json": (True, False, False),
            "config_experiment_adaptive_r.json": (True, True, False),
            "config_experiment_adaptive_r_recovery.json": (True, True, True),
            "config_experiment_full_method.json": (True, True, True),
        }

        for filename, expected_policy in expected_policies.items():
            with self.subTest(filename=filename):
                config = load_config(PROJECT_ROOT / "01_data" / filename)

                self.assertEqual(config.config_metadata.profile, "experiment")
                self.assertEqual(
                    (
                        config.fusion_policy.use_nis_rejection,
                        config.fusion_policy.use_adaptive_r,
                        config.fusion_policy.use_recovery_scale,
                    ),
                    expected_policy,
                )
                self.assertNotEqual(Path(config.config_path).resolve(), DEFAULT_CONFIG_PATH.resolve())
                self.assertTrue(config.results_dir.startswith("03_results_experiment_"))

    def test_non_default_config_cannot_claim_default_general_profile(self) -> None:
        payload = json.loads((PROJECT_ROOT / "01_data" / "config_00000422_decoded.json").read_text(encoding="utf-8"))
        payload["config_metadata"]["profile"] = "default_general"

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "bad_profile.json"
            config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "不能使用 profile=default_general"):
                load_config(config_path)


if __name__ == "__main__":
    unittest.main()
