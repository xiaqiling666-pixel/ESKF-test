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

    def test_sample_config_uses_non_default_profile(self) -> None:
        config = load_config(PROJECT_ROOT / "01_data" / "config_00000422_decoded.json")

        self.assertEqual(config.config_metadata.profile, "sample_validation")
        self.assertNotEqual(Path(config.config_path).resolve(), DEFAULT_CONFIG_PATH.resolve())

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
