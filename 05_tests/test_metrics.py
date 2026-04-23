from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.analysis.evaluator import compute_metrics, save_metrics


class MetricsTests(unittest.TestCase):
    def test_mode_duration_and_entry_metrics(self) -> None:
        result_df = pd.DataFrame(
            {
                "time": [0.0, 1.0, 2.0, 3.0, 4.0],
                "mode": ["GNSS_STABLE", "GNSS_STABLE", "INERTIAL_HOLD", "INERTIAL_HOLD", "GNSS_STABLE"],
                "mode_reason": [
                    "fresh_gnss_and_healthy_covariance",
                    "fresh_gnss_and_healthy_covariance",
                    "short_gnss_outage",
                    "short_gnss_outage",
                    "restoring_gnss_stability",
                ],
                "quality_score": [90.0, 90.0, 50.0, 50.0, 88.0],
                "used_gnss_pos": [1, 1, 0, 0, 1],
                "used_baro": [1, 1, 1, 1, 1],
                "used_mag": [1, 1, 1, 1, 1],
                "covariance_health": ["HEALTHY", "HEALTHY", "CAUTION", "UNHEALTHY", "HEALTHY"],
                "covariance_health_reason": [
                    "nominal",
                    "nominal",
                    "position_sigma_caution",
                    "multiple_unhealthy_sigmas",
                    "nominal",
                ],
                "covariance_caution": [False, False, True, True, False],
                "covariance_unhealthy": [False, False, False, True, False],
                "covariance_caution_duration_s": [0.0, 0.0, 0.2, 1.2, 0.0],
                "covariance_unhealthy_duration_s": [0.0, 0.0, 0.0, 0.6, 0.0],
                "mode_transition_pending": [False, False, True, False, False],
                "predict_skipped": [False, False, True, False, False],
                "predict_warning": [False, False, True, True, False],
                "dt_raw_s": [0.02, 0.02, 0.5, 0.02, 0.02],
                "dt_applied_s": [0.02, 0.02, 0.0, 0.02, 0.02],
                "predict_reason": [
                    "applied",
                    "applied",
                    "above_max_dt_skipped",
                    "applied",
                    "applied",
                ],
            }
        )

        metrics = compute_metrics(result_df)

        self.assertAlmostEqual(metrics["mode_duration_GNSS_STABLE_s"], 3.0, places=9)
        self.assertAlmostEqual(metrics["mode_duration_INERTIAL_HOLD_s"], 2.0, places=9)
        self.assertAlmostEqual(metrics["mode_share_GNSS_STABLE_pct"], 60.0, places=9)
        self.assertAlmostEqual(metrics["mode_share_INERTIAL_HOLD_pct"], 40.0, places=9)
        self.assertEqual(metrics["mode_entry_count_GNSS_STABLE"], 2.0)
        self.assertEqual(metrics["mode_entry_count_INERTIAL_HOLD"], 1.0)
        self.assertEqual(metrics["mode_transition_count"], 2.0)
        self.assertAlmostEqual(metrics["mode_reason_duration_fresh_gnss_and_healthy_covariance_s"], 2.0, places=9)
        self.assertAlmostEqual(metrics["mode_reason_duration_short_gnss_outage_s"], 2.0, places=9)
        self.assertAlmostEqual(metrics["mode_reason_duration_restoring_gnss_stability_s"], 1.0, places=9)
        self.assertEqual(metrics["mode_reason_entry_count_fresh_gnss_and_healthy_covariance"], 1.0)
        self.assertEqual(metrics["mode_reason_entry_count_short_gnss_outage"], 1.0)
        self.assertEqual(metrics["mode_reason_entry_count_restoring_gnss_stability"], 1.0)
        self.assertEqual(metrics["mode_reason_transition_count"], 2.0)
        self.assertAlmostEqual(metrics["covariance_health_duration_HEALTHY_s"], 3.0, places=9)
        self.assertAlmostEqual(metrics["covariance_health_duration_CAUTION_s"], 1.0, places=9)
        self.assertAlmostEqual(metrics["covariance_health_duration_UNHEALTHY_s"], 1.0, places=9)
        self.assertEqual(metrics["covariance_health_transition_count"], 3.0)
        self.assertAlmostEqual(metrics["covariance_health_reason_duration_nominal_s"], 3.0, places=9)
        self.assertAlmostEqual(metrics["covariance_health_reason_duration_position_sigma_caution_s"], 1.0, places=9)
        self.assertAlmostEqual(metrics["covariance_health_reason_duration_multiple_unhealthy_sigmas_s"], 1.0, places=9)
        self.assertEqual(metrics["covariance_caution_row_count"], 2.0)
        self.assertEqual(metrics["covariance_unhealthy_row_count"], 1.0)
        self.assertAlmostEqual(metrics["max_covariance_caution_duration_s"], 1.2, places=9)
        self.assertAlmostEqual(metrics["max_covariance_unhealthy_duration_s"], 0.6, places=9)
        self.assertEqual(metrics["pending_row_count"], 1.0)
        self.assertAlmostEqual(metrics["pending_duration_s"], 1.0, places=9)
        self.assertEqual(metrics["predict_skipped_count"], 1.0)
        self.assertEqual(metrics["predict_warning_count"], 2.0)
        self.assertAlmostEqual(metrics["max_dt_raw_s"], 0.5, places=9)
        self.assertAlmostEqual(metrics["max_dt_applied_s"], 0.02, places=9)
        self.assertAlmostEqual(metrics["predict_reason_duration_applied_s"], 4.0, places=9)
        self.assertAlmostEqual(metrics["predict_reason_duration_above_max_dt_skipped_s"], 1.0, places=9)

    def test_initialization_summary_metrics(self) -> None:
        result_df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "quality_score": [90.0, 92.0],
                "used_gnss_pos": [1, 1],
                "used_baro": [0, 0],
                "used_mag": [0, 0],
            }
        )

        initialization_summary = {
            "initialization_mode": "direct_static_coarse_alignment",
            "initialization_phase": "INITIALIZED",
            "initialization_ready_mode": "direct",
            "heading_source": "zero_yaw_fallback",
            "static_coarse_alignment_used": "true",
            "static_alignment_ready": "true",
            "initialization_wait_s": "1.500000",
        }

        metrics = compute_metrics(result_df, initialization_summary=initialization_summary)

        self.assertEqual(metrics["initialization_completed_flag"], 1.0)
        self.assertEqual(metrics["initialization_mode_direct_flag"], 1.0)
        self.assertEqual(metrics["initialization_mode_bootstrap_position_pair_flag"], 0.0)
        self.assertEqual(metrics["initialization_static_coarse_alignment_used_flag"], 1.0)
        self.assertEqual(metrics["initialization_static_alignment_ready_flag"], 1.0)
        self.assertEqual(metrics["initialization_zero_yaw_fallback_used_flag"], 1.0)
        self.assertAlmostEqual(metrics["initialization_wait_s"], 1.5, places=9)

    def test_save_metrics_writes_human_readable_initialization_summary(self) -> None:
        metrics = {
            "initialization_mode_direct_flag": 1.0,
            "initialization_zero_yaw_fallback_used_flag": 1.0,
            "initialization_wait_s": 1.5,
        }
        initialization_summary = {
            "initialization_phase": "INITIALIZED",
            "initialization_reason": "direct_init_completed",
            "initialization_ready_mode": "direct",
            "heading_source": "zero_yaw_fallback",
            "static_coarse_alignment_used": "false",
            "static_alignment_ready": "true",
            "static_alignment_reason": "static_alignment_ready",
            "initialization_wait_s": "1.500000",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_metrics(metrics, output_dir, initialization_summary=initialization_summary)
            summary_text = (output_dir / "metrics_summary.txt").read_text(encoding="utf-8")

        self.assertIn("Initialization Summary", summary_text)
        self.assertIn("phase: INITIALIZED", summary_text)
        self.assertIn("mode: direct", summary_text)
        self.assertIn("reason: direct_init_completed", summary_text)
        self.assertIn("heading_source: zero_yaw_fallback", summary_text)
        self.assertIn("static_coarse_alignment_used: no", summary_text)
        self.assertIn("static_alignment_ready: yes", summary_text)
        self.assertIn("zero_yaw_fallback_used: yes", summary_text)
        self.assertIn("initialization_wait_s: 1.500000", summary_text)
        self.assertIn("Metric Values", summary_text)


if __name__ == "__main__":
    unittest.main()
