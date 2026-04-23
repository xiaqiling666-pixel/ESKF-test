from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.analysis.evaluator import compute_metrics


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

    def test_pending_duration_counts_interval_after_pending_sample(self) -> None:
        result_df = pd.DataFrame(
            {
                "time": [0.0, 0.4, 1.0, 1.5],
                "quality_score": [80.0, 80.0, 80.0, 80.0],
                "used_gnss_pos": [1, 1, 1, 1],
                "used_baro": [1, 1, 1, 1],
                "used_mag": [1, 1, 1, 1],
                "mode_transition_pending": [True, False, True, False],
            }
        )

        metrics = compute_metrics(result_df)

        self.assertEqual(metrics["pending_row_count"], 2.0)
        self.assertAlmostEqual(metrics["pending_duration_s"], 0.9, places=9)


if __name__ == "__main__":
    unittest.main()
