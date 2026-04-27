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

from eskf_stack.analysis.evaluator import (
    compute_metrics,
    metric_category,
    metric_experiment_comparison_direction,
    metric_supports_experiment_delta,
    save_metrics,
)


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
                "available_gnss_pos": [1, 1, 0, 1, 1],
                "available_gnss_vel": [0, 1, 1, 1, 1],
                "available_baro": [1, 1, 0, 1, 1],
                "available_mag": [1, 0, 1, 1, 1],
                "used_gnss_pos": [1, 1, 0, 0, 0],
                "used_gnss_vel": [0, 0, 1, 0, 1],
                "used_baro": [1, 0, 1, 1, 1],
                "used_mag": [1, 1, 0, 1, 1],
                "gnss_pos_reject_streak": [0, 0, 0, 0, 1],
                "gnss_vel_reject_streak": [0, 0, 0, 0, 0],
                "baro_reject_streak": [0, 1, 0, 0, 0],
                "mag_reject_streak": [0, 0, 1, 0, 0],
                "gnss_pos_adaptive_streak": [0, 1, 2, 0, 0],
                "gnss_vel_adaptive_streak": [0, 0, 0, 0, 1],
                "baro_adaptive_streak": [0, 1, 0, 2, 0],
                "mag_adaptive_streak": [0, 0, 1, 0, 0],
                "gnss_pos_outage_s": [0.0, 0.0, 1.0, 2.0, 3.0],
                "gnss_vel_outage_s": [float("inf"), 0.0, 0.0, 1.0, 0.0],
                "baro_outage_s": [0.0, 1.0, 2.0, 0.0, 0.0],
                "mag_outage_s": [0.0, 1.0, 0.0, 0.0, 0.0],
                "auxiliary_outage_s": [0.0, 0.0, 0.0, 0.0, 0.0],
                "gnss_pos_innovation_norm": [1.0, 2.0, float("nan"), 4.0, 5.0],
                "gnss_vel_innovation_norm": [float("nan"), 1.5, 2.5, 3.5, 4.5],
                "baro_innovation_abs": [0.2, 0.4, float("nan"), 0.8, 1.0],
                "yaw_innovation_abs_deg": [5.0, float("nan"), 9.0, 11.0, 13.0],
                "gnss_pos_mode_scale": [1.0, 2.0, 1.0, 1.0, 1.25],
                "gnss_vel_mode_scale": [1.0, 1.0, 1.5, 1.0, 1.15],
                "gnss_pos_r_scale": [1.0, 2.0, 1.0, 1.0, 1.25],
                "gnss_vel_r_scale": [1.0, 1.0, 1.5, 1.0, 1.15],
                "gnss_pos_adaptation_scale": [1.0, 1.8, 1.0, 1.0, 1.0],
                "gnss_vel_adaptation_scale": [1.0, 1.0, 1.0, 1.0, 1.6],
                "gnss_pos_rejected": [False, False, False, False, True],
                "gnss_vel_rejected": [False, False, False, False, False],
                "baro_adaptation_scale": [1.0, 2.0, 1.0, 1.4, 1.0],
                "mag_adaptation_scale": [1.0, 1.2, 2.0, 1.0, 1.0],
                "baro_rejected": [False, True, False, False, False],
                "mag_rejected": [False, False, True, False, False],
                "gnss_pos_nis": [1.0, 2.0, float("nan"), 4.0, 5.0],
                "gnss_vel_nis": [float("nan"), 1.5, 2.5, 3.5, 4.5],
                "baro_nis": [0.5, 1.5, float("nan"), 2.5, 3.5],
                "mag_nis": [2.0, float("nan"), 4.0, 6.0, 8.0],
                "gnss_pos_nis_adapt_threshold": [3.0, 3.0, 3.0, 3.0, 3.0],
                "gnss_pos_nis_reject_threshold": [4.5, 4.5, 4.5, 4.5, 4.5],
                "gnss_vel_nis_adapt_threshold": [3.0, 3.0, 3.0, 3.0, 3.0],
                "gnss_vel_nis_reject_threshold": [4.0, 4.0, 4.0, 4.0, 4.0],
                "baro_nis_adapt_threshold": [2.0, 2.0, 2.0, 2.0, 2.0],
                "baro_nis_reject_threshold": [3.0, 3.0, 3.0, 3.0, 3.0],
                "mag_nis_adapt_threshold": [5.0, 5.0, 5.0, 5.0, 5.0],
                "mag_nis_reject_threshold": [7.0, 7.0, 7.0, 7.0, 7.0],
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
        self.assertEqual(metrics["gnss_pos_mode_scaled_measurements"], 2.0)
        self.assertEqual(metrics["gnss_pos_mode_scaled_updates"], 1.0)
        self.assertEqual(metrics["gnss_pos_mode_scaled_rejections"], 1.0)
        self.assertEqual(metrics["gnss_pos_available_measurements"], 4.0)
        self.assertEqual(metrics["gnss_pos_updates"], 2.0)
        self.assertEqual(metrics["max_gnss_pos_reject_streak"], 1.0)
        self.assertEqual(metrics["max_gnss_pos_adaptive_streak"], 2.0)
        self.assertAlmostEqual(metrics["max_gnss_pos_outage_s"], 3.0, places=9)
        self.assertEqual(metrics["gnss_vel_available_measurements"], 4.0)
        self.assertEqual(metrics["gnss_vel_updates"], 2.0)
        self.assertEqual(metrics["max_gnss_vel_reject_streak"], 0.0)
        self.assertEqual(metrics["max_gnss_vel_adaptive_streak"], 1.0)
        self.assertTrue(metrics["max_gnss_vel_outage_s"] > 1e6)
        self.assertEqual(metrics["gnss_vel_mode_scaled_measurements"], 2.0)
        self.assertEqual(metrics["gnss_vel_mode_scaled_updates"], 2.0)
        self.assertEqual(metrics["gnss_vel_mode_scaled_rejections"], 0.0)
        self.assertEqual(metrics["gnss_pos_adapted_updates"], 1.0)
        self.assertEqual(metrics["gnss_vel_adapted_updates"], 1.0)
        self.assertEqual(metrics["baro_available_measurements"], 4.0)
        self.assertEqual(metrics["baro_updates"], 4.0)
        self.assertEqual(metrics["max_baro_reject_streak"], 1.0)
        self.assertEqual(metrics["max_baro_adaptive_streak"], 2.0)
        self.assertAlmostEqual(metrics["max_baro_outage_s"], 2.0, places=9)
        self.assertEqual(metrics["mag_available_measurements"], 4.0)
        self.assertEqual(metrics["mag_updates"], 4.0)
        self.assertEqual(metrics["max_mag_reject_streak"], 1.0)
        self.assertEqual(metrics["max_mag_adaptive_streak"], 1.0)
        self.assertAlmostEqual(metrics["max_mag_outage_s"], 1.0, places=9)
        self.assertAlmostEqual(metrics["max_auxiliary_outage_s"], 0.0, places=9)
        self.assertEqual(metrics["baro_rejections"], 1.0)
        self.assertEqual(metrics["mag_rejections"], 1.0)
        self.assertEqual(metrics["baro_adapted_updates"], 1.0)
        self.assertEqual(metrics["mag_adapted_updates"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_gnss_pos_nis"], 3.0, places=9)
        self.assertAlmostEqual(metrics["max_gnss_pos_nis"], 5.0, places=9)
        self.assertEqual(metrics["gnss_pos_innovation_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_gnss_pos_innovation"], 3.0, places=9)
        self.assertAlmostEqual(metrics["max_gnss_pos_innovation"], 5.0, places=9)
        self.assertEqual(metrics["gnss_pos_nis_adapt_exceed_count"], 2.0)
        self.assertEqual(metrics["gnss_pos_nis_adapt_exceed_used_count"], 0.0)
        self.assertEqual(metrics["gnss_pos_nis_adapt_exceed_rejected_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_adapt_exceed_not_rejected_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_used_count"], 0.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_rejected_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_not_rejected_count"], 0.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_policy_bypass_count"], 0.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_policy_bypass_pct"], 0.0)
        self.assertEqual(metrics["gnss_vel_nis_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_gnss_vel_nis"], 3.0, places=9)
        self.assertAlmostEqual(metrics["max_gnss_vel_nis"], 4.5, places=9)
        self.assertEqual(metrics["gnss_vel_innovation_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_gnss_vel_innovation"], 3.0, places=9)
        self.assertAlmostEqual(metrics["max_gnss_vel_innovation"], 4.5, places=9)
        self.assertEqual(metrics["gnss_vel_nis_adapt_exceed_count"], 2.0)
        self.assertEqual(metrics["gnss_vel_nis_adapt_exceed_used_count"], 1.0)
        self.assertEqual(metrics["gnss_vel_nis_adapt_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["gnss_vel_nis_adapt_exceed_not_rejected_count"], 2.0)
        self.assertEqual(metrics["gnss_vel_nis_reject_exceed_count"], 1.0)
        self.assertEqual(metrics["gnss_vel_nis_reject_exceed_used_count"], 1.0)
        self.assertEqual(metrics["gnss_vel_nis_reject_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["gnss_vel_nis_reject_exceed_not_rejected_count"], 1.0)
        self.assertEqual(metrics["gnss_vel_nis_reject_policy_bypass_count"], 1.0)
        self.assertEqual(metrics["gnss_vel_nis_reject_policy_bypass_pct"], 100.0)
        self.assertEqual(metrics["baro_nis_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_baro_nis"], 2.0, places=9)
        self.assertAlmostEqual(metrics["max_baro_nis"], 3.5, places=9)
        self.assertEqual(metrics["baro_innovation_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_baro_innovation"], 0.6, places=9)
        self.assertAlmostEqual(metrics["max_baro_innovation"], 1.0, places=9)
        self.assertEqual(metrics["baro_nis_adapt_exceed_count"], 2.0)
        self.assertEqual(metrics["baro_nis_adapt_exceed_used_count"], 2.0)
        self.assertEqual(metrics["baro_nis_adapt_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["baro_nis_adapt_exceed_not_rejected_count"], 2.0)
        self.assertEqual(metrics["baro_nis_reject_exceed_count"], 1.0)
        self.assertEqual(metrics["baro_nis_reject_exceed_used_count"], 1.0)
        self.assertEqual(metrics["baro_nis_reject_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["baro_nis_reject_exceed_not_rejected_count"], 1.0)
        self.assertEqual(metrics["baro_nis_reject_policy_bypass_count"], 1.0)
        self.assertEqual(metrics["baro_nis_reject_policy_bypass_pct"], 100.0)
        self.assertEqual(metrics["mag_nis_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_mag_nis"], 5.0, places=9)
        self.assertAlmostEqual(metrics["max_mag_nis"], 8.0, places=9)
        self.assertEqual(metrics["mag_innovation_valid_count"], 4.0)
        self.assertAlmostEqual(metrics["mean_mag_innovation"], 9.5, places=9)
        self.assertAlmostEqual(metrics["max_mag_innovation"], 13.0, places=9)
        self.assertEqual(metrics["mag_nis_adapt_exceed_count"], 2.0)
        self.assertEqual(metrics["mag_nis_adapt_exceed_used_count"], 2.0)
        self.assertEqual(metrics["mag_nis_adapt_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["mag_nis_adapt_exceed_not_rejected_count"], 2.0)
        self.assertEqual(metrics["mag_nis_reject_exceed_count"], 1.0)
        self.assertEqual(metrics["mag_nis_reject_exceed_used_count"], 1.0)
        self.assertEqual(metrics["mag_nis_reject_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["mag_nis_reject_exceed_not_rejected_count"], 1.0)
        self.assertEqual(metrics["mag_nis_reject_policy_bypass_count"], 1.0)
        self.assertEqual(metrics["mag_nis_reject_policy_bypass_pct"], 100.0)

    def test_reject_policy_bypass_metrics_capture_disabled_rejection_case(self) -> None:
        result_df = pd.DataFrame(
            {
                "time": [0.0, 1.0],
                "quality_score": [80.0, 82.0],
                "used_gnss_pos": [1, 1],
                "used_baro": [0, 0],
                "used_mag": [0, 0],
                "gnss_pos_reject_bypassed": [True, False],
                "gnss_pos_rejected": [False, False],
                "gnss_pos_nis": [5.5, 2.0],
                "gnss_pos_nis_reject_threshold": [4.5, 4.5],
            }
        )

        metrics = compute_metrics(result_df)

        self.assertEqual(metrics["gnss_pos_reject_bypassed_updates"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_rejected_count"], 0.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_exceed_not_rejected_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_policy_bypass_count"], 1.0)
        self.assertEqual(metrics["gnss_pos_nis_reject_policy_bypass_pct"], 100.0)

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

    def test_initialization_transition_row_metrics(self) -> None:
        result_df = pd.DataFrame(
            {
                "time": [0.0, 0.2, 0.22],
                "quality_score": [float("nan"), 85.0, 86.0],
                "used_gnss_pos": [0, 1, 1],
                "used_gnss_vel": [0, 0, 0],
                "used_baro": [0, 0, 0],
                "used_mag": [0, 0, 0],
                "initialization_phase": ["WAITING_BOOTSTRAP_MOTION", "INITIALIZED", "INITIALIZED"],
                "initialization_reason": [
                    "awaiting_bootstrap_anchor",
                    "bootstrap_init_completed",
                    "bootstrap_init_completed",
                ],
                "initialization_heading_source": ["none", "position_pair_course", "position_pair_course"],
                "initialization_wait_s": [0.0, 0.2, 0.2],
                "initialization_completed_this_frame": [False, True, False],
            }
        )

        metrics = compute_metrics(result_df)

        self.assertEqual(metrics["initialization_row_count"], 3.0)
        self.assertEqual(metrics["initialization_pending_row_count"], 1.0)
        self.assertEqual(metrics["initialization_completed_row_count"], 1.0)
        self.assertAlmostEqual(metrics["initialization_phase_duration_WAITING_BOOTSTRAP_MOTION_s"], 0.2, places=9)
        self.assertAlmostEqual(metrics["initialization_phase_duration_INITIALIZED_s"], 0.13, places=9)
        self.assertAlmostEqual(metrics["max_initialization_wait_s"], 0.2, places=9)

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

    def test_save_metrics_writes_metric_categories(self) -> None:
        metrics = {
            "input_quality_row_count": 2.0,
            "initialization_completed_flag": 1.0,
            "position_rmse_m": 0.5,
            "gnss_pos_rejections": 1.0,
            "baro_adapted_updates": 1.0,
            "baro_nis_valid_count": 4.0,
            "mean_baro_nis": 2.0,
            "max_baro_nis": 3.5,
            "baro_nis_adapt_exceed_count": 2.0,
            "baro_nis_adapt_exceed_used_count": 2.0,
            "baro_nis_adapt_exceed_rejected_count": 0.0,
            "mag_rejections": 1.0,
            "mag_nis_valid_count": 4.0,
            "mean_mag_nis": 5.0,
            "max_mag_nis": 8.0,
            "mag_nis_reject_exceed_count": 1.0,
            "mag_nis_reject_exceed_rejected_count": 0.0,
            "mag_nis_reject_exceed_not_rejected_count": 1.0,
            "max_auxiliary_outage_s": 0.0,
            "max_baro_adaptive_streak": 2.0,
            "predict_warning_count": 0.0,
            "covariance_unhealthy_row_count": 0.0,
            "mode_transition_count": 2.0,
            "pipeline_runtime_s": 0.01,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            save_metrics(metrics, output_dir)
            metrics_frame = pd.read_csv(output_dir / "metrics.csv")

        self.assertEqual(list(metrics_frame.columns), ["metric", "value", "category"])
        categories = dict(zip(metrics_frame["metric"], metrics_frame["category"], strict=True))
        self.assertEqual(categories["input_quality_row_count"], "input_quality")
        self.assertEqual(categories["initialization_completed_flag"], "initialization")
        self.assertEqual(categories["position_rmse_m"], "estimation_error")
        self.assertEqual(categories["gnss_pos_rejections"], "measurement_management")
        self.assertEqual(categories["baro_adapted_updates"], "measurement_management")
        self.assertEqual(categories["baro_nis_valid_count"], "measurement_management")
        self.assertEqual(categories["mean_baro_nis"], "measurement_management")
        self.assertEqual(categories["max_baro_nis"], "measurement_management")
        self.assertEqual(categories["baro_nis_adapt_exceed_count"], "measurement_management")
        self.assertEqual(categories["baro_nis_adapt_exceed_used_count"], "measurement_management")
        self.assertEqual(categories["baro_nis_adapt_exceed_rejected_count"], "measurement_management")
        self.assertEqual(categories["mag_rejections"], "measurement_management")
        self.assertEqual(categories["mag_nis_valid_count"], "measurement_management")
        self.assertEqual(categories["mean_mag_nis"], "measurement_management")
        self.assertEqual(categories["max_mag_nis"], "measurement_management")
        self.assertEqual(categories["mag_nis_reject_exceed_count"], "measurement_management")
        self.assertEqual(categories["mag_nis_reject_exceed_rejected_count"], "measurement_management")
        self.assertEqual(categories["mag_nis_reject_exceed_not_rejected_count"], "measurement_management")
        self.assertEqual(categories["max_auxiliary_outage_s"], "measurement_management")
        self.assertEqual(categories["max_baro_adaptive_streak"], "measurement_management")
        self.assertEqual(categories["predict_warning_count"], "prediction_diagnostics")
        self.assertEqual(categories["covariance_unhealthy_row_count"], "covariance_health")
        self.assertEqual(categories["mode_transition_count"], "mode_state")
        self.assertEqual(categories["pipeline_runtime_s"], "runtime")

    def test_metric_category_defaults_unknown_metrics_to_other(self) -> None:
        self.assertEqual(metric_category("custom_future_metric"), "other")

    def test_metric_supports_experiment_delta_matches_supported_categories(self) -> None:
        self.assertTrue(metric_supports_experiment_delta("position_rmse_m"))
        self.assertTrue(metric_supports_experiment_delta("gnss_pos_rejections"))
        self.assertTrue(metric_supports_experiment_delta("initialization_completed_flag"))
        self.assertFalse(metric_supports_experiment_delta("custom_future_metric"))

    def test_metric_experiment_comparison_direction_reports_expected_preferences(self) -> None:
        self.assertEqual(metric_experiment_comparison_direction("position_rmse_m"), "lower_better")
        self.assertEqual(metric_experiment_comparison_direction("mean_quality_score"), "higher_better")
        self.assertEqual(metric_experiment_comparison_direction("initialization_completed_flag"), "higher_better")
        self.assertIsNone(metric_experiment_comparison_direction("gnss_pos_updates"))


if __name__ == "__main__":
    unittest.main()
