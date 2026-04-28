from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "02_src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from eskf_stack.analysis.quality import (
    CovarianceHealthTracker,
    SensorFreshnessTracker,
    classify_covariance_health,
    compute_quality_score,
    summarize_measurement_support,
)
from eskf_stack.analysis.state_machine import ModeDecision, ModeStateTracker, ModeThresholds, determine_mode


class HealthStateTests(unittest.TestCase):
    @staticmethod
    def _covariance_health(
        current_time: float,
        pos_sigma_norm_m: float,
        vel_sigma_norm_mps: float,
        att_sigma_norm_deg: float,
        tracker: CovarianceHealthTracker | None = None,
    ):
        tracker = tracker or CovarianceHealthTracker()
        health = classify_covariance_health(
            pos_sigma_norm_m=pos_sigma_norm_m,
            vel_sigma_norm_mps=vel_sigma_norm_mps,
            att_sigma_norm_deg=att_sigma_norm_deg,
        )
        return tracker.step(current_time, health)

    def test_tracker_records_reject_streak_and_outage(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 0.1, available=True, used=True, rejected=False, adaptation_scale=1.2)
        tracker.note_result("mag", 0.2, available=True, used=False, rejected=True)
        tracker.note_result("mag", 0.3, available=True, used=False, rejected=True)
        tracker.note_result("gnss_pos", 0.2, available=True, used=True, rejected=False, adaptation_scale=1.4)
        tracker.note_result("gnss_pos", 0.4, available=True, used=True, rejected=False, adaptation_scale=1.5)
        tracker.note_result("gnss_pos", 0.6, available=True, used=False, rejected=True)

        snapshot = tracker.snapshot(1.0)

        self.assertEqual(snapshot.gnss_pos_reject_streak, 1)
        self.assertEqual(snapshot.gnss_vel_reject_streak, 0)
        self.assertEqual(snapshot.baro_reject_streak, 0)
        self.assertEqual(snapshot.mag_reject_streak, 2)
        self.assertEqual(snapshot.gnss_pos_adaptive_streak, 0)
        self.assertEqual(snapshot.gnss_vel_adaptive_streak, 0)
        self.assertEqual(snapshot.baro_adaptive_streak, 1)
        self.assertEqual(snapshot.mag_adaptive_streak, 0)
        self.assertAlmostEqual(snapshot.gnss_pos_outage_s, 0.6, places=9)
        self.assertAlmostEqual(snapshot.gnss_vel_outage_s, 1.0, places=9)
        self.assertAlmostEqual(snapshot.gnss_outage_s, 0.6, places=9)
        self.assertAlmostEqual(snapshot.baro_outage_s, 0.9, places=9)
        self.assertTrue(snapshot.mag_outage_s > 1e6)
        self.assertAlmostEqual(snapshot.auxiliary_outage_s, 0.9, places=9)

    def test_tracker_skip_clears_reject_streak_without_counting_as_update(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_pos", 0.2, available=True, used=False, rejected=True)
        tracker.note_result("gnss_pos", 0.3, available=True, used=False, rejected=False, management_mode="skip")

        snapshot = tracker.snapshot(0.61)

        self.assertEqual(snapshot.gnss_pos_reject_streak, 0)
        self.assertFalse(snapshot.recent_gnss_pos)
        self.assertTrue(snapshot.recent_available_gnss_pos)
        self.assertEqual(snapshot.gnss_pos_skip_streak, 1)
        self.assertAlmostEqual(snapshot.gnss_pos_outage_s, 0.61, places=9)
        self.assertAlmostEqual(snapshot.gnss_pos_available_outage_s, 0.31, places=9)

    def test_tracker_records_auxiliary_availability_without_successful_updates(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("baro", 1.1, available=True, used=False, rejected=False, management_mode="skip")

        snapshot = tracker.snapshot(1.2)

        self.assertFalse(snapshot.recent_baro)
        self.assertTrue(snapshot.recent_available_baro)
        self.assertEqual(snapshot.baro_skip_streak, 2)
        self.assertTrue(snapshot.baro_outage_s > 1.0)
        self.assertAlmostEqual(snapshot.baro_available_outage_s, 0.1, places=9)
        self.assertAlmostEqual(snapshot.auxiliary_available_outage_s, 0.1, places=9)

    def test_support_summary_keeps_single_stable_auxiliary_backup(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("baro", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("baro", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("mag", 1.0, available=True, used=True, rejected=False, management_mode="recover")
        tracker.note_result("mag", 1.1, available=True, used=True, rejected=False, management_mode="recover")
        tracker.note_result("mag", 1.2, available=True, used=True, rejected=False, management_mode="recover")
        support = summarize_measurement_support(tracker.snapshot(1.3))

        self.assertTrue(support.stable_auxiliary_support)
        self.assertFalse(support.auxiliary_instability_without_backup)
        self.assertTrue(support.baro_hard_unstable)
        self.assertFalse(support.mag_hard_unstable)

    def test_support_summary_marks_gnss_available_without_successful_updates(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_pos", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_vel", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_pos", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_vel", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        support = summarize_measurement_support(tracker.snapshot(1.2))

        self.assertTrue(support.recent_any_gnss_available)
        self.assertFalse(support.recent_any_gnss)
        self.assertTrue(support.gnss_available_without_successful_updates)

    def test_support_summary_respects_custom_gnss_skip_threshold(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_pos", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_vel", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_pos", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_vel", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        support = summarize_measurement_support(tracker.snapshot(1.2), gnss_skip_streak_degraded=3)

        self.assertFalse(support.gnss_available_without_successful_updates)

    def test_state_machine_marks_gnss_degraded_on_rejections(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_pos", 0.2, available=True, used=False, rejected=True)
        tracker.note_result("gnss_pos", 0.3, available=True, used=False, rejected=True)
        tracker.note_result("gnss_pos", 0.4, available=True, used=False, rejected=True)
        snapshot = tracker.snapshot(0.5)

        decision = determine_mode(
            snapshot,
            quality_score=85.0,
            covariance_health=self._covariance_health(0.5, 0.5, 0.2, 2.0),
        )

        self.assertEqual(decision.mode, "GNSS_DEGRADED")
        self.assertEqual(decision.reason, "gnss_rejection_streak")

    def test_state_machine_uses_policy_aligned_custom_streak_thresholds(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_pos", 0.2, available=True, used=False, rejected=True)
        tracker.note_result("gnss_pos", 0.3, available=True, used=False, rejected=True)
        snapshot = tracker.snapshot(0.35)

        decision = determine_mode(
            snapshot,
            quality_score=85.0,
            covariance_health=self._covariance_health(0.35, 0.5, 0.2, 2.0),
            thresholds=ModeThresholds(
                gnss_reject_streak_degraded=2,
                gnss_adaptive_streak_degraded=2,
                auxiliary_reject_streak_degraded=2,
                auxiliary_adaptive_streak_degraded=2,
            ),
        )

        self.assertEqual(decision.mode, "GNSS_DEGRADED")
        self.assertEqual(decision.reason, "gnss_rejection_streak")

    def test_state_machine_marks_gnss_degraded_on_adaptive_scaling_streak(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False, adaptation_scale=1.3)
        tracker.note_result("gnss_pos", 0.1, available=True, used=True, rejected=False, adaptation_scale=1.4)
        tracker.note_result("gnss_pos", 0.2, available=True, used=True, rejected=False, adaptation_scale=1.5)
        tracker.note_result("gnss_vel", 0.2, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(0.25)

        decision = determine_mode(
            snapshot,
            quality_score=70.0,
            covariance_health=self._covariance_health(0.25, 0.5, 0.2, 2.0),
        )

        self.assertEqual(decision.mode, "GNSS_DEGRADED")
        self.assertEqual(decision.reason, "gnss_adaptive_scaling_streak")

    def test_state_machine_marks_gnss_degraded_on_reject_bypass_streak(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False, reject_bypassed=True)
        tracker.note_result("gnss_pos", 0.1, available=True, used=True, rejected=False, reject_bypassed=True)
        tracker.note_result("gnss_vel", 0.1, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(0.15)

        decision = determine_mode(
            snapshot,
            quality_score=72.0,
            covariance_health=self._covariance_health(0.15, 0.5, 0.2, 2.0),
        )

        self.assertEqual(decision.mode, "GNSS_DEGRADED")
        self.assertEqual(decision.reason, "gnss_reject_bypass_streak")

    def test_quality_score_does_not_penalize_missing_auxiliary_when_full_gnss_is_recent(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(0.2)

        score = compute_quality_score(
            snapshot,
            pos_innovation_norm=0.2,
            vel_innovation_norm=0.1,
            baro_innovation_abs=0.0,
            yaw_innovation_abs_deg=0.0,
            pos_sigma_norm_m=0.5,
            vel_sigma_norm_mps=0.2,
            att_sigma_norm_deg=2.0,
        )

        self.assertGreaterEqual(score, 75.0)

    def test_quality_score_preserves_auxiliary_backup_when_other_aux_source_degrades(self) -> None:
        tracker_with_backup = SensorFreshnessTracker()
        tracker_with_backup.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker_with_backup.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker_with_backup.note_result("baro", 1.0, available=True, used=False, rejected=True)
        tracker_with_backup.note_result("baro", 1.1, available=True, used=False, rejected=True)
        tracker_with_backup.note_result("mag", 1.0, available=True, used=True, rejected=False, management_mode="recover")
        tracker_with_backup.note_result("mag", 1.1, available=True, used=True, rejected=False, management_mode="recover")
        tracker_with_backup.note_result("mag", 1.2, available=True, used=True, rejected=False, management_mode="recover")
        snapshot_with_backup = tracker_with_backup.snapshot(1.3)

        tracker_without_backup = SensorFreshnessTracker()
        tracker_without_backup.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker_without_backup.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker_without_backup.note_result("baro", 1.0, available=True, used=False, rejected=True)
        tracker_without_backup.note_result("baro", 1.1, available=True, used=False, rejected=True)
        snapshot_without_backup = tracker_without_backup.snapshot(1.3)

        score_with_backup = compute_quality_score(
            snapshot_with_backup,
            pos_innovation_norm=0.2,
            vel_innovation_norm=0.1,
            baro_innovation_abs=0.0,
            yaw_innovation_abs_deg=1.0,
            pos_sigma_norm_m=0.8,
            vel_sigma_norm_mps=0.3,
            att_sigma_norm_deg=3.0,
        )
        score_without_backup = compute_quality_score(
            snapshot_without_backup,
            pos_innovation_norm=0.2,
            vel_innovation_norm=0.1,
            baro_innovation_abs=0.0,
            yaw_innovation_abs_deg=1.0,
            pos_sigma_norm_m=0.8,
            vel_sigma_norm_mps=0.3,
            att_sigma_norm_deg=3.0,
        )

        self.assertGreater(score_with_backup, score_without_backup)

    def test_quality_score_penalizes_gnss_available_without_successful_updates(self) -> None:
        tracker_with_skips = SensorFreshnessTracker()
        tracker_with_skips.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker_with_skips.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker_with_skips.note_result("gnss_pos", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker_with_skips.note_result("gnss_vel", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker_with_skips.note_result("gnss_pos", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        tracker_with_skips.note_result("gnss_vel", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        score_with_skips = compute_quality_score(
            tracker_with_skips.snapshot(1.2),
            pos_innovation_norm=0.2,
            vel_innovation_norm=0.1,
            baro_innovation_abs=0.0,
            yaw_innovation_abs_deg=1.0,
            pos_sigma_norm_m=0.8,
            vel_sigma_norm_mps=0.3,
            att_sigma_norm_deg=3.0,
        )

        tracker_without_skips = SensorFreshnessTracker()
        tracker_without_skips.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker_without_skips.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        score_without_skips = compute_quality_score(
            tracker_without_skips.snapshot(1.2),
            pos_innovation_norm=0.2,
            vel_innovation_norm=0.1,
            baro_innovation_abs=0.0,
            yaw_innovation_abs_deg=1.0,
            pos_sigma_norm_m=0.8,
            vel_sigma_norm_mps=0.3,
            att_sigma_norm_deg=3.0,
        )

        self.assertLess(score_with_skips, score_without_skips)

    def test_state_machine_marks_inertial_hold_on_short_outage(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 1.0, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(1.2)

        decision = determine_mode(
            snapshot,
            quality_score=55.0,
            covariance_health=self._covariance_health(1.2, 0.8, 0.3, 3.0),
        )

        self.assertEqual(decision.mode, "INERTIAL_HOLD")
        self.assertEqual(decision.reason, "short_gnss_outage")

    def test_state_machine_degrades_short_gnss_outage_without_auxiliary_support(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(1.2)

        decision = determine_mode(
            snapshot,
            quality_score=55.0,
            covariance_health=self._covariance_health(1.2, 0.8, 0.3, 3.0),
        )

        self.assertEqual(decision.mode, "DEGRADED")
        self.assertEqual(decision.reason, "gnss_outage_without_aux_support")

    def test_state_machine_distinguishes_available_but_unsuccessful_auxiliary_support(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("baro", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        snapshot = tracker.snapshot(1.2)

        decision = determine_mode(
            snapshot,
            quality_score=55.0,
            covariance_health=self._covariance_health(1.2, 0.8, 0.3, 3.0),
        )

        self.assertEqual(decision.mode, "DEGRADED")
        self.assertEqual(decision.reason, "auxiliary_available_without_successful_updates")

    def test_state_machine_marks_gnss_available_without_successful_updates(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 1.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_pos", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_vel", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_pos", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("gnss_vel", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        snapshot = tracker.snapshot(1.2)

        decision = determine_mode(
            snapshot,
            quality_score=55.0,
            covariance_health=self._covariance_health(1.2, 0.8, 0.3, 3.0),
        )

        self.assertEqual(decision.mode, "DEGRADED")
        self.assertEqual(decision.reason, "gnss_available_without_successful_updates")

    def test_state_machine_degrades_partial_gnss_without_auxiliary_support(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.5, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(0.8)

        decision = determine_mode(
            snapshot,
            quality_score=70.0,
            covariance_health=self._covariance_health(0.8, 0.5, 0.2, 2.0),
        )

        self.assertEqual(decision.mode, "GNSS_DEGRADED")
        self.assertEqual(decision.reason, "partial_gnss_without_aux_support")

    def test_state_machine_keeps_partial_gnss_available_with_single_adaptive_auxiliary_support(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 0.0, available=True, used=True, rejected=False, adaptation_scale=1.2)
        tracker.note_result("baro", 0.1, available=True, used=True, rejected=False, adaptation_scale=1.3)
        tracker.note_result("baro", 0.2, available=True, used=True, rejected=False, adaptation_scale=1.4)
        snapshot = tracker.snapshot(0.3)

        decision = determine_mode(
            snapshot,
            quality_score=50.0,
            covariance_health=self._covariance_health(0.3, 0.5, 0.2, 2.0),
        )

        self.assertEqual(decision.mode, "GNSS_AVAILABLE")
        self.assertEqual(decision.reason, "partial_or_recovering_gnss")

    def test_state_machine_keeps_auxiliary_backup_when_one_aux_source_remains_stable(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 1.0, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("baro", 1.1, available=True, used=False, rejected=False, management_mode="skip")
        tracker.note_result("mag", 1.0, available=True, used=True, rejected=False)
        tracker.note_result("mag", 1.1, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(1.2)

        decision = determine_mode(
            snapshot,
            quality_score=50.0,
            covariance_health=self._covariance_health(1.2, 0.8, 0.3, 3.0),
        )

        self.assertEqual(decision.mode, "INERTIAL_HOLD")
        self.assertEqual(decision.reason, "short_gnss_outage")

    def test_state_machine_treats_recovering_auxiliary_as_valid_backup_against_single_source_failure(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("baro", 1.0, available=True, used=False, rejected=True)
        tracker.note_result("baro", 1.1, available=True, used=False, rejected=True)
        tracker.note_result("mag", 1.0, available=True, used=True, rejected=False, management_mode="recover")
        tracker.note_result("mag", 1.1, available=True, used=True, rejected=False, management_mode="recover")
        tracker.note_result("mag", 1.2, available=True, used=True, rejected=False, management_mode="recover")
        snapshot = tracker.snapshot(1.3)

        decision = determine_mode(
            snapshot,
            quality_score=50.0,
            covariance_health=self._covariance_health(1.3, 0.8, 0.3, 3.0),
        )

        self.assertEqual(decision.mode, "INERTIAL_HOLD")
        self.assertEqual(decision.reason, "short_gnss_outage")

    def test_state_machine_respects_custom_auxiliary_instability_quality_floor(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("mag", 0.7, available=True, used=True, rejected=False)
        tracker.note_result("mag", 1.0, available=True, used=False, rejected=True)
        tracker.note_result("mag", 1.1, available=True, used=False, rejected=True)
        snapshot = tracker.snapshot(1.2)

        decision = determine_mode(
            snapshot,
            quality_score=50.0,
            covariance_health=self._covariance_health(1.2, 0.8, 0.3, 3.0),
            thresholds=ModeThresholds(short_outage_aux_instability_quality_floor=60.0),
        )

        self.assertEqual(decision.mode, "DEGRADED")
        self.assertEqual(decision.reason, "auxiliary_sensor_instability")

    def test_state_machine_uses_specific_covariance_reason_under_gnss(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(0.1)
        covariance_tracker = CovarianceHealthTracker()

        determine_mode(
            snapshot,
            quality_score=85.0,
            covariance_health=self._covariance_health(0.1, 1.8, 0.2, 4.0, covariance_tracker),
        )

        decision = determine_mode(
            snapshot,
            quality_score=85.0,
            covariance_health=self._covariance_health(0.6, 1.8, 0.2, 4.0, covariance_tracker),
        )

        self.assertEqual(decision.mode, "GNSS_DEGRADED")
        self.assertEqual(decision.reason, "position_sigma_caution")

    def test_state_machine_uses_specific_covariance_reason_without_gnss(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(6.0)
        covariance_tracker = CovarianceHealthTracker()

        determine_mode(
            snapshot,
            quality_score=25.0,
            covariance_health=self._covariance_health(5.9, 3.4, 1.7, 6.0, covariance_tracker),
        )

        decision = determine_mode(
            snapshot,
            quality_score=25.0,
            covariance_health=self._covariance_health(6.2, 3.4, 1.7, 6.0, covariance_tracker),
        )

        self.assertEqual(decision.mode, "DEGRADED")
        self.assertEqual(decision.reason, "multiple_unhealthy_sigmas")

    def test_covariance_health_tracker_accumulates_and_resets_duration(self) -> None:
        tracker = CovarianceHealthTracker()

        health0 = self._covariance_health(0.0, 1.8, 0.2, 4.0, tracker)
        health1 = self._covariance_health(0.5, 1.8, 0.2, 4.0, tracker)
        health2 = self._covariance_health(0.8, 0.8, 0.2, 4.0, tracker)

        self.assertAlmostEqual(health0.caution_duration_s, 0.0, places=9)
        self.assertAlmostEqual(health1.caution_duration_s, 0.5, places=9)
        self.assertAlmostEqual(health2.caution_duration_s, 0.0, places=9)

    def test_state_machine_does_not_degrade_on_short_covariance_caution(self) -> None:
        tracker = SensorFreshnessTracker()
        tracker.note_result("gnss_pos", 0.0, available=True, used=True, rejected=False)
        tracker.note_result("gnss_vel", 0.0, available=True, used=True, rejected=False)
        snapshot = tracker.snapshot(0.1)

        decision = determine_mode(
            snapshot,
            quality_score=85.0,
            covariance_health=self._covariance_health(0.1, 1.8, 0.2, 4.0),
        )

        self.assertEqual(decision.mode, "GNSS_STABLE")
        self.assertEqual(decision.reason, "fresh_gnss_and_healthy_covariance")

    def test_covariance_health_classifies_multiple_unhealthy_sources(self) -> None:
        health = classify_covariance_health(
            pos_sigma_norm_m=3.5,
            vel_sigma_norm_mps=1.8,
            att_sigma_norm_deg=12.0,
        )

        self.assertEqual(health.level, "UNHEALTHY")
        self.assertEqual(health.reason, "multiple_unhealthy_sigmas")
        self.assertTrue(health.caution)
        self.assertTrue(health.unhealthy)
        self.assertAlmostEqual(health.pos_excess_m, 0.5, places=9)
        self.assertAlmostEqual(health.vel_excess_mps, 0.3, places=9)
        self.assertAlmostEqual(health.att_excess_deg, 0.0, places=9)

    def test_covariance_health_classifies_single_caution_source(self) -> None:
        health = classify_covariance_health(
            pos_sigma_norm_m=1.7,
            vel_sigma_norm_mps=0.4,
            att_sigma_norm_deg=10.0,
        )

        self.assertEqual(health.level, "CAUTION")
        self.assertEqual(health.reason, "position_sigma_caution")
        self.assertTrue(health.caution)
        self.assertFalse(health.unhealthy)
        self.assertAlmostEqual(health.pos_excess_m, 0.2, places=9)

    def test_mode_tracker_holds_current_mode_until_confirmation(self) -> None:
        tracker = ModeStateTracker(min_mode_hold_s=0.4, confirmation_time_s={"GNSS_DEGRADED": 0.2})

        state0 = tracker.step(0.0, ModeDecision("GNSS_STABLE", "fresh"))
        state1 = tracker.step(0.1, ModeDecision("GNSS_DEGRADED", "rejects"))
        state2 = tracker.step(0.25, ModeDecision("GNSS_DEGRADED", "rejects"))

        self.assertEqual(state0.mode, "GNSS_STABLE")
        self.assertEqual(state1.mode, "GNSS_STABLE")
        self.assertTrue(state1.transition_pending)
        self.assertEqual(state2.mode, "GNSS_STABLE")
        self.assertTrue(state2.transition_pending)

    def test_mode_tracker_switches_after_hold_and_confirmation(self) -> None:
        tracker = ModeStateTracker(min_mode_hold_s=0.4, confirmation_time_s={"GNSS_DEGRADED": 0.2})

        tracker.step(0.0, ModeDecision("GNSS_STABLE", "fresh"))
        tracker.step(0.1, ModeDecision("GNSS_DEGRADED", "rejects"))
        tracker.step(0.35, ModeDecision("GNSS_DEGRADED", "rejects"))
        state = tracker.step(0.45, ModeDecision("GNSS_DEGRADED", "rejects"))

        self.assertEqual(state.mode, "GNSS_DEGRADED")
        self.assertEqual(state.reason, "rejects")
        self.assertFalse(state.transition_pending)

    def test_mode_tracker_inserts_recovering_before_return_to_gnss(self) -> None:
        tracker = ModeStateTracker(
            min_mode_hold_s=0.2,
            confirmation_time_s={"RECOVERING": 0.1, "GNSS_STABLE": 0.1},
        )

        tracker.step(0.0, ModeDecision("INERTIAL_HOLD", "short_gnss_outage"))
        tracker.step(0.05, ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance"))
        recovering = tracker.step(0.20, ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance"))

        self.assertEqual(recovering.mode, "RECOVERING")
        self.assertEqual(recovering.reason, "restoring_gnss_stability")

    def test_mode_tracker_leaves_recovering_after_confirmation(self) -> None:
        tracker = ModeStateTracker(
            min_mode_hold_s=0.2,
            confirmation_time_s={"RECOVERING": 0.1, "GNSS_STABLE": 0.1},
        )

        tracker.step(0.0, ModeDecision("INERTIAL_HOLD", "short_gnss_outage"))
        tracker.step(0.05, ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance"))
        tracker.step(0.20, ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance"))
        tracker.step(0.25, ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance"))
        stable = tracker.step(0.40, ModeDecision("GNSS_STABLE", "fresh_gnss_and_healthy_covariance"))

        self.assertEqual(stable.mode, "GNSS_STABLE")
        self.assertEqual(stable.reason, "fresh_gnss_and_healthy_covariance")


if __name__ == "__main__":
    unittest.main()
