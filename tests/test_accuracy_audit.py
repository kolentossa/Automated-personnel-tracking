from __future__ import annotations

from unittest import TestCase

from scripts.audit_false_positives import _event_metrics


class AccuracyAuditTests(TestCase):
    def test_event_metrics_count_image_level_outcomes_and_duplicates(self) -> None:
        predictions = [
            {
                "scenario": "phone-call",
                "trigger_event_counts": {"phone_call": 2},
                "trigger_durations_ms": {"phone_call": [1200.0, 1300.0]},
            },
            {
                "scenario": "phone-call",
                "trigger_event_counts": {},
                "trigger_durations_ms": {},
            },
            {
                "scenario": "general_negative",
                "trigger_event_counts": {"phone_call": 1},
                "trigger_durations_ms": {"phone_call": [1250.0]},
            },
            {
                "scenario": "general_negative",
                "trigger_event_counts": {},
                "trigger_durations_ms": {},
            },
        ]

        result = _event_metrics(predictions, "phone_call", {"phone-call"})

        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 1)
        self.assertEqual(result["false_negative"], 1)
        self.assertEqual(result["true_negative"], 1)
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["duplicate_events"], 1)
        self.assertEqual(result["average_trigger_ms"], 1250.0)

    def test_event_metrics_can_limit_the_evaluation_domain(self) -> None:
        predictions = [
            {
                "source": "rk3588-event-evidence",
                "scenario": "phone-call",
                "trigger_event_counts": {"phone_call": 1},
                "trigger_durations_ms": {"phone_call": [1200.0]},
            },
            {
                "source": "cigdet-v1",
                "scenario": "real_cigarette",
                "trigger_event_counts": {"phone_call": 1},
                "trigger_durations_ms": {"phone_call": [1200.0]},
            },
        ]

        result = _event_metrics(
            predictions,
            "phone_call",
            {"phone-call"},
            eligible_sources={"rk3588-event-evidence"},
        )

        self.assertEqual(result["evaluated_images"], 1)
        self.assertEqual(result["true_positive"], 1)
        self.assertEqual(result["false_positive"], 0)
