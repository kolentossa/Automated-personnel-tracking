from __future__ import annotations

from unittest import TestCase

from app.behavior.smoking import SmokingBehaviorStateMachine
from vision.types import Detection, TrackedPerson


class SmokingBehaviorTests(TestCase):
    def setUp(self) -> None:
        self.track = TrackedPerson(7, (0, 0, 200, 300), 0.9)

    def test_cigarette_away_from_mouth_does_not_trigger(self) -> None:
        machine = _machine()
        detections = [Detection((80, 230, 90, 240), 0.85, 0, "cigarette")]

        self.assertEqual(machine.update([self.track], detections, [], (300, 200), 0), [])
        self.assertEqual(machine.snapshot()["active_candidate_count"], 0)

    def test_cigarette_near_mouth_triggers_after_temporal_rule(self) -> None:
        machine = _machine()
        detections = [Detection((90, 55, 100, 65), 0.85, 0, "cigarette")]

        self.assertEqual(machine.update([self.track], detections, [], (300, 200), 0), [])
        triggers = machine.update([self.track], detections, [], (300, 200), 1)

        self.assertEqual(len(triggers), 1)
        self.assertIn("cigarette_near_mouth", triggers[0].evidence)

    def test_direct_smoking_is_not_overridden_by_distant_cigarette(self) -> None:
        machine = _machine()
        detections = [
            Detection((20, 20, 180, 280), 0.9, 1, "person smoking"),
            Detection((80, 230, 90, 240), 0.8, 0, "cigarette"),
        ]

        machine.update([self.track], detections, [], (300, 200), 0)
        triggers = machine.update([self.track], detections, [], (300, 200), 1)

        self.assertEqual(len(triggers), 1)
        self.assertIn("direct_smoking_model_output", triggers[0].evidence)

    def test_flame_and_lighter_without_direct_evidence_do_not_trigger(self) -> None:
        machine = _machine()
        detections = [
            Detection((90, 55, 100, 65), 0.9, 2, "flame"),
            Detection((95, 65, 110, 80), 0.9, 3, "lighter"),
        ]

        self.assertEqual(machine.update([self.track], detections, [], (300, 200), 0), [])
        self.assertEqual(machine.update([self.track], detections, [], (300, 200), 1), [])


def _machine() -> SmokingBehaviorStateMachine:
    return SmokingBehaviorStateMachine(
        {
            "enabled": True,
            "duration_ms": 0,
            "min_consecutive_frames": 2,
            "confidence_threshold": 0.2,
            "cooldown_ms": 1000,
            "max_gap_frames": 1,
            "rearm_absence_ms": 100,
            "max_mouth_distance_ratio": 1.1,
            "allow_persistent_cigarette": False,
        },
        {},
        {"person_expansion_ratio": 0.12, "stale_track_ms": 1000},
    )
