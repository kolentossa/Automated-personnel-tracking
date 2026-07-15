from __future__ import annotations

from copy import deepcopy
from unittest import TestCase

from app.behavior.config import DEFAULT_BEHAVIOR_CONFIG
from app.behavior.engine import BehaviorEngine
from app.behavior.state_machine import BehaviorRule, TemporalEventStateMachine
from app.behavior.types import BehaviorSignal
from vision.types import Detection, TrackedPerson


class TemporalEventStateMachineTests(TestCase):
    def setUp(self) -> None:
        self.machine = TemporalEventStateMachine()
        self.rule = BehaviorRule(100, 3, 0.3, 1000, max_gap_frames=1)
        self.signal = BehaviorSignal(
            "phone_call",
            7,
            True,
            0.8,
            (100, 100, 300, 500),
            {"phone": (205, 145, 230, 185)},
            ("phone_near_detected_face",),
        )

    def test_short_observation_does_not_trigger(self) -> None:
        self.assertIsNone(self.machine.update(self.signal, self.rule, 0))
        self.assertIsNone(self.machine.update(self.signal, self.rule, 50))

    def test_threshold_triggers_once_and_respects_cooldown(self) -> None:
        self.machine.update(self.signal, self.rule, 0)
        self.machine.update(self.signal, self.rule, 50)
        trigger = self.machine.update(self.signal, self.rule, 100)
        self.assertIsNotNone(trigger)
        self.assertEqual(trigger.event_type, "phone_call")
        self.assertIsNone(self.machine.update(self.signal, self.rule, 150))

        absent = BehaviorSignal("phone_call", 7, False, 0.0, self.signal.person_bbox)
        self.machine.update(absent, self.rule, 200)
        self.machine.update(absent, self.rule, 250)
        for timestamp in (300, 350, 400, 1050):
            self.assertIsNone(self.machine.update(self.signal, self.rule, timestamp))
        self.machine.update(absent, self.rule, 1060)
        self.machine.update(absent, self.rule, 1070)
        self.machine.update(self.signal, self.rule, 1100)
        self.machine.update(self.signal, self.rule, 1150)
        self.assertIsNotNone(self.machine.update(self.signal, self.rule, 1200))


class BehaviorEngineScenarioTests(TestCase):
    def setUp(self) -> None:
        self.config = _test_config()
        self.track = TrackedPerson(1, (100, 100, 300, 500), 0.95)
        self.face = (150, 110, 240, 210)

    def test_phone_near_ear_below_duration_does_not_alert(self) -> None:
        engine = BehaviorEngine(self.config)
        phone = Detection((205, 145, 230, 185), 0.9, 67, "cell phone")
        self.assertEqual(engine.update([self.track], [phone], [self.face], (720, 1280), 0).triggers, [])
        self.assertEqual(engine.update([self.track], [phone], [self.face], (720, 1280), 50).triggers, [])

    def test_phone_call_triggers_once(self) -> None:
        engine = BehaviorEngine(self.config)
        phone = Detection((205, 145, 230, 185), 0.9, 67, "cell phone")
        events = _run(engine, self.track, [phone], [self.face], (0, 50, 100, 150, 200))
        calls = [event for event in events if event.event_type == "phone_call"]
        self.assertEqual(len(calls), 1)
        self.assertIn("phone_near_detected_face", calls[0].evidence)

    def test_phone_playing_triggers(self) -> None:
        engine = BehaviorEngine(self.config)
        phone = Detection((175, 300, 210, 345), 0.9, 67, "cell phone")
        events = _run(engine, self.track, [phone], [self.face], (0, 50, 100, 150))
        playing = [event for event in events if event.event_type == "phone_playing"]
        self.assertEqual(len(playing), 1)
        self.assertIn("phone_below_face_attention_proxy", playing[0].evidence)

    def test_phone_raised_toward_prohibited_roi_triggers(self) -> None:
        engine = BehaviorEngine(self.config)
        phone = Detection((260, 235, 290, 275), 0.9, 67, "cell phone")
        events = _run(engine, self.track, [phone], [self.face], (0, 50, 100, 150))
        photos = [event for event in events if event.event_type == "unauthorized_photography"]
        self.assertEqual(len(photos), 1)
        self.assertIn("prohibited_roi", photos[0].object_bboxes)

    def test_person_inside_roi_does_not_trigger_photography(self) -> None:
        engine = BehaviorEngine(self.config)
        person = TrackedPerson(2, (420, 100, 550, 500), 0.95)
        phone = Detection((510, 210, 535, 250), 0.9, 67, "cell phone")
        events = _run(engine, person, [phone], [], (0, 50, 100, 150))
        self.assertFalse(any(event.event_type == "unauthorized_photography" for event in events))

    def test_person_outside_roi_with_low_phone_does_not_trigger_photography(self) -> None:
        engine = BehaviorEngine(self.config)
        phone = Detection((150, 390, 185, 440), 0.9, 67, "cell phone")
        events = _run(engine, self.track, [phone], [self.face], (0, 50, 100, 150, 200))
        self.assertFalse(any(event.event_type == "unauthorized_photography" for event in events))

    def test_cigarette_and_hand_to_mouth_trigger_smoking(self) -> None:
        engine = BehaviorEngine(self.config)
        detections = [
            Detection((205, 175, 217, 190), 0.85, 2, "cigarette"),
            Detection((175, 155, 225, 215), 0.8, 6, "hand"),
        ]
        events = _run(engine, self.track, detections, [self.face], (0, 50, 100, 150))
        smoking = [event for event in events if event.event_type == "smoking"]
        self.assertEqual(len(smoking), 1)
        self.assertIn("hand_to_mouth_with_cigarette", smoking[0].evidence)

    def test_smoke_without_person_action_does_not_trigger_smoking(self) -> None:
        engine = BehaviorEngine(self.config)
        smoke = Detection((140, 130, 260, 300), 0.95, 3, "smoke")
        events = _run(engine, self.track, [smoke], [self.face], (0, 50, 100, 150, 200))
        self.assertFalse(any(event.event_type == "smoking" for event in events))

    def test_missing_track_state_is_cleaned(self) -> None:
        engine = BehaviorEngine(self.config)
        phone = Detection((205, 145, 230, 185), 0.9, 67, "cell phone")
        engine.update([self.track], [phone], [self.face], (720, 1280), 0)
        engine.update([], [], [], (720, 1280), 500)
        snapshot = engine.snapshot()
        self.assertEqual(snapshot["phone_active_candidate_count"], 0)
        self.assertEqual(snapshot["smoking_active_candidate_count"], 0)


def _run(engine, track, detections, faces, timestamps):
    events = []
    for timestamp in timestamps:
        events.extend(engine.update([track], detections, faces, (720, 1280), timestamp).triggers)
    return events


def _test_config() -> dict:
    config = deepcopy(DEFAULT_BEHAVIOR_CONFIG)
    config["association"]["stale_track_ms"] = 200
    for name in ("phone_call", "phone_playing", "unauthorized_photography"):
        config["phone"][name].update({
            "duration_ms": 100,
            "min_consecutive_frames": 3,
            "confidence_threshold": 0.2,
            "cooldown_ms": 1000,
            "max_gap_frames": 1,
        })
    config["phone"]["prohibited_rois"] = [{
        "id": "equipment",
        "enabled": True,
        "normalized": False,
        "x1": 400,
        "y1": 100,
        "x2": 600,
        "y2": 400,
    }]
    config["smoking"].update({
        "duration_ms": 100,
        "min_consecutive_frames": 3,
        "confidence_threshold": 0.2,
        "cooldown_ms": 1000,
        "max_gap_frames": 1,
    })
    return config
