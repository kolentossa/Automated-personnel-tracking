from __future__ import annotations

from unittest import TestCase

from app.behavior.smoking import SmokingBehaviorStateMachine
from vision.types import Detection, TrackedPerson


class CigaretteVerificationTests(TestCase):
    def setUp(self) -> None:
        self.track = TrackedPerson(7, (100, 100, 300, 500), 0.95)

    def test_phone_overlap_never_becomes_verified_cigarette(self) -> None:
        machine = _machine()
        cigarette = Detection((205, 172, 225, 192), 0.45, 0, "cigarette")
        phone = Detection((195, 145, 245, 225), 0.55, 67, "cell phone")

        events = _run(machine, self.track, [cigarette, phone], 6)

        self.assertEqual(events, [])
        snapshot = machine.snapshot()
        self.assertGreater(
            snapshot["cigarette_filter_reasons"].get("phone_overlap", 0), 0
        )
        self.assertFalse(snapshot["cigarette_candidate_states"]["7"]["verified"])

    def test_phone_edge_overlap_at_calibrated_boundary_is_rejected(self) -> None:
        machine = _machine()
        cigarette = Detection((180, 150, 229, 168.33), 0.50, 0, "cigarette")
        phone = Detection((184, 156, 210.5, 181), 0.60, 67, "cell phone")

        events = _run(machine, self.track, [cigarette, phone], 6)

        self.assertEqual(events, [])
        self.assertGreater(
            machine.snapshot()["cigarette_filter_reasons"].get("phone_overlap", 0), 0
        )

    def test_screwdriver_overlap_never_confirms_smoking(self) -> None:
        machine = _machine()
        cigarette = Detection((205, 172, 225, 192), 0.48, 0, "cigarette")
        tool = Detection((198, 160, 232, 225), 0.72, 101, "screwdriver")

        events = _run(machine, self.track, [cigarette, tool], 6)

        self.assertEqual(events, [])
        self.assertGreater(
            machine.snapshot()["cigarette_filter_reasons"].get("tool_verifier", 0), 0
        )

    def test_long_tool_shaped_candidate_fails_person_size_context(self) -> None:
        machine = _machine()
        candidate = Detection((205, 150, 220, 245), 0.72, 0, "cigarette")

        events = _run(machine, self.track, [candidate], 6)

        self.assertEqual(events, [])
        self.assertGreater(
            machine.snapshot()["cigarette_filter_reasons"].get(
                "invalid_size_context", 0
            ),
            0,
        )

    def test_candidate_above_mouth_band_fails_person_context(self) -> None:
        machine = _machine()
        candidate = Detection((205, 112, 225, 126), 0.72, 0, "cigarette")

        events = _run(machine, self.track, [candidate], 6)

        self.assertEqual(events, [])
        self.assertGreater(
            machine.snapshot()["cigarette_filter_reasons"].get(
                "invalid_size_context", 0
            ),
            0,
        )

    def test_single_raw_candidate_does_not_confirm_smoking(self) -> None:
        machine = _machine()
        cigarette = Detection((205, 172, 218, 187), 0.72, 0, "cigarette")

        self.assertEqual(
            machine.update(
                [self.track],
                [cigarette],
                (720, 1280),
                0,
                frame_index=0,
                fresh_inference=True,
            ),
            [],
        )
        for frame_index in (1, 2, 3):
            self.assertEqual(
                machine.update(
                    [self.track],
                    [cigarette],
                    (720, 1280),
                    frame_index * 30,
                    frame_index=frame_index,
                    fresh_inference=False,
                ),
                [],
            )

    def test_continuous_real_cigarette_is_verified_and_confirms(self) -> None:
        machine = _machine()
        cigarette = Detection((205, 172, 218, 187), 0.72, 0, "cigarette")

        events = _run(machine, self.track, [cigarette], 4)

        self.assertEqual(len(events), 1)
        self.assertIn("verified_cigarette", events[0].evidence)
        self.assertIn("hand_to_mouth_context_proxy", events[0].evidence)

    def test_unassociated_cigarette_does_not_confirm_smoking(self) -> None:
        machine = _machine()
        cigarette = Detection((700, 172, 718, 187), 0.90, 0, "cigarette")

        events = _run(machine, self.track, [cigarette], 4)

        self.assertEqual(events, [])
        self.assertGreater(
            machine.snapshot()["cigarette_filter_reasons"].get(
                "no_person_association", 0
            ),
            0,
        )


def _run(machine, track, detections, frames):
    events = []
    for frame_index in range(frames):
        events.extend(
            machine.update(
                [track],
                detections,
                (720, 1280),
                frame_index * 100,
                frame_index=frame_index,
                fresh_inference=True,
            )
        )
    return events


def _machine() -> SmokingBehaviorStateMachine:
    return SmokingBehaviorStateMachine(
        {
            "enabled": True,
            "duration_ms": 0,
            "min_consecutive_frames": 1,
            "confidence_threshold": 0.2,
            "cooldown_ms": 1000,
            "max_gap_frames": 1,
            "rearm_absence_ms": 100,
            "max_mouth_distance_ratio": 1.1,
            "cigarette_raw_confidence": 0.3,
            "cigarette_verified_confidence": 0.4,
            "phone_confidence": 0.16,
            "phone_cigarette_ioa_threshold": 0.48,
            "phone_confidence_margin": 0.2,
            "temporal_window_frames": 8,
            "minimum_verified_frames": 3,
            "max_candidate_length_person_ratio": 0.13,
            "max_candidate_area_person_ratio": 0.012,
            "min_candidate_center_person_ratio": 0.125,
            "upper_body_ratio": 0.72,
            "candidate_motion_ratio": 0.12,
            "hard_negative_confidence": 0.22,
            "hard_negative_ioa_threshold": 0.5,
            "hard_negative_labels": ["screwdriver", "pen", "stick", "toothbrush"],
            "head_region": {
                "left_ratio": 0.22,
                "right_ratio": 0.78,
                "top_ratio": 0.0,
                "bottom_ratio": 0.32,
                "mouth_top_ratio": 0.16,
                "mouth_bottom_ratio": 0.36,
            },
        },
        {"phone": ["cell phone"], "cigarette": ["cigarette"]},
        {"person_expansion_ratio": 0.12, "stale_track_ms": 1000},
    )
