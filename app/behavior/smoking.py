"""Smoking evidence fusion and temporal state machine."""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence

from app.behavior.geometry import (
    associate_faces,
    associate_targets,
    estimated_face_box,
    highest_confidence,
    labels_for_group,
    near_face,
    normalise_label,
)
from app.behavior.state_machine import TemporalEventStateMachine, rule_from_config
from app.behavior.types import BehaviorSignal, BehaviorTrigger
from vision.types import BBox, Detection, TrackedPerson, bbox_centroid


class SmokingBehaviorStateMachine:
    EVENT_TYPE = "smoking"

    def __init__(self, config: dict, class_groups: dict, association_config: dict) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.groups = {
            name: labels_for_group(class_groups, name, fallback)
            for name, fallback in {
                "cigarette": ["cigarette", "cigar"],
                "smoke": ["smoke"],
                "flame": ["flame", "fire"],
                "lighter": ["lighter"],
                "hand": ["hand", "left hand", "right hand"],
                "direct_smoking": ["smoking", "person smoking", "hand with cigarette"],
            }.items()
        }
        all_labels = set().union(*self.groups.values())
        self.expansion_ratio = float(association_config.get("person_expansion_ratio") or 0.12)
        self.stale_track_ms = int(association_config.get("stale_track_ms") or 5000)
        self.max_mouth_distance_ratio = float(self.config.get("max_mouth_distance_ratio") or 1.1)
        self.allow_persistent_cigarette = bool(self.config.get("allow_persistent_cigarette", False))
        self._all_labels = all_labels
        self._rule = rule_from_config(self.config)
        self._machine = TemporalEventStateMachine()

    def update(
        self,
        tracks: Iterable[TrackedPerson],
        detections: Iterable[Detection],
        face_boxes: Iterable[BBox],
        frame_shape: Sequence[int],
        now_ms: float,
    ) -> List[BehaviorTrigger]:
        del frame_shape
        people = list(tracks)
        if not self.enabled:
            self._machine.cleanup([], now_ms, self.stale_track_ms)
            return []
        assignments = associate_targets(people, detections, self._all_labels, self.expansion_ratio)
        faces = associate_faces(people, face_boxes)
        triggers: List[BehaviorTrigger] = []
        for track in people:
            signal = self._signal_for_track(track, assignments.get(track.track_id, []), faces.get(track.track_id))
            trigger = self._machine.update(signal, self._rule, now_ms)
            if trigger is not None:
                triggers.append(trigger)
        self._machine.cleanup((track.track_id for track in people), now_ms, self.stale_track_ms)
        return triggers

    def reset(self) -> None:
        self._machine.reset()

    def snapshot(self) -> dict:
        data = self._machine.snapshot()
        data.update({
            "smoking_behavior_enabled": self.enabled,
            "smoking_required_classes": sorted(self.groups["cigarette"] | self.groups["smoke"] | self.groups["flame"]),
        })
        return data

    def _signal_for_track(
        self,
        track: TrackedPerson,
        detections: List[Detection],
        detected_face: BBox | None,
    ) -> BehaviorSignal:
        grouped = {
            name: [item for item in detections if normalise_label(item.label) in labels]
            for name, labels in self.groups.items()
        }
        cigarette = highest_confidence(grouped["cigarette"])
        smoke = highest_confidence(grouped["smoke"])
        flame = highest_confidence(grouped["flame"])
        lighter = highest_confidence(grouped["lighter"])
        hands = grouped["hand"]
        direct_smoking = highest_confidence(grouped["direct_smoking"])
        face = detected_face or estimated_face_box(track.bbox)
        real_face = detected_face is not None

        cigarette_near_mouth = False
        cigarette_relation = 0.0
        if cigarette is not None:
            cigarette_near_mouth, cigarette_relation = near_face(
                cigarette.bbox, face, self.max_mouth_distance_ratio
            )
        hand_to_mouth = any(near_face(hand.bbox, face, self.max_mouth_distance_ratio)[0] for hand in hands)
        hand_near_cigarette = bool(cigarette and any(_boxes_close(hand.bbox, cigarette.bbox, track.bbox) for hand in hands))
        ignition = bool(flame and (lighter or hand_to_mouth) and near_face(flame.bbox, face, 1.5)[0])

        observed = False
        evidence = []
        confidence_parts = []
        if direct_smoking is not None:
            observed = True
            confidence_parts.append(direct_smoking.confidence)
            evidence.append("direct_smoking_model_output")
        if cigarette is not None:
            observed = observed or self.allow_persistent_cigarette or cigarette_near_mouth or hand_near_cigarette
            confidence_parts.append(cigarette.confidence * (0.78 + 0.22 * cigarette_relation))
            evidence.append("cigarette_associated_with_person")
        if cigarette_near_mouth:
            evidence.append("cigarette_near_mouth")
        if hand_to_mouth and hand_near_cigarette:
            evidence.append("hand_to_mouth_with_cigarette")
            confidence_parts.append(0.85)
        if ignition:
            evidence.append("flame_or_lighter_near_mouth")
            if observed:
                confidence_parts.append(
                    max(flame.confidence if flame else 0.0, lighter.confidence if lighter else 0.0)
                )
        if smoke is not None:
            evidence.append("smoke_near_associated_person")
            if observed:
                confidence_parts.append(smoke.confidence * 0.8)
        if smoke is not None and not observed:
            evidence = ["smoke_without_cigarette_or_human_action_not_classified_as_smoking"]

        confidence = sum(confidence_parts) / len(confidence_parts) if confidence_parts else 0.0
        if not real_face and observed:
            confidence *= 0.9
            evidence.append("estimated_head_geometry")
        objects = {}
        for name, item in (("cigarette", cigarette), ("smoke", smoke), ("flame", flame), ("lighter", lighter)):
            if item is not None:
                objects[name] = item.bbox
        if direct_smoking is not None:
            objects["smoking"] = direct_smoking.bbox
        if hands:
            objects["hand"] = highest_confidence(hands).bbox
        objects["face"] = face
        return BehaviorSignal(
            event_type=self.EVENT_TYPE,
            track_id=track.track_id,
            observed=observed,
            confidence=min(1.0, confidence),
            person_bbox=track.bbox,
            object_bboxes=objects,
            evidence=tuple(evidence),
        )


def _boxes_close(first: BBox, second: BBox, person: BBox) -> bool:
    person_diagonal = max(1.0, math.hypot(person[2] - person[0], person[3] - person[1]))
    return math.dist(bbox_centroid(first), bbox_centroid(second)) / person_diagonal <= 0.18
