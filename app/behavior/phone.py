"""Phone-call, phone-playing, and restricted-photography state machine."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from app.behavior.geometry import (
    associate_targets,
    estimated_head_region,
    highest_confidence,
    labels_for_group,
    near_region,
    phone_aims_at_roi,
    phone_below_head,
    resolve_rois,
)
from app.behavior.state_machine import TemporalEventStateMachine, rule_from_config
from app.behavior.types import BehaviorSignal, BehaviorTrigger
from vision.types import BBox, Detection, TrackedPerson


class PhoneBehaviorStateMachine:
    EVENT_TYPES = ("phone_call", "phone_playing", "unauthorized_photography")

    def __init__(
        self, config: dict, class_groups: dict, association_config: dict
    ) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("enabled", True))
        self.phone_labels = labels_for_group(
            class_groups, "phone", ["cell phone", "phone"]
        )
        self.expansion_ratio = float(
            association_config.get("person_expansion_ratio") or 0.12
        )
        self.stale_track_ms = int(association_config.get("stale_track_ms") or 5000)
        self.head_region_config = dict(self.config.get("head_region") or {})
        self.roi_config = list(self.config.get("prohibited_rois") or [])
        self._rules = {
            event_type: rule_from_config(self.config.get(event_type, {}))
            for event_type in self.EVENT_TYPES
        }
        self._machine = TemporalEventStateMachine()

    def update(
        self,
        tracks: Iterable[TrackedPerson],
        detections: Iterable[Detection],
        frame_shape: Sequence[int],
        now_ms: float,
    ) -> List[BehaviorTrigger]:
        people = list(tracks)
        if not self.enabled:
            self._machine.cleanup([], now_ms, self.stale_track_ms)
            return []
        assignments = associate_targets(
            people, detections, self.phone_labels, self.expansion_ratio
        )
        rois = resolve_rois(self.roi_config, frame_shape)
        triggers: List[BehaviorTrigger] = []

        for track in people:
            phone = highest_confidence(assignments.get(track.track_id, []))
            head = estimated_head_region(track.bbox, self.head_region_config)
            call_signal = self._call_signal(track, phone, head)
            playing_signal = self._playing_signal(track, phone, head)
            photo_signal = self._photography_signal(track, phone, rois)
            for signal in (call_signal, playing_signal, photo_signal):
                trigger = self._machine.update(
                    signal, self._rules[signal.event_type], now_ms
                )
                if trigger is not None:
                    triggers.append(trigger)

        self._machine.cleanup(
            (track.track_id for track in people), now_ms, self.stale_track_ms
        )
        return triggers

    def reset(self) -> None:
        self._machine.reset()

    def snapshot(self) -> dict:
        data = self._machine.snapshot()
        data.update(
            {
                "phone_behavior_enabled": self.enabled,
                "phone_labels": sorted(self.phone_labels),
            }
        )
        return data

    def _call_signal(
        self,
        track: TrackedPerson,
        phone: Detection | None,
        head: BBox,
    ) -> BehaviorSignal:
        event_type = "phone_call"
        if phone is None:
            return _empty_signal(event_type, track)
        relation = self.config.get(event_type, {})
        matched, relation_score = near_region(
            phone.bbox,
            head,
            float(relation.get("max_head_distance_ratio") or 0.9),
        )
        matched = matched and _adjacent_to_head(
            phone.bbox,
            head,
            float(relation.get("max_horizontal_offset_ratio") or 1.0),
            float(relation.get("max_vertical_offset_ratio") or 0.75),
        )
        confidence = phone.confidence * (0.90 + 0.10 * relation_score) * 0.95
        evidence = ["phone_near_estimated_head_region"] if matched else []
        return BehaviorSignal(
            event_type=event_type,
            track_id=track.track_id,
            observed=matched,
            confidence=confidence,
            person_bbox=track.bbox,
            object_bboxes={"phone": phone.bbox, "head_region": head},
            evidence=tuple(evidence),
        )

    def _playing_signal(
        self,
        track: TrackedPerson,
        phone: Detection | None,
        head: BBox,
    ) -> BehaviorSignal:
        event_type = "phone_playing"
        if phone is None:
            return _empty_signal(event_type, track)
        below, relation_score = phone_below_head(phone.bbox, head, track.bbox)
        near, _ = near_region(
            phone.bbox,
            head,
            float(
                self.config.get("phone_playing", {}).get(
                    "max_head_exclusion_distance_ratio", 0.75
                )
            ),
        )
        observed = below and not near
        confidence = phone.confidence * (0.88 + 0.12 * relation_score) * 0.94
        evidence = ()
        if observed:
            evidence = (
                "phone_held_inside_upper_body",
                "phone_below_head_attention_proxy",
                "estimated_head_geometry",
            )
        return BehaviorSignal(
            event_type=event_type,
            track_id=track.track_id,
            observed=observed,
            confidence=confidence,
            person_bbox=track.bbox,
            object_bboxes={"phone": phone.bbox, "head_region": head},
            evidence=evidence,
        )

    def _photography_signal(
        self,
        track: TrackedPerson,
        phone: Detection | None,
        rois: list[dict],
    ) -> BehaviorSignal:
        event_type = "unauthorized_photography"
        if phone is None or not rois:
            return _empty_signal(event_type, track)
        max_angle = float(
            self.config.get(event_type, {}).get("max_alignment_angle_deg") or 35.0
        )
        matches = []
        for roi in rois:
            matched, score = phone_aims_at_roi(
                phone.bbox, track.bbox, roi["bbox"], max_angle
            )
            if matched:
                matches.append((score, roi))
        if not matches:
            return BehaviorSignal(
                event_type,
                track.track_id,
                False,
                0.0,
                track.bbox,
                {"phone": phone.bbox},
                (),
            )
        score, roi = max(matches, key=lambda item: item[0])
        confidence = phone.confidence * (0.7 + 0.3 * score)
        return BehaviorSignal(
            event_type=event_type,
            track_id=track.track_id,
            observed=True,
            confidence=confidence,
            person_bbox=track.bbox,
            object_bboxes={"phone": phone.bbox, "prohibited_roi": roi["bbox"]},
            evidence=(
                "phone_raised",
                f"phone_position_aligned_with_roi:{roi['id']}",
                "camera_direction_is_geometric_proxy",
            ),
        )


def _empty_signal(event_type: str, track: TrackedPerson) -> BehaviorSignal:
    return BehaviorSignal(event_type, track.track_id, False, 0.0, track.bbox, {}, ())


def _adjacent_to_head(
    phone_bbox: BBox,
    head_bbox: BBox,
    max_horizontal_offset_ratio: float,
    max_vertical_offset_ratio: float,
) -> bool:
    phone_x = (phone_bbox[0] + phone_bbox[2]) * 0.5
    phone_y = (phone_bbox[1] + phone_bbox[3]) * 0.5
    head_width = max(1.0, head_bbox[2] - head_bbox[0])
    head_height = max(1.0, head_bbox[3] - head_bbox[1])
    horizontal = max(head_bbox[0] - phone_x, 0.0, phone_x - head_bbox[2]) / head_width
    vertical = max(head_bbox[1] - phone_y, 0.0, phone_y - head_bbox[3]) / head_height
    return (
        horizontal <= max_horizontal_offset_ratio
        and vertical <= max_vertical_offset_ratio
    )
