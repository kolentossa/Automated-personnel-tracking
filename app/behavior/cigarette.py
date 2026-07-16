"""Context and temporal verification for cigarette detector candidates."""

from __future__ import annotations

import math
from collections import Counter, deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

from app.behavior.geometry import (
    box_area,
    expand_box,
    intersection_over_target,
    normalise_label,
    point_in_box,
)
from vision.types import BBox, Detection, TrackedPerson, bbox_centroid


@dataclass(frozen=True)
class CigaretteDecision:
    candidate: Optional[Detection]
    verified: bool
    filter_reason: str
    support_frames: int
    evidence: tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "bbox": list(self.candidate.bbox) if self.candidate else None,
            "confidence": round(float(self.candidate.confidence), 4)
            if self.candidate
            else 0.0,
            "verified": self.verified,
            "filter_reason": self.filter_reason,
            "support_frames": self.support_frames,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class _Observation:
    frame_index: int
    bbox: BBox
    confidence: float


class CigaretteVerifier:
    """Promote raw detections only after context, conflict, and time checks."""

    def __init__(
        self, config: dict, phone_labels: Iterable[str], tool_labels: Iterable[str]
    ) -> None:
        self.config = dict(config or {})
        self.phone_labels = {normalise_label(value) for value in phone_labels}
        self.tool_labels = {normalise_label(value) for value in tool_labels}
        self.raw_confidence = float(self.config.get("cigarette_raw_confidence", 0.30))
        self.verified_confidence = float(
            self.config.get("cigarette_verified_confidence", 0.42)
        )
        self.phone_confidence = float(self.config.get("phone_confidence", 0.16))
        self.phone_ioa = float(self.config.get("phone_cigarette_ioa_threshold", 0.48))
        self.phone_margin = float(self.config.get("phone_confidence_margin", 0.20))
        self.window_frames = max(1, int(self.config.get("temporal_window_frames", 18)))
        self.minimum_frames = max(1, int(self.config.get("minimum_verified_frames", 3)))
        self.max_length_ratio = float(
            self.config.get("max_candidate_length_person_ratio", 0.13)
        )
        self.max_area_ratio = float(
            self.config.get("max_candidate_area_person_ratio", 0.012)
        )
        self.min_center_ratio = float(
            self.config.get("min_candidate_center_person_ratio", 0.125)
        )
        self.upper_body_ratio = float(self.config.get("upper_body_ratio", 0.72))
        self.motion_ratio = float(self.config.get("candidate_motion_ratio", 0.12))
        self.tool_confidence = float(self.config.get("hard_negative_confidence", 0.22))
        self.tool_ioa = float(self.config.get("hard_negative_ioa_threshold", 0.50))
        self._history: dict[int, Deque[_Observation]] = {}
        self._last_decision: dict[int, CigaretteDecision] = {}
        self._last_fresh_frame: dict[int, int] = {}
        self._reason_counts: Counter[str] = Counter()
        self._raw_candidates = 0
        self._verified_candidates = 0

    def update(
        self,
        track: TrackedPerson,
        candidates: Iterable[Detection],
        phones: Iterable[Detection],
        tools: Iterable[Detection],
        frame_index: int,
        fresh_inference: bool,
    ) -> CigaretteDecision:
        track_id = int(track.track_id)
        if not fresh_inference:
            last_frame = self._last_fresh_frame.get(track_id, -self.window_frames - 1)
            if frame_index - last_frame <= self.window_frames:
                return self._last_decision.get(track_id, _empty_decision())
            return _empty_decision("insufficient_temporal_evidence")

        self._last_fresh_frame[track_id] = int(frame_index)
        candidate = _highest(candidates)
        if candidate is None:
            decision = self._reject(track_id, None, "insufficient_temporal_evidence")
            self._prune_history(track_id, frame_index)
            return decision

        self._raw_candidates += 1
        if candidate.confidence < self.raw_confidence:
            return self._reject(track_id, candidate, "low_verified_confidence")
        if not self._valid_person_context(track.bbox, candidate.bbox):
            return self._reject(track_id, candidate, "invalid_size_context", clear=True)
        if self._phone_conflict(candidate, phones):
            return self._reject(track_id, candidate, "phone_overlap", clear=True)
        if self._tool_conflict(candidate, tools):
            return self._reject(track_id, candidate, "tool_verifier", clear=True)

        history = self._history.setdefault(track_id, deque())
        self._prune_history(track_id, frame_index)
        if history and not self._position_is_continuous(
            history[-1].bbox, candidate.bbox, track.bbox
        ):
            history.clear()
        history.append(
            _Observation(int(frame_index), candidate.bbox, float(candidate.confidence))
        )
        support = len(history)
        average_confidence = sum(item.confidence for item in history) / max(1, support)
        verified = (
            support >= self.minimum_frames
            and average_confidence >= self.verified_confidence
        )
        reason = (
            ""
            if verified
            else (
                "low_verified_confidence"
                if support >= self.minimum_frames
                else "insufficient_temporal_evidence"
            )
        )
        evidence = (
            (
                "person_association",
                "valid_size_context",
                "temporal_continuity",
            )
            if verified
            else ()
        )
        decision = CigaretteDecision(candidate, verified, reason, support, evidence)
        self._last_decision[track_id] = decision
        if verified:
            self._verified_candidates += 1
        elif reason:
            self._reason_counts[reason] += 1
        return decision

    def record_unassociated(self, count: int) -> None:
        if count > 0:
            self._reason_counts["no_person_association"] += int(count)

    def cleanup(self, active_track_ids: Iterable[int]) -> None:
        active = {int(value) for value in active_track_ids}
        for mapping in (self._history, self._last_decision, self._last_fresh_frame):
            for track_id in list(mapping):
                if track_id not in active:
                    mapping.pop(track_id, None)

    def reset(self) -> None:
        self._history.clear()
        self._last_decision.clear()
        self._last_fresh_frame.clear()
        self._reason_counts.clear()
        self._raw_candidates = 0
        self._verified_candidates = 0

    def snapshot(self) -> dict:
        return {
            "raw_cigarette_candidates": self._raw_candidates,
            "verified_cigarette_candidates": self._verified_candidates,
            "cigarette_filter_reasons": dict(self._reason_counts),
            "cigarette_candidate_states": {
                str(track_id): decision.as_dict()
                for track_id, decision in sorted(self._last_decision.items())
            },
        }

    def _reject(
        self,
        track_id: int,
        candidate: Optional[Detection],
        reason: str,
        clear: bool = False,
    ) -> CigaretteDecision:
        if clear:
            self._history.pop(track_id, None)
        decision = CigaretteDecision(candidate, False, reason, 0, ())
        self._last_decision[track_id] = decision
        self._reason_counts[reason] += 1
        return decision

    def _prune_history(self, track_id: int, frame_index: int) -> None:
        history = self._history.setdefault(track_id, deque())
        minimum_frame = int(frame_index) - self.window_frames
        while history and history[0].frame_index < minimum_frame:
            history.popleft()

    def _valid_person_context(self, person: BBox, candidate: BBox) -> bool:
        px1, py1, px2, py2 = person
        person_height = max(1.0, py2 - py1)
        person_area = max(1.0, box_area(person))
        width = max(0.0, candidate[2] - candidate[0])
        height = max(0.0, candidate[3] - candidate[1])
        center = bbox_centroid(candidate)
        return bool(
            point_in_box(center, expand_box(person, 0.12))
            and center[1] >= py1 + self.min_center_ratio * person_height
            and center[1] <= py1 + self.upper_body_ratio * person_height
            and max(width, height) / person_height <= self.max_length_ratio
            and box_area(candidate) / person_area <= self.max_area_ratio
        )

    def _phone_conflict(
        self, candidate: Detection, phones: Iterable[Detection]
    ) -> bool:
        for phone in phones:
            if normalise_label(phone.label) not in self.phone_labels:
                continue
            overlap = intersection_over_target(
                candidate.bbox, expand_box(phone.bbox, 0.08)
            )
            confidence_wins = (
                phone.confidence + self.phone_margin >= candidate.confidence
            )
            if (
                overlap >= self.phone_ioa
                and phone.confidence >= self.phone_confidence
                and confidence_wins
            ):
                return True
        return False

    def _tool_conflict(self, candidate: Detection, tools: Iterable[Detection]) -> bool:
        for tool in tools:
            if normalise_label(tool.label) not in self.tool_labels:
                continue
            if tool.confidence < self.tool_confidence:
                continue
            if (
                intersection_over_target(candidate.bbox, expand_box(tool.bbox, 0.08))
                >= self.tool_ioa
            ):
                return True
        return False

    def _position_is_continuous(
        self, previous: BBox, current: BBox, person: BBox
    ) -> bool:
        diagonal = max(1.0, math.hypot(person[2] - person[0], person[3] - person[1]))
        distance = math.dist(bbox_centroid(previous), bbox_centroid(current))
        return distance / diagonal <= self.motion_ratio


def _highest(items: Iterable[Detection]) -> Optional[Detection]:
    values = list(items)
    return max(values, key=lambda item: item.confidence) if values else None


def _empty_decision(reason: str = "") -> CigaretteDecision:
    return CigaretteDecision(None, False, reason, 0, ())
