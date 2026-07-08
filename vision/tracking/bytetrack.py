"""Lightweight ByteTrack-style IoU tracker.

This keeps the public interface compatible with a future full ByteTrack library
while avoiding heavy dependencies for the first RK3588 demo.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Set, Tuple

from vision.tracking.tracker import Tracker
from vision.types import BBox, Detection, TrackedPerson, bbox_area


@dataclass
class _TrackState:
    track_id: int
    bbox: BBox
    confidence: float
    missed: int = 0


class ByteTrackTracker(Tracker):
    """Associate detections to tracks with greedy IoU matching."""

    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 20, min_confidence: float = 0.15) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_confidence = min_confidence
        self._tracks: Dict[int, _TrackState] = {}
        self._next_id = 1

    def update(self, detections: Iterable[Detection]) -> List[TrackedPerson]:
        detections_list = sorted(detections, key=lambda item: item.confidence, reverse=True)
        matches = self._match(detections_list)
        used_detection_indexes: Set[int] = set()
        updated_track_ids: Set[int] = set()

        for track_id, detection_index in matches:
            detection = detections_list[detection_index]
            self._tracks[track_id].bbox = detection.bbox
            self._tracks[track_id].confidence = detection.confidence
            self._tracks[track_id].missed = 0
            used_detection_indexes.add(detection_index)
            updated_track_ids.add(track_id)

        for track_id in list(self._tracks):
            if track_id not in updated_track_ids:
                self._tracks[track_id].missed += 1
                if self._tracks[track_id].missed > self.max_missing:
                    del self._tracks[track_id]

        for index, detection in enumerate(detections_list):
            if index in used_detection_indexes or detection.confidence < self.min_confidence:
                continue
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _TrackState(track_id, detection.bbox, detection.confidence, missed=0)
            updated_track_ids.add(track_id)

        visible = [
            TrackedPerson(state.track_id, state.bbox, state.confidence)
            for state in self._tracks.values()
            if state.missed == 0
        ]
        visible.sort(key=lambda item: item.track_id)
        return visible

    def _match(self, detections: List[Detection]) -> List[Tuple[int, int]]:
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for detection_index, detection in enumerate(detections):
                overlap = _iou(track.bbox, detection.bbox)
                if overlap >= self.iou_threshold:
                    candidates.append((overlap, track_id, detection_index))
        candidates.sort(reverse=True)
        matched_tracks: Set[int] = set()
        matched_detections: Set[int] = set()
        matches: List[Tuple[int, int]] = []
        for _, track_id, detection_index in candidates:
            if track_id in matched_tracks or detection_index in matched_detections:
                continue
            matched_tracks.add(track_id)
            matched_detections.add(detection_index)
            matches.append((track_id, detection_index))
        return matches


def _iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter = bbox_area((inter_x1, inter_y1, inter_x2, inter_y2))
    if inter <= 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union > 0 else 0.0
