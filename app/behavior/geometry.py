"""Target-to-person association and spatial behavior relations."""

from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from vision.types import BBox, Detection, TrackedPerson, bbox_centroid

Rect = Tuple[float, float, float, float]


def associate_targets(
    tracks: Iterable[TrackedPerson],
    detections: Iterable[Detection],
    labels: Iterable[str],
    expansion_ratio: float = 0.12,
) -> Dict[int, List[Detection]]:
    """Assign each target to at most one person using containment and distance."""

    people = list(tracks)
    wanted = {normalise_label(label) for label in labels}
    result: Dict[int, List[Detection]] = {track.track_id: [] for track in people}
    for detection in detections:
        if normalise_label(detection.label) not in wanted:
            continue
        center = bbox_centroid(detection.bbox)
        candidates = []
        for track in people:
            expanded = expand_box(track.bbox, expansion_ratio)
            if not point_in_box(center, expanded):
                continue
            candidates.append((_normalised_center_distance(center, track.bbox), track.track_id))
        if candidates:
            _, track_id = min(candidates)
            result[track_id].append(detection)
    return result


def associate_faces(tracks: Iterable[TrackedPerson], face_boxes: Iterable[BBox]) -> Dict[int, BBox]:
    faces = [tuple(float(value) for value in box) for box in face_boxes]
    result: Dict[int, BBox] = {}
    for track in tracks:
        inside = [face for face in faces if point_in_box(bbox_centroid(face), expand_box(track.bbox, 0.05))]
        if inside:
            result[track.track_id] = max(inside, key=lambda box: box_area(box))
    return result


def estimated_face_box(person_bbox: BBox) -> BBox:
    x1, y1, x2, y2 = person_bbox
    width = x2 - x1
    height = y2 - y1
    return (x1 + 0.22 * width, y1, x2 - 0.22 * width, y1 + 0.32 * height)


def near_face(target_bbox: BBox, face_bbox: BBox, max_distance_ratio: float) -> Tuple[bool, float]:
    face_center = bbox_centroid(face_bbox)
    target_center = bbox_centroid(target_bbox)
    face_scale = max(1.0, math.hypot(face_bbox[2] - face_bbox[0], face_bbox[3] - face_bbox[1]))
    distance_ratio = math.dist(face_center, target_center) / face_scale
    overlap = intersection_over_target(target_bbox, expand_box(face_bbox, 0.35))
    matched = overlap > 0.05 or distance_ratio <= max_distance_ratio
    score = max(overlap, max(0.0, 1.0 - distance_ratio / max(0.01, max_distance_ratio)))
    return matched, min(1.0, score)


def phone_below_face(phone_bbox: BBox, face_bbox: BBox, person_bbox: BBox) -> Tuple[bool, float]:
    px, py = bbox_centroid(phone_bbox)
    fx, fy = bbox_centroid(face_bbox)
    x1, y1, x2, y2 = person_bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    inside_upper_body = x1 <= px <= x2 and y1 + 0.18 * height <= py <= y1 + 0.78 * height
    below = py > fy + 0.08 * height
    horizontal = abs(px - fx) / width
    score = max(0.0, 1.0 - horizontal)
    return bool(inside_upper_body and below), score


def phone_aims_at_roi(
    phone_bbox: BBox,
    person_bbox: BBox,
    roi: Rect,
    max_angle_deg: float,
) -> Tuple[bool, float]:
    body = bbox_centroid(person_bbox)
    phone = bbox_centroid(phone_bbox)
    roi_center = bbox_centroid(roi)
    if point_in_box(body, roi):
        return False, 0.0
    person_height = max(1.0, person_bbox[3] - person_bbox[1])
    raised = phone[1] <= person_bbox[1] + 0.58 * person_height
    phone_vector = (phone[0] - body[0], phone[1] - body[1])
    roi_vector = (roi_center[0] - body[0], roi_center[1] - body[1])
    phone_norm = math.hypot(*phone_vector)
    roi_norm = math.hypot(*roi_vector)
    if not raised or phone_norm < 1.0 or roi_norm < 1.0:
        return False, 0.0
    cosine = max(-1.0, min(1.0, _dot(phone_vector, roi_vector) / (phone_norm * roi_norm)))
    angle = math.degrees(math.acos(cosine))
    aligned = angle <= max_angle_deg
    score = max(0.0, 1.0 - angle / max(1.0, max_angle_deg))
    return aligned, score


def resolve_rois(configured: Sequence[dict], frame_shape: Sequence[int]) -> List[dict]:
    height, width = int(frame_shape[0]), int(frame_shape[1])
    result = []
    for index, item in enumerate(configured or []):
        if not isinstance(item, dict) or not bool(item.get("enabled", True)):
            continue
        scale_x = width if bool(item.get("normalized", True)) else 1.0
        scale_y = height if bool(item.get("normalized", True)) else 1.0
        try:
            box = (
                float(item["x1"]) * scale_x,
                float(item["y1"]) * scale_y,
                float(item["x2"]) * scale_x,
                float(item["y2"]) * scale_y,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        result.append({"id": str(item.get("id") or f"roi-{index + 1}"), "bbox": box})
    return result


def labels_for_group(config: dict, name: str, fallback: Iterable[str]) -> set[str]:
    value = config.get(name, fallback)
    if isinstance(value, str):
        value = [value]
    return {normalise_label(item) for item in value}


def highest_confidence(items: Iterable[Detection]) -> Optional[Detection]:
    values = list(items)
    return max(values, key=lambda item: item.confidence) if values else None


def expand_box(box: BBox, ratio: float) -> BBox:
    x1, y1, x2, y2 = box
    dx = max(0.0, x2 - x1) * ratio
    dy = max(0.0, y2 - y1) * ratio
    return (x1 - dx, y1 - dy, x2 + dx, y2 + dy)


def point_in_box(point: Tuple[float, float], box: BBox) -> bool:
    return box[0] <= point[0] <= box[2] and box[1] <= point[1] <= box[3]


def box_area(box: BBox) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_over_target(target: BBox, region: BBox) -> float:
    x1, y1 = max(target[0], region[0]), max(target[1], region[1])
    x2, y2 = min(target[2], region[2]), min(target[3], region[3])
    intersection = box_area((x1, y1, x2, y2))
    return intersection / max(1.0, box_area(target))


def normalise_label(value: object) -> str:
    return str(value).strip().lower().replace("_", " ")


def _normalised_center_distance(point: Tuple[float, float], box: BBox) -> float:
    center = bbox_centroid(box)
    diagonal = max(1.0, math.hypot(box[2] - box[0], box[3] - box[1]))
    return math.dist(point, center) / diagonal


def _dot(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]
