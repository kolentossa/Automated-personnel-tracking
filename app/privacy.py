"""Face/head mosaic privacy processing before Web streaming."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

Box = Tuple[int, int, int, int]


def apply_privacy_mosaic(
    frame: np.ndarray,
    person_boxes: Optional[Iterable[Sequence[float]]] = None,
    face_mosaic_enabled: bool = True,
    head_fallback_enabled: bool = True,
    mosaic_strength: int = 14,
) -> np.ndarray:
    """Return an anonymised copy of the frame before Web display."""

    result = frame.copy()
    if not face_mosaic_enabled:
        return result
    face_boxes = detect_faces(result)
    for face_box in face_boxes:
        result = mosaic_region(result, face_box, mosaic_strength)
    if not face_boxes and head_fallback_enabled:
        for head_box in _head_fallback_boxes(result, person_boxes):
            result = mosaic_region(result, head_box, mosaic_strength)
    return result


def mosaic_region(frame: np.ndarray, box: Sequence[float], mosaic_strength: int = 14) -> np.ndarray:
    result = frame.copy()
    clipped = _clip_box(result, box)
    if clipped is None:
        return result
    x1, y1, x2, y2 = clipped
    roi = result[y1:y2, x1:x2]
    if roi.size == 0:
        return result
    strength = max(4, int(mosaic_strength))
    small_w = max(1, roi.shape[1] // strength)
    small_h = max(1, roi.shape[0] // strength)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    result[y1:y2, x1:x2] = mosaic
    return result


def detect_faces(frame: np.ndarray) -> List[Box]:
    cascade = _face_cascade()
    if cascade is None:
        return []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24))
    return [(int(x), int(y), int(x + w), int(y + h)) for x, y, w, h in faces]


@lru_cache(maxsize=1)
def _face_cascade():
    candidates: List[Path] = []
    data = getattr(cv2, "data", None)
    if data is not None and getattr(data, "haarcascades", ""):
        candidates.append(Path(data.haarcascades) / "haarcascade_frontalface_default.xml")
    candidates.extend(
        [
            Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"),
            Path("/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml"),
            Path(__file__).resolve().parents[1] / "models" / "haarcascade_frontalface_default.xml",
        ]
    )
    for path in candidates:
        if not path.exists():
            continue
        cascade = cv2.CascadeClassifier(str(path))
        if not cascade.empty():
            return cascade
    return None


def _head_fallback_boxes(frame: np.ndarray, person_boxes: Optional[Iterable[Sequence[float]]]) -> List[Box]:
    if not person_boxes:
        return []
    boxes: List[Box] = []
    for box in person_boxes:
        clipped = _clip_box(frame, box)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        head_y2 = y1 + max(1, int((y2 - y1) * 0.32))
        boxes.append((x1, y1, x2, head_y2))
    return boxes


def _clip_box(frame: np.ndarray, box: Sequence[float]) -> Optional[Box]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2