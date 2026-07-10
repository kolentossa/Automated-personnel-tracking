"""Asynchronous face detection and mosaic processing before Web streaming."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from math import ceil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config import project_path

Box = Tuple[int, int, int, int]


@dataclass
class _FlowFace:
    box: Box
    points: np.ndarray


class RetinaFaceONNXDetector:
    """Rockchip model-zoo RetinaFace Mobile320 running on ONNX Runtime CPU."""

    def __init__(
        self,
        model_path: Path,
        *,
        input_size: int = 320,
        confidence_threshold: float = 0.6,
        nms_threshold: float = 0.4,
        num_threads: int = 2,
    ) -> None:
        try:
            import onnxruntime as ort
        except Exception as exc:  # pragma: no cover - target dependency
            raise RuntimeError("onnxruntime is not available") from exc

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"RetinaFace model not found: {self.model_path}")
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, int(num_threads))
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(self.model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._priors = _prior_boxes(self.input_size)
        self.last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> List[Box]:
        input_image, ratio, offset_x, offset_y = _letterbox(frame, self.input_size, 114)
        tensor = input_image.astype(np.float32)
        tensor -= np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        started = time.monotonic()
        loc, confidence, _ = self._session.run(None, {self._input_name: tensor})
        self.last_inference_ms = _ms(time.monotonic() - started)

        boxes = _decode_boxes(loc.squeeze(0), self._priors)
        boxes *= float(self.input_size)
        scores = confidence.squeeze(0)[:, 1]
        selected = np.where(scores >= self.confidence_threshold)[0]
        if selected.size == 0:
            return []

        boxes = boxes[selected]
        scores = scores[selected]
        order = scores.argsort()[::-1][:500]
        boxes = boxes[order]
        scores = scores[order]
        boxes[:, 0::2] = np.clip((boxes[:, 0::2] - offset_x) / ratio, 0, frame.shape[1])
        boxes[:, 1::2] = np.clip((boxes[:, 1::2] - offset_y) / ratio, 0, frame.shape[0])
        keep = _nms(boxes, scores, self.nms_threshold)
        return [tuple(int(round(value)) for value in boxes[index]) for index in keep]


class FaceMosaicProcessor:
    """Detect faces asynchronously and propagate boxes with per-frame optical flow."""

    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = dict(config or {})
        self.enabled = bool(self.config.get("face_mosaic_enabled", True))
        self.head_fallback_enabled = bool(self.config.get("head_fallback_enabled", True))
        self.mosaic_strength = int(self.config.get("mosaic_strength") or 14)
        self.mosaic_padding = float(self.config.get("face_mosaic_padding") or 0.25)
        self.detect_every_n_frames = max(1, int(self.config.get("face_detect_every_n_frames") or 5))
        self.max_result_age_ms = max(100.0, float(self.config.get("face_result_max_age_ms") or 1000.0))
        self.tracking_enabled = bool(self.config.get("face_tracking_enabled", True))
        self.tracking_min_points = max(3, int(self.config.get("face_tracking_min_points") or 4))
        self.tracking_max_motion = max(8.0, float(self.config.get("face_tracking_max_motion_px") or 96.0))
        self.tracking_win_size = _odd(max(15, int(self.config.get("face_tracking_win_size") or 31)))
        self.tracking_max_level = max(0, int(self.config.get("face_tracking_max_level") or 3))
        self.detector_name = "disabled"
        self.detector_available = False
        self.warning = ""
        self._detector: Optional[RetinaFaceONNXDetector] = None
        self._lock = threading.RLock()
        self._wake_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_frame: Optional[np.ndarray] = None
        self._latest_boxes: List[Box] = []
        self._latest_gray: Optional[np.ndarray] = None
        self._latest_at = 0.0
        self._latest_generation = 0
        self._consumed_generation = 0
        self._flow_gray: Optional[np.ndarray] = None
        self._flow_faces: List[_FlowFace] = []
        self._frame_index = 0
        self._flow_frame_index = 0
        self._face_detection_ms = 0.0
        self._face_tracking_ms = 0.0
        self._faces_detected = 0
        self._fallback_regions = 0
        self._mode = "disabled" if not self.enabled else "initializing"
        self._initialise_detector()

    def _initialise_detector(self) -> None:
        if not self.enabled:
            return
        detector_type = str(self.config.get("face_detector") or "retinaface-onnx").lower()
        if detector_type not in {"retinaface", "retinaface-onnx"}:
            self.warning = f"Unsupported face detector: {detector_type}"
            self._mode = "head-fallback" if self.head_fallback_enabled else "unavailable"
            return
        try:
            self._detector = RetinaFaceONNXDetector(
                project_path(str(self.config.get("face_model_path") or "models/RetinaFace_mobile320.onnx")),
                input_size=int(self.config.get("face_input_size") or 320),
                confidence_threshold=float(self.config.get("face_confidence_threshold") or 0.6),
                nms_threshold=float(self.config.get("face_nms_threshold") or 0.4),
                num_threads=int(self.config.get("face_detector_threads") or 1),
            )
            self.detector_name = "retinaface-mobile320-onnx"
            self.detector_available = True
            self._mode = "ready"
        except Exception as exc:
            self.warning = str(exc)
            self.detector_name = "retinaface-unavailable"
            self._mode = "head-fallback" if self.head_fallback_enabled else "unavailable"

    def start(self) -> None:
        if not self.detector_available or self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker_loop, name="face-privacy-inference", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._wake_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3.0)

    def process(self, frame: np.ndarray, person_boxes: Optional[Iterable[Sequence[float]]] = None) -> np.ndarray:
        people = list(person_boxes or [])
        if not self.enabled:
            return frame.copy()

        self._frame_index += 1
        if people and self.detector_available and (self._frame_index - 1) % self.detect_every_n_frames == 0:
            with self._lock:
                self._pending_frame = frame.copy()
            self._wake_event.set()

        now = time.monotonic()
        with self._lock:
            age_ms = (now - self._latest_at) * 1000.0 if self._latest_at else float("inf")
            generation = self._latest_generation
            detected_boxes = list(self._latest_boxes)
            detection_gray = self._latest_gray

        tracking_started = time.monotonic()
        faces = self._tracked_boxes(frame, detected_boxes, detection_gray, generation, age_ms)
        self._face_tracking_ms = _ms(time.monotonic() - tracking_started)
        expanded_faces = [_expand_box(frame, box, self.mosaic_padding) for box in faces]
        expanded_faces = [box for box in expanded_faces if box is not None]
        fallback_boxes = _head_fallback_boxes(frame, people, expanded_faces) if self.head_fallback_enabled else []
        result = apply_privacy_mosaic(
            frame,
            person_boxes=people,
            face_mosaic_enabled=True,
            head_fallback_enabled=False,
            mosaic_strength=self.mosaic_strength,
            face_boxes=expanded_faces,
        )
        for box in fallback_boxes:
            _mosaic_region_in_place(result, box, self.mosaic_strength)

        with self._lock:
            self._fallback_regions = len(fallback_boxes)
            if expanded_faces:
                self._mode = "face-tracked" if self.tracking_enabled else "face-detected"
            elif fallback_boxes:
                self._mode = "head-fallback"
            elif self.detector_available:
                self._mode = "ready"
        return result

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            age_ms = (time.monotonic() - self._latest_at) * 1000.0 if self._latest_at else 0.0
            return {
                "face_detector": self.detector_name,
                "face_detector_available": self.detector_available,
                "face_privacy_mode": self._mode,
                "faces_detected": self._faces_detected,
                "face_tracked_boxes": len(self._flow_faces),
                "face_tracking_points": sum(len(face.points) for face in self._flow_faces),
                "face_fallback_regions": self._fallback_regions,
                "face_detection_interval_frames": self.detect_every_n_frames,
                "face_detection_ms": round(self._face_detection_ms, 1),
                "face_tracking_ms": round(self._face_tracking_ms, 1),
                "face_result_age_ms": round(age_ms, 1),
                "face_detector_error": self.warning,
            }

    def _tracked_boxes(
        self,
        frame: np.ndarray,
        detected_boxes: List[Box],
        detection_gray: Optional[np.ndarray],
        generation: int,
        age_ms: float,
    ) -> List[Box]:
        if not self.tracking_enabled:
            return detected_boxes if age_ms <= self.max_result_age_ms else []
        if age_ms > self.max_result_age_ms:
            self._flow_faces = []
            self._flow_gray = None
            return []

        has_new_detection = generation != self._consumed_generation
        if not has_new_detection and not self._flow_faces:
            return []
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self._flow_frame_index += 1
        if has_new_detection:
            self._consumed_generation = generation
            self._flow_faces = _adopt_face_detections(
                detection_gray,
                current_gray,
                detected_boxes,
                frame,
                self.tracking_min_points,
                self.tracking_win_size,
                self.tracking_max_level,
                self.tracking_max_motion,
            )
        elif self._flow_gray is not None:
            self._flow_faces = _advance_face_tracks(
                self._flow_gray,
                current_gray,
                self._flow_faces,
                frame,
                self.tracking_min_points,
                self.tracking_win_size,
                self.tracking_max_level,
                self.tracking_max_motion,
                self._flow_frame_index,
            )
        self._flow_gray = current_gray
        return [face.box for face in self._flow_faces]

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            self._wake_event.wait(timeout=0.2)
            self._wake_event.clear()
            if self._stop_event.is_set():
                break
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None
            if frame is None or self._detector is None:
                continue
            try:
                boxes = self._detector.detect(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                with self._lock:
                    self._latest_boxes = boxes
                    self._latest_gray = gray
                    self._latest_at = time.monotonic()
                    self._latest_generation += 1
                    self._faces_detected = len(boxes)
                    self._face_detection_ms = self._detector.last_inference_ms
                    self.warning = ""
            except Exception as exc:
                with self._lock:
                    self.warning = str(exc)


def _adopt_face_detections(
    source_gray: Optional[np.ndarray],
    current_gray: np.ndarray,
    boxes: Iterable[Box],
    frame: np.ndarray,
    min_points: int,
    win_size: int,
    max_level: int,
    max_motion: float,
) -> List[_FlowFace]:
    if source_gray is None or source_gray.shape != current_gray.shape:
        return []
    tracks: List[_FlowFace] = []
    for box in boxes:
        points = _seed_flow_points(source_gray, box)
        tracked = _track_flow_face(
            source_gray,
            current_gray,
            _FlowFace(box, points),
            frame,
            min_points,
            win_size,
            max_level,
            max_motion,
            reseed=True,
        )
        if tracked is not None:
            tracks.append(tracked)
    return tracks


def _advance_face_tracks(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    tracks: Iterable[_FlowFace],
    frame: np.ndarray,
    min_points: int,
    win_size: int,
    max_level: int,
    max_motion: float,
    frame_index: int,
) -> List[_FlowFace]:
    advanced: List[_FlowFace] = []
    for face in tracks:
        tracked = _track_flow_face(
            previous_gray,
            current_gray,
            face,
            frame,
            min_points,
            win_size,
            max_level,
            max_motion,
            reseed=frame_index % 5 == 0 or len(face.points) < 12,
        )
        if tracked is not None:
            advanced.append(tracked)
    return advanced


def _track_flow_face(
    previous_gray: np.ndarray,
    current_gray: np.ndarray,
    face: _FlowFace,
    frame: np.ndarray,
    min_points: int,
    win_size: int,
    max_level: int,
    max_motion: float,
    *,
    reseed: bool,
) -> Optional[_FlowFace]:
    points = face.points if len(face.points) >= min_points else _seed_flow_points(previous_gray, face.box)
    if len(points) < min_points:
        return None
    next_points, status, errors = cv2.calcOpticalFlowPyrLK(
        previous_gray,
        current_gray,
        points,
        None,
        winSize=(win_size, win_size),
        maxLevel=max_level,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
    )
    if next_points is None or status is None:
        return None
    valid = status.reshape(-1).astype(bool)
    if errors is not None:
        valid &= np.isfinite(errors.reshape(-1)) & (errors.reshape(-1) < 50.0)
    previous_good = points.reshape(-1, 2)[valid]
    current_good = next_points.reshape(-1, 2)[valid]
    if len(current_good) < min_points:
        return None

    moved_box = _flow_box(face.box, previous_good, current_good, frame, max_motion)
    if moved_box is None:
        return None
    tracked_points = current_good.reshape(-1, 1, 2).astype(np.float32)
    if reseed:
        seeded = _seed_flow_points(current_gray, moved_box)
        if len(seeded) >= min_points:
            tracked_points = seeded
    return _FlowFace(moved_box, tracked_points)


def _flow_box(
    box: Box,
    previous_points: np.ndarray,
    current_points: np.ndarray,
    frame: np.ndarray,
    max_motion: float,
) -> Optional[Box]:
    x1, y1, x2, y2 = [float(value) for value in box]
    old_center = np.asarray(((x1 + x2) / 2.0, (y1 + y2) / 2.0), dtype=np.float32)
    transformed: Optional[np.ndarray] = None
    if len(current_points) >= 4:
        matrix, _ = cv2.estimateAffinePartial2D(
            previous_points,
            current_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=3.0,
            maxIters=50,
            confidence=0.95,
            refineIters=5,
        )
        if matrix is not None:
            scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
            if 0.7 <= scale <= 1.4:
                corners = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
                transformed = cv2.transform(corners[None, ...], matrix)[0]
    if transformed is None:
        delta = np.median(current_points - previous_points, axis=0)
        transformed = np.asarray(
            ((x1 + delta[0], y1 + delta[1]), (x2 + delta[0], y2 + delta[1])),
            dtype=np.float32,
        )

    new_x1 = float(np.min(transformed[:, 0]))
    new_y1 = float(np.min(transformed[:, 1]))
    new_x2 = float(np.max(transformed[:, 0]))
    new_y2 = float(np.max(transformed[:, 1]))
    new_center = np.asarray(((new_x1 + new_x2) / 2.0, (new_y1 + new_y2) / 2.0), dtype=np.float32)
    if float(np.linalg.norm(new_center - old_center)) > max_motion:
        return None
    old_width, old_height = max(1.0, x2 - x1), max(1.0, y2 - y1)
    new_width, new_height = new_x2 - new_x1, new_y2 - new_y1
    if not (0.65 * old_width <= new_width <= 1.5 * old_width):
        return None
    if not (0.65 * old_height <= new_height <= 1.5 * old_height):
        return None
    return _clip_box(frame, (new_x1, new_y1, new_x2, new_y2))


def _seed_flow_points(gray: np.ndarray, box: Sequence[float]) -> np.ndarray:
    height, width = gray.shape[:2]
    x1, y1, x2, y2 = [int(round(float(value))) for value in box]
    x1, y1 = max(0, min(width - 1, x1)), max(0, min(height - 1, y1))
    x2, y2 = max(0, min(width, x2)), max(0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 1, 2), dtype=np.float32)
    points = cv2.goodFeaturesToTrack(
        gray[y1:y2, x1:x2],
        maxCorners=40,
        qualityLevel=0.01,
        minDistance=4,
        blockSize=3,
    )
    if points is not None and len(points) >= 4:
        points = points.astype(np.float32)
        points[:, 0, 0] += x1
        points[:, 0, 1] += y1
        return points
    xs = np.linspace(x1 + (x2 - x1) * 0.2, x2 - (x2 - x1) * 0.2, 4)
    ys = np.linspace(y1 + (y2 - y1) * 0.2, y2 - (y2 - y1) * 0.2, 4)
    return np.asarray([[[x, y]] for y in ys for x in xs], dtype=np.float32)


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def apply_privacy_mosaic(
    frame: np.ndarray,
    person_boxes: Optional[Iterable[Sequence[float]]] = None,
    face_mosaic_enabled: bool = True,
    head_fallback_enabled: bool = True,
    mosaic_strength: int = 14,
    face_boxes: Optional[Iterable[Sequence[float]]] = None,
) -> np.ndarray:
    """Return an anonymised copy; explicit face boxes take priority."""

    result = frame.copy()
    if not face_mosaic_enabled:
        return result
    detected_faces = list(face_boxes) if face_boxes is not None else detect_faces(result)
    for face_box in detected_faces:
        _mosaic_region_in_place(result, face_box, mosaic_strength)
    if head_fallback_enabled:
        for head_box in _head_fallback_boxes(result, person_boxes, detected_faces):
            _mosaic_region_in_place(result, head_box, mosaic_strength)
    return result


def mosaic_region(frame: np.ndarray, box: Sequence[float], mosaic_strength: int = 14) -> np.ndarray:
    result = frame.copy()
    _mosaic_region_in_place(result, box, mosaic_strength)
    return result


def _mosaic_region_in_place(frame: np.ndarray, box: Sequence[float], mosaic_strength: int) -> None:
    clipped = _clip_box(frame, box)
    if clipped is None:
        return
    x1, y1, x2, y2 = clipped
    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return
    strength = max(4, int(mosaic_strength))
    small_w = max(1, roi.shape[1] // strength)
    small_h = max(1, roi.shape[0] // strength)
    small = cv2.resize(roi, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic


def detect_faces(frame: np.ndarray) -> List[Box]:
    """Legacy Haar path retained for callers without a configured processor."""

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


def _head_fallback_boxes(
    frame: np.ndarray,
    person_boxes: Optional[Iterable[Sequence[float]]],
    face_boxes: Optional[Iterable[Sequence[float]]] = None,
) -> List[Box]:
    people = list(person_boxes or [])
    faces = list(face_boxes or [])
    boxes: List[Box] = []
    for box in people:
        clipped = _clip_box(frame, box)
        if clipped is None or any(_box_center_inside(face, clipped) for face in faces):
            continue
        x1, y1, x2, y2 = clipped
        head_y2 = y1 + max(1, int((y2 - y1) * 0.32))
        boxes.append((x1, y1, x2, head_y2))
    return boxes


def _letterbox(frame: np.ndarray, size: int, color: int) -> Tuple[np.ndarray, float, int, int]:
    height, width = frame.shape[:2]
    ratio = min(size / float(width), size / float(height))
    resized_width = max(1, int(round(width * ratio)))
    resized_height = max(1, int(round(height * ratio)))
    resized = cv2.resize(frame, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), color, dtype=np.uint8)
    offset_x = (size - resized_width) // 2
    offset_y = (size - resized_height) // 2
    canvas[offset_y : offset_y + resized_height, offset_x : offset_x + resized_width] = resized
    return canvas, ratio, offset_x, offset_y


@lru_cache(maxsize=4)
def _prior_boxes(image_size: int) -> np.ndarray:
    anchors: List[float] = []
    min_sizes = ((16, 32), (64, 128), (256, 512))
    steps = (8, 16, 32)
    for sizes, step in zip(min_sizes, steps):
        feature = ceil(image_size / step)
        for row, column in product(range(feature), repeat=2):
            for minimum in sizes:
                anchors.extend(
                    (
                        (column + 0.5) * step / image_size,
                        (row + 0.5) * step / image_size,
                        minimum / image_size,
                        minimum / image_size,
                    )
                )
    return np.asarray(anchors, dtype=np.float32).reshape(-1, 4)


def _decode_boxes(locations: np.ndarray, priors: np.ndarray) -> np.ndarray:
    centers = priors[:, :2] + locations[:, :2] * 0.1 * priors[:, 2:]
    sizes = priors[:, 2:] * np.exp(locations[:, 2:] * 0.2)
    return np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1 + 1.0) * np.maximum(0.0, y2 - y1 + 1.0)
    order = scores.argsort()[::-1]
    keep: List[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[index], x1[order[1:]])
        yy1 = np.maximum(y1[index], y1[order[1:]])
        xx2 = np.minimum(x2[index], x2[order[1:]])
        yy2 = np.minimum(y2[index], y2[order[1:]])
        width = np.maximum(0.0, xx2 - xx1 + 1.0)
        height = np.maximum(0.0, yy2 - yy1 + 1.0)
        overlap = width * height
        iou = overlap / np.maximum(areas[index] + areas[order[1:]] - overlap, 1e-6)
        order = order[np.where(iou <= threshold)[0] + 1]
    return keep


def _expand_box(frame: np.ndarray, box: Sequence[float], padding: float) -> Optional[Box]:
    x1, y1, x2, y2 = [float(value) for value in box]
    pad_x = (x2 - x1) * max(0.0, padding)
    pad_y = (y2 - y1) * max(0.0, padding)
    return _clip_box(frame, (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y))


def _box_center_inside(inner: Sequence[float], outer: Sequence[float]) -> bool:
    ix1, iy1, ix2, iy2 = [float(value) for value in inner]
    ox1, oy1, ox2, oy2 = [float(value) for value in outer]
    center_x = (ix1 + ix2) / 2.0
    center_y = (iy1 + iy2) / 2.0
    return ox1 <= center_x <= ox2 and oy1 <= center_y <= oy2


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


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
