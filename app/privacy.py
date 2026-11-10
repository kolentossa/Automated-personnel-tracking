"""Web 显示前的隐私处理：异步 RetinaFace 检测 + 光流跟随 + 头部兜底。

检测器隔帧工作；光流把最近人脸框跟到当前帧；没有可靠人脸框的人员使用
头部区域兜底。输出副本用于 Web 显示，不改动模型使用的原始图像。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from itertools import product
from math import ceil
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from app.config import project_path

Box = Tuple[int, int, int, int]


@dataclass
class _FlowFace:
    box: Box
    points: np.ndarray


def create_privacy(config: Dict[str, Any]) -> SimpleNamespace:
    """读取隐私设置，准备线程共享的数据，再尝试加载人脸模型。"""
    config = dict(config or {})
    privacy = SimpleNamespace(
        config=config,
        # 遮挡和检测频率。
        enabled=bool(config.get("face_mosaic_enabled", True)),
        head_fallback_enabled=bool(config.get("head_fallback_enabled", True)),
        mosaic_strength=int(config.get("mosaic_strength") or 14),
        mosaic_padding=float(config.get("face_mosaic_padding") or 0.25),
        detect_every_n_frames=max(1, int(config.get("face_detect_every_n_frames") or 5)),
        max_result_age_ms=max(100.0, float(config.get("face_result_max_age_ms") or 1000.0)),
        # 两次人脸检测之间，用光流跟随脸的位置。
        tracking_enabled=bool(config.get("face_tracking_enabled", True)),
        tracking_min_points=max(3, int(config.get("face_tracking_min_points") or 4)),
        tracking_max_motion=max(8.0, float(config.get("face_tracking_max_motion_px") or 96.0)),
        tracking_win_size=_odd(max(15, int(config.get("face_tracking_win_size") or 31))),
        tracking_max_level=max(0, int(config.get("face_tracking_max_level") or 3)),
        detector_name="disabled",
        detector_available=False,
        warning="",
        _detector=None,
        # 主线程提交最新画面；后台线程写回最新人脸框。
        _lock=threading.RLock(),
        _wake_event=threading.Event(),
        _stop_event=threading.Event(),
        _thread=None,
        _pending_frame=None,
        _latest_boxes=[],
        _latest_gray=None,
        _latest_at=0.0,
        _latest_generation=0,
        _consumed_generation=0,
        # 当前光流位置及状态接口显示的数据。
        _flow_gray=None,
        _flow_faces=[],
        _frame_index=0,
        _flow_frame_index=0,
        _face_detection_ms=0.0,
        _face_tracking_ms=0.0,
        _faces_detected=0,
        _fallback_regions=0,
    )
    privacy._mode = "disabled" if not privacy.enabled else "initializing"
    _initialise_detector(privacy)
    return privacy


def mask_faces(
    privacy: SimpleNamespace,
    frame: np.ndarray,
    person_boxes: Optional[Iterable[Sequence[float]]] = None,
) -> np.ndarray:
    """提交待检测画面，读取近期结果并跟随位置，再遮挡人脸及遗漏的头部。"""
    people = list(person_boxes or [])
    if not privacy.enabled:
        return frame.copy()

    privacy._frame_index += 1
    if people and privacy.detector_available and (privacy._frame_index - 1) % privacy.detect_every_n_frames == 0:
        with privacy._lock:
            privacy._pending_frame = frame.copy()
        privacy._wake_event.set()

    now = time.monotonic()
    with privacy._lock:
        age_ms = (now - privacy._latest_at) * 1000.0 if privacy._latest_at else float("inf")
        generation = privacy._latest_generation
        detected_boxes = list(privacy._latest_boxes)
        detection_gray = privacy._latest_gray

    tracking_started = time.monotonic()
    faces = _tracked_boxes(privacy, frame, detected_boxes, detection_gray, generation, age_ms)
    privacy._face_tracking_ms = _ms(time.monotonic() - tracking_started)
    expanded_faces = [_expand_box(frame, box, privacy.mosaic_padding) for box in faces]
    expanded_faces = [box for box in expanded_faces if box is not None]
    fallback_boxes = _head_fallback_boxes(frame, people, expanded_faces) if privacy.head_fallback_enabled else []
    # 只修改显示副本；先遮挡人脸，再遮挡漏检人员的头部。
    result = frame.copy()
    for box in expanded_faces + fallback_boxes:
        _mosaic_region_in_place(result, box, privacy.mosaic_strength)

    with privacy._lock:
        privacy._fallback_regions = len(fallback_boxes)
        if expanded_faces:
            privacy._mode = "face-tracked" if privacy.tracking_enabled else "face-detected"
        elif fallback_boxes:
            privacy._mode = "head-fallback"
        elif privacy.detector_available:
            privacy._mode = "ready"
    return result


def privacy_snapshot(privacy: SimpleNamespace) -> Dict[str, Any]:
    with privacy._lock:
        age_ms = (time.monotonic() - privacy._latest_at) * 1000.0 if privacy._latest_at else 0.0
        return {
            "face_detector": privacy.detector_name,
            "face_detector_available": privacy.detector_available,
            "face_privacy_mode": privacy._mode,
            "faces_detected": privacy._faces_detected,
            "face_tracked_boxes": len(privacy._flow_faces),
            "face_tracking_points": sum(len(face.points) for face in privacy._flow_faces),
            "face_fallback_regions": privacy._fallback_regions,
            "face_detection_interval_frames": privacy.detect_every_n_frames,
            "face_detection_ms": round(privacy._face_detection_ms, 1),
            "face_tracking_ms": round(privacy._face_tracking_ms, 1),
            "face_result_age_ms": round(age_ms, 1),
            "face_detector_error": privacy.warning,
        }


def start_privacy(privacy: SimpleNamespace) -> None:
    if not privacy.detector_available or privacy._thread and privacy._thread.is_alive():
        return
    privacy._stop_event.clear()
    privacy._thread = threading.Thread(
        target=_worker_loop, args=(privacy,), name="face-privacy-inference", daemon=True,
    )
    privacy._thread.start()


def stop_privacy(privacy: SimpleNamespace) -> None:
    privacy._stop_event.set()
    privacy._wake_event.set()
    if privacy._thread and privacy._thread.is_alive():
        privacy._thread.join(timeout=3.0)


def _worker_loop(privacy: SimpleNamespace) -> None:
    while not privacy._stop_event.is_set():
        privacy._wake_event.wait(timeout=0.2)
        privacy._wake_event.clear()
        if privacy._stop_event.is_set():
            break
        with privacy._lock:
            # 只取最新待处理帧；慢检测不会让旧画面排成越来越长的队列。
            frame = privacy._pending_frame
            privacy._pending_frame = None
        if frame is None:
            continue
        try:
            boxes = detect_faces(privacy._detector, frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            with privacy._lock:
                privacy._latest_boxes = boxes
                privacy._latest_gray = gray
                privacy._latest_at = time.monotonic()
                privacy._latest_generation += 1
                privacy._faces_detected = len(boxes)
                privacy._face_detection_ms = privacy._detector.last_inference_ms
                privacy.warning = ""
        except Exception as exc:
            with privacy._lock:
                privacy.warning = str(exc)


def _tracked_boxes(
    privacy: SimpleNamespace,
    frame: np.ndarray,
    detected_boxes: List[Box],
    detection_gray: Optional[np.ndarray],
    generation: int,
    age_ms: float,
) -> List[Box]:
    if not privacy.tracking_enabled:
        return detected_boxes if age_ms <= privacy.max_result_age_ms else []
    if age_ms > privacy.max_result_age_ms:
        # 过期结果不能继续代表当前人脸；清空光流状态，由头部兜底接手。
        privacy._flow_faces = []
        privacy._flow_gray = None
        return []

    has_new_detection = generation != privacy._consumed_generation
    if not has_new_detection and not privacy._flow_faces:
        return []
    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    privacy._flow_frame_index += 1
    previous_gray = privacy._flow_gray
    faces = privacy._flow_faces
    reseed = privacy._flow_frame_index % 5 == 0
    if has_new_detection:
        # 后台结果来自较早的帧，先把它跟随到当前帧再使用。
        privacy._consumed_generation = generation
        previous_gray = detection_gray
        faces = []
        if previous_gray is not None and previous_gray.shape == current_gray.shape:
            faces = [
                _FlowFace(box, _seed_flow_points(previous_gray, box))
                for box in detected_boxes
            ]
        # 新检测必须在当前帧重选角点，作为后续跟随的起点。
        reseed = True

    # 新检测和已有轨迹都走这一个循环；只保留跟随成功的人脸。
    if previous_gray is not None:
        advanced = []
        for face in faces:
            tracked = _track_flow_face(
                previous_gray,
                current_gray,
                face,
                frame,
                privacy.tracking_min_points,
                privacy.tracking_win_size,
                privacy.tracking_max_level,
                privacy.tracking_max_motion,
                reseed=reseed or len(face.points) < 12,
            )
            if tracked is not None:
                advanced.append(tracked)
        faces = advanced
    privacy._flow_faces = faces
    privacy._flow_gray = current_gray
    return [face.box for face in faces]


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
    """LK 光流移动角点；剔除失败点后估计框的位置，必要时补充角点。"""
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
        # 角点充足时估计平移、旋转和缩放；RANSAC 忽略少数跟错的点。
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
        # 仿射估计失败或缩放异常时，只采用角点位移的中位数。
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
    clipped = _clip_box(gray, box)
    if clipped is None:
        return np.empty((0, 1, 2), dtype=np.float32)
    x1, y1, x2, y2 = clipped
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
    # 纹理不足时在框内均匀取 4×4 个点，仍需通过下一步光流质量检查。
    sample_x = np.linspace(x1 + (x2 - x1) * 0.2, x2 - (x2 - x1) * 0.2, 4)
    sample_y = np.linspace(y1 + (y2 - y1) * 0.2, y2 - (y2 - y1) * 0.2, 4)
    return np.asarray([[[x, y]] for y in sample_y for x in sample_x], dtype=np.float32)


def _mosaic_region_in_place(frame: np.ndarray, box: Box, mosaic_strength: int) -> None:
    """遮挡已裁到图像内的非空整数框；调用方负责裁框。"""
    x1, y1, x2, y2 = box
    region = frame[y1:y2, x1:x2]
    strength = max(4, int(mosaic_strength))
    small_width = max(1, region.shape[1] // strength)
    small_height = max(1, region.shape[0] // strength)
    small = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_LINEAR)
    mosaic = cv2.resize(small, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
    frame[y1:y2, x1:x2] = mosaic


def _head_fallback_boxes(
    frame: np.ndarray,
    person_boxes: Optional[Iterable[Sequence[float]]],
    face_boxes: Optional[Iterable[Sequence[float]]] = None,
) -> List[Box]:
    """人员框内没有已知人脸时，遮挡其顶部 32%，补上漏检人脸。"""
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


def _initialise_detector(privacy: SimpleNamespace) -> None:
    if not privacy.enabled:
        return
    detector_type = str(privacy.config.get("face_detector") or "retinaface-onnx").lower()
    if detector_type not in {"retinaface", "retinaface-onnx"}:
        privacy.warning = f"Unsupported face detector: {detector_type}"
        privacy._mode = "head-fallback" if privacy.head_fallback_enabled else "unavailable"
        return
    try:
        privacy._detector = create_face_detector(
            project_path(str(privacy.config.get("face_model_path") or "models/RetinaFace_mobile320.onnx")),
            input_size=int(privacy.config.get("face_input_size") or 320),
            confidence_threshold=float(privacy.config.get("face_confidence_threshold") or 0.6),
            nms_threshold=float(privacy.config.get("face_nms_threshold") or 0.4),
            num_threads=int(privacy.config.get("face_detector_threads") or 1),
        )
        privacy.detector_name = "retinaface-mobile320-onnx"
        privacy.detector_available = True
        privacy._mode = "ready"
    except Exception as exc:
        privacy.warning = str(exc)
        privacy.detector_name = "retinaface-unavailable"
        privacy._mode = "head-fallback" if privacy.head_fallback_enabled else "unavailable"


def create_face_detector(
    model_path: Path,
    *,
    input_size: int = 320,
    confidence_threshold: float = 0.6,
    nms_threshold: float = 0.4,
    num_threads: int = 2,
) -> SimpleNamespace:
    """在 CPU 上加载 RetinaFace，保存会话、阈值和模型对应的先验框。"""
    try:
        import onnxruntime as ort
    except Exception as exc:  # pragma: no cover - target dependency
        raise RuntimeError("onnxruntime is not available") from exc

    detector = SimpleNamespace(model_path=Path(model_path))
    if not detector.model_path.is_file():
        raise FileNotFoundError(f"RetinaFace model not found: {detector.model_path}")
    detector.input_size = int(input_size)
    detector.confidence_threshold = float(confidence_threshold)
    detector.nms_threshold = float(nms_threshold)
    options = ort.SessionOptions()
    options.intra_op_num_threads = max(1, int(num_threads))
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    detector._session = ort.InferenceSession(
        str(detector.model_path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    detector._input_name = detector._session.get_inputs()[0].name
    detector._priors = _prior_boxes(detector.input_size)
    detector.last_inference_ms = 0.0
    return detector


def detect_faces(detector: SimpleNamespace, frame: np.ndarray) -> List[Box]:
    # 等比例缩放并补边，再按模型要求减去 BGR 均值、转成 NCHW 张量。
    input_image, ratio, offset_x, offset_y = _letterbox(frame, detector.input_size, 114)
    tensor = input_image.astype(np.float32)
    tensor -= np.asarray((104.0, 117.0, 123.0), dtype=np.float32)
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

    started = time.monotonic()
    locations, confidence, _ = detector._session.run(None, {detector._input_name: tensor})
    detector.last_inference_ms = _ms(time.monotonic() - started)

    # 模型给出相对先验框的偏移；解码、筛选后映射回原图，再去掉重叠框。
    boxes = _decode_boxes(locations.squeeze(0), detector._priors)
    boxes *= float(detector.input_size)
    scores = confidence.squeeze(0)[:, 1]
    selected = np.where(scores >= detector.confidence_threshold)[0]
    if selected.size == 0:
        return []

    boxes = boxes[selected]
    scores = scores[selected]
    order = scores.argsort()[::-1][:500]
    boxes = boxes[order]
    scores = scores[order]
    boxes[:, 0::2] = np.clip((boxes[:, 0::2] - offset_x) / ratio, 0, frame.shape[1])
    boxes[:, 1::2] = np.clip((boxes[:, 1::2] - offset_y) / ratio, 0, frame.shape[0])
    keep = _nms(boxes, scores, detector.nms_threshold)
    return [tuple(int(round(value)) for value in boxes[index]) for index in keep]


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


def _prior_boxes(image_size: int) -> np.ndarray:
    """按 RetinaFace 三层输出的步长和尺寸生成先验框，顺序须与模型输出一致。"""
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
    """依置信度保留框，删除与已选框交并比过高的重复检测。"""
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


def _odd(value: int) -> int:
    return value if value % 2 else value + 1


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
