"""把摄像头图像送入 CrowdHuman 模型，再把模型输出还原成人体框。"""

from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any, Sequence

import cv2
import numpy as np

from app.config import project_path
from app.types import Detection, Frame

MAX_CANDIDATES = 300
# 三组 anchor 是模型训练时使用的参考框尺寸，来源见模型元数据。
CROWDHUMAN_ANCHORS = (
    ((4.77734375, 5.26171875), (10.6484375, 12.203125), (18.359375, 22.09375)),
    ((19.453125, 51.6875), (34.59375, 43.1875), (40.25, 107.75)),
    ((68.875, 180.5), (110.75, 299.75), (230.5, 494.25)),
)


def load_detector(config: dict) -> SimpleNamespace:
    """保存模型设置并加载 NPU；加载失败时保存原因，检测函数返回空结果。"""
    detector = SimpleNamespace(
        model_path=str(config.get("model_path") or "").strip(),
        input_size=int(config.get("input_size") or 448),
        confidence_threshold=float(config.get("confidence_threshold") or 0.35),
        nms_threshold=float(config.get("nms_threshold") or 0.45),
        class_filter={str(item).lower() for item in (config.get("class_filter") or ["person"])},
        name="no-op-person-detector",
        npu_enabled=False,
        warning="",
        last_profile=_empty_profile(),
        rknn=None,
    )
    if not detector.model_path:
        _set_unavailable(detector, "No semantic person model configured.")
        return detector
    path = project_path(detector.model_path)
    if not path.exists():
        _set_unavailable(detector, f"Configured RKNN model not found: {path}")
        return detector
    try:
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:
            raise RuntimeError("rknnlite.api.RKNNLite is not available") from exc
        detector.rknn = RKNNLite()
        result = detector.rknn.load_rknn(str(path))
        if result != 0:
            raise RuntimeError(f"RKNN load_rknn failed with code {result}: {path}")
        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", None)
        if core_mask is not None:
            result = detector.rknn.init_runtime(core_mask=core_mask)
        else:
            result = detector.rknn.init_runtime()
        if result != 0:
            raise RuntimeError(f"RKNN init_runtime failed with code {result}")
        detector.npu_enabled = True
        detector.name = "rknn-yolov5-crowdhuman"
    except Exception as exc:
        _set_unavailable(detector, f"Could not load RKNN detector: {exc}")
    return detector


def detect_people(detector: SimpleNamespace, frame: Frame) -> list[Detection]:
    """等比缩放 → NPU 推理 → 解码为原图人体框，同时记录三个阶段耗时。"""
    started = time.monotonic()
    if not detector.npu_enabled:
        detector.last_profile = _empty_profile()
        detector.last_profile["inference_ms"] = _ms(time.monotonic() - started)
        return []
    preprocess_started = time.monotonic()
    input_image, ratio, pad = _preprocess(frame, detector.input_size)
    preprocessed = time.monotonic()
    outputs = detector.rknn.inference(inputs=[input_image])
    inferred = time.monotonic()
    detections = _postprocess(detector, outputs, frame.shape[:2], ratio, pad)
    finished = time.monotonic()
    detector.last_profile = {
        "preprocess_ms": _ms(preprocessed - preprocess_started),
        "inference_ms": _ms(inferred - preprocessed),
        "postprocess_ms": _ms(finished - inferred),
    }
    elapsed_ms = _ms(time.monotonic() - started)
    if not any(detector.last_profile.values()):
        detector.last_profile["inference_ms"] = elapsed_ms
    return detections


def release_detector(detector: SimpleNamespace) -> None:
    if detector.rknn is not None:
        try:
            detector.rknn.release()
        except Exception:
            pass


def _postprocess(
    detector: SimpleNamespace, outputs: Any, frame_shape: Sequence[int], ratio: float, pad: tuple[float, float],
) -> list[Detection]:
    """解码三组 sigmoid 输出，只把 person 全身框交给计数，不把 head 算作第二个人。

    Sigmoid 已包含在 RKNN 图内；直接量化未限幅 logits 会降低框偏移精度。
    """
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 3:
        raise ValueError("CrowdHuman RKNN requires three sigmoid convolution outputs")
    boxes, scores = [], []
    for tensor, anchors, stride in zip(outputs, CROWDHUMAN_ANCHORS, (8, 16, 32)):
        grid_size = detector.input_size // stride
        probabilities = np.asarray(tensor, dtype=np.float32)
        if probabilities.shape != (1, 21, grid_size, grid_size):
            raise ValueError(f"Unexpected CrowdHuman head shape: {probabilities.shape}")
        if not np.isfinite(probabilities).all() or probabilities.min() < -.01 or probabilities.max() > 1.01:
            raise ValueError("CrowdHuman outputs must be sigmoid probabilities, not raw logits")
        prediction = np.clip(probabilities, 0, 1).reshape(3, 7, grid_size, grid_size)
        # 七个值依次是四个框参数、objectness、person 分数、head 分数。
        # person 置信度 = objectness × person 分数，head 不创建人员检测框。
        confidence = prediction[:, 4] * prediction[:, 5]
        anchor_ids, ys, xs = np.where(confidence >= detector.confidence_threshold)
        if not anchor_ids.size:
            continue
        selected = prediction[anchor_ids, :, ys, xs]
        centers = (selected[:, :2] * 2 - 0.5 + np.column_stack((xs, ys))) * stride
        sizes = (selected[:, 2:4] * 2) ** 2 * np.asarray(anchors, np.float32)[anchor_ids]
        boxes.append(np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1))
        scores.append(confidence[anchor_ids, ys, xs])
    if not boxes or (detector.class_filter and "person" not in detector.class_filter):
        return []
    boxes, scores = np.concatenate(boxes), np.concatenate(scores)
    if scores.size > MAX_CANDIDATES:
        top = np.argpartition(scores, -MAX_CANDIDATES)[-MAX_CANDIDATES:]
        boxes, scores = boxes[top], scores[top]
    height, width = map(int, frame_shape[:2])
    boxes = _scale_boxes_from_letterbox(boxes, ratio, pad, width, height)
    valid = np.isfinite(boxes).all(axis=1)
    valid &= (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    boxes, scores = boxes[valid], scores[valid]
    return [Detection(tuple(float(v) for v in boxes[i]), float(scores[i]), 0, "person")
            for i in _nms_xyxy(boxes, scores, detector.nms_threshold)]


def _set_unavailable(detector: SimpleNamespace, reason: str) -> None:
    # 沿用原来的状态文字，网页与健康接口仍能显示模型失败原因。
    detector.warning = f"{reason} Motion fallback is disabled to avoid marking non-person moving objects."


def _preprocess(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, float, tuple[float, float]]:
    # 等比缩放后用灰色补边；保留缩放比例与补边量，输出框再还原到原图。
    height, width = frame.shape[:2]
    ratio = min(input_size / float(width), input_size / float(height))
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    pad_x = (input_size - new_width) // 2
    pad_y = (input_size - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0), ratio, (float(pad_x), float(pad_y))


def _scale_boxes_from_letterbox(
    boxes: np.ndarray,
    ratio: float,
    pad: tuple[float, float],
    width: int,
    height: int,
) -> np.ndarray:
    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - float(pad[0])) / float(ratio)
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - float(pad[1])) / float(ratio)
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0.0, float(width))
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0.0, float(height))
    return scaled


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, nms_threshold: float) -> list[int]:
    # 从高分到低分保留框，剔除与已保留框 IoU 超过阈值的重复候选。
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        union = areas[current] + areas[rest] - inter
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[iou <= nms_threshold]
    return keep


def _empty_profile() -> dict[str, float]:
    return {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
