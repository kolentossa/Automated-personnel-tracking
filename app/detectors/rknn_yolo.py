"""RKNN Lite YOLO person detector for RK3588 NPU inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.types import Detection, Frame


class RKNNYoloDetector:
    """Run a YOLO RKNN model through rknn-toolkit-lite2 on RK3588.

    The postprocess path handles common YOLOv8/YOLOv5 exported layouts that
    already include decoded boxes. Raw DFL head decoding can be added later
    without changing the pipeline contract.
    """

    PERSON_CLASS_ID = 0
    MAX_CANDIDATES = 300

    def __init__(
        self,
        model_path: Path,
        *,
        model_family: str = "yolov8",
        input_size: int = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        class_filter: Optional[Iterable[str]] = None,
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:  # pragma: no cover - depends on RK3588 runtime
            raise RuntimeError("rknnlite.api.RKNNLite is not available") from exc

        self.model_path = Path(model_path)
        self.model_family = str(model_family or "yolov8").lower()
        self.input_size = int(input_size)
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.class_filter = {str(item).lower() for item in (class_filter or ["person"])}
        self.last_profile: Dict[str, float] = _empty_profile()
        self.npu_enabled = False
        self.name = _detector_name(self.model_path, self.model_family)
        self._rknn = RKNNLite()

        ret = self._rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(f"RKNN load_rknn failed with code {ret}: {self.model_path}")

        core_mask = getattr(RKNNLite, "NPU_CORE_0_1_2", None)
        if core_mask is not None:
            ret = self._rknn.init_runtime(core_mask=core_mask)
        else:
            ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with code {ret}")
        self.npu_enabled = True

    def detect(self, frame: Frame) -> List[Detection]:
        t0 = time.monotonic()
        input_image, ratio, pad = _preprocess(frame, self.input_size)
        t1 = time.monotonic()
        outputs = self._rknn.inference(inputs=[input_image])
        t2 = time.monotonic()
        detections = self._postprocess(outputs, frame.shape[:2], ratio, pad)
        t3 = time.monotonic()
        self.last_profile = {
            "preprocess_ms": _ms(t1 - t0),
            "inference_ms": _ms(t2 - t1),
            "postprocess_ms": _ms(t3 - t2),
        }
        return detections

    def release(self) -> None:
        rknn = getattr(self, "_rknn", None)
        if rknn is not None:
            try:
                rknn.release()
            except Exception:
                pass

    def _postprocess(
        self,
        outputs: Any,
        frame_shape: Sequence[int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> List[Detection]:
        model_zoo_detections = self._postprocess_yolov8_heads(outputs, frame_shape, ratio, pad)
        if model_zoo_detections is not None:
            return model_zoo_detections

        rows = _rows_from_outputs(outputs)
        height, width = int(frame_shape[0]), int(frame_shape[1])
        boxes_xywh: List[List[int]] = []
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        confidences: List[float] = []

        for row in rows:
            decoded = self._decode_row(row)
            if decoded is None:
                continue
            x1, y1, x2, y2, score = decoded
            x1, y1, x2, y2 = _scale_box_from_letterbox((x1, y1, x2, y2), ratio, pad, width, height)
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append((x1, y1, x2, y2))
            boxes_xywh.append([int(round(x1)), int(round(y1)), int(round(x2 - x1)), int(round(y2 - y1))])
            confidences.append(float(score))

        keep = _nms(boxes_xywh, confidences, self.confidence_threshold, self.nms_threshold)
        detections = [
            Detection(boxes_xyxy[index], confidences[index], self.PERSON_CLASS_ID, "person")
            for index in keep
        ]
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def _postprocess_yolov8_heads(
        self,
        outputs: Any,
        frame_shape: Sequence[int],
        ratio: float,
        pad: Tuple[float, float],
    ) -> Optional[List[Detection]]:
        tensors = outputs if isinstance(outputs, (list, tuple)) else []
        if len(tensors) < 6 or len(tensors) % 3 != 0:
            return None
        arrays = [_as_float32(tensor) for tensor in tensors]
        if not all(array.ndim == 4 for array in arrays[:6]):
            return None

        branch_count = 3
        pair_per_branch = len(arrays) // branch_count
        branch_boxes: List[np.ndarray] = []
        branch_scores: List[np.ndarray] = []
        for branch_index in range(branch_count):
            box_tensor = arrays[pair_per_branch * branch_index]
            class_tensor = arrays[pair_per_branch * branch_index + 1]
            if box_tensor.shape[1] % 4 != 0 or class_tensor.shape[1] < 1:
                return None
            person_map = class_tensor[0, self.PERSON_CLASS_ID]
            ys, xs = np.where(person_map >= self.confidence_threshold)
            if ys.size == 0:
                continue
            scores = person_map[ys, xs].astype(np.float32, copy=False)
            branch_boxes.append(_yolov8_box_process_selected(box_tensor[0], ys, xs, self.input_size))
            branch_scores.append(scores)

        if not branch_boxes:
            return []

        boxes = np.concatenate(branch_boxes, axis=0)
        scores = np.concatenate(branch_scores, axis=0)
        if scores.size > self.MAX_CANDIDATES:
            top = np.argpartition(scores, -self.MAX_CANDIDATES)[-self.MAX_CANDIDATES :]
            order = top[np.argsort(scores[top])[::-1]]
            boxes = boxes[order]
            scores = scores[order]

        height, width = int(frame_shape[0]), int(frame_shape[1])
        boxes_xyxy = _scale_boxes_from_letterbox(boxes, ratio, pad, width, height)
        valid = (boxes_xyxy[:, 2] > boxes_xyxy[:, 0]) & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        if not np.any(valid):
            return []
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]

        keep = _nms_xyxy(boxes_xyxy, scores, self.nms_threshold)
        detections = [
            Detection(tuple(float(value) for value in boxes_xyxy[index]), float(scores[index]), self.PERSON_CLASS_ID, "person")
            for index in keep
        ]
        return detections

    def _decode_row(self, row: np.ndarray) -> Optional[Tuple[float, float, float, float, float]]:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 6:
            return None

        class_id = self.PERSON_CLASS_ID
        score = 0.0
        xyxy_format = False

        if values.size <= 7:
            score = float(values[4])
            class_id = int(round(float(values[5]))) if values.size >= 6 else self.PERSON_CLASS_ID
            xyxy_format = True
        elif values.size >= 85 and self.model_family in {"yolov5", "yolov6"}:
            objectness = float(values[4])
            class_scores = values[5:]
            class_id = int(np.argmax(class_scores))
            score = objectness * float(class_scores[class_id])
        elif values.size >= 85 and float(values[4]) <= 1.0:
            objectness = float(values[4])
            class_scores = values[5:]
            class_id = int(np.argmax(class_scores))
            score = objectness * float(class_scores[class_id])
        else:
            class_scores = values[4:]
            class_id = int(np.argmax(class_scores))
            score = float(class_scores[class_id])

        if class_id != self.PERSON_CLASS_ID or score < self.confidence_threshold:
            return None
        if self.class_filter and "person" not in self.class_filter:
            return None

        coords = values[:4].astype(np.float32)
        if float(np.max(coords)) <= 1.5:
            coords = coords * float(self.input_size)

        if xyxy_format:
            x1, y1, x2, y2 = [float(item) for item in coords]
        else:
            cx, cy, bw, bh = [float(item) for item in coords]
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
        return x1, y1, x2, y2, score


def _preprocess(frame: np.ndarray, input_size: int) -> Tuple[np.ndarray, float, Tuple[float, float]]:
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


def _rows_from_outputs(outputs: Any) -> List[np.ndarray]:
    if outputs is None:
        return []
    tensors = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    rows: List[np.ndarray] = []
    for tensor in tensors:
        arr = np.asarray(tensor)
        if arr.size == 0:
            continue
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        elif arr.ndim == 2:
            if arr.shape[0] in {6, 84, 85} and arr.shape[1] > arr.shape[0]:
                arr = arr.T
        elif arr.ndim >= 3:
            if arr.shape[-1] >= 6:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.shape[0] >= 6:
                arr = arr.reshape(arr.shape[0], -1).T
            else:
                continue
        if arr.ndim == 2 and arr.shape[1] >= 6:
            rows.extend(arr)
    return rows


def _as_float32(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


def _yolov8_box_process(position: np.ndarray, input_size: int) -> np.ndarray:
    _, channels, grid_h, grid_w = position.shape
    reg_max = channels // 4
    position = position.reshape(1, 4, reg_max, grid_h, grid_w)
    position = _softmax(position, axis=2)
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    distance = (position * bins).sum(axis=2)

    col, row = np.meshgrid(np.arange(grid_w, dtype=np.float32), np.arange(grid_h, dtype=np.float32))
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array([input_size // grid_w, input_size // grid_h], dtype=np.float32).reshape(1, 2, 1, 1)
    box_xy1 = grid + 0.5 - distance[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + distance[:, 2:4, :, :]
    return np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)


def _yolov8_box_process_selected(position: np.ndarray, ys: np.ndarray, xs: np.ndarray, input_size: int) -> np.ndarray:
    channels, grid_h, grid_w = position.shape
    reg_max = channels // 4
    logits = position.reshape(4, reg_max, grid_h, grid_w)[:, :, ys, xs]
    probs = _softmax(logits, axis=1)
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
    distance = (probs * bins).sum(axis=1)
    grid_x = xs.astype(np.float32, copy=False) + 0.5
    grid_y = ys.astype(np.float32, copy=False) + 0.5
    stride_x = float(input_size) / float(grid_w)
    stride_y = float(input_size) / float(grid_h)
    return np.stack(
        (
            (grid_x - distance[0]) * stride_x,
            (grid_y - distance[1]) * stride_y,
            (grid_x + distance[2]) * stride_x,
            (grid_y + distance[3]) * stride_y,
        ),
        axis=1,
    )


def _flatten_head(value: np.ndarray) -> np.ndarray:
    channels = value.shape[1]
    return value.transpose(0, 2, 3, 1).reshape(-1, channels)


def _softmax(value: np.ndarray, axis: int) -> np.ndarray:
    shifted = value - np.max(value, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _scale_box_from_letterbox(
    box: Sequence[float],
    ratio: float,
    pad: Tuple[float, float],
    width: int,
    height: int,
) -> Tuple[float, float, float, float]:
    pad_x, pad_y = pad
    x1 = (float(box[0]) - pad_x) / ratio
    y1 = (float(box[1]) - pad_y) / ratio
    x2 = (float(box[2]) - pad_x) / ratio
    y2 = (float(box[3]) - pad_y) / ratio
    return (
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    )


def _scale_boxes_from_letterbox(
    boxes: np.ndarray,
    ratio: float,
    pad: Tuple[float, float],
    width: int,
    height: int,
) -> np.ndarray:
    scaled = boxes.astype(np.float32, copy=True)
    scaled[:, [0, 2]] = (scaled[:, [0, 2]] - float(pad[0])) / float(ratio)
    scaled[:, [1, 3]] = (scaled[:, [1, 3]] - float(pad[1])) / float(ratio)
    scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0.0, float(width))
    scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0.0, float(height))
    return scaled


def _nms_xyxy(boxes: np.ndarray, scores: np.ndarray, nms_threshold: float) -> List[int]:
    if boxes.size == 0:
        return []
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep: List[int] = []
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


def _nms(boxes: List[List[int]], confidences: List[float], score_threshold: float, nms_threshold: float) -> List[int]:
    if not boxes:
        return []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    if indexes is None or len(indexes) == 0:
        return []
    return [int(index) for index in np.asarray(indexes).reshape(-1)]


def _detector_name(model_path: Path, model_family: str) -> str:
    stem = model_path.stem.lower()
    if "yolov8n" in stem:
        return "rknn-yolov8n"
    if "yolo11n" in stem:
        return "rknn-yolo11n"
    if "yolov5n" in stem:
        return "rknn-yolov5n"
    if "yolov6n" in stem:
        return "rknn-yolov6n"
    if model_family:
        return f"rknn-{model_family}"
    return "rknn-yolo"


def _empty_profile() -> Dict[str, float]:
    return {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
