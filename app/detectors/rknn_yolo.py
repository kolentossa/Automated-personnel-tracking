"""RKNN Lite YOLO object detector for RK3588 NPU inference."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from vision.types import Detection, Frame


COCO_CLASS_NAMES = (
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
)
DAMO_FAMILIES = {"damoyolo", "damo-yolo", "damo_yolo"}


class RKNNYoloDetector:
    """Run a YOLO RKNN model through rknn-toolkit-lite2 on RK3588.

    The postprocess path handles common YOLOv8/YOLOv5 exported layouts that
    already include decoded boxes. Raw DFL head decoding can be added later
    without changing the pipeline contract.
    """

    MAX_CANDIDATES = 300

    def __init__(
        self,
        model_path: Path,
        *,
        model_family: str = "yolov8",
        input_size: Any = 640,
        confidence_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        class_filter: Optional[Iterable[str]] = None,
        class_names: Optional[Any] = None,
        class_confidence_thresholds: Optional[Dict[Any, float]] = None,
        box_filter: Optional[Dict[str, Any]] = None,
        core_mask: Any = "0_1_2",
    ) -> None:
        try:
            from rknnlite.api import RKNNLite
        except Exception as exc:  # pragma: no cover - depends on RK3588 runtime
            raise RuntimeError("rknnlite.api.RKNNLite is not available") from exc

        self.model_path = Path(model_path)
        self.model_family = str(model_family or "yolov8").lower()
        self.input_height, self.input_width = _normalise_input_shape(input_size)
        self.input_size = (
            self.input_width
            if self.input_height == self.input_width
            else (self.input_height, self.input_width)
        )
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.class_names = _normalise_class_names(class_names)
        self.class_confidence_thresholds = _normalise_class_thresholds(
            class_confidence_thresholds, self.class_names
        )
        self.box_filter = _normalise_box_filter(box_filter)
        self.class_filter = {
            str(item).strip().lower() for item in (class_filter or ["person"])
        }
        self.selected_class_ids = {
            class_id
            for class_id, label in self.class_names.items()
            if not self.class_filter or label.lower() in self.class_filter
        }
        if self.class_filter and not self.selected_class_ids:
            raise ValueError(
                f"class_filter does not match class_names: {sorted(self.class_filter)}"
            )
        self.last_profile: Dict[str, float] = _empty_profile()
        self.npu_enabled = False
        self.name = _detector_name(self.model_path, self.model_family)
        self._rknn = RKNNLite()

        ret = self._rknn.load_rknn(str(self.model_path))
        if ret != 0:
            raise RuntimeError(
                f"RKNN load_rknn failed with code {ret}: {self.model_path}"
            )

        resolved_core_mask = _resolve_core_mask(RKNNLite, core_mask)
        if resolved_core_mask is not None:
            ret = self._rknn.init_runtime(core_mask=resolved_core_mask)
        else:
            ret = self._rknn.init_runtime()
        if ret != 0:
            raise RuntimeError(f"RKNN init_runtime failed with code {ret}")
        self.npu_enabled = True

    def detect(
        self,
        frame: Frame,
        *,
        class_confidence_thresholds: Optional[Dict[Any, float]] = None,
    ) -> List[Detection]:
        threshold_overrides = _normalise_class_thresholds(
            class_confidence_thresholds, self.class_names
        )
        t0 = time.monotonic()
        if self.model_family in DAMO_FAMILIES:
            input_image = _preprocess_damoyolo(frame, self.input_size)
            ratio, pad = 1.0, (0.0, 0.0)
        else:
            input_image, ratio, pad = _preprocess(frame, self.input_size)
        t1 = time.monotonic()
        outputs = self._rknn.inference(inputs=[input_image])
        t2 = time.monotonic()
        detections = self._postprocess(
            outputs, frame.shape[:2], ratio, pad, threshold_overrides
        )
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
        threshold_overrides: Optional[Dict[int, float]] = None,
    ) -> List[Detection]:
        if self.model_family in DAMO_FAMILIES:
            return self._postprocess_damoyolo(outputs, frame_shape, threshold_overrides)
        model_zoo_detections = self._postprocess_yolov8_heads(
            outputs, frame_shape, ratio, pad, threshold_overrides
        )
        if model_zoo_detections is not None:
            return model_zoo_detections

        rows = _rows_from_outputs(outputs, len(self.class_names))
        height, width = int(frame_shape[0]), int(frame_shape[1])
        boxes_xywh: List[List[int]] = []
        boxes_xyxy: List[Tuple[float, float, float, float]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for row in rows:
            decoded = self._decode_row(row, threshold_overrides)
            if decoded is None:
                continue
            x1, y1, x2, y2, score, class_id = decoded
            x1, y1, x2, y2 = _scale_box_from_letterbox(
                (x1, y1, x2, y2), ratio, pad, width, height
            )
            if x2 <= x1 or y2 <= y1:
                continue
            boxes_xyxy.append((x1, y1, x2, y2))
            boxes_xywh.append(
                [
                    int(round(x1)),
                    int(round(y1)),
                    int(round(x2 - x1)),
                    int(round(y2 - y1)),
                ]
            )
            confidences.append(float(score))
            class_ids.append(int(class_id))

        effective_thresholds = dict(getattr(self, "class_confidence_thresholds", {}))
        effective_thresholds.update(threshold_overrides or {})
        keep = _class_aware_nms(
            boxes_xywh,
            confidences,
            class_ids,
            self.confidence_threshold,
            self.nms_threshold,
            effective_thresholds,
        )
        detections = [
            Detection(
                boxes_xyxy[index],
                confidences[index],
                class_ids[index],
                self.class_names[class_ids[index]],
            )
            for index in keep
        ]
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections

    def _postprocess_damoyolo(
        self,
        outputs: Any,
        frame_shape: Sequence[int],
        threshold_overrides: Optional[Dict[int, float]] = None,
    ) -> List[Detection]:
        if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
            count = len(outputs) if isinstance(outputs, (list, tuple)) else 0
            raise ValueError(
                f"DAMO-YOLO output count mismatch: expected 2, got {count}"
            )
        scores = _as_float32(outputs[0])
        boxes = _as_float32(outputs[1])
        input_height, input_width = _detector_input_shape(self)
        expected_predictions = sum(
            (input_height // stride) * (input_width // stride) for stride in (8, 16, 32)
        )
        expected_classes = len(self.class_names)
        expected_scores = (1, expected_predictions, expected_classes)
        expected_boxes = (1, expected_predictions, 4)
        if tuple(scores.shape) != expected_scores:
            raise ValueError(
                f"DAMO-YOLO scores shape mismatch: expected {expected_scores}, got {tuple(scores.shape)}"
            )
        if tuple(boxes.shape) != expected_boxes:
            raise ValueError(
                f"DAMO-YOLO boxes shape mismatch: expected {expected_boxes}, got {tuple(boxes.shape)}"
            )
        if set(self.class_names) != set(range(expected_classes)):
            raise ValueError(
                "DAMO-YOLO class_names must use contiguous IDs starting at zero"
            )
        if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
            raise ValueError("DAMO-YOLO outputs contain NaN or infinity")

        selected_boxes: List[np.ndarray] = []
        selected_scores: List[np.ndarray] = []
        selected_class_ids: List[np.ndarray] = []
        for class_id in sorted(self.selected_class_ids):
            class_scores = scores[0, :, class_id]
            indexes = np.where(
                class_scores >= self._threshold_for_class(class_id, threshold_overrides)
            )[0]
            if indexes.size == 0:
                continue
            selected_boxes.append(boxes[0, indexes])
            selected_scores.append(class_scores[indexes])
            selected_class_ids.append(np.full(indexes.shape, class_id, dtype=np.int32))
        if not selected_boxes:
            return []

        candidate_boxes = np.concatenate(selected_boxes, axis=0).astype(
            np.float32, copy=True
        )
        candidate_scores = np.concatenate(selected_scores, axis=0).astype(
            np.float32, copy=False
        )
        candidate_class_ids = np.concatenate(selected_class_ids, axis=0)
        if candidate_scores.size > self.MAX_CANDIDATES:
            top = np.argpartition(candidate_scores, -self.MAX_CANDIDATES)[
                -self.MAX_CANDIDATES :
            ]
            order = top[np.argsort(candidate_scores[top])[::-1]]
            candidate_boxes = candidate_boxes[order]
            candidate_scores = candidate_scores[order]
            candidate_class_ids = candidate_class_ids[order]

        height, width = int(frame_shape[0]), int(frame_shape[1])
        candidate_boxes[:, [0, 2]] *= float(width) / float(input_width)
        candidate_boxes[:, [1, 3]] *= float(height) / float(input_height)
        candidate_boxes[:, [0, 2]] = np.clip(
            candidate_boxes[:, [0, 2]], 0.0, float(width)
        )
        candidate_boxes[:, [1, 3]] = np.clip(
            candidate_boxes[:, [1, 3]], 0.0, float(height)
        )
        box_filter = getattr(self, "box_filter", {})
        valid = _box_filter_mask(candidate_boxes, width, height, box_filter)
        candidate_boxes = candidate_boxes[valid]
        candidate_scores = candidate_scores[valid]
        candidate_class_ids = candidate_class_ids[valid]

        keep: List[int] = []
        for class_id in np.unique(candidate_class_ids):
            indexes = np.where(candidate_class_ids == class_id)[0]
            class_keep = _nms_xyxy(
                candidate_boxes[indexes],
                candidate_scores[indexes],
                self.nms_threshold,
                box_filter.get("duplicate_containment_threshold"),
            )
            keep.extend(int(indexes[index]) for index in class_keep)
        keep.sort(key=lambda index: float(candidate_scores[index]), reverse=True)
        return [
            Detection(
                tuple(float(value) for value in candidate_boxes[index]),
                float(candidate_scores[index]),
                int(candidate_class_ids[index]),
                self.class_names[int(candidate_class_ids[index])],
            )
            for index in keep
        ]

    def _postprocess_yolov8_heads(
        self,
        outputs: Any,
        frame_shape: Sequence[int],
        ratio: float,
        pad: Tuple[float, float],
        threshold_overrides: Optional[Dict[int, float]] = None,
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
        branch_class_ids: List[np.ndarray] = []
        for branch_index in range(branch_count):
            box_tensor = arrays[pair_per_branch * branch_index]
            class_tensor = arrays[pair_per_branch * branch_index + 1]
            if box_tensor.shape[1] % 4 != 0 or class_tensor.shape[1] < 1:
                return None
            for class_id in sorted(self.selected_class_ids):
                if class_id >= class_tensor.shape[1]:
                    continue
                class_map = class_tensor[0, class_id]
                ys, xs = np.where(
                    class_map
                    >= self._threshold_for_class(class_id, threshold_overrides)
                )
                if ys.size == 0:
                    continue
                scores = class_map[ys, xs].astype(np.float32, copy=False)
                branch_boxes.append(
                    _yolov8_box_process_selected(box_tensor[0], ys, xs, self.input_size)
                )
                branch_scores.append(scores)
                branch_class_ids.append(np.full(scores.shape, class_id, dtype=np.int32))

        if not branch_boxes:
            return []

        boxes = np.concatenate(branch_boxes, axis=0)
        scores = np.concatenate(branch_scores, axis=0)
        class_ids = np.concatenate(branch_class_ids, axis=0)
        if scores.size > self.MAX_CANDIDATES:
            top = np.argpartition(scores, -self.MAX_CANDIDATES)[-self.MAX_CANDIDATES :]
            order = top[np.argsort(scores[top])[::-1]]
            boxes = boxes[order]
            scores = scores[order]
            class_ids = class_ids[order]

        height, width = int(frame_shape[0]), int(frame_shape[1])
        boxes_xyxy = _scale_boxes_from_letterbox(boxes, ratio, pad, width, height)
        box_filter = getattr(self, "box_filter", {})
        valid = _box_filter_mask(boxes_xyxy, width, height, box_filter)
        if not np.any(valid):
            return []
        boxes_xyxy = boxes_xyxy[valid]
        scores = scores[valid]
        class_ids = class_ids[valid]

        keep: List[int] = []
        for class_id in np.unique(class_ids):
            indexes = np.where(class_ids == class_id)[0]
            class_keep = _nms_xyxy(
                boxes_xyxy[indexes],
                scores[indexes],
                self.nms_threshold,
                box_filter.get("duplicate_containment_threshold"),
            )
            keep.extend(int(indexes[index]) for index in class_keep)
        keep.sort(key=lambda index: float(scores[index]), reverse=True)
        detections = [
            Detection(
                tuple(float(value) for value in boxes_xyxy[index]),
                float(scores[index]),
                int(class_ids[index]),
                self.class_names[int(class_ids[index])],
            )
            for index in keep
        ]
        return detections

    def _decode_row(
        self,
        row: np.ndarray,
        threshold_overrides: Optional[Dict[int, float]] = None,
    ) -> Optional[Tuple[float, float, float, float, float, int]]:
        values = np.asarray(row, dtype=np.float32).reshape(-1)
        if values.size < 5:
            return None

        class_id = 0
        score = 0.0
        xyxy_format = False

        if values.size in {6, 7}:
            score = float(values[4])
            class_id = int(round(float(values[5]))) if values.size >= 6 else 0
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

        if class_id not in self.selected_class_ids or score < self._threshold_for_class(
            class_id, threshold_overrides
        ):
            return None

        coords = values[:4].astype(np.float32)
        if float(np.max(coords)) <= 1.5:
            input_height, input_width = _detector_input_shape(self)
            coords[[0, 2]] *= float(input_width)
            coords[[1, 3]] *= float(input_height)

        if xyxy_format:
            x1, y1, x2, y2 = [float(item) for item in coords]
        else:
            cx, cy, bw, bh = [float(item) for item in coords]
            x1 = cx - bw / 2.0
            y1 = cy - bh / 2.0
            x2 = cx + bw / 2.0
            y2 = cy + bh / 2.0
        return x1, y1, x2, y2, score, class_id

    def _threshold_for_class(
        self,
        class_id: int,
        threshold_overrides: Optional[Dict[int, float]] = None,
    ) -> float:
        if threshold_overrides and int(class_id) in threshold_overrides:
            return float(threshold_overrides[int(class_id)])
        thresholds = getattr(self, "class_confidence_thresholds", {})
        return float(thresholds.get(int(class_id), self.confidence_threshold))


def _normalise_input_shape(value: Any) -> Tuple[int, int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("RKNN input shape must contain height and width")
        height, width = int(value[0]), int(value[1])
    else:
        height = width = int(value)
    if height < 32 or width < 32:
        raise ValueError("RKNN input height and width must be at least 32")
    if height % 32 or width % 32:
        raise ValueError("RKNN input height and width must be divisible by 32")
    return height, width


def _detector_input_shape(detector: Any) -> Tuple[int, int]:
    height = getattr(detector, "input_height", None)
    width = getattr(detector, "input_width", None)
    if height is not None and width is not None:
        return int(height), int(width)
    return _normalise_input_shape(getattr(detector, "input_size", 640))


def _preprocess(
    frame: np.ndarray, input_size: Any
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    input_height, input_width = _normalise_input_shape(input_size)
    height, width = frame.shape[:2]
    ratio = min(input_width / float(width), input_height / float(height))
    new_width = int(round(width * ratio))
    new_height = int(round(height * ratio))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((input_height, input_width, 3), 114, dtype=np.uint8)
    pad_x = (input_width - new_width) // 2
    pad_y = (input_height - new_height) // 2
    canvas[pad_y : pad_y + new_height, pad_x : pad_x + new_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0), ratio, (float(pad_x), float(pad_y))


def _preprocess_damoyolo(frame: np.ndarray, input_size: Any) -> np.ndarray:
    input_height, input_width = _normalise_input_shape(input_size)
    resized = cv2.resize(
        frame, (input_width, input_height), interpolation=cv2.INTER_LINEAR
    )
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return np.expand_dims(rgb, axis=0)


def _rows_from_outputs(outputs: Any, expected_classes: int = 80) -> List[np.ndarray]:
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
            channel_counts = {
                4 + int(expected_classes),
                5 + int(expected_classes),
                6,
                84,
                85,
            }
            expected_channel_counts = {
                4 + int(expected_classes),
                5 + int(expected_classes),
            }
            if arr.shape[0] in expected_channel_counts and arr.shape[1] != arr.shape[0]:
                arr = arr.T
            elif arr.shape[0] in channel_counts and arr.shape[1] > arr.shape[0]:
                arr = arr.T
        elif arr.ndim >= 3:
            if arr.shape[-1] >= 5:
                arr = arr.reshape(-1, arr.shape[-1])
            elif arr.shape[0] >= 5:
                arr = arr.reshape(arr.shape[0], -1).T
            else:
                continue
        if arr.ndim == 2 and arr.shape[1] >= 5:
            rows.extend(arr)
    return rows


def _as_float32(value: Any) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == np.float32:
        return arr
    return arr.astype(np.float32, copy=False)


def _yolov8_box_process(position: np.ndarray, input_size: Any) -> np.ndarray:
    input_height, input_width = _normalise_input_shape(input_size)
    _, channels, grid_h, grid_w = position.shape
    reg_max = channels // 4
    position = position.reshape(1, 4, reg_max, grid_h, grid_w)
    position = _softmax(position, axis=2)
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, 1, reg_max, 1, 1)
    distance = (position * bins).sum(axis=2)

    col, row = np.meshgrid(
        np.arange(grid_w, dtype=np.float32), np.arange(grid_h, dtype=np.float32)
    )
    col = col.reshape(1, 1, grid_h, grid_w)
    row = row.reshape(1, 1, grid_h, grid_w)
    grid = np.concatenate((col, row), axis=1)
    stride = np.array(
        [input_width / grid_w, input_height / grid_h], dtype=np.float32
    ).reshape(1, 2, 1, 1)
    box_xy1 = grid + 0.5 - distance[:, 0:2, :, :]
    box_xy2 = grid + 0.5 + distance[:, 2:4, :, :]
    return np.concatenate((box_xy1 * stride, box_xy2 * stride), axis=1)


def _yolov8_box_process_selected(
    position: np.ndarray, ys: np.ndarray, xs: np.ndarray, input_size: Any
) -> np.ndarray:
    input_height, input_width = _normalise_input_shape(input_size)
    channels, grid_h, grid_w = position.shape
    reg_max = channels // 4
    logits = position.reshape(4, reg_max, grid_h, grid_w)[:, :, ys, xs]
    probs = _softmax(logits, axis=1)
    bins = np.arange(reg_max, dtype=np.float32).reshape(1, reg_max, 1)
    distance = (probs * bins).sum(axis=1)
    grid_x = xs.astype(np.float32, copy=False) + 0.5
    grid_y = ys.astype(np.float32, copy=False) + 0.5
    stride_x = float(input_width) / float(grid_w)
    stride_y = float(input_height) / float(grid_h)
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


def _nms_xyxy(
    boxes: np.ndarray,
    scores: np.ndarray,
    nms_threshold: float,
    containment_threshold: Optional[float] = None,
) -> List[int]:
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
        duplicate = iou > nms_threshold
        if containment_threshold is not None:
            smaller = np.minimum(areas[current], areas[rest])
            containment = np.divide(
                inter, smaller, out=np.zeros_like(inter), where=smaller > 0
            )
            duplicate |= containment >= float(containment_threshold)
        order = rest[~duplicate]
    return keep


def _nms(
    boxes: List[List[int]],
    confidences: List[float],
    score_threshold: float,
    nms_threshold: float,
) -> List[int]:
    if not boxes:
        return []
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold, nms_threshold)
    if indexes is None or len(indexes) == 0:
        return []
    return [int(index) for index in np.asarray(indexes).reshape(-1)]


def _class_aware_nms(
    boxes: List[List[int]],
    confidences: List[float],
    class_ids: List[int],
    score_threshold: float,
    nms_threshold: float,
    class_score_thresholds: Optional[Dict[int, float]] = None,
) -> List[int]:
    keep: List[int] = []
    for class_id in sorted(set(class_ids)):
        indexes = [index for index, value in enumerate(class_ids) if value == class_id]
        selected = _nms(
            [boxes[index] for index in indexes],
            [confidences[index] for index in indexes],
            float((class_score_thresholds or {}).get(class_id, score_threshold)),
            nms_threshold,
        )
        keep.extend(indexes[index] for index in selected)
    keep.sort(key=lambda index: confidences[index], reverse=True)
    return keep


def _normalise_class_names(value: Optional[Any]) -> Dict[int, str]:
    if value is None:
        return {index: label for index, label in enumerate(COCO_CLASS_NAMES)}
    if isinstance(value, dict):
        result = {
            int(class_id): str(label).strip() for class_id, label in value.items()
        }
    elif isinstance(value, (list, tuple)):
        result = {index: str(label).strip() for index, label in enumerate(value)}
    else:
        raise TypeError("class_names must be a list or class-id mapping")
    if not result or any(not label for label in result.values()):
        raise ValueError("class_names must contain non-empty labels")
    return result


def _normalise_class_thresholds(
    value: Optional[Dict[Any, float]], class_names: Dict[int, str]
) -> Dict[int, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError(
            "class_confidence_thresholds must be a class-id or label mapping"
        )
    labels = {
        label.strip().lower(): class_id for class_id, label in class_names.items()
    }
    result: Dict[int, float] = {}
    for key, raw_threshold in value.items():
        text = str(key).strip()
        if text.lstrip("-").isdigit():
            class_id = int(text)
        else:
            class_id = labels.get(text.lower(), -1)
        if class_id not in class_names:
            raise ValueError(f"Unknown class_confidence_thresholds key: {key}")
        threshold = float(raw_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"Class confidence threshold must be between 0 and 1: {key}"
            )
        result[class_id] = threshold
    return result


def _normalise_box_filter(value: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("box_filter must be a mapping")
    result: Dict[str, float] = {}
    limits = {
        "min_side_px": (0.0, None),
        "max_aspect_ratio": (1.0, None),
        "max_frame_area_ratio": (0.0, 1.0),
        "duplicate_containment_threshold": (0.0, 1.0),
    }
    for key, raw_value in value.items():
        if key not in limits:
            raise ValueError(f"Unknown box_filter option: {key}")
        number = float(raw_value)
        minimum, maximum = limits[key]
        if number < minimum or (maximum is not None and number > maximum):
            raise ValueError(f"box_filter.{key} is outside the supported range")
        result[key] = number
    return result


def _box_filter_mask(
    boxes: np.ndarray, frame_width: int, frame_height: int, config: Dict[str, float]
) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=bool)
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    valid = (widths > 0.0) & (heights > 0.0)
    min_side = float(config.get("min_side_px", 0.0))
    if min_side > 0.0:
        valid &= np.minimum(widths, heights) >= min_side
    max_aspect = config.get("max_aspect_ratio")
    if max_aspect is not None:
        aspect = np.maximum(
            np.divide(
                widths, heights, out=np.full_like(widths, np.inf), where=heights > 0
            ),
            np.divide(
                heights, widths, out=np.full_like(heights, np.inf), where=widths > 0
            ),
        )
        valid &= aspect <= float(max_aspect)
    max_area_ratio = config.get("max_frame_area_ratio")
    if max_area_ratio is not None:
        frame_area = max(1.0, float(frame_width) * float(frame_height))
        valid &= (widths * heights) / frame_area <= float(max_area_ratio)
    return valid


def _resolve_core_mask(rknn_lite_class: Any, value: Any) -> Optional[Any]:
    text = str(value or "auto").strip().lower().replace("npu_core_", "")
    if text in {"", "auto", "all"}:
        text = "0_1_2"
    aliases = {
        "0": "NPU_CORE_0",
        "1": "NPU_CORE_1",
        "2": "NPU_CORE_2",
        "0_1": "NPU_CORE_0_1",
        "0_1_2": "NPU_CORE_0_1_2",
    }
    attribute = aliases.get(text)
    if attribute is None:
        raise ValueError(f"Unsupported RKNN core mask: {value}")
    return getattr(rknn_lite_class, attribute, None)


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
    if model_family in DAMO_FAMILIES:
        return (
            "rknn-damoyolo-cigarette-int8"
            if "int8" in stem
            else "rknn-damoyolo-cigarette"
        )
    if model_family:
        return f"rknn-{model_family}"
    return "rknn-yolo"


def _empty_profile() -> Dict[str, float]:
    return {"preprocess_ms": 0.0, "inference_ms": 0.0, "postprocess_ms": 0.0}


def _ms(seconds: float) -> float:
    return round(float(seconds) * 1000.0, 1)
