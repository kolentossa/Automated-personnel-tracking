#!/usr/bin/env python3
"""Compare DAMO cigarette ONNX and RKNN outputs on a labeled image set."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Callable, Iterable, Sequence

import cv2
import numpy as np

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument(
        "--rknn",
        required=True,
        action="append",
        metavar="NAME=PATH",
        help="RKNN model to compare; repeat for FP and INT8",
    )
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--confidence-threshold", type=float, default=0.35)
    parser.add_argument("--nms-threshold", type=float, default=0.70)
    parser.add_argument("--max-map-drop", type=float, default=0.05)
    parser.add_argument("--min-mean-iou", type=float, default=0.50)
    parser.add_argument("--max-mean-confidence-difference", type=float, default=0.15)
    args = parser.parse_args()

    if not args.onnx.is_file():
        parser.error(f"ONNX model does not exist: {args.onnx}")
    if not args.images.is_dir():
        parser.error(f"Image directory does not exist: {args.images}")
    models = [_parse_named_path(value) for value in args.rknn]
    if len({name for name, _ in models}) != len(models):
        parser.error("RKNN model names must be unique")
    for _, path in models:
        if not path.is_file():
            parser.error(f"RKNN model does not exist: {path}")

    samples = _load_samples(args.images)
    if not samples:
        parser.error(f"No labeled images found in {args.images}")

    onnx_runner = _onnx_runner(args.onnx)
    baseline = _evaluate_runner(
        "onnx",
        onnx_runner,
        samples,
        args.input_size,
        args.confidence_threshold,
        args.nms_threshold,
    )
    comparisons = {}
    failures = []
    for name, path in models:
        runner, release = _rknn_runner(path)
        try:
            result = _evaluate_runner(
                name,
                runner,
                samples,
                args.input_size,
                args.confidence_threshold,
                args.nms_threshold,
            )
        finally:
            release()
        consistency = _compare_predictions(
            baseline["predictions"], result["predictions"], args.confidence_threshold
        )
        result["consistency_vs_onnx"] = consistency
        map_drop = float(baseline["ap50"] - result["ap50"])
        result["ap50_drop_vs_onnx"] = round(map_drop, 6)
        comparisons[name] = result
        if map_drop > args.max_map_drop:
            failures.append(f"{name}: AP50 drop {map_drop:.4f} exceeds {args.max_map_drop:.4f}")
        if consistency["matched_boxes"] and consistency["mean_iou"] < args.min_mean_iou:
            failures.append(
                f"{name}: mean matched IoU {consistency['mean_iou']:.4f} is below {args.min_mean_iou:.4f}"
            )
        if consistency["matched_boxes"] and consistency["mean_confidence_difference"] > args.max_mean_confidence_difference:
            failures.append(
                f"{name}: mean confidence difference "
                f"{consistency['mean_confidence_difference']:.4f} exceeds "
                f"{args.max_mean_confidence_difference:.4f}"
            )

    baseline.pop("predictions", None)
    for result in comparisons.values():
        result.pop("predictions", None)
    report = {
        "schema_version": 1,
        "dataset": str(args.images.resolve()),
        "images": len(samples),
        "input_size": args.input_size,
        "confidence_threshold": args.confidence_threshold,
        "nms_threshold": args.nms_threshold,
        "onnx": baseline,
        "rknn": comparisons,
        "acceptance": {
            "passed": not failures,
            "max_map_drop": args.max_map_drop,
            "min_mean_iou": args.min_mean_iou,
            "max_mean_confidence_difference": args.max_mean_confidence_difference,
            "failures": failures,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if not failures else 1


def _parse_named_path(value: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("--rknn must use NAME=PATH")
    return name.strip(), Path(raw_path).expanduser()


def _load_samples(root: Path) -> list[dict]:
    samples = []
    for image_path in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label_path = image_path.with_suffix(".txt")
        if not label_path.is_file():
            continue
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"Unreadable validation image: {image_path}")
        height, width = image.shape[:2]
        ground_truth = []
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
            class_id, cx, cy, box_width, box_height = (float(field) for field in fields)
            if int(class_id) != 0:
                continue
            ground_truth.append(_yolo_box(cx, cy, box_width, box_height, width, height))
        samples.append({"path": image_path, "image": image, "ground_truth": ground_truth})
    return samples


def _onnx_runner(path: Path) -> Callable[[np.ndarray], tuple[Sequence[np.ndarray], float]]:
    try:
        import onnxruntime as ort
    except Exception as exc:
        raise RuntimeError("onnxruntime is required for ONNX validation") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 1 or inputs[0].name != "images":
        raise ValueError(f"Expected one ONNX input named images, got {[item.name for item in inputs]}")
    output_names = [item.name for item in session.get_outputs()]
    if output_names != ["scores", "boxes"]:
        raise ValueError(f"Expected ONNX outputs ['scores', 'boxes'], got {output_names}")

    def run(rgb: np.ndarray) -> tuple[Sequence[np.ndarray], float]:
        tensor = np.expand_dims(rgb.transpose(2, 0, 1), axis=0).astype(np.float32, copy=False)
        started = time.perf_counter()
        outputs = session.run(output_names, {"images": tensor})
        return outputs, (time.perf_counter() - started) * 1000.0

    return run


def _rknn_runner(path: Path) -> tuple[
    Callable[[np.ndarray], tuple[Sequence[np.ndarray], float]], Callable[[], None]
]:
    try:
        from rknnlite.api import RKNNLite
    except Exception as exc:
        raise RuntimeError("rknn-toolkit-lite2 is required for RKNN validation") from exc
    rknn = RKNNLite()
    result = rknn.load_rknn(str(path))
    if result != 0:
        raise RuntimeError(f"RKNN load_rknn failed with code {result}: {path}")
    result = rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0_1_2)
    if result != 0:
        rknn.release()
        raise RuntimeError(f"RKNN init_runtime failed with code {result}: {path}")

    def run(rgb: np.ndarray) -> tuple[Sequence[np.ndarray], float]:
        started = time.perf_counter()
        outputs = rknn.inference(inputs=[np.expand_dims(rgb, axis=0)])
        return outputs, (time.perf_counter() - started) * 1000.0

    return run, rknn.release


def _evaluate_runner(
    name: str,
    runner: Callable[[np.ndarray], tuple[Sequence[np.ndarray], float]],
    samples: list[dict],
    input_size: int,
    confidence_threshold: float,
    nms_threshold: float,
) -> dict:
    predictions = []
    inference_times = []
    output_shapes = None
    for index, sample in enumerate(samples):
        image = sample["image"]
        rgb = cv2.cvtColor(cv2.resize(image, (input_size, input_size)), cv2.COLOR_BGR2RGB)
        outputs, elapsed_ms = runner(rgb)
        inference_times.append(elapsed_ms)
        scores, boxes = _strict_damo_outputs(outputs)
        shapes = [list(scores.shape), list(boxes.shape)]
        if output_shapes is None:
            output_shapes = shapes
        elif output_shapes != shapes:
            raise ValueError(f"{name} output shapes changed: {output_shapes} -> {shapes}")
        predictions.append(
            _decode_damo(
                scores,
                boxes,
                image.shape[1],
                image.shape[0],
                input_size,
                nms_threshold,
            )
        )
    ground_truth = [sample["ground_truth"] for sample in samples]
    threshold_metrics = _threshold_metrics(predictions, ground_truth, confidence_threshold)
    return {
        "output_shapes": output_shapes,
        "ap50": round(_average_precision(predictions, ground_truth, 0.50), 6),
        "threshold_metrics": threshold_metrics,
        "small_target_recall": round(
            _small_target_recall(predictions, ground_truth, confidence_threshold), 6
        ),
        "inference_ms": {
            "mean": round(statistics.fmean(inference_times), 3),
            "p50": round(_percentile(inference_times, 50), 3),
            "p95": round(_percentile(inference_times, 95), 3),
        },
        "predictions": predictions,
    }


def _strict_damo_outputs(outputs: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 2:
        raise ValueError(f"DAMO output count mismatch: expected 2, got {type(outputs).__name__}")
    scores = np.asarray(outputs[0], dtype=np.float32)
    boxes = np.asarray(outputs[1], dtype=np.float32)
    if scores.shape != (1, 8400, 2):
        raise ValueError(f"DAMO scores shape mismatch: expected (1, 8400, 2), got {scores.shape}")
    if boxes.shape != (1, 8400, 4):
        raise ValueError(f"DAMO boxes shape mismatch: expected (1, 8400, 4), got {boxes.shape}")
    if not np.isfinite(scores).all() or not np.isfinite(boxes).all():
        raise ValueError("DAMO outputs contain NaN or infinity")
    return scores, boxes


def _decode_damo(
    scores: np.ndarray,
    boxes: np.ndarray,
    width: int,
    height: int,
    input_size: int,
    nms_threshold: float,
) -> list[dict]:
    cigarette_scores = scores[0, :, 0]
    candidate_indexes = np.where(cigarette_scores >= 0.05)[0]
    if not candidate_indexes.size:
        return []
    if candidate_indexes.size > 300:
        top = np.argpartition(cigarette_scores[candidate_indexes], -300)[-300:]
        candidate_indexes = candidate_indexes[top]
    candidate_boxes = boxes[0, candidate_indexes].astype(np.float32, copy=True)
    candidate_scores = cigarette_scores[candidate_indexes]
    candidate_boxes[:, [0, 2]] *= float(width) / float(input_size)
    candidate_boxes[:, [1, 3]] *= float(height) / float(input_size)
    candidate_boxes[:, [0, 2]] = np.clip(candidate_boxes[:, [0, 2]], 0.0, float(width))
    candidate_boxes[:, [1, 3]] = np.clip(candidate_boxes[:, [1, 3]], 0.0, float(height))
    valid = (candidate_boxes[:, 2] > candidate_boxes[:, 0]) & (
        candidate_boxes[:, 3] > candidate_boxes[:, 1]
    )
    candidate_boxes = candidate_boxes[valid]
    candidate_scores = candidate_scores[valid]
    keep = _nms(candidate_boxes, candidate_scores, nms_threshold)
    return [
        {"bbox": [float(value) for value in candidate_boxes[item]], "confidence": float(candidate_scores[item])}
        for item in keep
    ]


def _nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    if boxes.size == 0:
        return []
    x1, y1, x2, y2 = (boxes[:, index] for index in range(4))
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        intersection = np.maximum(0.0, np.minimum(x2[current], x2[rest]) - np.maximum(x1[current], x1[rest])) * np.maximum(
            0.0, np.minimum(y2[current], y2[rest]) - np.maximum(y1[current], y1[rest])
        )
        union = areas[current] + areas[rest] - intersection
        overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
        order = rest[overlap <= threshold]
    return keep


def _threshold_metrics(predictions: list[list[dict]], ground_truth: list[list[list[float]]], threshold: float) -> dict:
    true_positive = false_positive = false_negative = 0
    matched_ious = []
    for image_predictions, image_truth in zip(predictions, ground_truth):
        matches, unmatched_predictions, unmatched_truth = _match(
            [item for item in image_predictions if item["confidence"] >= threshold], image_truth, 0.50
        )
        true_positive += len(matches)
        false_positive += len(unmatched_predictions)
        false_negative += len(unmatched_truth)
        matched_ious.extend(item[2] for item in matches)
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2.0 * precision * recall / max(1e-12, precision + recall), 6),
        "mean_matched_iou": round(statistics.fmean(matched_ious), 6) if matched_ious else 0.0,
    }


def _average_precision(predictions: list[list[dict]], ground_truth: list[list[list[float]]], iou_threshold: float) -> float:
    ranked = sorted(
        ((item["confidence"], image_index, item["bbox"]) for image_index, items in enumerate(predictions) for item in items),
        key=lambda item: item[0],
        reverse=True,
    )
    matched = [set() for _ in ground_truth]
    true_positive = []
    false_positive = []
    for _, image_index, box in ranked:
        candidates = ground_truth[image_index]
        best_index = -1
        best_iou = 0.0
        for truth_index, truth_box in enumerate(candidates):
            if truth_index in matched[image_index]:
                continue
            overlap = _iou(box, truth_box)
            if overlap > best_iou:
                best_iou = overlap
                best_index = truth_index
        is_match = best_index >= 0 and best_iou >= iou_threshold
        true_positive.append(1.0 if is_match else 0.0)
        false_positive.append(0.0 if is_match else 1.0)
        if is_match:
            matched[image_index].add(best_index)
    total_truth = sum(len(items) for items in ground_truth)
    if not ranked or total_truth == 0:
        return 0.0
    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recalls = cumulative_tp / float(total_truth)
    precisions = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-12)
    return float(np.mean([np.max(precisions[recalls >= level], initial=0.0) for level in np.linspace(0.0, 1.0, 101)]))


def _small_target_recall(predictions: list[list[dict]], ground_truth: list[list[list[float]]], threshold: float) -> float:
    small = found = 0
    for image_predictions, image_truth in zip(predictions, ground_truth):
        filtered = [item for item in image_predictions if item["confidence"] >= threshold]
        for truth in image_truth:
            if (truth[2] - truth[0]) * (truth[3] - truth[1]) > 32.0 * 32.0:
                continue
            small += 1
            if any(_iou(item["bbox"], truth) >= 0.50 for item in filtered):
                found += 1
    return found / max(1, small)


def _compare_predictions(baseline: list[list[dict]], candidate: list[list[dict]], threshold: float) -> dict:
    matched_ious = []
    confidence_differences = []
    baseline_unmatched = candidate_extra = 0
    for first, second in zip(baseline, candidate):
        first_filtered = [item for item in first if item["confidence"] >= threshold]
        second_filtered = [item for item in second if item["confidence"] >= threshold]
        matches, first_missing, second_missing = _match(first_filtered, second_filtered, 0.50, both_predictions=True)
        matched_ious.extend(item[2] for item in matches)
        confidence_differences.extend(
            abs(first_filtered[item[0]]["confidence"] - second_filtered[item[1]]["confidence"])
            for item in matches
        )
        baseline_unmatched += len(first_missing)
        candidate_extra += len(second_missing)
    return {
        "matched_boxes": len(matched_ious),
        "baseline_unmatched": baseline_unmatched,
        "candidate_extra": candidate_extra,
        "mean_iou": round(statistics.fmean(matched_ious), 6) if matched_ious else 0.0,
        "minimum_iou": round(min(matched_ious), 6) if matched_ious else 0.0,
        "mean_confidence_difference": round(statistics.fmean(confidence_differences), 6) if confidence_differences else 0.0,
        "maximum_confidence_difference": round(max(confidence_differences), 6) if confidence_differences else 0.0,
    }


def _match(first: list, second: list, threshold: float, *, both_predictions: bool = False) -> tuple[list, list, list]:
    candidates = []
    for first_index, first_item in enumerate(first):
        first_box = first_item["bbox"] if both_predictions else first_item["bbox"]
        for second_index, second_item in enumerate(second):
            second_box = second_item["bbox"] if both_predictions else second_item
            candidates.append((first_index, second_index, _iou(first_box, second_box)))
    candidates.sort(key=lambda item: item[2], reverse=True)
    used_first = set()
    used_second = set()
    matches = []
    for first_index, second_index, overlap in candidates:
        if overlap < threshold:
            break
        if first_index in used_first or second_index in used_second:
            continue
        used_first.add(first_index)
        used_second.add(second_index)
        matches.append((first_index, second_index, overlap))
    return (
        matches,
        [index for index in range(len(first)) if index not in used_first],
        [index for index in range(len(second)) if index not in used_second],
    )


def _iou(first: Sequence[float], second: Sequence[float]) -> float:
    x1 = max(float(first[0]), float(second[0]))
    y1 = max(float(first[1]), float(second[1]))
    x2 = min(float(first[2]), float(second[2]))
    y2 = min(float(first[3]), float(second[3]))
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    first_area = max(0.0, float(first[2]) - float(first[0])) * max(0.0, float(first[3]) - float(first[1]))
    second_area = max(0.0, float(second[2]) - float(second[0])) * max(0.0, float(second[3]) - float(second[1]))
    return intersection / max(1e-12, first_area + second_area - intersection)


def _yolo_box(cx: float, cy: float, width: float, height: float, image_width: int, image_height: int) -> list[float]:
    return [
        (cx - width / 2.0) * image_width,
        (cy - height / 2.0) * image_height,
        (cx + width / 2.0) * image_width,
        (cy + height / 2.0) * image_height,
    ]


def _percentile(values: Iterable[float], percentile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return 0.0
    index = int(round((percentile / 100.0) * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, index))]


if __name__ == "__main__":
    raise SystemExit(main())
