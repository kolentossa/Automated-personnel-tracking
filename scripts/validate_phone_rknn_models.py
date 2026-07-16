#!/usr/bin/env python3
"""Validate phone candidate and verifier RKNN models on a real image."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.detectors.rknn_classifier import RKNNPhoneVerifier  # noqa: E402
from app.detectors.rknn_yolo import RKNNYoloDetector, _preprocess  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--verifier", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--candidate-threshold", type=float, default=0.10)
    parser.add_argument("--candidate-input-size", type=int, default=640)
    parser.add_argument("--candidate-input-height", type=int)
    parser.add_argument("--candidate-input-width", type=int)
    parser.add_argument("--verifier-threshold", type=float, default=0.50)
    parser.add_argument("--core-mask", default="0_1_2")
    parser.add_argument(
        "--roi",
        help="Optional x1,y1,x2,y2 crop, matching the production person-ROI path",
    )
    args = parser.parse_args()
    for path in (args.candidate, args.verifier, args.image):
        if not path.is_file():
            parser.error(f"file does not exist: {path}")
    if args.iterations < 1:
        parser.error("--iterations must be positive")

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"could not decode image: {args.image}")
    source_shape = list(image.shape)
    roi = None
    if args.roi:
        values = [float(value.strip()) for value in args.roi.split(",")]
        if len(values) != 4:
            parser.error("--roi must contain x1,y1,x2,y2")
        height, width = image.shape[:2]
        x1, y1, x2, y2 = values
        left = max(0, int(math.floor(x1)))
        top = max(0, int(math.floor(y1)))
        right = min(width, int(math.ceil(x2)))
        bottom = min(height, int(math.ceil(y2)))
        image = image[top:bottom, left:right]
        if image.size == 0:
            parser.error("--roi produced an empty image")
        roi = [left, top, right, bottom]
    candidate_input_shape = args.candidate_input_size
    if args.candidate_input_height or args.candidate_input_width:
        if not args.candidate_input_height or not args.candidate_input_width:
            parser.error("candidate input height and width must be provided together")
        candidate_input_shape = (
            args.candidate_input_height,
            args.candidate_input_width,
        )
    candidate = RKNNYoloDetector(
        args.candidate,
        model_family="yolov8",
        input_size=candidate_input_shape,
        confidence_threshold=args.candidate_threshold,
        nms_threshold=0.45,
        class_filter=["cell phone"],
        class_names=["cell phone"],
        core_mask=args.core_mask,
    )
    verifier = RKNNPhoneVerifier(
        args.verifier, input_size=224, core_mask=args.core_mask
    )
    candidate_times = []
    verifier_times = []
    detections = []
    try:
        prepared, _, _ = _preprocess(image, candidate_input_shape)
        raw_outputs = candidate._rknn.inference(inputs=[prepared])
        primary_output = np.asarray(raw_outputs[0], dtype=np.float32).squeeze()
        top_indexes = np.argsort(primary_output[4])[-10:][::-1]
        raw_top_rows = [
            {
                "index": int(index),
                "values": [
                    round(float(value), 6) for value in primary_output[:, index]
                ],
            }
            for index in top_indexes
        ]
        raw_contract = [
            {
                "shape": list(output.shape),
                "dtype": str(output.dtype),
                "min": round(float(output.min()), 6),
                "max": round(float(output.max()), 6),
                "sample": [round(float(value), 6) for value in output.reshape(-1)[:12]],
            }
            for output in raw_outputs
        ]
        for _ in range(3):
            candidate.detect(image)
        for _ in range(args.iterations):
            detections = candidate.detect(image)
            candidate_times.append(sum(candidate.last_profile.values()))
        verified = []
        for detection in detections[:5]:
            crop = _context_crop(image, detection.bbox, 1.8)
            probability = verifier.predict_phone_probability(crop)
            verifier_times.append(sum(verifier.last_profile.values()))
            verified.append(
                {
                    "bbox": [round(float(value), 2) for value in detection.bbox],
                    "candidate_confidence": round(float(detection.confidence), 6),
                    "phone_probability": round(probability, 6),
                    "accepted": probability >= args.verifier_threshold,
                }
            )
        report = {
            "candidate_model": str(args.candidate),
            "candidate_sha256": _sha256(args.candidate),
            "verifier_model": str(args.verifier),
            "verifier_sha256": _sha256(args.verifier),
            "image": str(args.image),
            "source_image_shape": source_shape,
            "image_shape": list(image.shape),
            "roi": roi,
            "candidate_input_shape": list(candidate_input_shape)
            if isinstance(candidate_input_shape, tuple)
            else [candidate_input_shape, candidate_input_shape],
            "candidate_output_contract": "single-class YOLOv8 [1,5,N] or transpose",
            "candidate_raw_outputs": raw_contract,
            "candidate_raw_top_rows": raw_top_rows,
            "verifier_output_contract": "one finite two-class tensor [not_phone,phone]",
            "candidate_count": len(detections),
            "verified": verified,
            "candidate_latency_ms": _summary(candidate_times),
            "verifier_latency_ms": _summary(verifier_times),
            "npu_enabled": candidate.npu_enabled and verifier.npu_enabled,
        }
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        candidate.release()
        verifier.release()
    return 0


def _context_crop(image, bbox, scale: float):
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(value) for value in bbox)
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    side = max(12.0, max(x2 - x1, y2 - y1) * scale)
    left = max(0, int(math.floor(center_x - side / 2.0)))
    top = max(0, int(math.floor(center_y - side / 2.0)))
    right = min(width, int(math.ceil(center_x + side / 2.0)))
    bottom = min(height, int(math.ceil(center_y + side / 2.0)))
    return image[top:bottom, left:right]


def _summary(values):
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    ordered = sorted(values)
    return {
        "mean": round(statistics.fmean(values), 3),
        "p50": round(statistics.median(values), 3),
        "p95": round(
            ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)], 3
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
