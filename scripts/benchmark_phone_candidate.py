#!/usr/bin/env python3
"""Evaluate a PyTorch/ONNX phone candidate on the private fixed corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402
from app.detector import _normalise_phone_roi_config, _phone_roi_crop  # noqa: E402


POSITIVE_SCENARIOS = {
    "phone",
    "phone-call",
    "phone-context",
    "phone-playing",
    "phone-near-face",
}
THRESHOLDS = (0.10, 0.20, 0.30, 0.50)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--event-log", type=Path)
    parser.add_argument("--split", default="final_test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit(
            "Install the x86 audit dependency: pip install ultralytics"
        ) from exc

    rows = [
        json.loads(line)
        for line in (args.corpus / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == args.split]
    event_people = _load_event_people(args.event_log)
    model = YOLO(str(args.model), task="detect")
    phone_class_ids = _phone_class_ids(model.names)
    if not phone_class_ids:
        raise SystemExit(f"No phone class in model names: {model.names}")
    detector_config = load_config()["detector"]
    config = _normalise_phone_roi_config(detector_config["phone_roi_refinement"])
    modes = list(config["crop_modes"])

    scores: dict[str, list[dict[str, Any]]] = defaultdict(list)
    durations: list[float] = []
    for index, row in enumerate(rows):
        image_path = args.corpus / str(row["path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read {image_path}")
        started = time.monotonic()
        full_score = _predict_score(model, image, args.device, phone_class_ids)
        durations.append((time.monotonic() - started) * 1000.0)
        record: dict[str, Any] = {
            "path": str(row["path"]),
            "scenario": str(row["scenario"]),
            "full": full_score,
            "legacy_upper_body": 0.0,
            "multi_region": 0.0,
        }

        person_bbox = event_people.get(Path(str(row["path"])).name)
        if person_bbox is not None:
            height, width = image.shape[:2]
            legacy_box = _phone_roi_crop(person_bbox, width, height, config)
            record["legacy_upper_body"] = _predict_score(
                model, _crop(image, legacy_box), args.device, phone_class_ids
            )
            mode_scores = []
            for mode in modes:
                crop_box = _phone_roi_crop(person_bbox, width, height, config, mode)
                mode_scores.append(
                    _predict_score(
                        model, _crop(image, crop_box), args.device, phone_class_ids
                    )
                )
            record["multi_region"] = max(mode_scores, default=0.0)
        record["full_or_legacy"] = max(record["full"], record["legacy_upper_body"])
        record["full_or_multi_region"] = max(record["full"], record["multi_region"])
        bucket = "positive" if row["scenario"] in POSITIVE_SCENARIOS else "negative"
        scores[bucket].append(record)
        if (index + 1) % 25 == 0:
            print(f"processed={index + 1}/{len(rows)}", file=sys.stderr)

    result = {
        "schema_version": 1,
        "model": str(args.model),
        "model_sha256": _sha256(args.model),
        "model_names": {str(key): value for key, value in model.names.items()},
        "phone_class_ids": phone_class_ids,
        "split": args.split,
        "positive_scenarios": sorted(POSITIVE_SCENARIOS),
        "negative_scenarios": sorted(
            {
                row["scenario"]
                for row in rows
                if row["scenario"] not in POSITIVE_SCENARIOS
            }
        ),
        "positive_images": len(scores["positive"]),
        "negative_images": len(scores["negative"]),
        "mean_full_frame_inference_ms": round(statistics.fmean(durations), 2),
        "thresholds": {
            f"{threshold:.2f}": _metrics_at_threshold(scores, threshold)
            for threshold in THRESHOLDS
        },
        "production_profile": _production_metrics(
            scores,
            float(detector_config["class_confidence_thresholds"]["cell phone"]),
            float(config["confidence_threshold"]),
        ),
        "per_image": scores,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _metrics_at_threshold(
    scores: dict[str, list[dict[str, Any]]], threshold: float
) -> dict:
    paths = (
        "full",
        "legacy_upper_body",
        "multi_region",
        "full_or_legacy",
        "full_or_multi_region",
    )
    result = {}
    for path in paths:
        positive_hits = sum(float(row[path]) >= threshold for row in scores["positive"])
        negative_hits = sum(float(row[path]) >= threshold for row in scores["negative"])
        result[path] = {
            "positive_hits": positive_hits,
            "recall": round(positive_hits / max(1, len(scores["positive"])), 4),
            "negative_hits": negative_hits,
            "negative_trigger_rate": round(
                negative_hits / max(1, len(scores["negative"])), 4
            ),
        }
    for scenario in sorted(POSITIVE_SCENARIOS):
        selected = [row for row in scores["positive"] if row["scenario"] == scenario]
        result[f"scenario:{scenario}"] = {
            "images": len(selected),
            "full_or_multi_region_hits": sum(
                float(row["full_or_multi_region"]) >= threshold for row in selected
            ),
        }
    return result


def _production_metrics(
    scores: dict[str, list[dict[str, Any]]],
    full_threshold: float,
    roi_threshold: float,
) -> dict[str, Any]:
    def detected(row: dict[str, Any]) -> bool:
        return (
            float(row["full"]) >= full_threshold
            or float(row["multi_region"]) >= roi_threshold
        )

    true_positive = sum(detected(row) for row in scores["positive"])
    false_positive = sum(detected(row) for row in scores["negative"])
    false_negative = len(scores["positive"]) - true_positive
    true_negative = len(scores["negative"]) - false_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    scenarios = sorted(
        {row["scenario"] for bucket in scores.values() for row in bucket}
    )
    return {
        "full_frame_threshold": full_threshold,
        "roi_threshold": roi_threshold,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "per_scenario": {
            scenario: {
                "images": len(selected),
                "hits": sum(detected(row) for row in selected),
            }
            for scenario in scenarios
            if (
                selected := [
                    row
                    for bucket in scores.values()
                    for row in bucket
                    if row["scenario"] == scenario
                ]
            )
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_event_people(
    path: Path | None,
) -> dict[str, tuple[float, float, float, float]]:
    if path is None or not path.exists():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        person = (event.get("bboxes") or {}).get("person")
        snapshot = event.get("snapshot_path")
        if person and snapshot:
            result[Path(str(snapshot)).name] = tuple(float(value) for value in person)
    return result


def _predict_score(model, image, device: str, phone_class_ids: list[int]) -> float:
    predictions = model.predict(
        source=image,
        imgsz=640,
        conf=0.01,
        iou=0.45,
        device=device,
        classes=phone_class_ids,
        verbose=False,
    )
    boxes = predictions[0].boxes
    if boxes is None or len(boxes) == 0:
        return 0.0
    return max(float(value) for value in boxes.conf.cpu().tolist())


def _phone_class_ids(names: dict[int, str]) -> list[int]:
    accepted = {"cell phone", "mobile phone", "mobilephone", "phone"}
    result = []
    for class_id, name in names.items():
        normalised = str(name).strip().lower().replace("_", " ").replace("-", " ")
        normalised = " ".join(normalised.split())
        if normalised in accepted:
            result.append(int(class_id))
    return result


def _crop(image, box: tuple[int, int, int, int]):
    x1, y1, x2, y2 = box
    return image[y1:y2, x1:x2]


if __name__ == "__main__":
    raise SystemExit(main())
