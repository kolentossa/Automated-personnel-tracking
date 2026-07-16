#!/usr/bin/env python3
"""Compare full-frame and person-ROI phone recall on private validation images."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402
from app.detector import (  # noqa: E402
    PHONE_LABELS,
    PersonDetector,
    _label,
    _normalise_phone_roi_config,
    _phone_roi_crop,
)
from vision.types import Detection  # noqa: E402


DEFAULT_SCENARIOS = (
    "phone",
    "phone-call",
    "phone-context",
    "phone-playing",
    "phone-near-face",
)
ROI_THRESHOLD_CURVE = (0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.35)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "private_accuracy_validation" / "corpus",
    )
    parser.add_argument("--split", default="final_test")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--model-path")
    parser.add_argument("--model-family")
    parser.add_argument("--expected-sha256")
    parser.add_argument(
        "--include-negatives",
        action="store_true",
        help="Evaluate every image in the split and report precision/recall/F1.",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenarios = set(args.scenarios or DEFAULT_SCENARIOS)
    manifest_path = args.corpus / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [row for row in rows if row.get("split") == args.split]
    if not args.include_negatives:
        rows = [row for row in rows if row.get("scenario") in scenarios]
    if not rows:
        raise SystemExit(f"No matching validation rows in {manifest_path}")

    detector_config = dict(load_config()["detector"])
    if args.model_path:
        detector_config["model_path"] = args.model_path
    if args.model_family:
        detector_config["model_family"] = args.model_family
    if args.expected_sha256:
        detector_config["expected_sha256"] = args.expected_sha256
    detector = PersonDetector(detector_config)
    if not detector.npu_enabled:
        raise SystemExit(f"Primary RKNN detector is unavailable: {detector.warning}")
    roi_config = _normalise_phone_roi_config(detector.phone_roi_config)
    modes = list(roi_config["crop_modes"])
    if not modes:
        raise SystemExit("phone_roi_refinement.crop_modes is empty")

    counters = Counter()
    timings: dict[str, list[float]] = {
        "full_frame": [],
        "legacy_upper_body": [],
        **{str(mode["name"]): [] for mode in modes},
    }
    per_scenario: dict[str, Counter] = {}
    score_records: list[dict[str, object]] = []
    try:
        for row in rows:
            image_path = args.corpus / str(row["path"])
            frame = cv2.imread(str(image_path))
            if frame is None:
                raise RuntimeError(f"Could not read {image_path}")
            scenario = str(row["scenario"])
            is_positive = scenario in scenarios
            scenario_counts = per_scenario.setdefault(scenario, Counter())
            counters["images"] += 1
            counters["positive_images" if is_positive else "negative_images"] += 1
            scenario_counts["images"] += 1

            full = _timed_detect(detector._detector, frame, None, timings["full_frame"])
            people = [item for item in full if _label(item) == "person"]
            full_score = _max_phone_score(full)
            full_hit = full_score > 0.0
            if full_hit:
                _record_hit(counters, scenario_counts, "full_frame", is_positive)
            if not people:
                counters["person_misses"] += 1
                scenario_counts["person_misses"] += 1
                _record_classification(counters, full_hit, is_positive)
                score_records.append(
                    {
                        "scenario": scenario,
                        "positive": is_positive,
                        "full_frame": full_score,
                        "legacy_upper_body": 0.0,
                        "multi_region": 0.0,
                        "crop_scores": {},
                    }
                )
                continue

            person = max(people, key=lambda item: _area(item.bbox))
            height, width = frame.shape[:2]
            legacy_score = _detect_crop(
                detector,
                frame,
                person,
                _phone_roi_crop(person.bbox, width, height, roi_config),
                timings["legacy_upper_body"],
                None,
            )
            if legacy_score > 0.0:
                _record_hit(counters, scenario_counts, "legacy_upper_body", is_positive)
            if full_hit or legacy_score > 0.0:
                _record_hit(counters, scenario_counts, "full_or_legacy", is_positive)

            mode_scores: list[float] = []
            for mode in modes:
                name = str(mode["name"])
                crop_box = _phone_roi_crop(person.bbox, width, height, roi_config, mode)
                score = _detect_crop(
                    detector, frame, person, crop_box, timings[name], mode
                )
                mode_scores.append(score)
                if score > 0.0:
                    _record_hit(counters, scenario_counts, name, is_positive)
            if any(score > 0.0 for score in mode_scores):
                _record_hit(
                    counters, scenario_counts, "multi_region_union", is_positive
                )
            production_hit = full_hit or any(score > 0.0 for score in mode_scores)
            if production_hit:
                _record_hit(
                    counters, scenario_counts, "full_or_multi_region", is_positive
                )
            _record_classification(counters, production_hit, is_positive)
            score_records.append(
                {
                    "scenario": scenario,
                    "positive": is_positive,
                    "full_frame": full_score,
                    "legacy_upper_body": legacy_score,
                    "multi_region": max(mode_scores, default=0.0),
                    "crop_scores": {
                        str(mode["name"]): score
                        for mode, score in zip(modes, mode_scores)
                    },
                }
            )
    finally:
        detector.release()

    total = int(counters["images"])
    positive_total = int(counters["positive_images"])
    true_positive = int(counters["true_positive"])
    false_positive = int(counters["false_positive"])
    false_negative = int(counters["false_negative"])
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    result = {
        "schema_version": 2,
        "dataset": str(args.corpus),
        "detector": detector.name,
        "model_path": str(detector_config.get("model_path") or ""),
        "model_family": str(detector_config.get("model_family") or ""),
        "model_sha256": str(detector_config.get("expected_sha256") or ""),
        "split": args.split,
        "scenarios": sorted(scenarios),
        "images": total,
        "positive_images": positive_total,
        "negative_images": int(counters["negative_images"]),
        "person_misses": int(counters["person_misses"]),
        "hits": {
            key: int(value)
            for key, value in sorted(counters.items())
            if key.endswith("_hits") and not key.startswith(("positive_", "negative_"))
        },
        "false_positive_hits": {
            key.removeprefix("negative_").removesuffix("_hits"): int(value)
            for key, value in sorted(counters.items())
            if key.startswith("negative_") and key.endswith("_hits")
        },
        "recall": {
            key.removeprefix("positive_").removesuffix("_hits"): round(
                value / max(1, positive_total), 4
            )
            for key, value in sorted(counters.items())
            if key.startswith("positive_") and key.endswith("_hits")
        },
        "production_accuracy": {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": int(counters["true_negative"]),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "roi_threshold_curve": {
            f"{threshold:.2f}": _score_metrics(score_records, threshold)
            for threshold in ROI_THRESHOLD_CURVE
        },
        "mean_inference_ms": {
            key: round(statistics.fmean(values), 2) if values else 0.0
            for key, values in timings.items()
        },
        "per_scenario": {
            scenario: {key: int(value) for key, value in sorted(values.items())}
            for scenario, values in sorted(per_scenario.items())
        },
        "per_image_scores": score_records,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


def _timed_detect(detector, frame, thresholds, values: list[float]) -> list[Detection]:
    started = time.monotonic()
    if thresholds is None:
        detections = detector.detect(frame)
    else:
        detections = detector.detect(frame, class_confidence_thresholds=thresholds)
    values.append((time.monotonic() - started) * 1000.0)
    return detections


def _detect_crop(
    detector: PersonDetector,
    frame,
    person: Detection,
    crop_box: tuple[int, int, int, int],
    timings: list[float],
    mode: dict | None,
) -> float:
    del person
    x1, y1, x2, y2 = crop_box
    crop = frame[y1:y2, x1:x2]
    backend = detector._detector
    if (
        mode
        and str(mode.get("detector") or "primary") == "context"
        and detector._phone_context_detector is not None
    ):
        backend = detector._phone_context_detector
    threshold = (
        float(mode["confidence_threshold"])
        if mode and mode.get("confidence_threshold") is not None
        else float(detector.phone_roi_config["confidence_threshold"])
    )
    detections = _timed_detect(
        backend,
        crop,
        {detector._phone_label: threshold},
        timings,
    )
    return _max_phone_score(detections)


def _has_phone(detections: Iterable[Detection]) -> bool:
    return any(_label(item) in PHONE_LABELS for item in detections)


def _max_phone_score(detections: Iterable[Detection]) -> float:
    return max(
        (float(item.confidence) for item in detections if _label(item) in PHONE_LABELS),
        default=0.0,
    )


def _score_metrics(records: list[dict[str, object]], roi_threshold: float) -> dict:
    def detected(row: dict[str, object]) -> bool:
        return (
            float(row["full_frame"]) > 0.0
            or float(row["multi_region"]) >= roi_threshold
        )

    true_positive = sum(detected(row) for row in records if bool(row["positive"]))
    false_positive = sum(detected(row) for row in records if not bool(row["positive"]))
    positive_total = sum(bool(row["positive"]) for row in records)
    negative_total = len(records) - positive_total
    false_negative = positive_total - true_positive
    true_negative = negative_total - false_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, positive_total)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    scenarios = sorted({str(row["scenario"]) for row in records})
    return {
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
            if (selected := [row for row in records if row["scenario"] == scenario])
        },
    }


def _record_hit(
    counters: Counter,
    scenario_counts: Counter,
    name: str,
    is_positive: bool,
) -> None:
    key = f"{name}_hits"
    counters[key] += 1
    counters[f"{'positive' if is_positive else 'negative'}_{key}"] += 1
    scenario_counts[key] += 1


def _record_classification(counters: Counter, detected: bool, positive: bool) -> None:
    if detected and positive:
        counters["true_positive"] += 1
    elif detected:
        counters["false_positive"] += 1
    elif positive:
        counters["false_negative"] += 1
    else:
        counters["true_negative"] += 1


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


if __name__ == "__main__":
    raise SystemExit(main())
