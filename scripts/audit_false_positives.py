#!/usr/bin/env python3
"""Audit raw, verified, and confirmed behavior outputs on a fixed image manifest."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.behavior import BehaviorEngine, load_behavior_config  # noqa: E402
from app.behavior.detector import BehaviorObjectDetector  # noqa: E402
from app.config import load_config  # noqa: E402
from app.detector import PersonDetector  # noqa: E402
from vision.types import TrackedPerson  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="final_test")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--predictions-output", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    root = args.manifest.resolve().parent
    records = [
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = [item for item in records if item.get("split") == args.split]
    if args.scenarios:
        selected_scenarios = set(args.scenarios)
        records = [
            item for item in records if item.get("scenario") in selected_scenarios
        ]
    if args.limit > 0:
        records = records[: args.limit]
    if not records:
        parser.error(f"No records found for split {args.split}")

    config = load_config()
    behavior_config = load_behavior_config()
    primary = PersonDetector(
        config["detector"], fallback_config=config.get("detection", {})
    )
    behavior_detector = BehaviorObjectDetector(behavior_config["models"]["behavior"])
    behavior = BehaviorEngine(behavior_config)
    if not primary.npu_enabled:
        raise RuntimeError(f"Primary NPU detector is required: {primary.warning}")
    if not behavior_detector.smoking_detection_available:
        raise RuntimeError(
            f"Behavior NPU detector is required: {behavior_detector.error}"
        )

    predictions = []
    started = time.monotonic()
    try:
        for sample_index, record in enumerate(records):
            image_path = root / record["path"]
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f"Unreadable image: {image_path}")
            primary.reset_temporal_state()
            scene = primary.detect_scene(image)
            people = [
                item for item in scene if item.label == "person" or item.class_id == 0
            ]
            tracks = [
                TrackedPerson(index + 1, item.bbox, item.confidence)
                for index, item in enumerate(people)
            ]
            behavior.reset()
            triggers = []
            raw_cigarettes = []
            repeat_count = max(
                1,
                int(behavior_config["smoking"].get("minimum_verified_frames") or 1),
                int(behavior_config["smoking"].get("min_consecutive_frames") or 1),
            )
            for repeat in range(repeat_count + 2):
                detector_frame = repeat * behavior_detector.detect_every_n_frames
                auxiliary = behavior_detector.detect(image, detector_frame)
                raw_cigarettes = [
                    item for item in auxiliary if item.label == "cigarette"
                ]
                result = behavior.update(
                    tracks,
                    list(scene) + auxiliary,
                    image.shape[:2],
                    now_ms=float(repeat * 750),
                    frame_index=detector_frame,
                    behavior_model_fresh=behavior_detector.last_result_is_fresh,
                )
                triggers.extend(result.triggers)
            snapshot = behavior.snapshot()
            phones = [
                item
                for item in scene
                if item.label in {"cell phone", "phone", "mobile phone"}
            ]
            trigger_counts = Counter(trigger.event_type for trigger in triggers)
            trigger_durations = {
                event_type: [
                    round(float(trigger.duration_ms), 3)
                    for trigger in triggers
                    if trigger.event_type == event_type
                ]
                for event_type in sorted(trigger_counts)
            }
            predictions.append(
                {
                    "path": record["path"],
                    "scenario": record.get("scenario", "unknown"),
                    "source": record.get("source", "unknown"),
                    "person_count": len(people),
                    "person_bboxes": [list(item.bbox) for item in people],
                    "phone_count": len(phones),
                    "phone_confidences": [round(item.confidence, 4) for item in phones],
                    "phone_bboxes": [list(item.bbox) for item in phones],
                    "raw_cigarette_count": len(raw_cigarettes),
                    "raw_cigarette_confidences": [
                        round(item.confidence, 4) for item in raw_cigarettes
                    ],
                    "raw_cigarette_bboxes": [
                        list(item.bbox) for item in raw_cigarettes
                    ],
                    "verified_cigarette_count": sum(
                        1
                        for state in snapshot.get(
                            "cigarette_candidate_states", {}
                        ).values()
                        if state.get("verified")
                    ),
                    "confirmed_smoking_count": sum(
                        1 for trigger in triggers if trigger.event_type == "smoking"
                    ),
                    "trigger_event_counts": dict(trigger_counts),
                    "trigger_durations_ms": trigger_durations,
                    "filter_reasons": snapshot.get("cigarette_filter_reasons", {}),
                    "candidate_states": snapshot.get("cigarette_candidate_states", {}),
                }
            )
            if (sample_index + 1) % 25 == 0:
                print(f"processed={sample_index + 1}/{len(records)}", file=sys.stderr)
    finally:
        primary.release()
        behavior_detector.release()

    summary = _summarize(predictions, time.monotonic() - started)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    prediction_path = args.predictions_output or args.output.with_name(
        f"{args.output.stem}_predictions.jsonl"
    )
    with prediction_path.open("w", encoding="utf-8") as handle:
        for item in predictions:
            handle.write(json.dumps(item, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _summarize(predictions: list[dict], elapsed: float) -> dict:
    scenarios = {}
    for scenario in sorted({item["scenario"] for item in predictions}):
        items = [item for item in predictions if item["scenario"] == scenario]
        raw = sum(item["raw_cigarette_count"] for item in items)
        verified = sum(item["verified_cigarette_count"] for item in items)
        confirmed = sum(item["confirmed_smoking_count"] for item in items)
        scenarios[scenario] = {
            "images": len(items),
            "raw_cigarette_candidates": raw,
            "verified_cigarettes": verified,
            "confirmed_smoking_events": confirmed,
            "candidate_reduction_percent": round((raw - verified) / raw * 100.0, 2)
            if raw
            else 0.0,
        }
    reasons: Counter[str] = Counter()
    for item in predictions:
        reasons.update(item.get("filter_reasons", {}))
    positives = [item for item in predictions if item["scenario"] == "real_cigarette"]
    negatives = [item for item in predictions if item["scenario"] != "real_cigarette"]
    true_positive = sum(item["confirmed_smoking_count"] > 0 for item in positives)
    false_positive = sum(item["confirmed_smoking_count"] > 0 for item in negatives)
    false_negative = len(positives) - true_positive
    true_negative = len(negatives) - false_positive
    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "schema_version": 1,
        "images": len(predictions),
        "elapsed_seconds": round(elapsed, 3),
        "images_per_second": round(len(predictions) / max(0.001, elapsed), 3),
        "raw_cigarette_candidates": sum(
            item["raw_cigarette_count"] for item in predictions
        ),
        "verified_cigarettes": sum(
            item["verified_cigarette_count"] for item in predictions
        ),
        "confirmed_smoking_events": sum(
            item["confirmed_smoking_count"] for item in predictions
        ),
        "phone_detections": sum(item["phone_count"] for item in predictions),
        "filter_reasons": dict(reasons),
        "image_level_accuracy": {
            "positive_scenario": "real_cigarette",
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "true_negative": true_negative,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        },
        "event_level_accuracy": {
            "phone_call": _event_metrics(
                predictions,
                "phone_call",
                {"phone-call"},
                eligible_sources={"rk3588-event-evidence"},
            ),
            "phone_playing": _event_metrics(
                predictions,
                "phone_playing",
                {"phone-playing"},
                eligible_sources={"rk3588-event-evidence"},
            ),
            "smoking": _event_metrics(predictions, "smoking", {"real_cigarette"}),
        },
        "scenarios": scenarios,
    }


def _event_metrics(
    predictions: list[dict],
    event_type: str,
    positive_scenarios: set[str],
    eligible_sources: set[str] | None = None,
) -> dict:
    if eligible_sources is not None:
        predictions = [
            item for item in predictions if item.get("source") in eligible_sources
        ]
    true_positive = false_positive = false_negative = true_negative = 0
    duplicate_events = 0
    durations = []
    for item in predictions:
        count = int(item.get("trigger_event_counts", {}).get(event_type, 0))
        expected = item.get("scenario") in positive_scenarios
        predicted = count > 0
        if expected and predicted:
            true_positive += 1
        elif expected:
            false_negative += 1
        elif predicted:
            false_positive += 1
        else:
            true_negative += 1
        duplicate_events += max(0, count - 1)
        durations.extend(item.get("trigger_durations_ms", {}).get(event_type, []))

    precision = true_positive / max(1, true_positive + false_positive)
    recall = true_positive / max(1, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return {
        "evaluated_images": len(predictions),
        "eligible_sources": sorted(eligible_sources) if eligible_sources else ["all"],
        "positive_scenarios": sorted(positive_scenarios),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_negative": true_negative,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "duplicate_events": duplicate_events,
        "average_trigger_ms": round(statistics.mean(durations), 3)
        if durations
        else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
