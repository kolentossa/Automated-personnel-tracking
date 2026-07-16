#!/usr/bin/env python3
"""Build a deterministic, scene-balanced RKNN calibration image list."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--splits", default="train,tune")
    parser.add_argument("--event-log", type=Path)
    parser.add_argument(
        "--materialize-dir",
        type=Path,
        help="Write production-like square person ROI inputs for calibration",
    )
    parser.add_argument("--input-size", type=int, default=640)
    args = parser.parse_args()
    if not args.manifest.is_file():
        parser.error(f"manifest does not exist: {args.manifest}")
    if args.count < 200:
        parser.error("--count must be at least 200")
    splits = {value.strip() for value in args.splits.split(",") if value.strip()}

    groups = defaultdict(list)
    eligible = []
    for line in args.manifest.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("split") not in splits:
            continue
        path = (args.corpus_root / str(row["path"])).resolve()
        if not path.is_file():
            continue
        key = (
            str(row.get("source") or "unknown"),
            str(row.get("scenario") or "unknown"),
        )
        item = (path, row)
        groups[key].append(item)
        eligible.append(item)
    queues = {
        key: deque(sorted(items, key=lambda item: str(item[0])))
        for key, items in sorted(groups.items())
    }
    selected = []
    while queues and len(selected) < args.count:
        for key in list(queues):
            queue = queues[key]
            if queue:
                selected.append(queue.popleft())
                if len(selected) >= args.count:
                    break
            if not queue:
                queues.pop(key, None)
    if len(selected) < args.count:
        raise SystemExit(f"only {len(selected)} calibration images are available")

    paths = [item[0] for item in selected]
    materialized_roi_count = 0
    if args.materialize_dir:
        event_people = _load_event_people(args.event_log)
        args.materialize_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        event_items = [item for item in eligible if item[0].name in event_people]
        materialization_items = [
            (source, row, mode_index)
            for source, row in event_items
            for mode_index in (0, 1)
        ]
        selected_keys = {
            (str(source), int(mode)) for source, _, mode in materialization_items
        }
        for index, (source, row) in enumerate(selected):
            mode_index = index % 2
            if (str(source), mode_index) not in selected_keys:
                materialization_items.append((source, row, mode_index))
        if len(materialization_items) < args.count:
            raise SystemExit(
                f"only {len(materialization_items)} materialized calibration inputs are available"
            )
        for index, (source, row, mode_index) in enumerate(
            materialization_items[: args.count]
        ):
            image = cv2.imread(str(source))
            if image is None:
                raise RuntimeError(f"could not decode calibration image: {source}")
            person = event_people.get(source.name)
            if person is not None:
                image = _person_roi(image, person, mode_index)
                materialized_roi_count += 1
            image = _letterbox(image, args.input_size)
            destination = (
                args.materialize_dir / f"calibration_{index:04d}.jpg"
            ).resolve()
            if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 92]):
                raise RuntimeError(f"could not write calibration image: {destination}")
            paths.append(destination)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(str(path) for path in paths) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "count": len(selected),
                "source_scenario_groups": len(groups),
                "splits": sorted(splits),
                "materialized": bool(args.materialize_dir),
                "person_roi_count": materialized_roi_count,
            },
            sort_keys=True,
        )
    )
    return 0


def _load_event_people(path: Path | None):
    if path is None or not path.is_file():
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line)
        person = (event.get("bboxes") or {}).get("person")
        snapshot = Path(str(event.get("snapshot_path") or "")).name
        if snapshot and isinstance(person, list) and len(person) == 4:
            result[snapshot] = tuple(float(value) for value in person)
    return result


def _person_roi(image, person, mode_index: int):
    frame_height, frame_width = image.shape[:2]
    x1, y1, x2, y2 = person
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    if mode_index == 0:
        left_ratio, top_ratio, right_ratio, bottom_ratio = -0.28, -0.06, 1.28, 0.52
    else:
        left_ratio, top_ratio, right_ratio, bottom_ratio = -0.22, 0.18, 1.22, 0.88
    left = max(0, int(np.floor(x1 + left_ratio * width)))
    top = max(0, int(np.floor(y1 + top_ratio * height)))
    right = min(frame_width, int(np.ceil(x1 + right_ratio * width)))
    bottom = min(frame_height, int(np.ceil(y1 + bottom_ratio * height)))
    if right <= left or bottom <= top:
        return image
    return image[top:bottom, left:right]


def _letterbox(image, input_size: int):
    height, width = image.shape[:2]
    ratio = min(input_size / float(width), input_size / float(height))
    resized = cv2.resize(
        image,
        (int(round(width * ratio)), int(round(height * ratio))),
        interpolation=cv2.INTER_LINEAR,
    )
    canvas = np.full((input_size, input_size, 3), 114, dtype=np.uint8)
    left = (input_size - resized.shape[1]) // 2
    top = (input_size - resized.shape[0]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


if __name__ == "__main__":
    raise SystemExit(main())
