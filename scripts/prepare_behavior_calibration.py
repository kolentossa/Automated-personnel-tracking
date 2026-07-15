#!/usr/bin/env python3
"""Build a deterministic, validated RKNN calibration image list."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Iterable

import cv2

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cigdet-train", required=True, type=Path)
    parser.add_argument("--coco-images", required=True, type=Path)
    parser.add_argument("--curated-negatives", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--positive-count", type=int, default=220)
    parser.add_argument("--coco-count", type=int, default=73)
    parser.add_argument("--curated-count", type=int, default=7)
    parser.add_argument("--seed", type=int, default=3588)
    args = parser.parse_args()

    requested = args.positive_count + args.coco_count + args.curated_count
    if requested < 200:
        parser.error("Calibration requires at least 200 images")

    rng = random.Random(args.seed)
    groups = {
        "cigdet_positive": _sample(args.cigdet_train, args.positive_count, rng),
        "coco_negative": _sample(args.coco_images, args.coco_count, rng),
        "curated_negative": _sample(args.curated_negatives, args.curated_count, rng),
    }
    selected = [path for paths in groups.values() for path in paths]
    if len({str(path.resolve()) for path in selected}) != requested:
        raise RuntimeError("Calibration selection contains duplicate paths")

    dimensions = {}
    for path in selected:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"Calibration image is unreadable: {path}")
        dimensions[str(path.resolve())] = [int(image.shape[1]), int(image.shape[0])]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(str(path.resolve()) for path in selected) + "\n"
    args.output.write_text(content, encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "seed": args.seed,
        "total_images": len(selected),
        "groups": {name: len(paths) for name, paths in groups.items()},
        "list_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "sources": {
            "cigdet_positive": "CigDet v1 (Mendeley Data, CC BY 4.0)",
            "coco_negative": "COCO128 subset (COCO image licenses; see bundled LICENSE)",
            "curated_negative": "Wikimedia Commons files documented in behavior model selection notes",
        },
        "images": [
            {"path": str(path.resolve()), "width": dimensions[str(path.resolve())][0], "height": dimensions[str(path.resolve())][1]}
            for path in selected
        ],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in ("total_images", "groups", "list_sha256")}, indent=2))
    return 0


def _sample(root: Path, count: int, rng: random.Random) -> list[Path]:
    if count < 0:
        raise ValueError("Sample counts must be non-negative")
    if not root.is_dir():
        raise FileNotFoundError(f"Calibration source directory does not exist: {root}")
    candidates = sorted(_images(root), key=lambda item: str(item).lower())
    if len(candidates) < count:
        raise RuntimeError(f"Requested {count} images from {root}, but only {len(candidates)} exist")
    return sorted(rng.sample(candidates, count), key=lambda item: str(item).lower())


def _images(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            yield path


if __name__ == "__main__":
    raise SystemExit(main())
