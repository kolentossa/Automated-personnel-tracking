#!/usr/bin/env python3
"""Build a scene-separated private validation corpus without committing media."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cigdet", required=True, type=Path, help="CigDet_dataset directory"
    )
    parser.add_argument("--coco", required=True, type=Path, help="coco128 directory")
    parser.add_argument("--curated-negatives", required=True, type=Path)
    parser.add_argument("--board-evidence", type=Path)
    parser.add_argument("--board-review-index", type=Path)
    parser.add_argument("--board-review-labels", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        if not args.force:
            parser.error(f"Output exists: {args.output}; pass --force to rebuild")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)

    records: list[dict] = []
    _add_cigdet(args.cigdet, args.output, records)
    _add_coco(args.coco, args.output, records)
    _add_curated(args.curated_negatives, args.output, records)
    if bool(args.board_review_index) != bool(args.board_review_labels):
        parser.error(
            "--board-review-index and --board-review-labels must be provided together"
        )
    board_review = _load_board_review(args.board_review_index, args.board_review_labels)
    if args.board_evidence:
        _add_board(args.board_evidence, args.output, records, board_review)

    manifest = args.output / "manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for record in sorted(records, key=lambda item: item["path"]):
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True) + "\n")
    summary = {
        "schema_version": 1,
        "images": len(records),
        "splits": dict(Counter(item["split"] for item in records)),
        "sources": dict(Counter(item["source"] for item in records)),
        "scenarios": dict(Counter(item["scenario"] for item in records)),
        "manifest_sha256": _sha256(manifest),
        "split_policy": "CigDet official split; all board sessions split by UTC date; reviewed board smoking evidence uses fixed manual labels; independent COCO/curated images split deterministically by source item",
    }
    (args.output / "dataset_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    if len(records) < 500:
        raise RuntimeError(
            f"Validation corpus has {len(records)} images; at least 500 are required"
        )
    return 0


def _add_cigdet(source: Path, output: Path, records: list[dict]) -> None:
    for official_split in ("train", "test"):
        directory = source / official_split
        for image in _images(directory):
            split = (
                "final_test" if official_split == "test" else _hash_split(image.stem)
            )
            label = image.with_suffix(".txt")
            destination = _copy_pair(image, label, output, split, "cigdet")
            records.append(
                _record(
                    destination,
                    output,
                    split,
                    "cigdet-v1",
                    "real_cigarette",
                    "cigarette",
                    {"0": "cigarette"},
                )
            )


def _add_coco(source: Path, output: Path, records: list[dict]) -> None:
    image_root = source / "images" / "train2017"
    label_root = source / "labels" / "train2017"
    class_map = {
        "0": "person",
        "39": "bottle",
        "42": "fork",
        "43": "knife",
        "44": "spoon",
        "65": "remote",
        "67": "cell phone",
        "76": "scissors",
        "79": "toothbrush",
    }
    for image in _images(image_root):
        label = label_root / f"{image.stem}.txt"
        labels = _label_ids(label)
        scenario = "phone" if 67 in labels else "general_negative"
        if labels.intersection({39, 42, 43, 44, 65, 76, 79}):
            scenario = "elongated_object_negative"
        split = (
            "final_test"
            if int(hashlib.sha256(image.stem.encode()).hexdigest()[:2], 16) % 3 == 0
            else "tune"
        )
        destination = _copy_pair(image, label, output, split, "coco128")
        records.append(
            _record(destination, output, split, "coco128", scenario, "coco", class_map)
        )


def _add_curated(source: Path, output: Path, records: list[dict]) -> None:
    items = _images(source)
    for image in items:
        scenario = image.stem.replace("_", "-")
        if image.stem.startswith("screwdriver_openimages_"):
            scenario = "screwdriver_negative"
            split = "final_test"
        else:
            bucket = int(hashlib.sha256(image.name.encode("utf-8")).hexdigest()[:8], 16)
            split = "final_test" if bucket % 2 == 0 else "tune"
        destination = _copy_pair(image, None, output, split, "curated")
        records.append(
            _record(
                destination, output, split, "curated-negatives", scenario, "none", {}
            )
        )


def _add_board(
    source: Path,
    output: Path,
    records: list[dict],
    review_by_name: dict[str, str],
) -> None:
    for image in _images(source):
        if not image.name.endswith("_evidence.jpg"):
            continue
        utc_day = image.name[:8]
        split = "final_test" if utc_day >= "20260716" else "tune"
        if "_phone_call_" in image.name:
            scenario = "phone-call"
        elif "_phone_playing_" in image.name:
            scenario = "phone-playing"
        elif "_smoking_" in image.name:
            manual_label = review_by_name.get(image.name)
            scenario = {
                "phone_context": "phone-context",
                "tool_or_pen": "board-tool-negative",
                "screwdriver_or_thin_tool": "board-screwdriver-negative",
                "other_false_positive": "board-other-negative",
            }.get(manual_label, "board-smoking-alert-review")
        else:
            scenario = "board-event"
        destination = _copy_pair(image, None, output, split, "board")
        record = _record(
            destination,
            output,
            split,
            "rk3588-event-evidence",
            scenario,
            "manual_reviewed"
            if image.name in review_by_name
            else "manual_review_required",
            {},
        )
        if image.name in review_by_name:
            record["manual_label"] = review_by_name[image.name]
        records.append(record)


def _load_board_review(
    index_path: Path | None, labels_path: Path | None
) -> dict[str, str]:
    if index_path is None or labels_path is None:
        return {}
    index = json.loads(index_path.read_text(encoding="utf-8"))
    labels_document = json.loads(labels_path.read_text(encoding="utf-8"))
    labels = labels_document.get("labels") or {}
    result: dict[str, str] = {}
    for row in index:
        label = labels.get(str(row.get("index")))
        name = str(row.get("name") or "").strip()
        if name and label:
            result[name] = str(label)
    return result


def _record(
    path: Path,
    root: Path,
    split: str,
    source: str,
    scenario: str,
    annotation_type: str,
    class_map: dict,
) -> dict:
    label = path.with_suffix(".txt")
    return {
        "path": path.relative_to(root).as_posix(),
        "label_path": label.relative_to(root).as_posix() if label.is_file() else None,
        "split": split,
        "group_id": f"{source}:{path.stem}",
        "source": source,
        "scenario": scenario,
        "annotation_type": annotation_type,
        "class_map": class_map,
        "sha256": _sha256(path),
    }


def _copy_pair(
    image: Path, label: Path | None, root: Path, split: str, source: str
) -> Path:
    directory = root / split / source
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / image.name
    shutil.copy2(image, destination)
    if label and label.is_file():
        shutil.copy2(label, destination.with_suffix(".txt"))
    return destination


def _images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return sorted(
        (path for path in directory.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES),
        key=lambda path: path.name.lower(),
    )


def _label_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    return {
        int(float(line.split()[0]))
        for line in path.read_text().splitlines()
        if line.split()
    }


def _hash_split(value: str) -> str:
    bucket = int(hashlib.sha256(value.encode()).hexdigest()[:2], 16) % 5
    return "train" if bucket < 3 else "tune"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
