"""检测和跟踪之间传递的数据：人体框，以及带编号的人体框。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

BBox = tuple[float, float, float, float]
Frame = Any


@dataclass(frozen=True)
class Detection:
    """单帧检测结果；bbox 依次为左、上、右、下坐标。"""

    bbox: BBox
    confidence: float
    class_id: int = 0
    label: str = "person"


@dataclass(frozen=True)
class TrackedPerson:
    """同一人的跨帧编号；observed=False 表示本帧只有预测，没有检测。"""

    track_id: int
    bbox: BBox
    confidence: float
    observed: bool = True
    observation_bbox: BBox | None = None
    recovered: bool = False


def bbox_centroid(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)
