"""Entry/exit counting using anonymous virtual line crossings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from vision.types import CountingEvent, TrackedPerson, bbox_centroid

Point = Tuple[float, float]


@dataclass(frozen=True)
class LineCrossingConfig:
    line_start: Point
    line_end: Point
    enter_direction: str = "positive_to_negative"


class EntryExitCounter:
    """Convert track centroid line crossings into ENTER and EXIT events."""

    def __init__(self, config: LineCrossingConfig) -> None:
        if config.enter_direction not in {"positive_to_negative", "negative_to_positive"}:
            raise ValueError("enter_direction must be positive_to_negative or negative_to_positive")
        self.config = config
        self.current_people_inside = 0
        self.total_entered = 0
        self.total_exited = 0
        self._last_sides: Dict[int, int] = {}

    def update(self, tracks: Iterable[TrackedPerson]) -> List[CountingEvent]:
        events: List[CountingEvent] = []
        for track in tracks:
            side = self._side(bbox_centroid(track.bbox))
            if side == 0:
                continue
            previous = self._last_sides.get(track.track_id)
            self._last_sides[track.track_id] = side
            if previous is None or previous == side:
                continue
            direction = self._direction(previous, side)
            event_type = "ENTER" if direction == self.config.enter_direction else "EXIT"
            event = CountingEvent.now(event_type, track.track_id)
            events.append(event)
            if event_type == "ENTER":
                self.total_entered += 1
                self.current_people_inside += 1
            else:
                self.total_exited += 1
                self.current_people_inside = max(0, self.current_people_inside - 1)
        return events

    def snapshot(self) -> dict:
        return {
            "current_people": self.current_people_inside,
            "today_entered": self.total_entered,
            "today_exited": self.total_exited,
        }

    def _side(self, point: Point) -> int:
        x1, y1 = self.config.line_start
        x2, y2 = self.config.line_end
        px, py = point
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross > 0:
            return 1
        if cross < 0:
            return -1
        return 0

    @staticmethod
    def _direction(previous: int, current: int) -> str:
        before = "positive" if previous > 0 else "negative"
        after = "positive" if current > 0 else "negative"
        return f"{before}_to_{after}"


def create_default_counter(frame_width: int, frame_height: int) -> EntryExitCounter:
    x = frame_width / 2.0
    return EntryExitCounter(
        LineCrossingConfig(
            line_start=(x, 0.0),
            line_end=(x, float(frame_height)),
            enter_direction="positive_to_negative",
        )
    )
