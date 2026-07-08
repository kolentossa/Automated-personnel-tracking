"""Local anonymous event storage."""

from __future__ import annotations

import json
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List

from vision.types import CountingEvent


class EventStore:
    """Store recent anonymous events in memory and append JSONL locally."""

    def __init__(self, log_path: Path, max_events: int = 200) -> None:
        self.log_path = log_path
        self.max_events = max_events
        self._events: Deque[Dict[str, object]] = deque(maxlen=max_events)
        self._lock = threading.RLock()
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def add(self, event: CountingEvent) -> None:
        payload = event.as_dict()
        with self._lock:
            self._events.appendleft(payload)
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def recent(self, limit: int = 50) -> List[Dict[str, object]]:
        with self._lock:
            return list(self._events)[:limit]
