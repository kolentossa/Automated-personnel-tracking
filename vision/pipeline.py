"""Frame processing pipeline from camera to anonymous counting events."""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, Optional

from vision.camera.base import Camera
from vision.counting import EntryExitCounter, create_default_counter
from vision.detection import Detector
from vision.tracking import Tracker
from vision.types import CountingEvent

EventHandler = Callable[[CountingEvent], None]


class PersonTrackingPipeline:
    """Compose camera, detection, tracking, and entry/exit counting."""

    def __init__(
        self,
        camera: Camera,
        detector: Detector,
        tracker: Tracker,
        counter: Optional[EntryExitCounter] = None,
        event_handler: Optional[EventHandler] = None,
    ) -> None:
        self.camera = camera
        self.detector = detector
        self.tracker = tracker
        self.counter = counter
        self.event_handler = event_handler
        self.frames_processed = 0
        self.measured_fps = 0.0
        self.last_error = ""
        self._fps_window_start = time.monotonic()

    def process_once(self) -> bool:
        frame = self.camera.read()
        if frame is None:
            return False
        if self.counter is None:
            height, width = frame.shape[:2]
            self.counter = create_default_counter(width, height)

        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        events = self.counter.update(tracks)
        for event in events:
            if self.event_handler is not None:
                self.event_handler(event)

        self.frames_processed += 1
        if self.frames_processed % 15 == 0:
            now = time.monotonic()
            elapsed = max(0.001, now - self._fps_window_start)
            self.measured_fps = 15.0 / elapsed
            self._fps_window_start = now
        return True

    def run_forever(self, stop_event: Optional[threading.Event] = None) -> None:
        delay = 1.0 / max(1.0, min(60.0, self.camera.fps or 25.0))
        while stop_event is None or not stop_event.is_set():
            started = time.monotonic()
            try:
                ok = self.process_once()
                self.last_error = ""
            except Exception as exc:  # pragma: no cover - backend resilience path
                ok = False
                self.last_error = str(exc)
            elapsed = time.monotonic() - started
            if not ok:
                time.sleep(0.2)
            elif elapsed < delay:
                time.sleep(delay - elapsed)

    def snapshot(self) -> Dict[str, object]:
        counter_stats = self.counter.snapshot() if self.counter is not None else {
            "current_people": 0,
            "today_entered": 0,
            "today_exited": 0,
        }
        return {
            **counter_stats,
            "camera_status": self.camera.status,
            "fps": round(self.measured_fps or self.camera.fps or 0.0, 2),
            "frames_processed": self.frames_processed,
            "last_error": self.last_error,
        }

    def release(self) -> None:
        self.camera.release()
