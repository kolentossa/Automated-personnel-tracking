"""Application state and pipeline lifecycle management."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Dict, Optional

from backend.database import EventStore
from vision.camera import VideoFileCamera
from vision.detection import create_detector
from vision.pipeline import PersonTrackingPipeline
from vision.tracking import ByteTrackTracker

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class AppState:
    """Own the local pipeline thread and expose API-ready snapshots."""

    def __init__(self, project_root: Path = PROJECT_ROOT) -> None:
        self.project_root = project_root.resolve()
        self.events = EventStore(self.project_root / "logs" / "events.jsonl")
        self._lock = threading.RLock()
        self._pipeline: Optional[PersonTrackingPipeline] = None
        self._thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._startup_error = ""

    def start_from_env(self) -> None:
        video = os.environ.get("PERSON_TRACKING_VIDEO")
        model = os.environ.get("PERSON_TRACKING_MODEL")
        self.start(video_path=video, model_path=model)

    def start(self, video_path: Optional[str] = None, model_path: Optional[str] = None) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            try:
                source = self._project_path(video_path or "data/sample.mp4")
                if not source.exists():
                    self._startup_error = f"Video source not found: {source}"
                    return
                model = None
                if model_path:
                    model_candidate = self._project_path(model_path)
                    if model_candidate.exists():
                        model = str(model_candidate)
                camera = VideoFileCamera(str(source), loop=True)
                detector = create_detector(model)
                tracker = ByteTrackTracker()
                self._pipeline = PersonTrackingPipeline(
                    camera=camera,
                    detector=detector,
                    tracker=tracker,
                    event_handler=self.events.add,
                )
                self._stop_event = threading.Event()
                self._thread = threading.Thread(
                    target=self._pipeline.run_forever,
                    args=(self._stop_event,),
                    name="person-tracking-pipeline",
                    daemon=True,
                )
                self._thread.start()
                self._startup_error = ""
            except Exception as exc:  # pragma: no cover - startup resilience path
                self._startup_error = str(exc)

    def stop(self) -> None:
        with self._lock:
            if self._stop_event is not None:
                self._stop_event.set()
            if self._thread is not None:
                self._thread.join(timeout=3.0)
            if self._pipeline is not None:
                self._pipeline.release()
            self._thread = None
            self._stop_event = None
            self._pipeline = None

    def status(self) -> Dict[str, object]:
        with self._lock:
            if self._pipeline is None:
                return {
                    "current_people": 0,
                    "camera_status": "offline",
                    "fps": 0,
                    "frames_processed": 0,
                    "last_error": self._startup_error,
                }
            return self._pipeline.snapshot()

    def statistics(self) -> Dict[str, object]:
        status = self.status()
        return {
            "today_entered": int(status.get("today_entered", 0)),
            "today_exited": int(status.get("today_exited", 0)),
            "current_people": int(status.get("current_people", 0)),
        }

    def _project_path(self, value: str) -> Path:
        candidate = Path(value).expanduser()
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        resolved = candidate.resolve()
        if resolved != self.project_root and self.project_root not in resolved.parents:
            raise ValueError(f"Path must stay inside project directory: {value}")
        return resolved
