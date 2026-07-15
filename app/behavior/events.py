"""Asynchronous privacy-safe evidence persistence and replaceable callbacks."""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections import Counter, deque
from dataclasses import replace
from pathlib import Path
from typing import Callable, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from app.behavior.types import BehaviorEvent, BehaviorTrigger
from app.config import PROJECT_ROOT

EventCallback = Callable[[dict], None]


class BehaviorEventManager:
    """Persist evidence off the inference thread and publish unified events.

    Input frames must already have face/head mosaic applied. The unannotated
    snapshot is therefore original event evidence without behavior overlays,
    while still honoring the project's no-unmasked-frame storage policy.
    """

    def __init__(self, config: dict, camera_id: str, project_root: Path = PROJECT_ROOT) -> None:
        self.config = dict(config or {})
        self.camera_id = str(camera_id)
        self.project_root = project_root.resolve()
        self.save_unannotated = bool(self.config.get("save_unannotated_snapshot", True))
        self.save_annotated = bool(self.config.get("save_annotated_snapshot", True))
        self.save_video_clip = bool(self.config.get("save_video_clip", False))
        self.jpeg_quality = max(40, min(100, int(self.config.get("jpeg_quality") or 90)))
        self._queue: queue.Queue[Optional[Tuple[dict, np.ndarray]]] = queue.Queue(
            maxsize=max(1, int(self.config.get("queue_size") or 8))
        )
        self._lock = threading.RLock()
        self._recent: Deque[dict] = deque(maxlen=200)
        self._callbacks: List[EventCallback] = []
        self._counts: Counter[str] = Counter()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dropped = 0
        self._write_failures = 0
        self._last_error = ""
        self._history_loaded = False
        self._logger = logging.getLogger("person_tracking.behavior_events")
        level_name = str(self.config.get("logging_level") or "INFO").upper()
        self._logger.setLevel(getattr(logging, level_name, logging.INFO))

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._load_history()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._worker, name="behavior-event-writer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def register_callback(self, callback: EventCallback) -> None:
        self._callbacks.append(callback)

    def emit(self, trigger: BehaviorTrigger, privacy_safe_frame: np.ndarray) -> dict:
        event = BehaviorEvent.from_trigger(self.camera_id, trigger)
        event, path_error = self._assign_paths(event)
        payload = event.as_dict()
        payload.update({"persistence_status": "queued", "persistence_error": path_error})
        with self._lock:
            self._recent.appendleft(payload)
            self._counts[event.event_type] += 1
        try:
            self._queue.put_nowait((payload, privacy_safe_frame.copy()))
        except queue.Full:
            self._dropped += 1
            self._update_event(event.event_id, "failed", "Evidence queue is full")
        return dict(payload)

    def recent(self, limit: int = 50) -> List[dict]:
        with self._lock:
            return [dict(item) for item in list(self._recent)[: max(1, min(200, int(limit)))]]

    def reset(self) -> None:
        with self._lock:
            self._recent.clear()
            self._counts.clear()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "behavior_events": list(self._recent)[:20],
                "behavior_event_counts": dict(self._counts),
                "behavior_event_queue_depth": self._queue.qsize(),
                "behavior_event_dropped": self._dropped,
                "behavior_event_write_failures": self._write_failures,
                "behavior_event_last_error": self._last_error,
            }

    def _worker(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                item = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if item is None:
                self._queue.task_done()
                continue
            payload, frame = item
            errors: List[str] = [str(payload.get("persistence_error"))] if payload.get("persistence_error") else []
            self._persist_images(payload, frame, errors)
            payload["persistence_status"] = "persisted" if not errors else "partial"
            payload["persistence_error"] = "; ".join(errors)
            self._append_log(payload, errors)
            status = "persisted" if not errors else "partial" if payload.get("snapshot_path") else "failed"
            payload["persistence_status"] = status
            payload["persistence_error"] = "; ".join(errors)
            if errors:
                self._write_failures += 1
                self._last_error = "; ".join(errors)
            else:
                self._last_error = ""
            self._update_event(payload["event_id"], payload["persistence_status"], payload["persistence_error"])
            for callback in list(self._callbacks):
                try:
                    callback(dict(payload))
                except Exception as exc:
                    self._logger.warning("Behavior event callback failed: %s", exc)
            self._queue.task_done()

    def _load_history(self) -> None:
        if self._history_loaded:
            return
        self._history_loaded = True
        try:
            path = self._safe_path(
                str(self.config.get("event_log_path") or "logs/behavior_events.jsonl")
            )
            if not path.is_file():
                return
            payloads = []
            for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines()[-200:]:
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict) and payload.get("event_id"):
                    payloads.append(payload)
            with self._lock:
                for payload in payloads:
                    self._recent.appendleft(payload)
                    self._counts[str(payload.get("event_type") or "unknown")] += 1
        except Exception as exc:
            self._last_error = f"Could not restore behavior event history: {exc}"

    def _persist_images(self, payload: dict, frame: np.ndarray, errors: List[str]) -> None:
        if payload.get("snapshot_path"):
            self._write_image(payload["snapshot_path"], frame, "unannotated snapshot", errors)
        if payload.get("annotated_snapshot_path"):
            annotated = _draw_event_annotation(frame.copy(), payload)
            self._write_image(payload["annotated_snapshot_path"], annotated, "annotated snapshot", errors)

    def _write_image(self, relative_path: str, frame: np.ndarray, label: str, errors: List[str]) -> None:
        try:
            path = self._safe_path(relative_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            ok = cv2.imwrite(str(path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if not ok:
                raise OSError("cv2.imwrite returned false")
        except Exception as exc:
            errors.append(f"Could not write {label}: {exc}")

    def _append_log(self, payload: dict, errors: List[str]) -> None:
        try:
            path = self._safe_path(str(self.config.get("event_log_path") or "logs/behavior_events.jsonl"))
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
        except Exception as exc:
            errors.append(f"Could not append event log: {exc}")

    def _assign_paths(self, event: BehaviorEvent) -> Tuple[BehaviorEvent, str]:
        stem = f"{event.timestamp.replace(':', '').replace('-', '')}_{event.event_type}_track-{event.track_id}_{event.event_id[:8]}"
        try:
            snapshot_dir = self._safe_path(str(self.config.get("snapshot_dir") or "data/behavior_events/snapshots"))
            raw = self._relative(snapshot_dir / f"{stem}_evidence.jpg") if self.save_unannotated else None
            annotated = self._relative(snapshot_dir / f"{stem}_annotated.jpg") if self.save_annotated else None
            clip = None
            warning = ""
            if self.save_video_clip:
                warning = "Video clip recording is configured but not implemented; video_clip_path remains null"
            return replace(event, snapshot_path=raw, annotated_snapshot_path=annotated, video_clip_path=clip), warning
        except Exception as exc:
            return event, str(exc)

    def _safe_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.project_root / path
        resolved = path.resolve()
        if resolved != self.project_root and self.project_root not in resolved.parents:
            raise ValueError(f"Evidence path must stay inside project directory: {value}")
        return resolved

    def _relative(self, path: Path) -> str:
        return path.resolve().relative_to(self.project_root).as_posix()

    def _update_event(self, event_id: str, status: str, error: str) -> None:
        with self._lock:
            for item in self._recent:
                if item.get("event_id") == event_id:
                    item["persistence_status"] = status
                    item["persistence_error"] = error
                    break


def _draw_event_annotation(frame: np.ndarray, payload: dict) -> np.ndarray:
    colors: Dict[str, Tuple[int, int, int]] = {
        "person": (42, 166, 85),
        "phone": (0, 165, 255),
        "cigarette": (46, 46, 220),
        "smoke": (170, 170, 170),
        "flame": (0, 90, 255),
        "lighter": (160, 80, 190),
        "hand": (190, 150, 30),
        "prohibited_roi": (40, 40, 210),
        "face": (190, 150, 30),
    }
    for label, values in payload.get("bboxes", {}).items():
        if not isinstance(values, (list, tuple)) or len(values) != 4:
            continue
        x1, y1, x2, y2 = [int(round(float(value))) for value in values]
        color = colors.get(label, (120, 120, 220))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 2)
    title = (
        f"ID {payload.get('track_id')} {payload.get('event_type')} "
        f"conf={float(payload.get('confidence') or 0.0):.2f} duration={int(payload.get('duration_ms') or 0)}ms"
    )
    cv2.putText(frame, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 230), 2)
    return frame
