#!/usr/bin/env python3
"""Exercise the live counting configuration API and persisted config reload."""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import load_config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Test GET/POST /api/config/counting.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8001", help="Running FastAPI service URL.")
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    original = request_json("GET", f"{base_url}/api/config/counting")
    candidate = build_candidate(original)

    try:
        posted = request_json("POST", f"{base_url}/api/config/counting", candidate)
        assert_config_matches(posted, candidate, "POST response")

        fetched = request_json("GET", f"{base_url}/api/config/counting")
        assert_config_matches(fetched, candidate, "GET after POST")

        persisted = load_config()["counting"]
        persisted_direction = persisted.get("direction")
        if isinstance(persisted_direction, dict):
            persisted_direction = persisted_direction.get("mode")
        persisted_api_shape = {
            "line": persisted["line"],
            "direction": persisted_direction,
        }
        assert_config_matches(persisted_api_shape, candidate, "config.yaml reload")
    finally:
        request_json(
            "POST",
            f"{base_url}/api/config/counting",
            {"line": original["line"], "direction": original["direction"]},
        )

    restored = request_json("GET", f"{base_url}/api/config/counting")
    assert_config_matches(restored, {"line": original["line"], "direction": original["direction"]}, "restore")

    print("GET config: ok")
    print("POST config: ok")
    print("GET confirms saved config: ok")
    print("config.yaml reload: ok")
    print("restore original config: ok")
    return 0


def request_json(method: str, url: str, payload: Dict[str, Any] | None = None) -> Dict[str, Any]:
    data = None
    headers = {"Accept": "application/json"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url} failed: {exc.code} {body}") from exc


def build_candidate(original: Dict[str, Any]) -> Dict[str, Any]:
    frame_size = original.get("frame_size") or {"width": 1280, "height": 720}
    width = float(frame_size.get("width") or 1280)
    height = float(frame_size.get("height") or 720)
    direction = "right_to_left" if original.get("direction") == "left_to_right" else "left_to_right"
    return {
        "line": {
            "x1": round(width * 0.52, 1),
            "y1": 0,
            "x2": round(width * 0.52, 1),
            "y2": round(height, 1),
        },
        "direction": direction,
    }


def assert_config_matches(actual: Dict[str, Any], expected: Dict[str, Any], label: str) -> None:
    actual_line = actual.get("line") or {}
    expected_line = expected.get("line") or {}
    for key in ("x1", "y1", "x2", "y2"):
        if abs(float(actual_line[key]) - float(expected_line[key])) > 0.01:
            raise AssertionError(f"{label}: line.{key} expected {expected_line[key]}, got {actual_line[key]}")
    if actual.get("direction") != expected.get("direction"):
        raise AssertionError(f"{label}: direction expected {expected.get('direction')}, got {actual.get('direction')}")


if __name__ == "__main__":
    raise SystemExit(main())
