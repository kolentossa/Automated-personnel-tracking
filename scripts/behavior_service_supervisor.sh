#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT="${PERSON_TRACKING_PROJECT_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PERSON_TRACKING_PYTHON:?PERSON_TRACKING_PYTHON is required}"
HOST="${PERSON_TRACKING_HOST:-0.0.0.0}"
PORT="${PERSON_TRACKING_PORT:-8001}"
RESTART_SECONDS="${PERSON_TRACKING_RESTART_SECONDS:-3}"
GRACEFUL_SECONDS="${PERSON_TRACKING_GRACEFUL_SECONDS:-5}"
child_pid=""
stopping=0

stop_child() {
  stopping=1
  if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
    kill -TERM "$child_pid"
    for _ in $(seq 1 $((GRACEFUL_SECONDS * 2 + 2))); do
      kill -0 "$child_pid" 2>/dev/null || break
      sleep 0.5
    done
    if kill -0 "$child_pid" 2>/dev/null; then
      kill -KILL "$child_pid"
    fi
    wait "$child_pid" 2>/dev/null || true
  fi
}

trap stop_child TERM INT
cd "$PROJECT_ROOT"

while [[ "$stopping" -eq 0 ]]; do
  env \
    MALLOC_ARENA_MAX="${PERSON_TRACKING_MALLOC_ARENA_MAX:-2}" \
    MALLOC_TRIM_THRESHOLD_="${PERSON_TRACKING_MALLOC_TRIM_THRESHOLD:-131072}" \
    "$PYTHON_BIN" -m uvicorn app.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --timeout-graceful-shutdown "$GRACEFUL_SECONDS" &
  child_pid=$!
  wait "$child_pid"
  exit_code=$?
  child_pid=""
  if [[ "$stopping" -ne 0 ]]; then
    break
  fi
  printf 'uvicorn exited with code %s; restarting in %ss\n' "$exit_code" "$RESTART_SECONDS"
  sleep "$RESTART_SECONDS"
done
