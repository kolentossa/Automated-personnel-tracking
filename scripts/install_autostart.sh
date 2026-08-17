#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIT_NAME="person-tracking.service"
UNIT_SOURCE="$PROJECT_ROOT/deploy/$UNIT_NAME"
USER_UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"

export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"

mkdir -p "$USER_UNIT_DIR"
ln -sfn "$UNIT_SOURCE" "$USER_UNIT_DIR/$UNIT_NAME"
mkdir -p "$USER_UNIT_DIR/default.target.wants"
ln -sfn "$UNIT_SOURCE" "$USER_UNIT_DIR/default.target.wants/$UNIT_NAME"
loginctl enable-linger "$USER"

systemctl --user daemon-reload
systemctl --user stop "$UNIT_NAME" 2>/dev/null || true

# Remove a legacy manually launched instance before systemd claims the camera and port.
manual_pids="$(pgrep -f 'uvicorn app.main:app --host 0.0.0.0 --port 8001$' || true)"
if [[ -n "$manual_pids" ]]; then
    kill $manual_pids 2>/dev/null || true
    for _ in {1..25}; do
        if ! pgrep -f 'uvicorn app.main:app --host 0.0.0.0 --port 8001$' >/dev/null; then
            break
        fi
        sleep 0.2
    done
    remaining_pids="$(pgrep -f 'uvicorn app.main:app --host 0.0.0.0 --port 8001$' || true)"
    if [[ -n "$remaining_pids" ]]; then
        kill -KILL $remaining_pids 2>/dev/null || true
    fi
fi

for _ in {1..20}; do
    if ! fuser /dev/video11 >/dev/null 2>&1; then
        break
    fi
    sleep 0.1
done
if fuser /dev/video11 >/dev/null 2>&1; then
    echo "/dev/video11 is still busy" >&2
    exit 1
fi

systemctl --user enable --now "$UNIT_NAME"

for _ in {1..40}; do
    if curl -fsS http://127.0.0.1:8001/api/health >/dev/null 2>&1; then
        systemctl --user --no-pager --full status "$UNIT_NAME"
        exit 0
    fi
    sleep 0.5
done

systemctl --user --no-pager --full status "$UNIT_NAME" || true
exit 1
