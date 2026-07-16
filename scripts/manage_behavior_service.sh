#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/run"
LOG_DIR="$PROJECT_ROOT/logs"
PID_FILE="$RUN_DIR/behavior-8001-supervisor.pid"
LOG_FILE="$LOG_DIR/behavior-8001.log"
SUPERVISOR="$PROJECT_ROOT/scripts/behavior_service_supervisor.sh"
HOST="${PERSON_TRACKING_HOST:-0.0.0.0}"
PORT="${PERSON_TRACKING_PORT:-8001}"

find_python() {
  if [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$PROJECT_ROOT/.venv/bin/python"
    return
  fi
  local main_worktree
  main_worktree="$(git -C "$PROJECT_ROOT" worktree list --porcelain | awk '/^worktree / {sub(/^worktree /, ""); print; exit}')"
  if [[ -n "$main_worktree" && -x "$main_worktree/.venv/bin/python" ]]; then
    printf '%s\n' "$main_worktree/.venv/bin/python"
    return
  fi
  printf 'Could not locate a project virtualenv Python\n' >&2
  return 1
}

read_pid() {
  if [[ -s "$PID_FILE" ]]; then
    tr -cd '0-9' < "$PID_FILE"
  fi
}

is_our_supervisor() {
  local pid="$1"
  [[ -n "$pid" && -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq "$SUPERVISOR"
}

port_is_listening() {
  ss -ltn "( sport = :$PORT )" 2>/dev/null | grep -q LISTEN
}

start_service() {
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  local existing python_bin supervisor_pid
  existing="$(read_pid || true)"
  if [[ -n "$existing" ]] && is_our_supervisor "$existing"; then
    printf 'behavior service already supervised: pid=%s port=%s\n' "$existing" "$PORT"
    return
  fi
  if port_is_listening; then
    printf 'Port %s is already occupied; refusing to replace an unrelated service\n' "$PORT" >&2
    return 1
  fi
  python_bin="$(find_python)"
  nohup env \
    PERSON_TRACKING_PROJECT_ROOT="$PROJECT_ROOT" \
    PERSON_TRACKING_PYTHON="$python_bin" \
    PERSON_TRACKING_HOST="$HOST" \
    PERSON_TRACKING_PORT="$PORT" \
    "$SUPERVISOR" >> "$LOG_FILE" 2>&1 < /dev/null &
  supervisor_pid=$!
  printf '%s\n' "$supervisor_pid" > "$PID_FILE"
  for _ in $(seq 1 40); do
    if ! is_our_supervisor "$supervisor_pid"; then
      printf 'Behavior supervisor exited during startup; see %s\n' "$LOG_FILE" >&2
      return 1
    fi
    if curl -fsS --max-time 2 "http://127.0.0.1:$PORT/api/health" | grep -q '"status":"ok"'; then
      printf 'behavior service started: supervisor_pid=%s port=%s log=%s\n' \
        "$supervisor_pid" "$PORT" "$LOG_FILE"
      return
    fi
    sleep 1
  done
  printf 'Behavior service did not become healthy; see %s\n' "$LOG_FILE" >&2
  return 1
}

stop_service() {
  local pid
  pid="$(read_pid || true)"
  if [[ -z "$pid" ]]; then
    printf 'behavior service is not supervised (empty PID file)\n'
    return
  fi
  if [[ ! -d "/proc/$pid" ]]; then
    : > "$PID_FILE"
    printf 'cleared stale behavior supervisor PID: %s\n' "$pid"
    return
  fi
  if ! is_our_supervisor "$pid"; then
    printf 'PID %s does not belong to this project supervisor; refusing to kill it\n' "$pid" >&2
    return 1
  fi
  kill -TERM "$pid"
  for _ in $(seq 1 20); do
    if ! kill -0 "$pid" 2>/dev/null; then
      : > "$PID_FILE"
      printf 'behavior service stopped: supervisor_pid=%s\n' "$pid"
      return
    fi
    sleep 0.5
  done
  printf 'Supervisor %s did not stop within 10 seconds\n' "$pid" >&2
  return 1
}

status_service() {
  local pid
  pid="$(read_pid || true)"
  if [[ -n "$pid" ]] && is_our_supervisor "$pid"; then
    printf 'supervisor=running pid=%s\n' "$pid"
  else
    printf 'supervisor=stopped\n'
  fi
  ss -ltnp "( sport = :$PORT )" 2>/dev/null || true
  curl -fsS --max-time 3 "http://127.0.0.1:$PORT/api/health" || true
  printf '\n'
}

case "${1:-status}" in
  start) start_service ;;
  stop) stop_service ;;
  restart) stop_service; start_service ;;
  status) status_service ;;
  *) printf 'Usage: %s {start|stop|restart|status}\n' "$0" >&2; exit 2 ;;
esac
