#!/usr/bin/env bash
# READ ONLY. Identify the independent BetSightly server's real code checkout.
# Do not print environment variables or secret values.
# Usage: sudo bash deploy/audit_vps_baseline.sh [port]
set -euo pipefail
if [ "$#" -gt 0 ]; then PORT="$1"; else PORT=5050; fi
if ! [[ "$PORT" =~ ^[0-9]+$ ]] || (( PORT < 1 || PORT > 65535 )); then
  echo "Expected numeric TCP port (1..65535)" >&2
  exit 2
fi
echo "=== Independent BetSightly VPS read-only baseline ==="
date --iso-8601=seconds
echo "Port: $PORT"
if command -v ss >/dev/null 2>&1; then
  ss -ltnp "( sport = :$PORT )" || true
fi
if ! command -v fuser >/dev/null 2>&1; then
  echo "fuser unavailable; inspect listener PID above."
  exit 0
fi
PIDS="$(fuser -n tcp "$PORT" 2>/dev/null || true)"
if [ -z "$PIDS" ]; then
  echo "No accessible listener PID on this port."
  exit 0
fi
for PID in $PIDS; do
  [[ "$PID" =~ ^[0-9]+$ ]] || continue
  echo
  echo "== PID $PID =="
  ps -p "$PID" -o pid=,comm=,lstart= || true
  ROOT="$(readlink -f "/proc/$PID/cwd" 2>/dev/null || true)"
  EXE="$(readlink -f "/proc/$PID/exe" 2>/dev/null || true)"
  echo "Working directory: $ROOT"
  echo "Executable: $EXE"
  if [ -f "$ROOT/dashboard/app.py" ]; then
    CODE_ROOT="$ROOT"
  elif [ -f "$ROOT/app.py" ] && [ -f "$ROOT/../predict_today.py" ]; then
    CODE_ROOT="$(readlink -f "$ROOT/..")"
  else
    echo "Source root not identifiable from process cwd."
    continue
  fi
  echo "Source root: $CODE_ROOT"
  if git -C "$CODE_ROOT" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "Git commit: $(git -C "$CODE_ROOT" rev-parse HEAD)"
    echo "Git branch: $(git -C "$CODE_ROOT" branch --show-current)"
    echo "Tracked changed filenames (no contents):"
    git -C "$CODE_ROOT" diff --name-only || true
  else
    echo "No Git checkout at process cwd."
  fi
  echo "Source hashes:"
  for name in dashboard/app.py dashboard/templates/index.html dashboard/static/theme.css predict_today.py predict_sportybet.py fixtures_livescore.py; do
    if [ -f "$CODE_ROOT/$name" ]; then
      sha256sum "$CODE_ROOT/$name"
    fi
  done
  echo "Fixture state metadata:"
  for name in data/fixtures/fixture_store.sqlite3 cache/apifootball; do
    if [ -e "$CODE_ROOT/$name" ]; then
      stat -c '%n | %y | %s bytes' "$CODE_ROOT/$name"
    fi
  done
done
echo "Finished: no server changes or files written."
