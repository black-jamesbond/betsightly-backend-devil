#!/usr/bin/env bash
# OPTIONAL heavy training timer; verify CPU, memory and source freshness first.
# Usage: sudo bash deploy/install_training_timer.sh /absolute/repo /absolute/python service-user
set -euo pipefail
if [[ "$EUID" -ne 0 || "$#" -ne 3 ]]; then
  echo "Usage: sudo bash $0 /absolute/repo /absolute/python service-user" >&2
  exit 2
fi
ROOT="$(realpath "$1")"
PYTHON="$2"
SERVICE_USER="$3"
if [[ ! -f "$ROOT/scripts/stage_weekly_training.py" || ! -x "$PYTHON" ]]; then
  echo "Training runner / Python missing" >&2
  exit 2
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "Unknown service user" >&2
  exit 2
fi
install -d -m 755 "$ROOT/models/staging" "$ROOT/cache/training"
chown "$SERVICE_USER" "$ROOT/models/staging" "$ROOT/cache/training"
cat > /etc/systemd/system/betsightly-training-stage.service <<SERVICE
[Unit]
Description=Standalone BetSightly stage weekly model candidates
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$ROOT
ExecStart=/usr/bin/flock -n $ROOT/cache/training/stage.lock $PYTHON scripts/stage_weekly_training.py
TimeoutStartSec=7600
SERVICE
cat > /etc/systemd/system/betsightly-training-stage.timer <<TIMER
[Unit]
Description=Stage model candidates each Sunday (never auto-promote)

[Timer]
OnCalendar=Sun *-*-* 03:30:00 Africa/Lagos
Persistent=true
AccuracySec=1min
Unit=betsightly-training-stage.service

[Install]
WantedBy=timers.target
TIMER
systemctl daemon-reload
systemctl enable --now betsightly-training-stage.timer
systemctl list-timers 'betsightly-training-stage*'
echo "Staging timer installed; candidate promotion remains manual."
