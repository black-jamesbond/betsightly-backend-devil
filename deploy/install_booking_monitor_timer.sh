#!/usr/bin/env bash
# Configure non-betting periodic refresh of issued bookmaker share codes.
# This will contact SportyBet to re-issue a code only if the live price changes.
# Usage: sudo bash deploy/install_booking_monitor_timer.sh /absolute/repo /absolute/python service-user
set -euo pipefail
if [[ "$EUID" -ne 0 || "$#" -ne 3 ]]; then
  echo "Usage: sudo bash $0 /absolute/repo /absolute/python service-user" >&2
  exit 2
fi
ROOT="$(realpath "$1")"
PYTHON="$2"
USER_NAME="$3"
if [[ "$ROOT" == *" "* || "$PYTHON" == *" "* || "$PYTHON" != /* ]]; then
  echo "Paths must be absolute and not contain spaces" >&2
  exit 2
fi
if [[ ! -x "$PYTHON" || ! -f "$ROOT/scripts/refresh_booking_codes.py" ]]; then
  echo "Python or bookmaker monitor script missing" >&2
  exit 2
fi
if ! id "$USER_NAME" >/dev/null 2>&1; then
  echo "Unknown service user" >&2
  exit 2
fi
install -d -m 755 "$ROOT/cache"
chown "$USER_NAME" "$ROOT/cache"
cat > /etc/systemd/system/betsightly-booking-monitor.service <<SERVICE
[Unit]
Description=Check and renew changed bookmaker share codes for standalone BetSightly
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$USER_NAME
WorkingDirectory=$ROOT
ExecStart=/usr/bin/flock -n $ROOT/cache/bookmaker_monitor.lock $PYTHON scripts/refresh_booking_codes.py
TimeoutStartSec=1200
SERVICE
cat > /etc/systemd/system/betsightly-booking-monitor.timer <<TIMER
[Unit]
Description=Recheck active issued booking codes every ten minutes

[Timer]
OnCalendar=*-*-* *:00/10:00 Africa/Lagos
Persistent=true
AccuracySec=1min
Unit=betsightly-booking-monitor.service

[Install]
WantedBy=timers.target
TIMER
systemctl daemon-reload
systemctl enable --now betsightly-booking-monitor.timer
echo "Booking monitor timer installed; not started immediately."
systemctl list-timers 'betsightly-booking-monitor*'
