#!/usr/bin/env bash
# Configure periodic settlement of the latest three fixture days.
# REVIEW actual source-root and service identity before executing on VPS.
# Usage: sudo bash deploy/install_settlement_timer.sh /absolute/repo /absolute/python service-user
set -euo pipefail
if [[ "$EUID" -ne 0 || "$#" -ne 3 ]]; then
  echo "Usage: sudo bash $0 /absolute/repo /absolute/python service-user" >&2
  exit 2
fi
ROOT="$(realpath "$1")"
PYTHON="$2"
SERVICE_USER="$3"
if [[ "$ROOT" == *" "* || "$PYTHON" == *" "* || "$PYTHON" != /* ]]; then
  echo "Expected absolute paths without spaces" >&2
  exit 2
fi
if [[ ! -f "$ROOT/scripts/settle_results.py" || ! -x "$PYTHON" ]]; then
  echo "Runner or Python missing" >&2
  exit 2
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "Unknown service user" >&2
  exit 2
fi
install -d -m 755 "$ROOT/cache/reports" "$ROOT/data/fixtures"
chown "$SERVICE_USER" "$ROOT/cache/reports" "$ROOT/data/fixtures"
cat > /etc/systemd/system/betsightly-settlement.service <<SERVICE
[Unit]
Description=Standalone BetSightly confirmed final-score settlement
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$ROOT
ExecStart=/usr/bin/flock -w 600 $ROOT/cache/apifootball/fixture_collection.lock $PYTHON scripts/settle_results.py --days 3
TimeoutStartSec=1800
SERVICE
cat > /etc/systemd/system/betsightly-settlement.timer <<TIMER
[Unit]
Description=Refresh settled results every three hours

[Timer]
OnCalendar=*-*-* 00,03,06,09,12,15,18,21:20:00 Africa/Lagos
Persistent=true
AccuracySec=1min
Unit=betsightly-settlement.service

[Install]
WantedBy=timers.target
TIMER
systemctl daemon-reload
systemctl enable --now betsightly-settlement.timer
echo "Installed settlement timer (no immediate manual run)."
systemctl list-timers 'betsightly-settlement*'
