#!/usr/bin/env bash
# Install fixture collector timers on the separate BetSightly VPS.
# Usage: sudo bash deploy/install_fixture_timers.sh /absolute/repo /absolute/python service-user
set -euo pipefail

if [[ "$EUID" -ne 0 || "$#" -ne 3 ]]; then
  echo "Usage: sudo bash $0 /absolute/repo /absolute/python service-user" >&2
  exit 2
fi
APP_ROOT="$(realpath "$1")"
PYTHON="$(realpath "$2")"
SERVICE_USER="$3"
if [[ "$APP_ROOT" == *" "* || "$PYTHON" == *" "* ]]; then
  echo "Paths containing spaces are not supported" >&2
  exit 2
fi
if [[ ! -f "$APP_ROOT/scripts/fixture_collection.py" || ! -x "$PYTHON" ]]; then
  echo "Missing fixture_collection.py or executable Python" >&2
  exit 2
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "No such service user: $SERVICE_USER" >&2
  exit 2
fi

install -d -m 755 "$APP_ROOT/cache/apifootball" "$APP_ROOT/data/fixtures"
chown "$SERVICE_USER" "$APP_ROOT/cache/apifootball" "$APP_ROOT/data/fixtures"

for mode in weekly daily; do
  unit="betsightly-fixtures-$mode"
  if [[ "$mode" == weekly ]]; then
    calendar="Sun *-*-* 01:15:00 Africa/Lagos"
  else
    calendar="*-*-* 05:15:00 Africa/Lagos"
  fi
  cat > "/etc/systemd/system/$unit.service" <<SERVICE
[Unit]
Description=Independent BetSightly $mode fixture collection
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$APP_ROOT
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/flock -n $APP_ROOT/cache/apifootball/fixture_collection.lock $PYTHON scripts/fixture_collection.py $mode --publish-cache
TimeoutStartSec=1200
SERVICE

  cat > "/etc/systemd/system/$unit.timer" <<TIMER
[Unit]
Description=Schedule independent BetSightly $mode fixture collection

[Timer]
OnCalendar=$calendar
Persistent=true
AccuracySec=1min
Unit=$unit.service

[Install]
WantedBy=timers.target
TIMER
done

systemctl daemon-reload
systemctl enable --now betsightly-fixtures-weekly.timer betsightly-fixtures-daily.timer

echo "Installed timers. This did NOT execute collection immediately."
systemctl list-timers 'betsightly-fixtures-*'
echo "To test collection, run:"
echo "  sudo systemctl start betsightly-fixtures-daily.service"
echo "  journalctl -u betsightly-fixtures-daily.service -n 50 --no-pager"
