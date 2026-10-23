#!/usr/bin/env bash
# Review this against actual VPS paths and running code before installation.
# Usage: sudo bash deploy/install_prediction_timers.sh /absolute/repo /absolute/python service-user
set -euo pipefail

if [[ "$EUID" -ne 0 || "$#" -ne 3 ]]; then
  echo "Usage: sudo bash $0 /absolute/repo /absolute/python service-user" >&2
  exit 2
fi
APP_ROOT="$(realpath "$1")"
PYTHON="$2"
SERVICE_USER="$3"
if [[ "$APP_ROOT" == *" "* || "$PYTHON" == *" "* || "$PYTHON" != /* ]]; then
  echo "Expected absolute, space-free paths" >&2
  exit 2
fi
if [[ ! -f "$APP_ROOT/scripts/publish_daily_predictions.py" || ! -x "$PYTHON" ]]; then
  echo "Missing publisher script or executable Python" >&2
  exit 2
fi
if ! id "$SERVICE_USER" >/dev/null 2>&1; then
  echo "Unknown service user: $SERVICE_USER" >&2
  exit 2
fi
install -d -m 755 "$APP_ROOT/cache/apifootball" "$APP_ROOT/data/fixtures"
chown "$SERVICE_USER" "$APP_ROOT/cache/apifootball" "$APP_ROOT/data/fixtures"

for mode in preview early refresh; do
  case "$mode" in
    preview)
      calendar="*-*-* 21:30:00 Africa/Lagos"
      flags="--offset 1 --import-if-missing"
      ;;
    early)
      calendar="*-*-* 00:30:00 Africa/Lagos"
      flags="--offset 0"
      ;;
    refresh)
      calendar="*-*-* 05:45:00 Africa/Lagos"
      flags="--offset 0"
      ;;
  esac
  unit="betsightly-predict-$mode"
  cat > "/etc/systemd/system/$unit.service" <<SERVICE
[Unit]
Description=Independent BetSightly $mode predictions
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
User=$SERVICE_USER
WorkingDirectory=$APP_ROOT
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/flock -w 900 $APP_ROOT/cache/apifootball/fixture_collection.lock $PYTHON scripts/publish_daily_predictions.py run $flags
TimeoutStartSec=2500
SERVICE
  cat > "/etc/systemd/system/$unit.timer" <<TIMER
[Unit]
Description=Schedule independent BetSightly $mode predictions

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
systemctl enable --now betsightly-predict-preview.timer betsightly-predict-early.timer betsightly-predict-refresh.timer
echo "Installed timers, without running predictions immediately."
systemctl list-timers 'betsightly-predict-*'
