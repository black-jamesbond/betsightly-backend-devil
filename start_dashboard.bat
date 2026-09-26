@echo off
REM BetSightly dashboard.
REM
REM Double-click this (or run it from a terminal). It stops any dashboard already
REM listening on 5050 first, so you can never end up with two servers sharing the
REM port and one of them quietly serving stale code.
REM
REM The server needs outbound internet for live scores, SportyBet odds and booking
REM codes, so run it from a normal shell -- never a restricted/sandboxed one, where
REM you would get "WinError 10061: No connection could be made".
cd /d "%~dp0"

echo Stopping any dashboard already listening on port 5050...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Get-NetTCPConnection -LocalPort 5050 -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }"

echo Starting BetSightly dashboard on http://localhost:5050
echo auto-reload is ON - code edits apply without restarting
python dashboard\app.py --port 5050
pause
