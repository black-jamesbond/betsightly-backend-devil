"""
BetSightly Daily Automation
Runs every night to generate predictions and bot picks for tomorrow.
Scheduled via Windows Task Scheduler at 11:00 PM.
"""
import subprocess
import sys
import os
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

BASE_DIR = Path(__file__).parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

log_file = LOG_DIR / f"daily_{datetime.now().strftime('%Y-%m-%d')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    handlers=[
        logging.FileHandler(log_file, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("daily")

PYTHON = sys.executable


def run_script(name, args=None):
    cmd = [PYTHON, str(BASE_DIR / name)]
    if args:
        cmd.extend(args)
    log.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600, cwd=str(BASE_DIR)
        )
        if result.returncode == 0:
            last_lines = result.stdout.strip().split("\n")[-5:]
            for line in last_lines:
                log.info(f"  {line}")
            return True
        else:
            log.error(f"  FAILED (exit {result.returncode})")
            for line in result.stderr.strip().split("\n")[-10:]:
                log.error(f"  {line}")
            return False
    except subprocess.TimeoutExpired:
        log.error(f"  TIMEOUT after 10 minutes")
        return False


def main():
    tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    today = datetime.now().strftime("%Y-%m-%d")

    log.info("=" * 60)
    log.info(f"BetSightly Daily Automation")
    log.info(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"Generating predictions for: {tomorrow}")
    log.info("=" * 60)

    # Step 1 (weekly, Mondays): refit the Dixon-Coles ratings on fresh results
    if datetime.now().weekday() == 0:
        log.info(f"\n[1/5] Weekly Dixon-Coles refit")
        run_script("fit_dixon_coles.py")
    else:
        log.info(f"\n[1/5] Dixon-Coles refit skipped (runs Mondays)")

    # Step 2: Run predictions for tomorrow (hybrid: DC 1X2 + calibrated markets)
    log.info(f"\n[2/5] Predictions for {tomorrow}")
    run_script("predict_today.py", ["--date", tomorrow])

    # Step 3: Run bot system for tomorrow
    log.info(f"\n[3/5] Bot picks for {tomorrow}")
    run_script("bot_system.py", ["--date", tomorrow])

    # Step 4: Re-run today's bots to settle results
    log.info(f"\n[4/5] Settling today's results ({today})")
    run_script("bot_system.py", ["--date", today])

    # Step 5: Restart the dashboard server
    log.info(f"\n[5/5] Restarting dashboard server")
    restart_dashboard()

    # Summary
    pred_file = BASE_DIR / "cache" / "apifootball" / f"predictions_{tomorrow}.json"
    bot_file = BASE_DIR / "bot_results" / f"bots_{tomorrow}.json"

    pred_count = 0
    bot_count = 0
    if pred_file.exists():
        with open(pred_file, encoding="utf-8") as f:
            pred_count = len(json.load(f))
    if bot_file.exists():
        with open(bot_file, encoding="utf-8") as f:
            data = json.load(f)
            bot_count = sum(len(b.get("history", [])) for b in data.get("bots", []))

    log.info("\n" + "=" * 60)
    log.info(f"DONE - {tomorrow}")
    log.info(f"  Predictions: {pred_count} matches")
    log.info(f"  Bot picks:   {bot_count} bets")
    log.info(f"  Log: {log_file}")
    log.info("=" * 60)


def restart_dashboard():
    import signal
    try:
        # Kill existing Flask on port 5050
        result = subprocess.run(
            ["netstat", "-ano"], capture_output=True, text=True
        )
        for line in result.stdout.split("\n"):
            if ":5050" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                try:
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=10)
                    log.info(f"  Killed old server (PID {pid})")
                except:
                    pass

        # Start new server
        subprocess.Popen(
            [PYTHON, "app.py"],
            cwd=str(BASE_DIR / "dashboard"),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS
        )
        log.info("  Dashboard server restarted on port 5050")
    except Exception as e:
        log.error(f"  Failed to restart dashboard: {e}")


if __name__ == "__main__":
    main()
