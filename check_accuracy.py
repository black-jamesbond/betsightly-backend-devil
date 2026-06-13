#!/usr/bin/env python3
"""
check_accuracy.py
-----------------
Fetch today's fixtures + results from API-Football, generate predictions,
compare them against actual outcomes, and print a full accuracy report.

Usage:
    python check_accuracy.py                  # today
    python check_accuracy.py 2026-06-04       # specific date
"""

from __future__ import annotations

import os
import sys
import json
import logging
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from typing import Optional

warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv()

import requests

API_KEY = os.getenv("API_FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS = {"x-apisports-key": API_KEY}
CACHE_DIR = ROOT / "cache" / "api_football"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ── helpers ───────────────────────────────────────────────────────────────────

def _cache_path(date_str: str) -> Path:
    return CACHE_DIR / f"results_{date_str}.json"


def fetch_fixtures(date_str: str) -> list:
    """Fetch all fixtures for a date from API-Football (with local cache)."""
    cache = _cache_path(date_str)

    # Use cache when all games are finished (past date) or same-day cache
    if cache.exists():
        try:
            with open(cache, "r", encoding="utf-8") as f:
                data = json.load(f)
            print(f"[cache] Loaded {len(data)} fixtures from {cache.name}")
            return data
        except Exception:
            pass

    if not API_KEY:
        print("ERROR: API_FOOTBALL_API_KEY not set in .env")
        sys.exit(1)

    print(f"[api]   Fetching fixtures for {date_str} …")
    r = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"date": date_str},
        timeout=30,
    )
    if r.status_code != 200:
        print(f"ERROR: API returned {r.status_code}")
        sys.exit(1)

    body = r.json()
    errors = body.get("errors", {})
    if errors:
        print(f"ERROR: API error — {errors}")
        sys.exit(1)

    fixtures = body.get("response", [])
    remaining = r.headers.get("x-ratelimit-requests-remaining", "?")
    print(f"[api]   Got {len(fixtures)} fixtures  ({remaining} requests remaining today)")

    # Cache only if the target date is in the past (results are final)
    today = datetime.now().strftime("%Y-%m-%d")
    if date_str < today:
        with open(cache, "w", encoding="utf-8") as f:
            json.dump(fixtures, f, ensure_ascii=False)
        print(f"[cache] Saved to {cache.name}")

    return fixtures


def extract_actual_outcome(fixture: dict) -> Optional[dict]:
    """
    Return the actual result of a finished fixture, or None if not finished.

    Keys returned:
        home_team, away_team, league,
        home_goals, away_goals,
        result   : "Home Win" | "Draw" | "Away Win"
        btts     : True | False
        over_2_5 : True | False
        over_1_5 : True | False
    """
    status = fixture.get("fixture", {}).get("status", {}).get("short", "")
    # FT = Full Time, AET = After Extra Time, PEN = Penalty shootout
    if status not in ("FT", "AET", "PEN"):
        return None

    home_goals = fixture.get("goals", {}).get("home")
    away_goals = fixture.get("goals", {}).get("away")
    if home_goals is None or away_goals is None:
        return None

    home_goals = int(home_goals)
    away_goals = int(away_goals)
    total = home_goals + away_goals

    if home_goals > away_goals:
        result = "Home Win"
    elif away_goals > home_goals:
        result = "Away Win"
    else:
        result = "Draw"

    return {
        "home_team":  fixture["teams"]["home"]["name"],
        "away_team":  fixture["teams"]["away"]["name"],
        "league":     fixture.get("league", {}).get("name", "Unknown"),
        "country":    fixture.get("league", {}).get("country", ""),
        "home_goals": home_goals,
        "away_goals": away_goals,
        "result":     result,
        "btts":       home_goals > 0 and away_goals > 0,
        "over_2_5":   total > 2,
        "over_1_5":   total > 1,
        "status":     status,
    }


def generate_predictions(fixtures: list) -> list:
    """
    Run the prediction pipeline on a list of raw API-Football fixtures.
    Returns a list of (prediction_dict, actual_outcome_dict) pairs — only
    for finished games.
    """
    import services.advanced_prediction_service as aps

    svc = aps.advanced_prediction_service
    svc._ensure_loaded()

    paired = []
    for fix in fixtures:
        actual = extract_actual_outcome(fix)
        if actual is None:
            continue  # skip unfinished / cancelled

        pred = svc._predict_fixture_advanced(fix)
        if pred is None:
            continue

        paired.append({"prediction": pred, "actual": actual})

    return paired


# ── accuracy evaluation ───────────────────────────────────────────────────────

BET_CHECKERS = {
    "Home Win":              lambda a: a["result"] == "Home Win",
    "Away Win":              lambda a: a["result"] == "Away Win",
    "Draw":                  lambda a: a["result"] == "Draw",
    "Both Teams to Score":   lambda a: a["btts"],
    "BTTS - Yes":            lambda a: a["btts"],
    "BTTS - No":             lambda a: not a["btts"],
    "Over 2.5":              lambda a: a["over_2_5"],
    "Under 2.5":             lambda a: not a["over_2_5"],
    "Over 1.5":              lambda a: a["over_1_5"],
    "Under 1.5":             lambda a: not a["over_1_5"],
}


def evaluate_prediction(pred: dict, actual: dict) -> bool | None:
    """
    Return True (correct), False (wrong), or None (unknown bet type).
    """
    bet = pred.get("prediction", pred.get("bet_type", ""))
    checker = BET_CHECKERS.get(bet)
    if checker is None:
        return None
    return checker(actual)


# ── report printer ────────────────────────────────────────────────────────────

def _a(s: str, n: int) -> str:
    """ASCII-safe truncate/pad."""
    s = str(s).encode("ascii", "replace").decode()
    return s[:n].ljust(n)


def print_report(paired: list, date_str: str) -> None:
    finished = [p for p in paired]
    evaluated = [(p, evaluate_prediction(p["prediction"], p["actual"])) for p in finished]

    correct   = sum(1 for _, ok in evaluated if ok is True)
    wrong     = sum(1 for _, ok in evaluated if ok is False)
    unknown   = sum(1 for _, ok in evaluated if ok is None)
    total_ev  = correct + wrong

    print()
    print("=" * 80)
    print(f"  BETSIGHTLY ACCURACY REPORT — {date_str}")
    print("=" * 80)
    print(f"  Finished games evaluated : {len(finished)}")
    print(f"  Correct predictions      : {correct}")
    print(f"  Wrong predictions        : {wrong}")
    print(f"  Unknown bet type         : {unknown}")
    if total_ev > 0:
        pct = correct / total_ev * 100
        print(f"  Accuracy                 : {pct:.1f}%  ({correct}/{total_ev})")
    print()

    # ── Per bet-type breakdown ──
    by_bet: dict[str, dict] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for p, ok in evaluated:
        bet = p["prediction"].get("prediction", p["prediction"].get("bet_type", "?"))
        if ok is True:
            by_bet[bet]["correct"] += 1
        elif ok is False:
            by_bet[bet]["wrong"] += 1

    if by_bet:
        print("  BY BET TYPE")
        print("  " + "-" * 50)
        for bet, counts in sorted(by_bet.items()):
            c, w = counts["correct"], counts["wrong"]
            t = c + w
            acc = f"{c/t*100:.0f}%" if t else "n/a"
            print(f"  {_a(bet,28)} {c:>3} / {t:<3}  ({acc})")
        print()

    # ── Per league breakdown ──
    by_league: dict[str, dict] = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for p, ok in evaluated:
        league = p["actual"]["league"]
        if ok is True:
            by_league[league]["correct"] += 1
        elif ok is False:
            by_league[league]["wrong"] += 1

    leagues_sorted = sorted(by_league.items(), key=lambda x: -(x[1]["correct"] + x[1]["wrong"]))
    if leagues_sorted:
        print("  BY LEAGUE (top 20)")
        print("  " + "-" * 62)
        for league, counts in leagues_sorted[:20]:
            c, w = counts["correct"], counts["wrong"]
            t = c + w
            acc = f"{c/t*100:.0f}%" if t else "n/a"
            print(f"  {_a(league,35)} {c:>3} / {t:<3}  ({acc})")
        print()

    # ── Full game-by-game table ──
    print("  GAME-BY-GAME RESULTS")
    print("  " + "-" * 78)
    header = f"  {'HOME TEAM':<22} {'AWAY TEAM':<22} {'SCORE':<7} {'BET':<22} {'?':<5}"
    print(header)
    print("  " + "-" * 78)

    for p, ok in evaluated:
        actual = p["actual"]
        pred   = p["prediction"]
        home   = _a(actual["home_team"], 22)
        away   = _a(actual["away_team"], 22)
        score  = f"{actual['home_goals']}-{actual['away_goals']}"
        bet    = _a(pred.get("prediction", pred.get("bet_type", "?")), 22)
        result = "YES" if ok is True else ("NO " if ok is False else "?  ")
        conf   = pred.get("confidence", 0)
        conf_s = f"{conf:.0%}" if isinstance(conf, float) and conf <= 1 else f"{conf:.1f}%"
        print(f"  {home} {away} {score:<7} {bet} {result}  {conf_s}")

    print("=" * 80)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Check prediction accuracy for a date")
    parser.add_argument("date", nargs="?", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--save", metavar="FILE", help="Save report to a JSON file")
    args = parser.parse_args()

    date_str = args.date
    print(f"\nBetSightly Accuracy Checker - {date_str}\n")

    # 1. Fetch fixtures (with results for finished games)
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        print("No fixtures found for this date.")
        return

    finished = [f for f in fixtures
                if f.get("fixture", {}).get("status", {}).get("short") in ("FT", "AET", "PEN")]
    total    = len(fixtures)
    print(f"[info]  {total} fixtures total, {len(finished)} finished ({total - len(finished)} not yet finished)\n")

    # 2. Generate predictions + pair with actuals
    print("[ml]    Generating predictions …")
    paired = generate_predictions(fixtures)
    print(f"[ml]    Generated {len(paired)} predictions for finished games\n")

    if not paired:
        print("No finished games with predictions found.")
        print("Tip: run this script after the day's games are complete.")
        return

    # 3. Print report
    print_report(paired, date_str)

    # 4. Optionally save to JSON
    if args.save:
        report = []
        for p in paired:
            ok = evaluate_prediction(p["prediction"], p["actual"])
            report.append({
                "home_team":  p["actual"]["home_team"],
                "away_team":  p["actual"]["away_team"],
                "league":     p["actual"]["league"],
                "score":      f"{p['actual']['home_goals']}-{p['actual']['away_goals']}",
                "result":     p["actual"]["result"],
                "btts":       p["actual"]["btts"],
                "over_2_5":   p["actual"]["over_2_5"],
                "prediction": p["prediction"].get("prediction", ""),
                "confidence": p["prediction"].get("confidence", 0),
                "correct":    ok,
            })
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"date": date_str, "results": report}, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to {args.save}")


if __name__ == "__main__":
    main()
