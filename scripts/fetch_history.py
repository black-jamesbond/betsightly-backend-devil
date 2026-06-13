"""
Fetch 5 seasons of historical fixtures from API-Football.

Saves to data/api-football/matches.csv with team names that exactly
match the live fixture API ? eliminating all name-mismatch issues.

Usage:
    py scripts/fetch_history.py

Resume-safe: tracks progress in data/api-football/progress.json so you
can stop and restart without re-fetching completed league/season combos.

API cost: ~80 requests (16 leagues ? 5 seasons). Free tier = 100/day.
"""

import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
API_KEY  = os.getenv("API_FOOTBALL_API_KEY", "bbfc08f4961fb2ef3476a129b8cb1cd9")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

OUTPUT_DIR      = Path("data/api-football")
OUTPUT_CSV      = OUTPUT_DIR / "matches.csv"
PROGRESS_FILE   = OUTPUT_DIR / "progress.json"

# Free plan allows 2022-2024 only
SEASONS = [2022, 2023, 2024]

# League ID ? (display name, country, tier)
TARGET_LEAGUES = {
    39:  ("Premier League",    "England",     1),
    40:  ("Championship",      "England",     2),
    61:  ("Ligue 1",           "France",      1),
    62:  ("Ligue 2",           "France",      2),
    78:  ("Bundesliga",        "Germany",     1),
    79:  ("2. Bundesliga",     "Germany",     2),
    135: ("Serie A",           "Italy",       1),
    136: ("Serie B",           "Italy",       2),
    140: ("La Liga",           "Spain",       1),
    141: ("Segunda Division",  "Spain",       2),
    94:  ("Primeira Liga",     "Portugal",    1),
    88:  ("Eredivisie",        "Netherlands", 1),
    144: ("Pro League",        "Belgium",     1),
    203: ("Super Lig",         "Turkey",      1),
    2:   ("Champions League",  "Europe",      0),
    3:   ("Europa League",     "Europe",      0),
}

CSV_COLUMNS = [
    "home_team", "away_team", "date",
    "home_score", "away_score",
    "ht_home_score", "ht_away_score",
    "league_id", "league_name", "country", "league_tier", "season",
    "home_team_id", "away_team_id",
]

# ---------------------------------------------------------------------------

def load_progress() -> set:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return {tuple(x) for x in json.load(f)}
    return set()


def save_progress(done: set):
    with open(PROGRESS_FILE, "w") as f:
        json.dump([list(x) for x in done], f)


def fetch_season(league_id: int, season: int) -> list:
    """Fetch all finished fixtures for one league/season."""
    resp = requests.get(
        f"{BASE_URL}/fixtures",
        headers=HEADERS,
        params={"league": league_id, "season": season, "status": "FT"},
        timeout=30,
    )

    if resp.status_code == 429:
        print("  Rate limited - waiting 65s...")
        time.sleep(65)
        # Retry once
        resp = requests.get(
            f"{BASE_URL}/fixtures",
            headers=HEADERS,
            params={"league": league_id, "season": season, "status": "FT"},
            timeout=30,
        )

    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code}: {resp.text[:120]}")
        return []

    data = resp.json()
    errors = data.get("errors", {})
    if errors:
        print(f"  API errors: {errors}")
        return []

    return data.get("response", [])


def parse_fixture(f: dict, league_id: int, league_name: str,
                  country: str, tier: int, season: int) -> dict | None:
    score   = f.get("score", {})
    ft      = score.get("fulltime", {})
    ht      = score.get("halftime", {})
    goals   = f.get("goals", {})

    home_score = ft.get("home") if ft.get("home") is not None else goals.get("home")
    away_score = ft.get("away") if ft.get("away") is not None else goals.get("away")

    if home_score is None or away_score is None:
        return None

    raw_date = f.get("fixture", {}).get("date", "")
    date_str = raw_date[:10] if raw_date else ""

    return {
        "home_team":     f["teams"]["home"]["name"],
        "away_team":     f["teams"]["away"]["name"],
        "date":          date_str,
        "home_score":    int(home_score),
        "away_score":    int(away_score),
        "ht_home_score": ht.get("home", ""),
        "ht_away_score": ht.get("away", ""),
        "league_id":     league_id,
        "league_name":   league_name,
        "country":       country,
        "league_tier":   tier,
        "season":        season,
        "home_team_id":  f["teams"]["home"]["id"],
        "away_team_id":  f["teams"]["away"]["id"],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done = load_progress()

    tasks = [
        (lid, season)
        for lid in TARGET_LEAGUES
        for season in SEASONS
        if (lid, season) not in done
    ]

    total_tasks     = len(TARGET_LEAGUES) * len(SEASONS)
    completed_tasks = total_tasks - len(tasks)
    print(f"Progress: {completed_tasks}/{total_tasks} league/season combos already done")
    print(f"Remaining: {len(tasks)} fetches (~{len(tasks) * 1.2:.0f}s)\n")

    if not tasks:
        print("Nothing to fetch ? all done!")
        return

    file_exists = OUTPUT_CSV.exists() and OUTPUT_CSV.stat().st_size > 50
    mode = "a" if file_exists else "w"
    total_written = 0
    requests_used = 0

    with open(OUTPUT_CSV, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()

        for league_id, season in tasks:
            name, country, tier = TARGET_LEAGUES[league_id]
            print(f"[{requests_used + 1}/{len(tasks)}] {name} ? {season}/{season+1}...")

            fixtures = fetch_season(league_id, season)
            requests_used += 1

            rows = [parse_fixture(fx, league_id, name, country, tier, season)
                    for fx in fixtures]
            rows = [r for r in rows if r]

            for row in rows:
                writer.writerow(row)

            total_written += len(rows)
            done.add((league_id, season))
            save_progress(done)

            print(f"  Saved {len(rows)} matches  (running total: {total_written:,})")
            time.sleep(1.2)   # Stay well under rate limit

    print(f"\nOK  Done! {total_written:,} matches ? {OUTPUT_CSV}")
    print(f"   API requests used today: {requests_used}")
    print(f"\nNext step: py scripts/retrain_models.py")


if __name__ == "__main__":
    main()
