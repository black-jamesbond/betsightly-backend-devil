"""
World Cup 2026 — Data Collection

Fetches:
1. All 72 WC 2026 fixtures + real bookmaker odds from The Odds API
2. All 64 WC 2022 matches from API-Football (historical training data)
3. Stores everything in local JSON files for model training

API Budget: ~2 API-Football calls + 1 Odds API call
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
FOOTBALL_API_KEY = os.getenv("API_FOOTBALL_API_KEY", "")
FOOTBALL_API_BASE = "https://v3.football.api-sports.io"
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)


def _api_football_get(endpoint: str, params: dict) -> dict:
    """API-Football request with retry."""
    for attempt in range(3):
        resp = requests.get(
            f"{FOOTBALL_API_BASE}/{endpoint}",
            headers={"x-apisports-key": FOOTBALL_API_KEY},
            params=params,
            timeout=30,
        )
        if resp.status_code == 429:
            time.sleep(10 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    raise Exception("API-Football rate limit exceeded")


def fetch_odds() -> list:
    """Fetch all WC 2026 fixtures with real bookmaker odds from The Odds API."""
    logger.info("Fetching World Cup 2026 odds...")

    resp = requests.get(
        "https://api.the-odds-api.com/v4/sports/soccer_fifa_world_cup/odds",
        params={
            "apiKey": ODDS_API_KEY,
            "regions": "uk,eu",
            "markets": "h2h,totals,spreads",
            "oddsFormat": "decimal",
        },
        timeout=30,
    )
    resp.raise_for_status()
    raw = resp.json()

    with open(DATA_DIR / "wc_odds_raw.json", "w") as f:
        json.dump(raw, f, indent=2)

    # Parse into clean format
    fixtures = []
    for fx in raw:
        match = {
            "id": fx["id"],
            "home_team": fx["home_team"],
            "away_team": fx["away_team"],
            "commence_time": fx["commence_time"],
            "bookmakers": [],
        }

        for bm in fx.get("bookmakers", []):
            bm_data = {"name": bm["title"], "markets": {}}
            for market in bm.get("markets", []):
                bm_data["markets"][market["key"]] = {
                    o["name"]: o["price"] for o in market.get("outcomes", [])
                }
            match["bookmakers"].append(bm_data)

        # Aggregate odds across bookmakers
        h2h = {"home": [], "away": [], "draw": []}
        totals_over = []
        totals_under = []

        for bm in match["bookmakers"]:
            if "h2h" in bm["markets"]:
                m = bm["markets"]["h2h"]
                if match["home_team"] in m:
                    h2h["home"].append(m[match["home_team"]])
                if match["away_team"] in m:
                    h2h["away"].append(m[match["away_team"]])
                if "Draw" in m:
                    h2h["draw"].append(m["Draw"])
            if "totals" in bm["markets"]:
                m = bm["markets"]["totals"]
                if "Over" in m:
                    totals_over.append(m["Over"])
                if "Under" in m:
                    totals_under.append(m["Under"])

        match["best_odds"] = {
            "home_win": round(max(h2h["home"]), 2) if h2h["home"] else None,
            "away_win": round(max(h2h["away"]), 2) if h2h["away"] else None,
            "draw": round(max(h2h["draw"]), 2) if h2h["draw"] else None,
            "over_2_5": round(max(totals_over), 2) if totals_over else None,
            "under_2_5": round(max(totals_under), 2) if totals_under else None,
        }

        def avg_prob(odds_list):
            if not odds_list:
                return None
            return round(1.0 / (sum(odds_list) / len(odds_list)), 4)

        match["implied_prob"] = {
            "home_win": avg_prob(h2h["home"]),
            "away_win": avg_prob(h2h["away"]),
            "draw": avg_prob(h2h["draw"]),
            "over_2_5": avg_prob(totals_over),
            "under_2_5": avg_prob(totals_under),
        }

        fixtures.append(match)

    fixtures.sort(key=lambda x: x["commence_time"])

    with open(DATA_DIR / "wc_fixtures.json", "w") as f:
        json.dump(fixtures, f, indent=2)

    logger.info(f"Saved {len(fixtures)} fixtures with odds")
    return fixtures


def fetch_wc2022_history() -> list:
    """Fetch all 64 WC 2022 matches for training data."""
    cache = DATA_DIR / "wc2022_matches.json"
    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        logger.info(f"Loaded {len(data)} WC 2022 matches from cache")
        return data

    logger.info("Fetching WC 2022 matches from API-Football...")
    resp = _api_football_get("fixtures", {"league": 1, "season": 2022})

    matches = []
    for fx in resp.get("response", []):
        fixture = fx["fixture"]
        teams = fx["teams"]
        goals = fx["goals"]

        home_goals = goals["home"]
        away_goals = goals["away"]

        if home_goals is not None and away_goals is not None:
            if home_goals > away_goals:
                result = "home_win"
            elif away_goals > home_goals:
                result = "away_win"
            else:
                result = "draw"
            total_goals = home_goals + away_goals
        else:
            result = None
            total_goals = None

        matches.append({
            "fixture_id": fixture["id"],
            "date": fixture["date"],
            "home_team": teams["home"]["name"],
            "away_team": teams["away"]["name"],
            "home_team_id": teams["home"]["id"],
            "away_team_id": teams["away"]["id"],
            "home_goals": home_goals,
            "away_goals": away_goals,
            "total_goals": total_goals,
            "result": result,
            "status": fixture["status"]["long"],
            "round": fx["league"]["round"],
        })

    with open(cache, "w") as f:
        json.dump(matches, f, indent=2)

    logger.info(f"Saved {len(matches)} WC 2022 matches")
    return matches


def build_wc_team_profiles(wc2022: list) -> dict:
    """
    Build team profiles from WC 2022 data.
    Returns stats for teams that were in WC 2022 (many overlap with 2026).
    """
    profiles = {}

    for match in wc2022:
        if match["result"] is None:
            continue

        for side in ["home", "away"]:
            team = match[f"{side}_team"]
            other = "away" if side == "home" else "home"

            if team not in profiles:
                profiles[team] = {
                    "matches": 0, "wins": 0, "draws": 0, "losses": 0,
                    "goals_for": 0, "goals_against": 0,
                    "results": [],
                }

            p = profiles[team]
            gf = match[f"{side}_goals"]
            ga = match[f"{other}_goals"]

            p["matches"] += 1
            p["goals_for"] += gf
            p["goals_against"] += ga
            p["results"].append(match["result"] if side == "home" else
                              ("home_win" if match["result"] == "away_win" else
                               "away_win" if match["result"] == "home_win" else "draw"))

            actual_result = p["results"][-1]
            # Normalize to team perspective
            if (side == "home" and match["result"] == "home_win") or \
               (side == "away" and match["result"] == "away_win"):
                p["wins"] += 1
            elif match["result"] == "draw":
                p["draws"] += 1
            else:
                p["losses"] += 1

    # Compute rates
    for team, p in profiles.items():
        n = p["matches"]
        if n > 0:
            p["win_rate"] = round(p["wins"] / n, 3)
            p["draw_rate"] = round(p["draws"] / n, 3)
            p["loss_rate"] = round(p["losses"] / n, 3)
            p["goals_scored_avg"] = round(p["goals_for"] / n, 2)
            p["goals_conceded_avg"] = round(p["goals_against"] / n, 2)
            p["goal_diff_avg"] = round((p["goals_for"] - p["goals_against"]) / n, 2)
            p["clean_sheet_rate"] = round(sum(1 for r in p["results"] if r == "draw" or r == "home_win") / n * 0.3, 3)  # rough estimate
        else:
            p["win_rate"] = 0.0
            p["draw_rate"] = 0.0
            p["loss_rate"] = 0.0
            p["goals_scored_avg"] = 0.0
            p["goals_conceded_avg"] = 0.0
            p["goal_diff_avg"] = 0.0
            p["clean_sheet_rate"] = 0.0

    with open(DATA_DIR / "wc_team_profiles.json", "w") as f:
        json.dump(profiles, f, indent=2)

    logger.info(f"Built profiles for {len(profiles)} teams from WC 2022")
    return profiles


def compute_wc_stats(wc2022: list) -> dict:
    """Compute World Cup-specific statistics for the prediction model."""
    finished = [m for m in wc2022 if m["result"] is not None]
    n = len(finished)

    home_wins = sum(1 for m in finished if m["result"] == "home_win")
    draws = sum(1 for m in finished if m["result"] == "draw")
    away_wins = sum(1 for m in finished if m["result"] == "away_win")
    total_goals = sum(m["total_goals"] for m in finished if m["total_goals"] is not None)
    over_2_5 = sum(1 for m in finished if m["total_goals"] is not None and m["total_goals"] > 2.5)

    stats = {
        "total_matches": n,
        "home_win_rate": round(home_wins / n, 3) if n else 0,
        "draw_rate": round(draws / n, 3) if n else 0,
        "away_win_rate": round(away_wins / n, 3) if n else 0,
        "avg_goals_per_match": round(total_goals / n, 2) if n else 0,
        "over_2_5_rate": round(over_2_5 / n, 3) if n else 0,
        "avg_home_goals": round(sum(m["home_goals"] for m in finished if m["home_goals"] is not None) / n, 2) if n else 0,
        "avg_away_goals": round(sum(m["away_goals"] for m in finished if m["away_goals"] is not None) / n, 2) if n else 0,
    }

    with open(DATA_DIR / "wc_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"WC stats: {stats}")
    return stats


def run():
    """Run full data collection pipeline."""
    logger.info("=== World Cup 2026 Data Collection ===")

    # Step 1: Fetch 2026 fixtures + odds
    fixtures = fetch_odds()
    logger.info(f"Step 1: {len(fixtures)} fixtures with odds")

    # Step 2: Fetch WC 2022 training data
    wc2022 = fetch_wc2022_history()
    logger.info(f"Step 2: {len(wc2022)} WC 2022 matches")

    # Step 3: Build team profiles
    profiles = build_wc_team_profiles(wc2022)
    logger.info(f"Step 3: {len(profiles)} team profiles")

    # Step 4: Compute WC stats
    stats = compute_wc_stats(wc2022)
    logger.info(f"Step 4: WC avg goals/match: {stats['avg_goals_per_match']}")

    # Summary
    teams_2026 = set()
    for fx in fixtures:
        teams_2026.add(fx["home_team"])
        teams_2026.add(fx["away_team"])

    teams_with_profile = sum(1 for t in teams_2026 if t in profiles)
    logger.info(f"\n=== Summary ===")
    logger.info(f"2026 fixtures: {len(fixtures)}")
    logger.info(f"2026 teams: {len(teams_2026)}")
    logger.info(f"Teams with WC 2022 profile: {teams_with_profile}/{len(teams_2026)}")
    logger.info(f"Teams without profile (new to WC): {len(teams_2026) - teams_with_profile}")

    return fixtures, wc2022, profiles, stats


if __name__ == "__main__":
    run()
