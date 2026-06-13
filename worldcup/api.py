"""
World Cup 2026 — API Endpoints

Provides:
- GET  /api/worldcup/fixtures            — all matches with odds
- GET  /api/worldcup/predictions         — all predictions (filterable)
- GET  /api/worldcup/predictions/{id}    — single match prediction
- GET  /api/worldcup/groups              — group stage view
- GET  /api/worldcup/accumulators        — daily accumulator picks
- GET  /api/worldcup/teams               — team list with form stats
- GET  /api/worldcup/value-bets          — best value bets
- GET  /api/worldcup/performance         — prediction accuracy tracker
- POST /api/worldcup/refresh             — re-fetch odds + regenerate
"""

import json
import logging
import threading
import time
from pathlib import Path
from datetime import datetime, date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data"

router = APIRouter(prefix="/api/worldcup", tags=["World Cup 2026"])

# ── World Cup 2026 Groups ──────────────────────────────────
WC_GROUPS = {
    "A": ["Mexico", "South Africa", "Sweden", "Tunisia"],
    "B": ["South Korea", "Czech Republic", "Netherlands", "Japan"],
    "C": ["Canada", "Bosnia & Herzegovina", "Ivory Coast", "Ecuador"],
    "D": ["USA", "Paraguay", "Saudi Arabia", "Uruguay"],
    "E": ["Qatar", "Switzerland", "Belgium", "Egypt"],
    "F": ["Brazil", "Morocco", "Iraq", "Norway"],
    "G": ["Haiti", "Scotland", "Senegal", "Algeria"],
    "H": ["Australia", "Turkey", "Iran", "Jordan"],
    "I": ["Germany", "Curaçao", "Cape Verde", "DR Congo"],
    "J": ["Spain", "Cape Verde", "Croatia", "New Zealand"],
    "K": ["England", "Ghana", "Austria", "Uzbekistan"],
    "L": ["France", "Colombia", "Panama", "Qatar"],
}

# Build reverse lookup: team → group
TEAM_TO_GROUP = {}
for group, teams in WC_GROUPS.items():
    for team in teams:
        TEAM_TO_GROUP[team] = group


def _load_json(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return [] if filename != "wc_team_ids.json" else {}
    with open(path) as f:
        return json.load(f)


def _enrich_logos(predictions: list, team_map: dict) -> list:
    for p in predictions:
        p["home_team_logo"] = team_map.get(p.get("home_team", ""), {}).get("logo")
        p["away_team_logo"] = team_map.get(p.get("away_team", ""), {}).get("logo")
        p["group"] = TEAM_TO_GROUP.get(p.get("home_team", ""), "?")
    return predictions


# ── Auto-refresh background thread ─────────────────────────

_refresh_lock = threading.Lock()
_last_refresh: Optional[str] = None


def _background_refresh():
    """Auto-refresh odds every 6 hours."""
    global _last_refresh
    while True:
        time.sleep(6 * 3600)  # 6 hours
        if _refresh_lock.acquire(blocking=False):
            try:
                logger.info("Auto-refreshing World Cup odds...")
                from worldcup.fetch_data import fetch_odds
                from worldcup.model import generate_all_predictions
                fetch_odds()
                generate_all_predictions()
                _last_refresh = datetime.now().isoformat()
                logger.info(f"Auto-refresh complete at {_last_refresh}")
            except Exception as e:
                logger.error(f"Auto-refresh failed: {e}")
            finally:
                _refresh_lock.release()


# Start auto-refresh thread on import
_refresh_thread = threading.Thread(target=_background_refresh, daemon=True)
_refresh_thread.start()


# ── Endpoints ───────────────────────────────────────────────


@router.get("/fixtures")
async def get_fixtures():
    """Get all World Cup fixtures with odds."""
    fixtures = _load_json("wc_fixtures.json")
    if not fixtures:
        raise HTTPException(404, "No fixtures data. Run fetch_data.py first.")

    team_map = _load_json("wc_team_ids.json")
    for fx in fixtures:
        fx["home_team_logo"] = team_map.get(fx["home_team"], {}).get("logo")
        fx["away_team_logo"] = team_map.get(fx["away_team"], {}).get("logo")
        fx["group"] = TEAM_TO_GROUP.get(fx["home_team"], "?")

    return {
        "status": "success",
        "total": len(fixtures),
        "tournament": "FIFA World Cup 2026",
        "start_date": "2026-06-11",
        "last_refresh": _last_refresh,
        "fixtures": fixtures,
    }


@router.get("/predictions")
async def get_predictions(
    min_confidence: float = Query(0.0),
    risk_level: Optional[str] = Query(None),
    date: Optional[str] = Query(None),
    group: Optional[str] = Query(None, description="Filter by group (A-L)"),
    market: Optional[str] = Query(None, description="Filter by market type"),
):
    """Get all match predictions with filters."""
    predictions = _load_json("wc_predictions.json")
    if not predictions:
        raise HTTPException(404, "No predictions. Run the model first.")

    team_map = _load_json("wc_team_ids.json")
    _enrich_logos(predictions, team_map)

    # Filters
    if min_confidence > 0:
        predictions = [p for p in predictions if p["confidence"] >= min_confidence]
    if risk_level:
        predictions = [p for p in predictions if p["risk_level"] == risk_level]
    if date:
        predictions = [p for p in predictions if p["commence_time"].startswith(date)]
    if group:
        predictions = [p for p in predictions if p.get("group", "").upper() == group.upper()]
    if market:
        predictions = [p for p in predictions if p.get("prediction_market") == market]

    return {
        "status": "success",
        "total": len(predictions),
        "last_refresh": _last_refresh,
        "predictions": predictions,
    }


@router.get("/predictions/{match_id}")
async def get_prediction(match_id: str):
    """Get prediction for a single match."""
    predictions = _load_json("wc_predictions.json")
    team_map = _load_json("wc_team_ids.json")
    for p in predictions:
        if p["match_id"] == match_id:
            _enrich_logos([p], team_map)
            return {"status": "success", "prediction": p}
    raise HTTPException(404, f"Match {match_id} not found")


@router.get("/groups")
async def get_groups():
    """Get World Cup group stage view with matches and predictions per group."""
    predictions = _load_json("wc_predictions.json")
    team_map = _load_json("wc_team_ids.json")

    if not predictions:
        raise HTTPException(404, "No predictions available.")

    groups = {}
    for group_name, team_names in WC_GROUPS.items():
        teams = []
        for name in team_names:
            info = team_map.get(name, {})
            teams.append({"name": name, "logo": info.get("logo")})

        # Find matches for this group
        group_matches = []
        for p in predictions:
            if TEAM_TO_GROUP.get(p["home_team"]) == group_name:
                _enrich_logos([p], team_map)
                group_matches.append(p)

        group_matches.sort(key=lambda x: x["commence_time"])

        groups[group_name] = {
            "teams": teams,
            "matches": group_matches,
            "total_matches": len(group_matches),
        }

    return {
        "status": "success",
        "total_groups": len(groups),
        "groups": groups,
    }


@router.get("/accumulators")
async def get_accumulators(
    date: Optional[str] = Query(None, description="Date (YYYY-MM-DD), defaults to next match day"),
):
    """
    Get daily accumulator picks for World Cup matches.
    Returns 3 categories: Safe (2x), Moderate (5x), Bold (10x).
    """
    predictions = _load_json("wc_predictions.json")
    team_map = _load_json("wc_team_ids.json")

    if not predictions:
        raise HTTPException(404, "No predictions available.")

    _enrich_logos(predictions, team_map)

    # Filter to date (or find next match day)
    if date:
        day_preds = [p for p in predictions if p["commence_time"].startswith(date)]
    else:
        # Find the next date with matches
        today = datetime.now().strftime("%Y-%m-%d")
        future = sorted(set(
            p["commence_time"][:10] for p in predictions
            if p["commence_time"][:10] >= today
        ))
        if not future:
            return {"status": "success", "date": today, "accumulators": {}, "message": "No upcoming matches"}
        date = future[0]
        day_preds = [p for p in predictions if p["commence_time"].startswith(date)]

    if not day_preds:
        return {"status": "success", "date": date, "accumulators": {}, "message": "No matches on this date"}

    # Sort all tips across all matches by confidence
    all_tips = []
    for p in day_preds:
        for tip in p.get("top_tips", []):
            all_tips.append({
                "match": f"{p['home_team']} vs {p['away_team']}",
                "match_id": p["match_id"],
                "home_team": p["home_team"],
                "away_team": p["away_team"],
                "home_team_logo": p.get("home_team_logo"),
                "away_team_logo": p.get("away_team_logo"),
                "commence_time": p["commence_time"],
                "group": p.get("group", "?"),
                "tip": tip["tip"],
                "market": tip["market"],
                "confidence": tip["confidence"],
                "odds": tip.get("odds"),
            })

    all_tips.sort(key=lambda x: x["confidence"], reverse=True)

    # Build accumulators — pick from different matches
    def build_accu(tips, min_conf, max_picks, target_label):
        """Pick top tips above min_conf from different matches."""
        picks = []
        used_matches = set()
        for t in tips:
            if t["match_id"] in used_matches:
                continue
            if t["confidence"] >= min_conf:
                picks.append(t)
                used_matches.add(t["match_id"])
            if len(picks) >= max_picks:
                break

        total_odds = 1.0
        for pick in picks:
            # Estimate odds from confidence if not available
            est_odds = pick["odds"] if pick["odds"] else round(1.0 / max(pick["confidence"], 0.1), 2)
            pick["estimated_odds"] = est_odds
            total_odds *= est_odds

        return {
            "picks": picks,
            "total_picks": len(picks),
            "total_odds": round(total_odds, 2),
            "label": target_label,
        }

    accumulators = {
        "safe": build_accu(all_tips, 0.60, 3, "Safe Picks (2-3x)"),
        "moderate": build_accu(all_tips, 0.45, 5, "Moderate (5-8x)"),
        "bold": build_accu(all_tips, 0.30, 7, "Bold (10x+)"),
    }

    return {
        "status": "success",
        "date": date,
        "total_matches": len(day_preds),
        "accumulators": accumulators,
    }


@router.get("/teams")
async def get_teams():
    """Get all World Cup teams with form stats and group."""
    team_map = _load_json("wc_team_ids.json")
    profiles = _load_json("wc_team_profiles.json")
    wc2022 = _load_json("wc2022_matches.json")

    if not team_map:
        raise HTTPException(404, "No team data available.")

    teams = []
    for name, info in team_map.items():
        profile = profiles.get(name, {})

        # Find WC 2022 matches for this team
        recent = []
        for m in (wc2022 or []):
            if m["home_team"] == name or m["away_team"] == name:
                is_home = m["home_team"] == name
                recent.append({
                    "opponent": m["away_team"] if is_home else m["home_team"],
                    "result": m["result"] if is_home else (
                        "home_win" if m["result"] == "away_win" else
                        "away_win" if m["result"] == "home_win" else "draw"
                    ),
                    "score": f"{m['home_goals']}-{m['away_goals']}",
                    "round": m.get("round", ""),
                })

        teams.append({
            "name": name,
            "logo": info.get("logo"),
            "group": TEAM_TO_GROUP.get(name, "?"),
            "wc_stats": {
                "matches": profile.get("matches", 0),
                "wins": profile.get("wins", 0),
                "draws": profile.get("draws", 0),
                "losses": profile.get("losses", 0),
                "goals_for": profile.get("goals_for", 0),
                "goals_against": profile.get("goals_against", 0),
                "win_rate": profile.get("win_rate", 0),
                "goals_scored_avg": profile.get("goals_scored_avg", 0),
                "goals_conceded_avg": profile.get("goals_conceded_avg", 0),
            },
            "wc2022_matches": recent[:7],
        })

    teams.sort(key=lambda t: t["wc_stats"]["win_rate"], reverse=True)

    return {"status": "success", "total": len(teams), "teams": teams}


@router.get("/value-bets")
async def get_value_bets(min_edge: float = Query(0.0)):
    """Get best value bets across all matches."""
    predictions = _load_json("wc_predictions.json")
    if not predictions:
        raise HTTPException(404, "No predictions available.")

    team_map = _load_json("wc_team_ids.json")
    value_bets = []
    for p in predictions:
        for vb in p.get("value_bets", []):
            if vb["edge"] >= min_edge:
                value_bets.append({
                    "match": f"{p['home_team']} vs {p['away_team']}",
                    "match_id": p["match_id"],
                    "commence_time": p["commence_time"],
                    "group": TEAM_TO_GROUP.get(p["home_team"], "?"),
                    "home_team_logo": team_map.get(p["home_team"], {}).get("logo"),
                    "away_team_logo": team_map.get(p["away_team"], {}).get("logo"),
                    **vb,
                })

    value_bets.sort(key=lambda x: x["expected_value"], reverse=True)

    return {"status": "success", "total": len(value_bets), "value_bets": value_bets}


@router.get("/performance")
async def get_performance():
    """
    Get prediction accuracy tracker.
    Compares predictions against actual results (once matches are played).
    """
    predictions = _load_json("wc_predictions.json")
    results = _load_json("wc_results.json")  # Will be empty until matches are played

    if not predictions:
        raise HTTPException(404, "No predictions available.")

    if not results:
        return {
            "status": "success",
            "message": "No results yet — tournament hasn't started",
            "total_predictions": len(predictions),
            "resolved": 0,
            "correct": 0,
            "accuracy": 0,
            "by_market": {},
            "by_risk": {},
        }

    # Match results to predictions
    result_map = {r["match_id"]: r for r in results}
    correct = 0
    resolved = 0
    by_market = {}
    by_risk = {}

    for p in predictions:
        result = result_map.get(p["match_id"])
        if not result:
            continue
        resolved += 1

        market = p.get("prediction_market", "match_result")
        risk = p.get("risk_level", "medium")

        if market not in by_market:
            by_market[market] = {"total": 0, "correct": 0}
        if risk not in by_risk:
            by_risk[risk] = {"total": 0, "correct": 0}

        by_market[market]["total"] += 1
        by_risk[risk]["total"] += 1

        if result.get("prediction_correct"):
            correct += 1
            by_market[market]["correct"] += 1
            by_risk[risk]["correct"] += 1

    # Calculate rates
    for m in by_market.values():
        m["accuracy"] = round(m["correct"] / m["total"] * 100, 1) if m["total"] > 0 else 0
    for r in by_risk.values():
        r["accuracy"] = round(r["correct"] / r["total"] * 100, 1) if r["total"] > 0 else 0

    return {
        "status": "success",
        "total_predictions": len(predictions),
        "resolved": resolved,
        "correct": correct,
        "accuracy": round(correct / resolved * 100, 1) if resolved > 0 else 0,
        "by_market": by_market,
        "by_risk": by_risk,
    }


@router.get("/daily-accumulators")
async def get_daily_accumulators():
    """
    Get daily accumulator picks from World Cup data.
    Same format as /accumulators/today — used as fallback when leagues are off-season.
    """
    try:
        from worldcup.daily_feed import build_daily_accumulators
        result = build_daily_accumulators()
        if not result:
            raise HTTPException(404, "No WC predictions available")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error building daily accumulators: {e}", exc_info=True)
        raise HTTPException(500, str(e))


@router.post("/refresh")
async def refresh_predictions():
    """Re-fetch odds and regenerate predictions."""
    global _last_refresh
    if not _refresh_lock.acquire(blocking=False):
        return {"status": "busy", "message": "Refresh already in progress"}

    try:
        from worldcup.fetch_data import fetch_odds
        from worldcup.model import generate_all_predictions

        fixtures = fetch_odds()
        predictions = generate_all_predictions()
        _last_refresh = datetime.now().isoformat()

        return {
            "status": "success",
            "fixtures_updated": len(fixtures),
            "predictions_generated": len(predictions),
            "timestamp": _last_refresh,
        }
    except Exception as e:
        logger.error(f"Error refreshing predictions: {e}", exc_info=True)
        raise HTTPException(500, f"Refresh failed: {str(e)}")
    finally:
        _refresh_lock.release()
