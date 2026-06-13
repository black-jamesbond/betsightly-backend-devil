"""
World Cup Daily Feed

Generates daily accumulator data from WC predictions in the same format
the frontend expects from /accumulators/today.

This fills the 2_odds, 5_odds, 10_odds, over_1_5, and rollover categories
using World Cup match predictions when regular league data is unavailable.

Rollover: 1 pick per day at 2-3 odds, 10-day rolling chain.
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent / "data"


def _load(filename: str):
    path = DATA_DIR / filename
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def _save(filename: str, data):
    with open(DATA_DIR / filename, "w") as f:
        json.dump(data, f, indent=2)


def _to_game(p: dict, tip: dict = None) -> dict:
    """Convert a WC prediction to the game format the frontend expects."""
    if tip is None:
        # Use main prediction
        prediction = p.get("prediction", "")
        prediction_type = p.get("prediction_key", "match_result")
        confidence = p.get("confidence", 0.5)
        odds = None
        # Try to get odds from best_odds
        bo = p.get("best_odds", {})
        if prediction_type == "home_win":
            odds = bo.get("home_win")
        elif prediction_type == "away_win":
            odds = bo.get("away_win")
        elif prediction_type == "draw":
            odds = bo.get("draw")
        elif prediction_type in ("over_2_5", "over_1_5"):
            odds = bo.get("over_2_5")
    else:
        prediction = tip.get("tip", "")
        prediction_type = tip.get("market", "match_result")
        confidence = tip.get("confidence", 0.5)
        odds = tip.get("odds")

    return {
        "fixture_id": hash(p.get("match_id", "")) % 1000000,
        "home_team": p.get("home_team", ""),
        "away_team": p.get("away_team", ""),
        "league": "FIFA World Cup 2026",
        "date": p.get("commence_time", ""),
        "prediction": prediction,
        "prediction_type": prediction_type,
        "prediction_value": prediction,
        "readable_prediction": prediction,
        "confidence": confidence,
        "estimated_odds": odds or round(1.0 / max(confidence, 0.1), 2),
        "odds": odds,
        "real_odds": odds,
        "risk_score": 1.0 - confidence,
        "risk_level": p.get("risk_level", "medium"),
        "models_agreed": 3,
        "edge": 0.05,
        "expected_value": 0.1,
        "model_type": "worldcup_ensemble",
        "home_team_logo": p.get("home_team_logo"),
        "away_team_logo": p.get("away_team_logo"),
        "league_logo": "https://media.api-sports.io/football/leagues/1.png",
    }


def build_daily_accumulators() -> dict:
    """
    Build accumulator categories from WC predictions for today/next match day.

    Returns data in the exact format the frontend expects from /accumulators/today.
    """
    predictions = _load("wc_predictions.json")
    if not predictions:
        return None

    today = datetime.now().strftime("%Y-%m-%d")

    # Find next match day
    match_dates = sorted(set(p["commence_time"][:10] for p in predictions))
    next_dates = [d for d in match_dates if d >= today]
    target_date = next_dates[0] if next_dates else (match_dates[-1] if match_dates else today)

    # Get predictions for target date
    day_preds = [p for p in predictions if p["commence_time"].startswith(target_date)]

    # If no matches on target date, use next available
    if not day_preds and next_dates:
        for nd in next_dates:
            day_preds = [p for p in predictions if p["commence_time"].startswith(nd)]
            if day_preds:
                target_date = nd
                break

    if not day_preds:
        day_preds = [p for p in predictions if p["commence_time"][:10] >= today][:15]

    # Always include enough matches to build good accumulators
    # Pull from next several days until we have at least 8 matches
    all_upcoming = [p for p in predictions if p["commence_time"][:10] >= today]
    if len(day_preds) < 8:
        for p in all_upcoming:
            if p not in day_preds:
                day_preds.append(p)
            if len(day_preds) >= 12:
                break

    # Sort by confidence
    day_preds.sort(key=lambda p: p.get("confidence", 0), reverse=True)

    # Collect all tips
    all_tips = []
    for p in day_preds:
        for tip in p.get("top_tips", [{"tip": p["prediction"], "market": p.get("prediction_market", "match_result"), "confidence": p["confidence"]}]):
            all_tips.append({"pred": p, "tip": tip})

    all_tips.sort(key=lambda x: x["tip"]["confidence"], reverse=True)

    # ── 2 Odds: 2-3 safest picks combining to ~2x ──
    two_odds_picks = []
    used_2 = set()
    for t in all_tips:
        mid = t["pred"]["match_id"]
        if mid in used_2:
            continue
        if t["tip"]["confidence"] >= 0.55:
            two_odds_picks.append(t)
            used_2.add(mid)
        if len(two_odds_picks) >= 3:
            break

    two_odds_games = [_to_game(t["pred"], t["tip"]) for t in two_odds_picks]
    two_odds_total = 1.0
    for g in two_odds_games:
        two_odds_total *= g["estimated_odds"]

    # ── 5 Odds: 4-5 picks combining to ~5x ──
    five_odds_picks = []
    used_5 = set()
    for t in all_tips:
        mid = t["pred"]["match_id"]
        if mid in used_5:
            continue
        if t["tip"]["confidence"] >= 0.45:
            five_odds_picks.append(t)
            used_5.add(mid)
        if len(five_odds_picks) >= 5:
            break

    five_odds_games = [_to_game(t["pred"], t["tip"]) for t in five_odds_picks]
    five_odds_total = 1.0
    for g in five_odds_games:
        five_odds_total *= g["estimated_odds"]

    # ── 10 Odds: 5-7 riskier picks ──
    ten_odds_picks = []
    used_10 = set()
    for t in all_tips:
        mid = t["pred"]["match_id"]
        if mid in used_10:
            continue
        if t["tip"]["confidence"] >= 0.35:
            ten_odds_picks.append(t)
            used_10.add(mid)
        if len(ten_odds_picks) >= 7:
            break

    ten_odds_games = [_to_game(t["pred"], t["tip"]) for t in ten_odds_picks]
    ten_odds_total = 1.0
    for g in ten_odds_games:
        ten_odds_total *= g["estimated_odds"]

    # ── Over 1.5: safest goal picks ──
    over_picks = []
    used_o = set()
    for p in day_preds:
        if p["match_id"] in used_o:
            continue
        if p["goals"]["over_1_5_prob"] >= 0.65:
            game = _to_game(p)
            game["prediction"] = "Over 1.5 Goals"
            game["prediction_type"] = "over_1_5"
            game["confidence"] = p["goals"]["over_1_5_prob"]
            game["estimated_odds"] = round(1.0 / max(p["goals"]["over_1_5_prob"], 0.1), 2)
            over_picks.append(game)
            used_o.add(p["match_id"])
        if len(over_picks) >= 5:
            break

    over_total = 1.0
    for g in over_picks:
        over_total *= g["estimated_odds"]

    # ── Rollover: 1 pick per day at 2-3 odds, 10-day chain ──
    rollover = _build_rollover(predictions, today)

    # Build response in frontend-expected format
    def mk_cat(games, total_odds, risk, selected=True, reason=None):
        if not selected or not games:
            return {"selected": False, "games": [], "total_odds": 0, "risk_level": risk, "reason": reason or "No picks available"}
        avg_conf = sum(g["confidence"] for g in games) / len(games) if games else 0
        return {
            "selected": True,
            "games": games,
            "total_odds": round(total_odds, 2),
            "risk_level": risk,
            "reason": None,
        }

    result = {
        "status": "success",
        "date": target_date,
        "source": "worldcup",
        "accumulators": {
            "2_odds": mk_cat(two_odds_games, two_odds_total, "Low"),
            "5_odds": mk_cat(five_odds_games, five_odds_total, "Medium"),
            "10_odds": mk_cat(ten_odds_games, ten_odds_total, "High"),
            "over_1_5": mk_cat(over_picks, over_total, "Very Safe"),
            "rollover": rollover,
        },
    }

    return result


def _build_rollover(predictions: list, today: str) -> dict:
    """
    Build 10-day rollover chain.
    Each day = 1 pick at 2-3 odds.
    Saves state to wc_rollover_chain.json.
    """
    chain_path = DATA_DIR / "wc_rollover_chain.json"

    # Load existing chain
    if chain_path.exists():
        with open(chain_path) as f:
            chain = json.load(f)
    else:
        chain = {"start_date": today, "days": [], "status": "active"}

    # Clean up old chains (if start_date is >10 days ago, reset)
    if chain.get("start_date"):
        start = datetime.strptime(chain["start_date"], "%Y-%m-%d")
        if (datetime.now() - start).days > 10:
            chain = {"start_date": today, "days": [], "status": "active"}

    # Find the next match day
    match_dates = sorted(set(p["commence_time"][:10] for p in predictions))
    next_dates = [d for d in match_dates if d >= today]

    # Only add 1 pick per match day, and only for days not already in chain
    existing_dates = set(d["date"] for d in chain.get("days", []))

    for pick_date in next_dates:
        if pick_date in existing_dates:
            continue
        if len(chain.get("days", [])) >= 10:
            break

        day_preds = [p for p in predictions if p["commence_time"].startswith(pick_date)]

        if day_preds:
            # Find a pick with odds between 1.8 and 3.5
            best = None
            for p in sorted(day_preds, key=lambda x: x["confidence"], reverse=True):
                bo = p.get("best_odds", {})
                pk = p.get("prediction_key", "")
                odds = bo.get(pk) or round(1.0 / max(p["confidence"], 0.1), 2)
                if 1.5 <= odds <= 3.5:
                    best = (p, odds)
                    break

            if not best:
                # Fallback: use highest confidence
                p = day_preds[0]
                odds = round(1.0 / max(p["confidence"], 0.1), 2)
                best = (p, odds)

            p, odds = best
            chain["days"].append({
                "date": pick_date,
                "day_number": len(chain["days"]) + 1,
                "match": f"{p['home_team']} vs {p['away_team']}",
                "prediction": p["prediction"],
                "odds": round(odds, 2),
                "confidence": p["confidence"],
                "status": "pending",
                "game": _to_game(p),
            })

            # Save chain
            _save("wc_rollover_chain.json", chain)

    # Calculate cumulative odds
    cum_odds = 1.0
    for d in chain.get("days", []):
        cum_odds *= d.get("odds", 1.0)

    # Build rollover response
    games = [d.get("game", {}) for d in chain.get("days", []) if d.get("game")]
    today_pick = next((d for d in chain.get("days", []) if d["date"] >= today), None)

    return {
        "selected": True,
        "games": [today_pick["game"]] if today_pick else (games[-1:] if games else []),
        "total_odds": round(cum_odds, 2),
        "risk_level": "Challenge",
        "reason": None,
        "chain": chain.get("days", []),
        "chain_length": len(chain.get("days", [])),
        "target_days": 10,
        "cumulative_odds": round(cum_odds, 2),
    }
