#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Daily Predictions Script — run this once per day to get today's accumulators.

Usage:
    python scripts/daily_predictions.py                 # today's predictions
    python scripts/daily_predictions.py 2026-05-10      # specific date
    python scripts/daily_predictions.py --force          # re-run even if cached

How it works:
    1. Fetches today's fixtures from API-Football (cached — 1 API call/day)
    2. Loads ML models from disk (instant — no retraining)
    3. Runs 8 models per fixture (XGB, LGBM, ELO, Dixon-Coles + BMA)
    4. Quality-gates predictions (45%+ for match result, 60%+ for binary)
    5. Builds 4 accumulators: 2_odds, 5_odds, 10_odds, rollover
    6. Stores results in database for the API to serve

API calls used: ~1 per day (fixtures are cached).
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from pathlib import Path

# Fix Windows console encoding for team names with special characters
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Ensure project root is on path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def run(target_date: str = None, force: bool = False):
    """Generate predictions for a date and print results."""
    import warnings
    warnings.filterwarnings("ignore")

    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n{'='*60}")
    print(f"  BetSightly Daily Predictions — {target_date}")
    print(f"{'='*60}\n")

    # --- Step 1: Fetch fixtures ---
    from services.apifootball_service import APIFootballService
    svc = APIFootballService()
    fixtures = svc.get_daily_fixtures(target_date)
    upcoming = [f for f in fixtures if f.get("status") in ("NS", "TBD")]

    if not upcoming:
        print("No upcoming fixtures found for this date.")
        print("This could mean:")
        print("  - No games in tracked leagues today")
        print("  - All games already started/finished")
        return

    print(f"Fixtures found: {len(upcoming)} upcoming games\n")
    for fx in upcoming:
        print(f"  {fx['home_team']} vs {fx['away_team']} ({fx['league_name']})")

    # --- Step 2: Load models (from disk — instant) ---
    print(f"\nLoading models...", end=" ", flush=True)
    import time
    t0 = time.time()
    from api.endpoints.ml_predictions import RealMLPredictionService, _elo_system, _dixon_coles
    ml = RealMLPredictionService()
    print(f"done ({time.time()-t0:.1f}s)")

    elo_teams = len(getattr(_elo_system, "ratings", {}))
    dc_teams = len(getattr(_dixon_coles, "teams", [])) if _dixon_coles else 0
    print(f"  6 ML models + ELO ({elo_teams} teams) + Dixon-Coles ({dc_teams} teams)\n")

    # --- Step 3: Fetch real bookmaker odds ---
    odds_map = {}
    try:
        from services.odds_service import OddsService
        odds_svc = OddsService()
        if odds_svc.api_key:
            print("Fetching real bookmaker odds...", end=" ", flush=True)
            odds_map = odds_svc.get_odds_for_fixtures(upcoming)
            print(f"matched {len(odds_map)}/{len(upcoming)} fixtures")
            if odds_map:
                print(f"  Strategy: VALUE-BASED (using real market odds)\n")
            else:
                print(f"  No odds matched — using confidence-based fallback\n")
        else:
            print("  No ODDS_API_KEY set — using confidence-based selection")
            print("  (Set ODDS_API_KEY in .env for value-based picks)\n")
    except ImportError:
        print("  Odds service not available — using confidence-based selection\n")
    except Exception as e:
        logger.warning(f"Odds fetch failed: {e} — using fallback")
        print(f"  Odds unavailable ({e}) — using confidence-based fallback\n")

    # --- Step 4: Generate predictions ---
    print("Generating predictions...")
    all_predictions = []
    for fx in upcoming:
        try:
            pred = ml.generate_predictions_for_fixture(fx)
            if "error" not in pred:
                all_predictions.append(pred)
        except Exception as e:
            logger.warning(f"Skipped {fx['home_team']} vs {fx['away_team']}: {e}")

    print(f"  {len(all_predictions)}/{len(upcoming)} fixtures predicted\n")

    # --- Step 5: Build accumulators ---
    from services.accumulator_builder import AccumulatorBuilder
    builder = AccumulatorBuilder()
    result = builder.build_accumulators(all_predictions, odds_map=odds_map or None)

    strategy = result.get("strategy", "confidence_based")
    print(f"Strategy: {strategy.upper().replace('_', ' ')}")
    print(f"Quality picks: {result.get('quality_selections', 0)}")
    print(f"Safe picks: {result.get('safe_selections', 0)}")
    if result.get("value_selections") is not None:
        print(f"Value bets (edge > 3%): {result.get('value_selections', 0)}")
    print()

    # --- Step 6: Display accumulators ---
    accumulators = result.get("accumulators", {})
    any_selected = False

    for cat, acc in accumulators.items():
        if acc.get("selected"):
            any_selected = True
            print(f"{'='*50}")
            header = f"  {cat.upper()}: {acc['num_games']} games @ {acc['total_odds']:.2f} odds"
            if acc.get("average_edge"):
                header += f"  [AVG EDGE: {acc['average_edge']*100:.1f}%]"
            print(header)
            print(f"  Confidence: {acc['average_confidence']*100:.1f}%  |  Risk: {acc['risk_level']}")
            if acc.get("value_rating"):
                print(f"  Value Rating: {acc['value_rating']}")
            print(f"{'='*50}")
            for g in acc["games"]:
                conf_bar = "=" * int(g["confidence"] * 20)
                print(f"  {g['home_team']} vs {g['away_team']}")
                odds_str = f"odds {g.get('real_odds') or g.get('odds', '?')}"
                edge_str = ""
                if g.get("edge") and g["edge"] > 0:
                    edge_str = f"  EDGE: +{g['edge']*100:.1f}%"
                print(f"    -> {g['prediction']}  [{conf_bar}] {g['confidence']*100:.1f}%  ({odds_str}){edge_str}")
            print()
        else:
            print(f"  {cat.upper()}: SKIPPED — {acc.get('reason', 'not enough qualifying bets')}")

    if not any_selected:
        print("\nNo accumulators could be built today.")
        if odds_map:
            print("No bets with sufficient value (edge > 3%) found.")
            print("This means the market is priced efficiently today — no edge to exploit.")
        else:
            print("Not enough confident + safe predictions to hit any target odds.")
        return

    # --- Step 6: Store in database ---
    try:
        from services.daily_predictions_service import DailyPredictionsService
        db_service = DailyPredictionsService()

        if force:
            # Delete existing predictions for this date to force regeneration
            from database import get_db
            from services.daily_predictions_service import DailyPredictionSummary
            db = next(get_db())
            existing = db.query(DailyPredictionSummary).filter(
                DailyPredictionSummary.prediction_date == datetime.strptime(target_date, "%Y-%m-%d").date()
            ).first()
            if existing:
                existing.generation_status = "pending"
                db.commit()
            db.close()

        db_result = db_service.generate_daily_predictions(target_date)
        if db_result["status"] == "success":
            print(f"Stored in database for API to serve.")
        elif db_result["status"] == "already_exists":
            print(f"Already in database (run with --force to regenerate).")
        else:
            print(f"Database storage: {db_result.get('message', 'unknown')}")
    except Exception as e:
        logger.warning(f"Could not store in database (predictions still shown above): {e}")

    # --- Summary ---
    summary = result.get("summary", {})
    print(f"\n{'='*60}")
    print(f"  Accumulators: {summary.get('categories_with_accumulators', 0)}/{summary.get('total_categories', 4)}")
    print(f"  Total games across accumulators: {summary.get('total_games_in_accumulators', 0)}")
    print(f"  API calls used: ~1 (fixtures cached for the day)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BetSightly Daily Predictions")
    parser.add_argument("date", nargs="?", default=None, help="Date in YYYY-MM-DD format (default: today)")
    parser.add_argument("--force", action="store_true", help="Force regeneration even if already cached")
    args = parser.parse_args()
    run(target_date=args.date, force=args.force)
