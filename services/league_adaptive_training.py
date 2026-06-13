"""
league_adaptive_training.py
----------------------------
For every league/tournament represented in today's fixtures, this service:

  1. Identifies leagues that are NEW (not yet in the training CSV) or STALE
     (last fetched > REFRESH_DAYS ago).
  2. Fetches up to MAX_SEASONS seasons of historical results for those leagues
     from API-Football.
  3. Merges the new rows into data/api-football/matches.csv (dedup by
     fixture_id if available, otherwise by date+home_team+away_team).
  4. Re-runs the full model-training pipeline so the models loaded by
     advanced_prediction_service are aware of these leagues.
  5. Hot-reloads the API-Football models inside the running prediction service
     so the next prediction call benefits immediately -- no restart needed.

Designed to be called once per day, just before predictions are generated:

    from services.league_adaptive_training import league_adaptive_trainer
    report = league_adaptive_trainer.run_for_today()          # uses today's date
    report = league_adaptive_trainer.run_for_date("2026-06-05")

Or from the CLI:

    python services/league_adaptive_training.py
    python services/league_adaptive_training.py --date 2026-06-05 --dry-run
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

# ── project root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger(__name__)

# ── configuration ─────────────────────────────────────────────────────────────
API_KEY  = os.getenv("API_FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
HEADERS  = {"x-apisports-key": API_KEY}

MATCHES_CSV    = ROOT / "data" / "api-football" / "matches.csv"
LEAGUE_LOG     = ROOT / "data" / "api-football" / "fetched_leagues.json"
MODELS_DIR     = ROOT / "models" / "api_football"
CACHE_DIR      = ROOT / "cache" / "api_football"

# How many past seasons to fetch for a brand-new league
MAX_SEASONS = 3
# Re-fetch a league's data if last fetch was more than this many days ago
REFRESH_DAYS = 30
# Minimum games needed to include a league in training
MIN_GAMES_TO_TRAIN = 20
# Pause between API calls (seconds) to stay within rate limits
API_PAUSE = 1.5

CSV_COLUMNS = [
    "home_team", "away_team", "date",
    "home_score", "away_score",
    "ht_home_score", "ht_away_score",
    "league_id", "league_name", "country", "league_tier", "season",
    "home_team_id", "away_team_id",
]


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering (mirrors retrain_models.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

FORM_WINDOW = 5
H2H_WINDOW  = 10


def _team_stats(df_past: pd.DataFrame, team: str, n: int) -> dict:
    games = df_past[
        (df_past["home_team"] == team) | (df_past["away_team"] == team)
    ].tail(n)
    if len(games) == 0:
        return {"win_rate": 0.5, "draw_rate": 0.25,
                "goals_scored": 1.2, "goals_conceded": 1.2}
    wins = draws = gs = gc = 0
    for _, r in games.iterrows():
        if r["home_team"] == team:
            s, c = r["home_score"], r["away_score"]
        else:
            s, c = r["away_score"], r["home_score"]
        gs += s; gc += c
        if s > c: wins += 1
        elif s == c: draws += 1
    n_g = len(games)
    return {"win_rate": wins/n_g, "draw_rate": draws/n_g,
            "goals_scored": gs/n_g, "goals_conceded": gc/n_g}


def _team_home_stats(df_past: pd.DataFrame, team: str, n: int) -> dict:
    games = df_past[df_past["home_team"] == team].tail(n)
    if len(games) == 0:
        return {"win_rate": 0.5, "goals_scored": 1.5}
    wins  = sum(1 for _, r in games.iterrows() if r["home_score"] > r["away_score"])
    goals = sum(r["home_score"] for _, r in games.iterrows())
    n_g   = len(games)
    return {"win_rate": wins/n_g, "goals_scored": goals/n_g}


def _team_away_stats(df_past: pd.DataFrame, team: str, n: int) -> dict:
    games = df_past[df_past["away_team"] == team].tail(n)
    if len(games) == 0:
        return {"win_rate": 0.35, "goals_scored": 1.1}
    wins  = sum(1 for _, r in games.iterrows() if r["away_score"] > r["home_score"])
    goals = sum(r["away_score"] for _, r in games.iterrows())
    n_g   = len(games)
    return {"win_rate": wins/n_g, "goals_scored": goals/n_g}


def _h2h_stats(df_past: pd.DataFrame, home: str, away: str, n: int) -> dict:
    h2h = df_past[
        ((df_past["home_team"] == home) & (df_past["away_team"] == away)) |
        ((df_past["home_team"] == away) & (df_past["away_team"] == home))
    ].tail(n)
    if len(h2h) == 0:
        return {"home_win_rate": 0.45, "avg_goals": 2.5, "btts_rate": 0.5, "n": 0}
    home_wins = btts = total_goals = 0
    for _, r in h2h.iterrows():
        hg = r["home_score"] if r["home_team"] == home else r["away_score"]
        ag = r["away_score"] if r["home_team"] == home else r["home_score"]
        total_goals += hg + ag
        if hg > ag: home_wins += 1
        if hg > 0 and ag > 0: btts += 1
    n_g = len(h2h)
    return {"home_win_rate": home_wins/n_g, "avg_goals": total_goals/n_g,
            "btts_rate": btts/n_g, "n": n_g}


def compute_features(home: str, away: str, date,
                     df_past: pd.DataFrame, league_tier: int = 1) -> list:
    h5  = _team_stats(df_past, home, 5)
    h10 = _team_stats(df_past, home, 10)
    hh5 = _team_home_stats(df_past, home, 5)
    a5  = _team_stats(df_past, away, 5)
    a10 = _team_stats(df_past, away, 10)
    aa5 = _team_away_stats(df_past, away, 5)
    h2h = _h2h_stats(df_past, home, away, H2H_WINDOW)
    return [
        h5["win_rate"],   h10["win_rate"],  h5["draw_rate"],
        h5["goals_scored"], h5["goals_conceded"],
        hh5["win_rate"],  hh5["goals_scored"],
        a5["win_rate"],   a10["win_rate"],  a5["draw_rate"],
        a5["goals_scored"], a5["goals_conceded"],
        aa5["win_rate"],  aa5["goals_scored"],
        h2h["home_win_rate"], h2h["avg_goals"], h2h["btts_rate"],
        min(h2h["n"], 10) / 10,
        league_tier / 2,
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main service
# ─────────────────────────────────────────────────────────────────────────────

class LeagueAdaptiveTrainer:
    """
    Detects new/stale leagues from today's fixtures, fetches their historical
    data, augments the training CSV, retrains all models, and hot-reloads
    them in the running prediction service.
    """

    def __init__(self):
        MATCHES_CSV.parent.mkdir(parents=True, exist_ok=True)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ── public entry points ───────────────────────────────────────────────────

    def run_for_today(self, dry_run: bool = False) -> Dict[str, Any]:
        """Run adaptive training for today's date."""
        return self.run_for_date(datetime.now().strftime("%Y-%m-%d"), dry_run=dry_run)

    def run_for_date(self, date_str: str, dry_run: bool = False) -> Dict[str, Any]:
        """
        Full pipeline for a specific date.

        Returns a report dict with keys:
            date, leagues_found, leagues_new, leagues_fetched,
            rows_added, models_retrained, errors
        """
        report: Dict[str, Any] = {
            "date": date_str,
            "leagues_found": [],
            "leagues_new": [],
            "leagues_fetched": [],
            "rows_added": 0,
            "models_retrained": False,
            "errors": [],
        }

        logger.info(f"[LeagueAdaptiveTrainer] Starting for {date_str} (dry_run={dry_run})")

        # 1. Get today's leagues
        fixtures = self._load_fixtures(date_str)
        if not fixtures:
            report["errors"].append("No fixtures found for this date")
            return report

        leagues = self._extract_leagues(fixtures)
        report["leagues_found"] = [
            {"id": lid, "name": lname, "country": lct}
            for lid, (lname, lct) in leagues.items()
        ]
        logger.info(f"  Found {len(leagues)} distinct leagues in today's fixtures")

        # 2. Determine which leagues need data
        log = self._load_league_log()
        leagues_to_fetch = self._leagues_needing_update(leagues, log)
        report["leagues_new"] = [
            {"id": lid, "name": leagues[lid][0]}
            for lid in leagues_to_fetch
        ]
        logger.info(f"  {len(leagues_to_fetch)} leagues need historical data fetch")

        if not leagues_to_fetch:
            logger.info("  All leagues are up-to-date — skipping fetch and training")
            return report

        if dry_run:
            logger.info("  DRY RUN — would fetch these leagues: " +
                        ", ".join(str(l) for l in leagues_to_fetch))
            return report

        # 3. Fetch historical data for each new/stale league
        all_new_rows: List[dict] = []
        for league_id in leagues_to_fetch:
            league_name, country = leagues[league_id]
            rows, err = self._fetch_league_history(league_id, league_name, country)
            if err:
                report["errors"].append(f"{league_name}: {err}")
                continue
            if rows:
                all_new_rows.extend(rows)
                log[str(league_id)] = {
                    "name": league_name,
                    "country": country,
                    "last_fetched": datetime.now().isoformat(),
                    "rows": len(rows),
                }
                report["leagues_fetched"].append({
                    "id": league_id,
                    "name": league_name,
                    "rows_fetched": len(rows),
                })
                logger.info(f"  [{league_name}] fetched {len(rows)} historical matches")
            else:
                logger.warning(f"  [{league_name}] no historical data returned")

        # 4. Merge new rows into matches.csv
        if all_new_rows:
            added = self._merge_into_csv(all_new_rows)
            report["rows_added"] = added
            self._save_league_log(log)
            logger.info(f"  Added {added} new rows to {MATCHES_CSV}")
        else:
            logger.info("  No new rows to add")
            return report

        # 5. Retrain models on the augmented dataset
        if report["rows_added"] > 0:
            logger.info("  Retraining models on augmented dataset …")
            train_ok, train_msg = self._retrain_models()
            report["models_retrained"] = train_ok
            if not train_ok:
                report["errors"].append(f"Training error: {train_msg}")
            else:
                logger.info("  Model training complete")
                # 6. Hot-reload models in the live prediction service
                self._hot_reload_models()

        return report

    # ── fixtures loading ──────────────────────────────────────────────────────

    def _load_fixtures(self, date_str: str) -> List[dict]:
        """
        Load fixtures from local cache (written by check_accuracy.py or the
        prediction pipeline) or via a fresh API call.
        """
        # Try cache written by check_accuracy.py
        cache_path = CACHE_DIR / f"results_{date_str}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"  Loaded {len(data)} fixtures from cache ({cache_path.name})")
                return data
            except Exception:
                pass

        # Try data/fixtures/ directory (CSV format, for compatibility)
        csv_path = ROOT / "data" / "fixtures" / f"fixtures_{date_str}.json"
        if csv_path.exists():
            try:
                with open(csv_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                logger.info(f"  Loaded {len(data)} fixtures from {csv_path.name}")
                return data
            except Exception:
                pass

        # Last resort: live API call
        if not API_KEY:
            logger.warning("  No API key configured — cannot fetch fixtures")
            return []

        logger.info(f"  Fetching fixtures from API for {date_str} …")
        try:
            r = requests.get(f"{BASE_URL}/fixtures", headers=HEADERS,
                             params={"date": date_str}, timeout=30)
            body = r.json()
            if body.get("errors"):
                logger.error(f"  API error: {body['errors']}")
                return []
            fixtures = body.get("response", [])
            # Save to cache for future use
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(fixtures, f, ensure_ascii=False)
            return fixtures
        except Exception as e:
            logger.error(f"  Failed to fetch fixtures: {e}")
            return []

    # ── league discovery ──────────────────────────────────────────────────────

    def _extract_leagues(self, fixtures: List[dict]) -> Dict[int, Tuple[str, str]]:
        """Return {league_id: (league_name, country)} for all fixtures."""
        leagues: Dict[int, Tuple[str, str]] = {}
        for f in fixtures:
            league = f.get("league", {})
            lid    = league.get("id")
            lname  = league.get("name", "Unknown")
            country = league.get("country", "")
            if lid and lid not in leagues:
                leagues[lid] = (lname, country)
        return leagues

    def _leagues_needing_update(
        self,
        leagues: Dict[int, Tuple[str, str]],
        log: dict,
    ) -> List[int]:
        """
        Return league IDs that either have never been fetched or whose
        last fetch was older than REFRESH_DAYS.
        """
        cutoff = datetime.now() - timedelta(days=REFRESH_DAYS)
        needs: List[int] = []
        for lid in leagues:
            entry = log.get(str(lid))
            if entry is None:
                needs.append(lid)
            else:
                last = datetime.fromisoformat(entry["last_fetched"])
                if last < cutoff:
                    needs.append(lid)
        return needs

    # ── historical data fetch ─────────────────────────────────────────────────

    def _fetch_league_history(
        self,
        league_id: int,
        league_name: str,
        country: str,
    ) -> Tuple[List[dict], Optional[str]]:
        """
        Fetch finished matches for this league across up to MAX_SEASONS
        seasons ending this year.

        Returns (rows_list, error_string_or_None).
        Each row matches CSV_COLUMNS schema.
        """
        if not API_KEY:
            return [], "API_FOOTBALL_API_KEY not configured"

        current_year = datetime.now().year
        seasons = list(range(current_year - MAX_SEASONS + 1, current_year + 1))
        # Try the current season too (some tournaments run within one year)
        if current_year not in seasons:
            seasons.append(current_year)

        all_rows: List[dict] = []

        for season in seasons:
            logger.info(f"    Fetching {league_name} season {season} …")
            try:
                time.sleep(API_PAUSE)  # respect rate limit
                r = requests.get(
                    f"{BASE_URL}/fixtures",
                    headers=HEADERS,
                    params={"league": league_id, "season": season, "status": "FT"},
                    timeout=30,
                )

                if r.status_code == 429:
                    logger.warning("    Rate limited — waiting 65s …")
                    time.sleep(65)
                    r = requests.get(
                        f"{BASE_URL}/fixtures",
                        headers=HEADERS,
                        params={"league": league_id, "season": season, "status": "FT"},
                        timeout=30,
                    )

                if r.status_code != 200:
                    logger.warning(f"    HTTP {r.status_code} for {league_name} {season}")
                    continue

                body = r.json()
                if body.get("errors"):
                    errs = body["errors"]
                    if isinstance(errs, dict) and errs.get("access"):
                        return [], f"API account issue: {errs['access']}"
                    logger.warning(f"    API errors: {errs}")
                    continue

                raw = body.get("response", [])
                for f in raw:
                    row = self._parse_fixture(f, league_id, league_name, country, season)
                    if row:
                        all_rows.append(row)

                logger.info(f"    {len(raw)} raw fixtures → {sum(1 for f in raw if f)} rows parsed")

            except requests.exceptions.RequestException as e:
                logger.warning(f"    Network error for {league_name} {season}: {e}")
                continue

        return all_rows, None

    def _parse_fixture(
        self,
        f: dict,
        league_id: int,
        league_name: str,
        country: str,
        season: int,
    ) -> Optional[dict]:
        """Parse a raw API-Football fixture into a CSV row dict."""
        try:
            score  = f.get("score", {})
            ft     = score.get("fulltime", {})
            ht     = score.get("halftime", {})
            goals  = f.get("goals", {})

            home_score = ft.get("home") if ft.get("home") is not None else goals.get("home")
            away_score = ft.get("away") if ft.get("away") is not None else goals.get("away")

            if home_score is None or away_score is None:
                return None

            raw_date = f.get("fixture", {}).get("date", "")
            date_str = raw_date[:10] if raw_date else ""
            if not date_str:
                return None

            return {
                "home_team":      f["teams"]["home"]["name"],
                "away_team":      f["teams"]["away"]["name"],
                "date":           date_str,
                "home_score":     int(home_score),
                "away_score":     int(away_score),
                "ht_home_score":  ht.get("home", ""),
                "ht_away_score":  ht.get("away", ""),
                "league_id":      league_id,
                "league_name":    league_name,
                "country":        country,
                "league_tier":    2,   # default; top leagues already in training data
                "season":         season,
                "home_team_id":   f["teams"]["home"]["id"],
                "away_team_id":   f["teams"]["away"]["id"],
            }
        except Exception as e:
            logger.debug(f"    Fixture parse error: {e}")
            return None

    # ── CSV management ────────────────────────────────────────────────────────

    def _merge_into_csv(self, new_rows: List[dict]) -> int:
        """
        Append new_rows to MATCHES_CSV, skipping rows that already exist
        (matched on date + home_team + away_team).

        Returns the count of genuinely new rows added.
        """
        # Load existing CSV
        existing_keys: Set[Tuple] = set()
        existing_rows: List[dict] = []

        if MATCHES_CSV.exists():
            try:
                df_existing = pd.read_csv(MATCHES_CSV, low_memory=False)
                for _, row in df_existing.iterrows():
                    key = (str(row.get("date", ""))[:10],
                           str(row.get("home_team", "")),
                           str(row.get("away_team", "")))
                    existing_keys.add(key)
                existing_rows = df_existing.to_dict("records")
                logger.info(f"    Existing CSV: {len(existing_rows):,} rows")
            except Exception as e:
                logger.warning(f"    Could not read existing CSV: {e}")

        added = 0
        for row in new_rows:
            key = (str(row.get("date", ""))[:10],
                   str(row.get("home_team", "")),
                   str(row.get("away_team", "")))
            if key not in existing_keys:
                existing_rows.append(row)
                existing_keys.add(key)
                added += 1

        if added == 0:
            return 0

        # Write back sorted by date
        df_out = pd.DataFrame(existing_rows)
        # Ensure all required columns present
        for col in CSV_COLUMNS:
            if col not in df_out.columns:
                df_out[col] = ""

        df_out = df_out.sort_values("date").reset_index(drop=True)
        df_out[CSV_COLUMNS].to_csv(MATCHES_CSV, index=False, encoding="utf-8")
        return added

    # ── model training ────────────────────────────────────────────────────────

    def _retrain_models(self) -> Tuple[bool, str]:
        """
        Retrain the full model set using the current matches.csv.
        Mirrors the logic in scripts/retrain_models.py as a direct call
        (no subprocess) so it works in the same process.
        """
        try:
            import joblib
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import accuracy_score

            # Load data
            if not MATCHES_CSV.exists():
                return False, f"{MATCHES_CSV} not found"

            df = pd.read_csv(MATCHES_CSV, low_memory=False)
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df = df.dropna(subset=["date", "home_team", "away_team",
                                   "home_score", "away_score"])
            df["home_score"] = df["home_score"].astype(int)
            df["away_score"] = df["away_score"].astype(int)
            df = df.sort_values("date").reset_index(drop=True)

            logger.info(f"    Training on {len(df):,} matches across "
                        f"{df.get('league_name', pd.Series()).nunique()} leagues")

            if len(df) < MIN_GAMES_TO_TRAIN:
                return False, f"Only {len(df)} matches — need ≥ {MIN_GAMES_TO_TRAIN}"

            # Build feature matrix
            X, y_result, y_over15, y_over25, y_btts = [], [], [], [], []
            skipped = 0

            for idx, row in df.iterrows():
                home = row["home_team"]
                away = row["away_team"]
                date = row["date"]
                tier = int(row.get("league_tier", 1))
                df_past = df[df["date"] < date]

                home_count = ((df_past["home_team"] == home) |
                              (df_past["away_team"] == home)).sum()
                away_count = ((df_past["home_team"] == away) |
                              (df_past["away_team"] == away)).sum()
                if home_count < 5 or away_count < 5:
                    skipped += 1
                    continue

                feats = compute_features(home, away, date, df_past, tier)
                X.append(feats)
                hg, ag = row["home_score"], row["away_score"]
                y_result.append(2 if hg > ag else (1 if hg == ag else 0))
                y_over15.append(1 if hg + ag > 1 else 0)
                y_over25.append(1 if hg + ag > 2 else 0)
                y_btts.append(1 if hg > 0 and ag > 0 else 0)

            X = np.array(X)
            logger.info(f"    Built {len(X):,} training samples (skipped {skipped:,})")

            if len(X) < MIN_GAMES_TO_TRAIN:
                return False, f"Only {len(X)} usable samples after filtering"

            targets = {
                "match_result": np.array(y_result),
                "over_1_5":     np.array(y_over15),
                "over_2_5":     np.array(y_over25),
                "btts":         np.array(y_btts),
            }

            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            model_weights: Dict[str, Dict[str, float]] = {}
            trained_count = 0

            for target_name, y in targets.items():
                logger.info(f"    Training target: {target_name}")
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=42, stratify=y
                )
                model_weights[target_name] = {}

                # XGBoost
                m = self._train_xgboost(X_tr, y_tr, target_name)
                if m:
                    joblib.dump(m, MODELS_DIR / f"{target_name}_xgboost.joblib")
                    acc = accuracy_score(y_te, m.predict(X_te))
                    model_weights[target_name]["xgboost"] = float(acc)
                    trained_count += 1
                    logger.info(f"      xgboost {target_name}: {acc:.3f}")

                # LightGBM
                m = self._train_lightgbm(X_tr, y_tr, target_name)
                if m:
                    joblib.dump(m, MODELS_DIR / f"{target_name}_lightgbm.joblib")
                    acc = accuracy_score(y_te, m.predict(X_te))
                    model_weights[target_name]["lightgbm"] = float(acc)
                    trained_count += 1
                    logger.info(f"      lightgbm {target_name}: {acc:.3f}")

                # Random Forest
                m = self._train_random_forest(X_tr, y_tr, target_name)
                if m:
                    joblib.dump(m, MODELS_DIR / f"{target_name}_random_forest.joblib")
                    acc = accuracy_score(y_te, m.predict(X_te))
                    model_weights[target_name]["random_forest"] = float(acc)
                    trained_count += 1
                    logger.info(f"      random_forest {target_name}: {acc:.3f}")

                # Neural Network (MLP)
                result = self._train_neural_network(X_tr, y_tr, target_name)
                if result:
                    model_obj, scaler = result
                    joblib.dump(result, MODELS_DIR / f"{target_name}_neural_network.joblib")
                    X_te_sc = scaler.transform(X_te)
                    acc = accuracy_score(y_te, model_obj.predict(X_te_sc))
                    model_weights[target_name]["neural_network"] = float(acc)
                    trained_count += 1
                    logger.info(f"      neural_network {target_name}: {acc:.3f}")

            # Save metadata
            meta = {
                "trained_at": datetime.now().isoformat(),
                "training_rows": len(X),
                "skipped_rows": skipped,
                "targets": list(targets.keys()),
                "feature_columns": [
                    "home_win_rate_5", "home_win_rate_10", "home_draw_rate_5",
                    "home_goals_scored_5", "home_goals_conceded_5",
                    "home_home_win_rate_5", "home_home_goals_5",
                    "away_win_rate_5", "away_win_rate_10", "away_draw_rate_5",
                    "away_goals_scored_5", "away_goals_conceded_5",
                    "away_away_win_rate_5", "away_away_goals_5",
                    "h2h_home_win_rate", "h2h_avg_goals", "h2h_btts_rate",
                    "h2h_meetings", "league_tier",
                ],
                "models_trained": trained_count,
            }
            with open(MODELS_DIR / "meta.json", "w") as f:
                json.dump(meta, f, indent=2)
            with open(MODELS_DIR / "model_weights.json", "w") as f:
                json.dump(model_weights, f, indent=2)

            logger.info(f"    Trained {trained_count} models, saved to {MODELS_DIR}")
            return True, f"Trained {trained_count} models"

        except Exception as e:
            logger.exception(f"    Training error: {e}")
            return False, str(e)

    # ── individual model trainers ─────────────────────────────────────────────

    @staticmethod
    def _train_xgboost(X_tr, y_tr, label: str):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            return None
        n = len(np.unique(y_tr))
        kw = dict(num_class=n) if n > 2 else {}
        m = XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multi:softprob" if n > 2 else "binary:logistic",
            eval_metric="mlogloss" if n > 2 else "logloss",
            use_label_encoder=False, random_state=42, n_jobs=-1, **kw,
        )
        m.fit(X_tr, y_tr)
        return m

    @staticmethod
    def _train_lightgbm(X_tr, y_tr, label: str):
        try:
            import lightgbm as lgb
        except ImportError:
            return None
        n = len(np.unique(y_tr))
        kw = dict(num_class=n) if n > 2 else {}
        m = lgb.LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            objective="multiclass" if n > 2 else "binary",
            random_state=42, n_jobs=-1, verbose=-1, **kw,
        )
        m.fit(X_tr, y_tr)
        return m

    @staticmethod
    def _train_random_forest(X_tr, y_tr, label: str):
        from sklearn.ensemble import RandomForestClassifier
        m = RandomForestClassifier(
            n_estimators=400, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1,
        )
        m.fit(X_tr, y_tr)
        return m

    @staticmethod
    def _train_neural_network(X_tr, y_tr, label: str):
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_sc = scaler.fit_transform(X_tr)
        m = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), activation="relu",
            solver="adam", learning_rate="adaptive",
            max_iter=500, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20,
            random_state=42, verbose=False,
        )
        m.fit(X_sc, y_tr)
        return (m, scaler)

    # ── hot-reload ─────────────────────────────────────────────────────────────

    def _hot_reload_models(self) -> None:
        """
        Reload the API-Football models inside the already-running
        advanced_prediction_service singleton so next predictions use
        the freshly trained weights — no process restart needed.
        """
        try:
            from services.advanced_prediction_service import advanced_prediction_service
            advanced_prediction_service._load_api_football_models()
            logger.info("    Hot-reload: API-Football models reloaded in prediction service")
        except Exception as e:
            logger.warning(f"    Hot-reload failed (non-critical): {e}")

    # ── league log persistence ─────────────────────────────────────────────────

    @staticmethod
    def _load_league_log() -> dict:
        if LEAGUE_LOG.exists():
            try:
                with open(LEAGUE_LOG, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    @staticmethod
    def _save_league_log(log: dict) -> None:
        LEAGUE_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(LEAGUE_LOG, "w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)

    # ── status / info ──────────────────────────────────────────────────────────

    def get_status(self) -> Dict[str, Any]:
        """Return a summary of what leagues have been fetched and when."""
        log = self._load_league_log()
        csv_rows = 0
        if MATCHES_CSV.exists():
            try:
                df = pd.read_csv(MATCHES_CSV, usecols=["league_name"], low_memory=False)
                csv_rows = len(df)
            except Exception:
                pass
        return {
            "leagues_tracked":    len(log),
            "training_csv_rows":  csv_rows,
            "training_csv_path":  str(MATCHES_CSV),
            "models_dir":         str(MODELS_DIR),
            "leagues": [
                {
                    "id":           int(lid),
                    "name":         entry["name"],
                    "country":      entry.get("country", ""),
                    "last_fetched": entry["last_fetched"],
                    "rows_fetched": entry.get("rows", 0),
                }
                for lid, entry in sorted(log.items(), key=lambda x: x[1]["last_fetched"], reverse=True)
            ],
        }


# ── module-level singleton ────────────────────────────────────────────────────
league_adaptive_trainer = LeagueAdaptiveTrainer()


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    parser = argparse.ArgumentParser(description="League-adaptive model trainer")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to process (YYYY-MM-DD, default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be fetched without actually doing it")
    parser.add_argument("--status", action="store_true",
                        help="Print current league tracking status and exit")
    args = parser.parse_args()

    trainer = LeagueAdaptiveTrainer()

    if args.status:
        status = trainer.get_status()
        print(f"\nLeague Adaptive Trainer Status")
        print(f"  Training CSV : {status['training_csv_path']} ({status['training_csv_rows']:,} rows)")
        print(f"  Models dir   : {status['models_dir']}")
        print(f"  Leagues tracked: {status['leagues_tracked']}")
        print()
        if status["leagues"]:
            print(f"  {'ID':<8} {'NAME':<35} {'COUNTRY':<18} {'LAST FETCHED':<22} {'ROWS'}")
            print("  " + "-" * 95)
            for lg in status["leagues"]:
                print(f"  {lg['id']:<8} {lg['name'][:34]:<35} {lg['country'][:17]:<18} "
                      f"{lg['last_fetched'][:19]:<22} {lg['rows_fetched']}")
        sys.exit(0)

    print(f"\nLeague Adaptive Trainer — {args.date}")
    print(f"Dry run: {args.dry_run}\n")

    report = trainer.run_for_date(args.date, dry_run=args.dry_run)

    print(f"Leagues found today : {len(report['leagues_found'])}")
    print(f"Leagues needing fetch: {len(report['leagues_new'])}")
    print(f"Leagues fetched     : {len(report['leagues_fetched'])}")
    print(f"Rows added to CSV   : {report['rows_added']}")
    print(f"Models retrained    : {report['models_retrained']}")

    if report["leagues_fetched"]:
        print("\nFetched:")
        for lg in report["leagues_fetched"]:
            print(f"  [{lg['id']}] {lg['name']} — {lg['rows_fetched']} rows")

    if report["errors"]:
        print("\nErrors:")
        for e in report["errors"]:
            print(f"  ! {e}")
