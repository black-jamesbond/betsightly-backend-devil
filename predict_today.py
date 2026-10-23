"""
Predict Today's Games - Using apifootball.com API + GitHub historical data (228K matches)

Generates predictions for all upcoming matches today using:
1. Historical win rates, goal stats, and form data from GitHub dataset
2. ELO ratings where available
3. Home advantage calibration
4. H2H records

Usage:
    python predict_today.py [--date YYYY-MM-DD] [--league FILTER] [--save FILE]
"""
from __future__ import annotations

import json
import os
import sys
import math
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np


# -- Config --------------------------------------------------------------
# Read from the environment (put it in .env, which is gitignored) so the key is
# never committed. Empty means the apifootball calls are skipped and the free
# LiveScore fixture source is used instead.
API_KEY = os.environ.get("APIFOOTBALL_API_KEY", "")
CACHE_DIR = Path("cache/apifootball")
GITHUB_DATA = Path("data/github-football/Matches.csv")
API_FOOTBALL_DATA = Path("data/api-football/matches.csv")
# Same-provider history as the live fixtures — exact team-name matches, 4000+
# leagues. This is the highest-coverage source (see load_historical).
APIFOOTBALL_HISTORY = Path("data/apifootball/history.csv")


# Kick-off times are stored in UTC — canonical, and what the feeds send — so
# anything printed for a human is shifted to local time. The dashboard does the
# same at its own display layer; only the stored values stay UTC.
try:
    from fixtures_livescore import to_local_hhmm as _local_hhmm
except Exception:            # module missing: fall back to showing raw UTC
    def _local_hhmm(value, *_args, **_kwargs):
        return value


def _local(value):
    """UTC 'HH:MM' as the reader's local time, for display only."""
    return _local_hhmm(value)



# -- Fetch fixtures -----------------------------------------------------
def fetch_fixtures(date_str: str) -> List[Dict]:
    """Fetch fixtures from cache or API."""
    cache_file = CACHE_DIR / f"events_{date_str}.json"

    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"  Loaded {len(data)} matches from cache")
        return data

    # Fetch from API
    import urllib.request
    import time
    url = (
        f"https://apiv3.apifootball.com/?action=get_events"
        f"&from={date_str}&to={date_str}&APIkey={API_KEY}"
    )
    print(f"  Fetching from apifootball.com ...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    raw = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                raw = resp.read().decode()
            json.loads(raw)  # ensure the payload is complete/parseable
            break
        except Exception as e:
            wait = 2 * (attempt + 1)
            print(f"  Fetch attempt {attempt + 1} failed ({e}); retrying in {wait}s...")
            time.sleep(wait)
            raw = None
    if raw is None:
        print("  Fetch failed after retries.")
        return _livescore_fallback(date_str)
    data = json.loads(raw)

    if isinstance(data, dict) and "error" in data:
        print(f"  API Error: {data}")
        print("  Primary feed unavailable - using the free LiveScore source.")
        return _livescore_fallback(date_str)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(raw)
    print(f"  Fetched {len(data)} matches, cached to {cache_file}")
    return data


def _livescore_fallback(date_str: str) -> List[Dict]:
    """Free fallback fixture source (see fixtures_livescore.py).

    Used only when apiv3.apifootball.com returns nothing. Results are written in
    the apifootball schema, including to the shared events_<date>.json cache that
    the dashboard reads, so every consumer stays consistent.
    """
    try:
        from fixtures_livescore import fetch_day, write_events_cache

        matches = fetch_day(date_str)
        if not matches:
            print("  Fallback source returned no fixtures either.")
            return []
        dest = write_events_cache(date_str, matches)
        print(f"  Fallback: {len(matches)} matches from LiveScore, cached to {dest}")
        return matches
    except Exception as exc:
        print(f"  Fallback source failed: {exc}")
        return []


# -- Load historical data -----------------------------------------------
def load_historical() -> Optional[pd.DataFrame]:
    """Load and merge all available historical data."""
    frames = []

    # GitHub dataset (228K matches)
    if GITHUB_DATA.exists():
        df = pd.read_csv(GITHUB_DATA, low_memory=False)
        df = df.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "MatchDate": "date", "FTHome": "home_score", "FTAway": "away_score",
            "Division": "league", "HomeElo": "home_elo", "AwayElo": "away_elo",
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
        df["home_score"] = df["home_score"].astype(int)
        df["away_score"] = df["away_score"].astype(int)
        df["source"] = "github"
        frames.append(df)
        print(f"  GitHub data: {len(df):,} matches")

    # API-Football dataset (16K matches)
    if API_FOOTBALL_DATA.exists():
        df2 = pd.read_csv(API_FOOTBALL_DATA, low_memory=False)
        for col in ["home_team", "away_team", "date", "home_score", "away_score"]:
            if col not in df2.columns:
                break
        else:
            df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
            df2["home_score"] = pd.to_numeric(df2["home_score"], errors="coerce")
            df2["away_score"] = pd.to_numeric(df2["away_score"], errors="coerce")
            df2 = df2.dropna(subset=["date", "home_score", "away_score"])
            df2["source"] = "api-football"
            # Give the Dixon-Coles model a league key from this (fresher) source.
            # Namespaced by country so same-named leagues don't collide, and
            # prefixed "af:" so it never clashes with the GitHub division codes.
            if "league_name" in df2.columns:
                ctry = df2.get("country", "").astype(str) if "country" in df2.columns else ""
                df2["league"] = ("af:" + (ctry + ":" if "country" in df2.columns else "")
                                 + df2["league_name"].astype(str))
            frames.append(df2)
            print(f"  API-Football data: {len(df2):,} matches")

    # apifootball.com history — SAME provider as the live fixtures, so team names
    # match exactly and coverage spans 4000+ leagues (incl. lower/youth/obscure).
    if APIFOOTBALL_HISTORY.exists():
        df3 = pd.read_csv(APIFOOTBALL_HISTORY, low_memory=False)
        df3 = df3.rename(columns={"home_team": "home_team", "away_team": "away_team"})
        df3["date"] = pd.to_datetime(df3["date"], errors="coerce")
        df3["home_score"] = pd.to_numeric(df3["home_score"], errors="coerce")
        df3["away_score"] = pd.to_numeric(df3["away_score"], errors="coerce")
        df3 = df3.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
        df3["home_score"] = df3["home_score"].astype(int)
        df3["away_score"] = df3["away_score"].astype(int)
        df3["source"] = "apifootball"
        # League key matching how live fixtures are keyed (country:league_name),
        # so the Dixon-Coles model can cover these leagues too.
        if "league_name" in df3.columns:
            ctry = df3.get("country", "").astype(str) if "country" in df3.columns else ""
            df3["league"] = ("apif:" + (ctry + ":" if "country" in df3.columns else "")
                             + df3["league_name"].astype(str))
        frames.append(df3)
        print(f"  apifootball history: {len(df3):,} matches")

    if not frames:
        print("  WARNING: No historical data found!")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)

    # Pre-compute normalized team names ONCE (huge speedup). Uses the SAME
    # normalize_name as query time so keys can never drift out of sync.
    print("  Pre-computing normalized team names...")
    combined["_home_norm"] = combined["home_team"].apply(normalize_name)
    combined["_away_norm"] = combined["away_team"].apply(normalize_name)
    print(f"  Total historical: {len(combined):,} matches")
    return combined


# -- Team name fuzzy matching -------------------------------------------
# Generic club-type tokens carry no identity — drop them so "Cambridge Utd"
# and "Cambridge United FC" collapse to the same key.
_DROP_TOKENS = {
    "fc", "sc", "cf", "ac", "fk", "sk", "if", "bk", "sv", "ss", "us", "rc",
    "afc", "ca", "ec", "sd", "ad", "nk", "ks", "kf", "cd", "club", "calcio",
    "futbol", "football", "fussball", "the", "de", "cska",
}
# Unambiguous word-abbreviation expansions -> a single canonical spelling.
_EXPAND = {
    "utd": "united", "atl": "atletico", "ath": "athletic", "dep": "deportivo",
    "spt": "sporting", "din": "dinamo", "dyn": "dynamo", "st": "saint",
    "wdr": "wanderers", "cty": "city", "utdd": "united",
}


def normalize_name(name: str) -> str:
    """Normalize a team name for cross-source matching: HTML-unescape, strip
    accents, expand common abbreviations, and drop generic club words."""
    import re, html, unicodedata
    n = html.unescape(str(name))                       # &apos; -> '
    n = unicodedata.normalize("NFKD", n).encode("ascii", "ignore").decode()  # accents
    n = n.lower()
    n = re.sub(r"[^a-z0-9 ]", " ", n)                  # punctuation -> space
    tokens = n.split()
    out = []
    for t in tokens:
        t = _EXPAND.get(t, t)
        if t in _DROP_TOKENS:
            continue
        out.append(t)
    if not out:                                         # everything got dropped
        out = tokens
    return " ".join(out).strip()


def find_team_stats(team_name: str, hist: pd.DataFrame, last_n: int = 30) -> Dict[str, float]:
    """Find historical stats for a team using pre-computed normalized names.
    Uses recency-weighted stats (recent matches count more)."""
    norm = normalize_name(team_name)

    # Try exact match first (fast)
    home_mask = hist["home_team"].str.strip() == team_name
    away_mask = hist["away_team"].str.strip() == team_name

    if home_mask.sum() + away_mask.sum() == 0:
        home_mask = hist["_home_norm"] == norm
        away_mask = hist["_away_norm"] == norm

    if home_mask.sum() + away_mask.sum() == 0:
        home_mask = hist["_home_norm"].str.contains(norm, na=False, regex=False)
        away_mask = hist["_away_norm"].str.contains(norm, na=False, regex=False)

    total = home_mask.sum() + away_mask.sum()
    if total == 0:
        return {"found": False, "matches": 0}

    home_games = hist[home_mask].tail(last_n)
    away_games = hist[away_mask].tail(last_n)

    # Recency weights: most recent match = 1.0, oldest = 0.3
    def _weights(n):
        if n <= 1:
            return np.ones(n)
        return np.linspace(0.3, 1.0, n)

    hw = _weights(len(home_games))
    aw = _weights(len(away_games))

    # Weighted win/draw/loss counts
    h_win_mask = (home_games["home_score"].values > home_games["away_score"].values).astype(float)
    h_draw_mask = (home_games["home_score"].values == home_games["away_score"].values).astype(float)
    h_loss_mask = (home_games["home_score"].values < home_games["away_score"].values).astype(float)

    a_win_mask = (away_games["away_score"].values > away_games["home_score"].values).astype(float)
    a_draw_mask = (away_games["home_score"].values == away_games["away_score"].values).astype(float)
    a_loss_mask = (away_games["away_score"].values < away_games["home_score"].values).astype(float)

    w_home_wins = (h_win_mask * hw).sum()
    w_home_draws = (h_draw_mask * hw).sum()
    w_home_losses = (h_loss_mask * hw).sum()
    w_away_wins = (a_win_mask * aw).sum()
    w_away_draws = (a_draw_mask * aw).sum()
    w_away_losses = (a_loss_mask * aw).sum()

    w_total_home = hw.sum()
    w_total_away = aw.sum()
    w_total = w_total_home + w_total_away

    total_w_wins = w_home_wins + w_away_wins
    total_w_draws = w_home_draws + w_away_draws
    total_w_losses = w_home_losses + w_away_losses

    # Goal stats (unweighted is fine here)
    total_games = len(home_games) + len(away_games)
    gs_h = home_games["home_score"].sum() if len(home_games) else 0
    gc_h = home_games["away_score"].sum() if len(home_games) else 0
    gs_a = away_games["away_score"].sum() if len(away_games) else 0
    gc_a = away_games["home_score"].sum() if len(away_games) else 0
    avg_scored = (gs_h + gs_a) / total_games if total_games else 1.2
    avg_conceded = (gc_h + gc_a) / total_games if total_games else 1.2

    # Home/away specific win rates (weighted)
    home_win_rate = w_home_wins / w_total_home if w_total_home > 0 else 0.33
    away_win_rate = w_away_wins / w_total_away if w_total_away > 0 else 0.33

    # ELO
    elo = None
    if "home_elo" in hist.columns:
        elo_vals = list(home_games["home_elo"].dropna()) + list(away_games["away_elo"].dropna())
        if elo_vals:
            elo = float(np.mean(elo_vals[-5:])) if len(elo_vals) >= 5 else float(np.mean(elo_vals))

    # BTTS and O2.5 rates
    h_btts = ((home_games["home_score"] > 0) & (home_games["away_score"] > 0)).sum()
    a_btts = ((away_games["home_score"] > 0) & (away_games["away_score"] > 0)).sum()
    btts_rate = (h_btts + a_btts) / total_games if total_games else 0.5

    h_o25 = ((home_games["home_score"] + home_games["away_score"]) > 2.5).sum()
    a_o25 = ((away_games["home_score"] + away_games["away_score"]) > 2.5).sum()
    over25_rate = (h_o25 + a_o25) / total_games if total_games else 0.45

    # Recent form (last 5 matches only) - W/D/L
    recent_home = home_games.tail(5)
    recent_away = away_games.tail(5)
    recent_n = len(recent_home) + len(recent_away)
    if recent_n > 0:
        rw = ((recent_home["home_score"] > recent_home["away_score"]).sum() +
              (recent_away["away_score"] > recent_away["home_score"]).sum())
        rd = ((recent_home["home_score"] == recent_home["away_score"]).sum() +
              (recent_away["home_score"] == recent_away["away_score"]).sum())
        recent_form = (rw * 3 + rd) / (recent_n * 3)
    else:
        recent_form = 0.33

    return {
        "found": True,
        "matches": total,
        "win_rate": total_w_wins / w_total if w_total > 0 else 0.33,
        "draw_rate": total_w_draws / w_total if w_total > 0 else 0.33,
        "loss_rate": total_w_losses / w_total if w_total > 0 else 0.33,
        "home_win_rate": home_win_rate,
        "away_win_rate": away_win_rate,
        "avg_scored": avg_scored,
        "avg_conceded": avg_conceded,
        "elo": elo,
        "btts_rate": btts_rate,
        "over25_rate": over25_rate,
        "recent_form": recent_form,
    }


# -- H2H lookup ---------------------------------------------------------
def get_h2h(home: str, away: str, hist: pd.DataFrame) -> Dict[str, float]:
    """Get head-to-head record using pre-computed normalized names."""
    hn = normalize_name(home)
    an = normalize_name(away)

    mask = ((hist["_home_norm"] == hn) & (hist["_away_norm"] == an)) | \
           ((hist["_home_norm"] == an) & (hist["_away_norm"] == hn))
    h2h = hist[mask]

    if len(h2h) == 0:
        return {"meetings": 0}

    # Home team perspective
    home_as_home = h2h[(h2h["_home_norm"] == hn) & (h2h["_away_norm"] == an)]
    home_as_away = h2h[(h2h["_home_norm"] == an) & (h2h["_away_norm"] == hn)]

    home_wins = (
        (home_as_home["home_score"] > home_as_home["away_score"]).sum()
        + (home_as_away["away_score"] > home_as_away["home_score"]).sum()
    )
    draws = (
        (home_as_home["home_score"] == home_as_home["away_score"]).sum()
        + (home_as_away["home_score"] == home_as_away["away_score"]).sum()
    )
    total_goals = (
        h2h["home_score"].sum() + h2h["away_score"].sum()
    )

    return {
        "meetings": len(h2h),
        "home_win_rate": home_wins / len(h2h),
        "draw_rate": draws / len(h2h),
        "avg_goals": total_goals / len(h2h),
    }


# -- Global base rates (calibrated from evaluation data) ----------------
# Real-world outcome frequencies across all leagues
BASE_HOME = 0.36
BASE_DRAW = 0.28
BASE_AWAY = 0.36
BASE_O25 = 0.47
BASE_BTTS = 0.47
HOME_ADVANTAGE = 0.03  # reduced from 0.08 — actual advantage is small in lower leagues


# -- Calibration + hybrid layer -----------------------------------------
# Calibration maps raw probabilities onto honest ones ("70%" wins ~70%); the
# hybrid predictor additionally swaps in Dixon-Coles for 1X2 where it covers a
# match (it beat the current engine on match-result prediction in backtest).
# Loaded once; both degrade gracefully to no-ops if their files are missing.
try:
    from calibration import Calibrator as _Calibrator
    _CALIBRATOR = _Calibrator.load()
    if not _CALIBRATOR.is_fitted():
        _CALIBRATOR = None
except Exception:
    _CALIBRATOR = None

try:
    from hybrid import HybridPredictor as _HybridPredictor
    _HYBRID = _HybridPredictor()
    if not _HYBRID.ready:
        _HYBRID = None
except Exception:
    _HYBRID = None


# -- Prediction engine --------------------------------------------------
def predict_match(
    home_name: str,
    away_name: str,
    home_stats: Dict,
    away_stats: Dict,
    h2h: Dict,
) -> Dict[str, Any]:
    """Generate prediction for a single match.

    Key fixes over v1:
    - Draw probability anchored to realistic base rate (~28%)
    - Home advantage reduced from 8% to 3%
    - Recency-weighted form blended with base rates (Bayesian shrinkage)
    - Strength differential drives predictions, not raw win rates
    - Over 2.5 / BTTS calibrated down to match actual ~47% rates
    - Confidence tied to sample size AND prediction margin
    """

    both_found = home_stats["found"] and away_stats["found"]
    either_found = home_stats["found"] or away_stats["found"]

    if both_found:
        # Shrinkage: blend team rates with base rates proportional to sample size
        # Small samples lean heavily on base rates; large samples trust the data
        h_n = min(home_stats["matches"], 50)
        a_n = min(away_stats["matches"], 50)
        h_shrink = h_n / (h_n + 15)  # at 15 matches, 50/50 blend; at 50, 77% data
        a_shrink = a_n / (a_n + 15)

        # Home team's attacking strength (relative to average)
        h_attack = h_shrink * home_stats["avg_scored"] + (1 - h_shrink) * 1.2
        h_defend = h_shrink * home_stats["avg_conceded"] + (1 - h_shrink) * 1.2
        a_attack = a_shrink * away_stats["avg_scored"] + (1 - a_shrink) * 1.2
        a_defend = a_shrink * away_stats["avg_conceded"] + (1 - a_shrink) * 1.2

        # Expected goals
        expected_home_goals = (h_attack * a_defend) / 1.2  # normalized to league avg
        expected_away_goals = (a_attack * h_defend) / 1.2
        expected_home_goals = max(0.3, min(3.5, expected_home_goals))
        expected_away_goals = max(0.3, min(3.5, expected_away_goals))

        # Win probability from strength differential
        strength_diff = (home_stats["win_rate"] - home_stats["loss_rate"]) - \
                        (away_stats["win_rate"] - away_stats["loss_rate"])

        # Blend recent form (30%) with overall record (70%)
        h_form = home_stats.get("recent_form", 0.33)
        a_form = away_stats.get("recent_form", 0.33)
        form_diff = h_form - a_form

        # Convert to probabilities using logistic function
        combined_diff = 0.6 * strength_diff + 0.4 * form_diff
        home_p = BASE_HOME + combined_diff * 0.3
        away_p = BASE_AWAY - combined_diff * 0.3

        # Draw probability: higher when teams are close in strength
        closeness = 1.0 - min(1.0, abs(combined_diff))
        # Also factor in both teams' draw rates
        team_draw_avg = (home_stats["draw_rate"] + away_stats["draw_rate"]) / 2
        draw_p = 0.5 * (BASE_DRAW * (0.6 + 0.8 * closeness)) + 0.5 * team_draw_avg

        # ELO adjustment
        if home_stats.get("elo") and away_stats.get("elo"):
            elo_diff = home_stats["elo"] - away_stats["elo"]
            elo_home_exp = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))
            home_p = 0.7 * home_p + 0.3 * elo_home_exp
            away_p = 0.7 * away_p + 0.3 * (1 - elo_home_exp)

        # Confidence based on sample size and prediction clarity
        min_matches = min(h_n, a_n)
        sample_conf = min(0.7, 0.2 + 0.01 * min_matches)
        margin = abs(home_p - away_p)
        margin_conf = min(0.3, margin * 0.5)
        confidence = sample_conf + margin_conf

    elif home_stats["found"]:
        h_n = min(home_stats["matches"], 50)
        h_shrink = h_n / (h_n + 15)
        home_p = h_shrink * home_stats["home_win_rate"] + (1 - h_shrink) * BASE_HOME
        draw_p = h_shrink * home_stats["draw_rate"] + (1 - h_shrink) * BASE_DRAW
        away_p = 1 - home_p - draw_p
        expected_home_goals = h_shrink * home_stats["avg_scored"] + (1 - h_shrink) * 1.2
        expected_away_goals = h_shrink * home_stats["avg_conceded"] + (1 - h_shrink) * 1.2
        confidence = 0.25

    elif away_stats["found"]:
        a_n = min(away_stats["matches"], 50)
        a_shrink = a_n / (a_n + 15)
        away_p = a_shrink * away_stats["away_win_rate"] + (1 - a_shrink) * BASE_AWAY
        draw_p = a_shrink * away_stats["draw_rate"] + (1 - a_shrink) * BASE_DRAW
        home_p = 1 - away_p - draw_p
        expected_home_goals = a_shrink * away_stats["avg_conceded"] + (1 - a_shrink) * 1.2
        expected_away_goals = a_shrink * away_stats["avg_scored"] + (1 - a_shrink) * 1.2
        confidence = 0.25

    else:
        home_p, draw_p, away_p = BASE_HOME, BASE_DRAW, BASE_AWAY
        expected_home_goals = 1.2
        expected_away_goals = 1.0
        confidence = 0.10

    # --- H2H adjustment (small weight) ---
    if h2h.get("meetings", 0) >= 3:
        h2h_w = min(0.15, 0.03 * h2h["meetings"])
        home_p = home_p * (1 - h2h_w) + h2h["home_win_rate"] * h2h_w
        draw_p = draw_p * (1 - h2h_w) + h2h["draw_rate"] * h2h_w
        away_p = 1 - home_p - draw_p
        if h2h["meetings"] >= 5:
            confidence = min(0.95, confidence + 0.03)

    # --- Home advantage (small) ---
    home_p += HOME_ADVANTAGE
    away_p -= HOME_ADVANTAGE * 0.6
    draw_p -= HOME_ADVANTAGE * 0.4

    # --- Normalize to valid probability distribution ---
    home_p = max(0.08, home_p)
    draw_p = max(0.12, draw_p)  # draw floor is higher — draws are common
    away_p = max(0.08, away_p)
    total = home_p + draw_p + away_p
    home_p /= total
    draw_p /= total
    away_p /= total

    # --- Over/Under (Poisson model, calibrated) ---
    expected_total = expected_home_goals + expected_away_goals
    # Scale down slightly — raw Poisson over-predicts goals
    expected_total *= 0.92

    over15_p = 1 - math.exp(-expected_total) * (1 + expected_total)
    over25_p = 1 - math.exp(-expected_total) * (1 + expected_total + expected_total ** 2 / 2)
    over35_p = 1 - math.exp(-expected_total) * (
        1 + expected_total + expected_total ** 2 / 2 + expected_total ** 3 / 6
    )

    # Blend with historical rates and anchor toward base rate
    if both_found:
        hist_o25 = (home_stats["over25_rate"] + away_stats["over25_rate"]) / 2
        over25_p = 0.4 * over25_p + 0.3 * hist_o25 + 0.3 * BASE_O25
    else:
        over25_p = 0.5 * over25_p + 0.5 * BASE_O25

    over25_p = max(0.1, min(0.85, over25_p))

    # --- BTTS ---
    if both_found:
        btts_hist = (home_stats["btts_rate"] + away_stats["btts_rate"]) / 2
        # Scoring likelihood: both teams need to score
        score_factor = min(1.0, (home_stats["avg_scored"] * away_stats["avg_scored"]) / (1.2 * 1.2))
        btts_p = 0.45 * btts_hist + 0.30 * score_factor + 0.25 * BASE_BTTS
    else:
        btts_p = BASE_BTTS

    btts_p = max(0.15, min(0.80, btts_p))

    # --- Pick best bet (draw-aware; shared with the calibration layer) ---
    try:
        from calibration import pick_1x2
        best_result, _ = pick_1x2(home_p, draw_p, away_p)
    except Exception:
        result_map = {"Home Win": home_p, "Draw": draw_p, "Away Win": away_p}
        best_result = max(result_map, key=result_map.get)

    confidence = max(0.05, min(0.95, confidence))

    return {
        "home_win": round(home_p * 100, 1),
        "draw": round(draw_p * 100, 1),
        "away_win": round(away_p * 100, 1),
        "prediction": best_result,
        "over_1.5": round(over15_p * 100, 1),
        "over_2.5": round(over25_p * 100, 1),
        "over_3.5": round(over35_p * 100, 1),
        "btts": round(btts_p * 100, 1),
        "expected_home_goals": round(expected_home_goals, 2),
        "expected_away_goals": round(expected_away_goals, 2),
        "confidence": round(confidence * 100, 1),
        "data_quality": (
            "HIGH" if (both_found and min(home_stats["matches"], away_stats["matches"]) >= 10)
            else "MEDIUM" if either_found
            else "LOW"
        ),
        "h2h_meetings": h2h.get("meetings", 0),
    }


# -- Main ----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Predict today's games")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--league", default=None, help="Filter by league name (substring)")
    parser.add_argument("--save", default=None, help="Save predictions to JSON file")
    parser.add_argument("--fixture-file", default=None, help="Read already-imported fixtures from this JSON file")
    parser.add_argument("--no-cache", action="store_true", help="Only write --save output, for supervised atomic publication")
    parser.add_argument("--require-history", action="store_true", help="Fail instead of predicting from default priors")
    parser.add_argument("--top", type=int, default=None, help="Show only top N most confident")
    parser.add_argument("--raw", action="store_true",
                        help="Skip the calibration layer (show raw engine probabilities)")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"  BETSIGHTLY PREDICTIONS - {args.date}")
    print(f"{'='*70}")

    # 1. Load fixtures
    print("\n[1/3] Loading fixtures...")
    if args.fixture_file:
        with open(args.fixture_file, "r", encoding="utf-8") as f:
            fixtures = json.load(f)
        if not isinstance(fixtures, list):
            raise SystemExit("Fixture file must contain a JSON list")
    else:
        fixtures = fetch_fixtures(args.date)
    if not fixtures:
        print("No fixtures found!")
        if args.fixture_file:
            raise SystemExit(2)
        return

    # Remember SportyBet's own identifiers for today's board while the games are
    # still listed. Their feed drops a fixture the moment it kicks off, and the
    # booking code needs that id long afterwards — this is what lets a ticket
    # that has started still be re-booked. Best-effort: never fatal.
    try:
        from sportybet_betting import snapshot_ids
        ids = snapshot_ids()
        print(f"  SportyBet ids: {ids['board']} on the board, {ids['live']} live, "
              f"{ids['ledger']} remembered")
    except Exception as exc:
        print(f"  SportyBet ids unavailable ({type(exc).__name__}: {exc})")

    # Predict ALL of the day's matches — including games already in-play or
    # finished (early kickoffs). The engine only uses pre-match historical
    # data, so these are the same predictions it would have made before
    # kickoff, and the dashboard can score them against final results.
    upcoming = [m for m in fixtures if m.get("match_status") != "Cancelled"]

    # Optional league filter
    if args.league:
        filt = args.league.lower()
        upcoming = [m for m in upcoming if filt in m.get("league_name", "").lower()
                     or filt in m.get("country_name", "").lower()]
        print(f"  Filtered to {len(upcoming)} matches matching '{args.league}'")

    n_played = sum(1 for m in upcoming if m.get("match_hometeam_score", "") != "")
    print(f"  {len(upcoming)} matches to predict ({n_played} already started/finished)")

    # 2. Load historical data
    print("\n[2/3] Loading historical data...")
    hist = load_historical()
    if hist is None:
        if args.require_history:
            raise SystemExit("Historical data missing; refusing prior-only automated card")
        print("No historical data - predictions will use default priors only.")

    # 3. Generate predictions
    print("\n[3/3] Generating predictions...")
    predictions = []
    team_cache = {}

    # Live market odds (odds-api.io) — used as the primary 1X2 signal where
    # available. No-op unless ODDS_API_IO_KEY is set. Precedence: market > DC > current.
    odds_map = {}
    if not args.raw:
        try:
            import odds_api_io
            if odds_api_io.enabled():
                odds_map = odds_api_io.build_odds_map(args.date)
                print(f"  Loaded live market odds for {len(odds_map)} matches")
        except Exception as e:
            print(f"  Market odds unavailable ({e}) — using model only")

    for i, match in enumerate(upcoming):
        home = match["match_hometeam_name"]
        away = match["match_awayteam_name"]
        league = match.get("league_name", "?")
        country = match.get("country_name", "?")
        kick_off = match.get("match_time", "?")

        # Cache team stats
        if hist is not None:
            if home not in team_cache:
                team_cache[home] = find_team_stats(home, hist)
            if away not in team_cache:
                team_cache[away] = find_team_stats(away, hist)
            home_stats = team_cache[home]
            away_stats = team_cache[away]
            h2h = get_h2h(home, away, hist)
        else:
            home_stats = {"found": False, "matches": 0}
            away_stats = {"found": False, "matches": 0}
            h2h = {"meetings": 0}

        pred = predict_match(home, away, home_stats, away_stats, h2h)
        if not args.raw:
            if _HYBRID is not None:
                pred = _HYBRID.apply(pred, normalize_name(home), normalize_name(away))
            elif _CALIBRATOR is not None:
                pred = _CALIBRATOR.apply(pred)
            # Market odds override 1X2 when available (best available signal)
            if odds_map:
                mkey = (normalize_name(home), normalize_name(away))
                if mkey in odds_map:
                    import odds_api_io
                    pred = odds_api_io.apply_market(pred, odds_map[mkey])
        pred["home_team"] = home
        pred["away_team"] = away
        pred["league"] = league
        pred["country"] = country
        pred["kick_off"] = kick_off
        pred["match_id"] = match.get("match_id", "")
        pred["match_date"] = match.get("match_date", args.date)
        predictions.append(pred)

        if (i + 1) % 50 == 0:
            print(f"  ... {i+1}/{len(upcoming)} predicted")

    print(f"  Done! {len(predictions)} predictions generated.\n")

    # Sort by confidence
    predictions.sort(key=lambda x: -x["confidence"])

    if args.top:
        predictions = predictions[:args.top]

    # -- Display ---------------------------------------------------------
    # Group by league
    by_league = {}
    for p in predictions:
        key = f"{p['league']} ({p['country']})"
        by_league.setdefault(key, []).append(p)

    for league_name, preds in sorted(by_league.items()):
        safe_name = league_name.encode("ascii", "replace").decode()
        print(f"\n{'-'*70}")
        print(f"  {safe_name}")
        print(f"{'-'*70}")
        print(f"  {'Time':<6} {'Match':<40} {'Pred':<10} {'Conf':<6} {'O2.5':<6} {'BTTS':<6}")
        print(f"  {'-'*5} {'-'*39} {'-'*9} {'-'*5} {'-'*5} {'-'*5}")

        for p in sorted(preds, key=lambda x: x["kick_off"]):
            home_short = p["home_team"][:18]
            away_short = p["away_team"][:18]
            match_str = f"{home_short} v {away_short}"
            if len(match_str) > 39:
                match_str = match_str[:39]
            safe_match = match_str.encode("ascii", "replace").decode()

            pred_str = p["prediction"]
            if pred_str == "Home Win":
                pred_str = f"1 ({p['home_win']}%)"
            elif pred_str == "Away Win":
                pred_str = f"2 ({p['away_win']}%)"
            else:
                pred_str = f"X ({p['draw']}%)"

            o25 = f"{p['over_2.5']}%"
            btts = f"{p['btts']}%"
            conf = f"{p['confidence']}%"

            # Kick-offs are stored in UTC; show them in the reader's local time.
            print(f"  {_local(p['kick_off']):<6} {safe_match:<40} {pred_str:<10} {conf:<6} {o25:<6} {btts:<6}")

    # Summary stats
    high_conf = [p for p in predictions if p["confidence"] >= 60]
    med_conf = [p for p in predictions if 40 <= p["confidence"] < 60]
    low_conf = [p for p in predictions if p["confidence"] < 40]

    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  Total predictions: {len(predictions)}")
    print(f"  HIGH confidence (>=60%): {len(high_conf)}")
    print(f"  MEDIUM confidence (40-59%): {len(med_conf)}")
    print(f"  LOW confidence (<40%): {len(low_conf)}")
    print(f"  Data quality: HIGH={sum(1 for p in predictions if p['data_quality']=='HIGH')}"
          f"  MED={sum(1 for p in predictions if p['data_quality']=='MEDIUM')}"
          f"  LOW={sum(1 for p in predictions if p['data_quality']=='LOW')}")

    # Best bets (high confidence, clear favorite)
    best_bets = [p for p in predictions if p["confidence"] >= 50 and p["data_quality"] in ("HIGH", "MEDIUM")]
    if best_bets:
        print(f"\n  TOP PICKS (high confidence + good data):")
        for p in best_bets[:15]:
            home_s = p["home_team"][:20].encode("ascii","replace").decode()
            away_s = p["away_team"][:20].encode("ascii","replace").decode()
            print(f"    {_local(p['kick_off'])} {home_s} v {away_s} -> {p['prediction']} "
                  f"({p['confidence']}% conf, {p['data_quality']} data, H2H:{p['h2h_meetings']})")

    # Save
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved to {args.save}")

    # Scheduled publisher uses --no-cache to validate first and publish atomically.
    if not args.no_cache:
        pred_cache = CACHE_DIR / f"predictions_{args.date}.json"
        with open(pred_cache, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        print(f"  Predictions cached to {pred_cache}")


if __name__ == "__main__":
    main()
