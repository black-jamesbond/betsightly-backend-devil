"""
BetSightly 10-Bot Prediction System

10 automated bots, each with a unique strategy, running hour-by-hour
from 00:00 to midnight. Each bot picks ~2.0 odds bets and tries to
chain 10 successful predictions in a row, rolling winnings forward.

Usage:
    python bot_system.py [--date YYYY-MM-DD] [--stake 100] [--target-odds 2.0]
    python bot_system.py --simulate --days 3
"""
from __future__ import annotations

import json
import os
import sys
import math
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from collections import defaultdict

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Import prediction engine from predict_today.py
# ---------------------------------------------------------------------------
from predict_today import (
    fetch_fixtures,
    load_historical,
    find_team_stats,
    get_h2h,
    predict_match,
    normalize_name,
    BASE_HOME, BASE_DRAW, BASE_AWAY, BASE_O25, BASE_BTTS, HOME_ADVANTAGE,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TARGET_ODDS = 2.0
ODDS_RANGE = (1.70, 2.40)  # acceptable odds window around 2.0
CHAIN_TARGET = 10           # 10 wins in a row
DEFAULT_STAKE = 100         # starting bankroll per bot

RESULTS_DIR = Path("bot_results")


# ---------------------------------------------------------------------------
# Odds helpers
# ---------------------------------------------------------------------------
def prob_to_odds(prob: float) -> float:
    """Convert probability (0-1) to decimal odds."""
    if prob <= 0:
        return 99.0
    return round(1.0 / prob, 2)


def odds_to_prob(odds: float) -> float:
    """Convert decimal odds to implied probability."""
    return 1.0 / odds if odds > 0 else 0.0


def is_in_odds_range(prob: float, low: float = ODDS_RANGE[0], high: float = ODDS_RANGE[1]) -> bool:
    """Check if a probability translates to odds within our target range."""
    odds = prob_to_odds(prob)
    return low <= odds <= high


# ---------------------------------------------------------------------------
# Match data enrichment
# ---------------------------------------------------------------------------
def enrich_match(match: Dict, pred: Dict, home_stats: Dict, away_stats: Dict, h2h: Dict) -> Dict:
    """Combine raw match data with prediction into a single enriched dict."""
    return {
        "match_id": match.get("match_id", ""),
        "home_team": match["match_hometeam_name"],
        "away_team": match["match_awayteam_name"],
        "league": match.get("league_name", "?"),
        "country": match.get("country_name", "?"),
        "kick_off": match.get("match_time", "00:00"),
        "match_date": match.get("match_date", ""),
        "home_score": match.get("match_hometeam_score", ""),
        "away_score": match.get("match_awayteam_score", ""),
        # Prediction data
        "home_win_p": pred["home_win"] / 100,
        "draw_p": pred["draw"] / 100,
        "away_win_p": pred["away_win"] / 100,
        "prediction": pred["prediction"],
        "over15_p": pred["over_1.5"] / 100,
        "over25_p": pred["over_2.5"] / 100,
        "over35_p": pred.get("over_3.5", 30) / 100,
        "btts_p": pred["btts"] / 100,
        "confidence": pred["confidence"] / 100,
        "data_quality": pred["data_quality"],
        "expected_home_goals": pred["expected_home_goals"],
        "expected_away_goals": pred["expected_away_goals"],
        "h2h_meetings": pred["h2h_meetings"],
        # Odds
        "home_odds": prob_to_odds(pred["home_win"] / 100),
        "draw_odds": prob_to_odds(pred["draw"] / 100),
        "away_odds": prob_to_odds(pred["away_win"] / 100),
        "over25_odds": prob_to_odds(pred["over_2.5"] / 100),
        "btts_odds": prob_to_odds(pred["btts"] / 100),
        # Stats
        "home_stats": home_stats,
        "away_stats": away_stats,
        "h2h": h2h,
    }


def get_actual_result(match: Dict) -> Optional[str]:
    """Determine the actual result from a finished match."""
    hs = match.get("home_score", "")
    as_ = match.get("away_score", "")
    if hs == "" or as_ == "":
        return None
    try:
        h = int(hs)
        a = int(as_)
    except (ValueError, TypeError):
        return None
    if h > a:
        return "Home Win"
    elif h < a:
        return "Away Win"
    else:
        return "Draw"


def get_actual_goals(match: Dict) -> Optional[int]:
    """Get total goals from a finished match."""
    hs = match.get("home_score", "")
    as_ = match.get("away_score", "")
    if hs == "" or as_ == "":
        return None
    try:
        return int(hs) + int(as_)
    except (ValueError, TypeError):
        return None


def get_actual_btts(match: Dict) -> Optional[bool]:
    """Check if both teams scored."""
    hs = match.get("home_score", "")
    as_ = match.get("away_score", "")
    if hs == "" or as_ == "":
        return None
    try:
        return int(hs) > 0 and int(as_) > 0
    except (ValueError, TypeError):
        return None


# ---------------------------------------------------------------------------
# Bet types
# ---------------------------------------------------------------------------
class Bet:
    """Represents a single bet placed by a bot."""
    def __init__(self, match: Dict, market: str, selection: str, odds: float, prob: float):
        self.match = match
        self.market = market        # "result", "over25", "btts", "over15"
        self.selection = selection   # "Home Win", "Draw", "Away Win", "Over 2.5", "BTTS Yes"
        self.odds = odds
        self.prob = prob
        self.result = None          # "win", "loss", "pending"
        self.kick_off = match.get("kick_off", "00:00")
        self.match_id = match.get("match_id", "")

    def resolve(self, match_data: Dict) -> str:
        """Resolve the bet using actual match results."""
        if self.market == "result":
            actual = get_actual_result(match_data)
            if actual is None:
                self.result = "pending"
            elif actual == self.selection:
                self.result = "win"
            else:
                self.result = "loss"
        elif self.market == "over25":
            goals = get_actual_goals(match_data)
            if goals is None:
                self.result = "pending"
            elif goals > 2:
                self.result = "win"
            else:
                self.result = "loss"
        elif self.market == "over15":
            goals = get_actual_goals(match_data)
            if goals is None:
                self.result = "pending"
            elif goals > 1:
                self.result = "win"
            else:
                self.result = "loss"
        elif self.market == "under25":
            goals = get_actual_goals(match_data)
            if goals is None:
                self.result = "pending"
            elif goals <= 2:
                self.result = "win"
            else:
                self.result = "loss"
        elif self.market == "btts":
            btts = get_actual_btts(match_data)
            if btts is None:
                self.result = "pending"
            elif btts:
                self.result = "win"
            else:
                self.result = "loss"
        return self.result

    def to_dict(self) -> Dict:
        return {
            "match_id": self.match_id,
            "home_team": self.match.get("home_team", ""),
            "away_team": self.match.get("away_team", ""),
            "league": self.match.get("league", ""),
            "kick_off": self.kick_off,
            "market": self.market,
            "selection": self.selection,
            "odds": self.odds,
            "prob": round(self.prob, 3),
            "result": self.result,
        }


# ---------------------------------------------------------------------------
# Bot base class
# ---------------------------------------------------------------------------
class PredictionBot:
    """Base class for all prediction bots."""

    name = "BaseBot"
    description = "Base bot"
    emoji = "🤖"

    def __init__(self, stake: float = DEFAULT_STAKE):
        self.initial_stake = stake
        self.bankroll = stake
        self.chain_count = 0        # current win streak
        self.total_bets = 0
        self.wins = 0
        self.losses = 0
        self.best_chain = 0
        self.chains_completed = 0   # how many 10-chains hit
        self.history: List[Dict] = []
        self.active = True          # False once chain breaks or completes

    def pick_bet(self, matches: List[Dict]) -> Optional[Bet]:
        """Pick the best bet from available matches. Override in subclass."""
        raise NotImplementedError

    def place_bet(self, bet: Bet) -> Dict:
        """Place a bet with current bankroll."""
        stake = self.bankroll
        self.total_bets += 1

        entry = {
            "step": self.chain_count + 1,
            "stake": round(stake, 2),
            **bet.to_dict(),
        }

        if bet.result == "win":
            payout = stake * bet.odds
            self.bankroll = payout
            self.chain_count += 1
            self.wins += 1
            self.best_chain = max(self.best_chain, self.chain_count)
            entry["payout"] = round(payout, 2)
            entry["status"] = "WIN"

            if self.chain_count >= CHAIN_TARGET:
                self.chains_completed += 1
                entry["chain_complete"] = True
                entry["final_payout"] = round(self.bankroll, 2)
                self.reset_chain()
        elif bet.result == "loss":
            self.bankroll = self.initial_stake  # reset to starting stake
            self.chain_count = 0
            self.losses += 1
            entry["payout"] = 0
            entry["status"] = "LOSS"
        else:
            entry["status"] = "PENDING"

        self.history.append(entry)
        return entry

    def reset_chain(self):
        """Reset for a new chain attempt."""
        self.bankroll = self.initial_stake
        self.chain_count = 0

    def get_available_bets(self, matches: List[Dict], market: str = "result") -> List[Tuple[Dict, str, float, float]]:
        """Get all bets in the odds range for a given market."""
        bets = []
        for m in matches:
            if market == "result":
                for selection, p_key in [("Home Win", "home_win_p"), ("Draw", "draw_p"), ("Away Win", "away_win_p")]:
                    p = m[p_key]
                    odds = prob_to_odds(p)
                    if ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]:
                        bets.append((m, selection, odds, p))
            elif market == "over25":
                p = m["over25_p"]
                odds = prob_to_odds(p)
                if ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]:
                    bets.append((m, "Over 2.5", odds, p))
            elif market == "over15":
                p = m["over15_p"]
                odds = prob_to_odds(p)
                if ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]:
                    bets.append((m, "Over 1.5", odds, p))
            elif market == "btts":
                p = m["btts_p"]
                odds = prob_to_odds(p)
                if ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]:
                    bets.append((m, "BTTS Yes", odds, p))
        return bets

    def summary(self) -> Dict:
        return {
            "bot": self.name,
            "emoji": self.emoji,
            "description": self.description,
            "total_bets": self.total_bets,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.wins / self.total_bets * 100, 1) if self.total_bets else 0,
            "best_chain": self.best_chain,
            "chains_completed": self.chains_completed,
            "current_chain": self.chain_count,
            "current_bankroll": round(self.bankroll, 2),
            "history": self.history,
        }


# ---------------------------------------------------------------------------
# 10 Bot strategies
# ---------------------------------------------------------------------------

class ValueHunter(PredictionBot):
    """Picks the highest-confidence prediction closest to 2.0 odds."""
    name = "Value Hunter"
    description = "Highest confidence picks near 2.0 odds"
    emoji = "1"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            if m["data_quality"] == "LOW":
                continue
            for sel, p_key in [("Home Win", "home_win_p"), ("Draw", "draw_p"), ("Away Win", "away_win_p")]:
                p = m[p_key]
                odds = prob_to_odds(p)
                if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                    continue
                # Score: confidence * closeness to 2.0
                closeness = 1.0 - abs(odds - TARGET_ODDS) / 0.5
                score = m["confidence"] * max(0, closeness)
                if score > best_score:
                    best_score = score
                    best = Bet(m, "result", sel, odds, p)
        return best


class HomeFortress(PredictionBot):
    """Only bets on strong home teams with good home records."""
    name = "Home Fortress"
    description = "Strong home teams with proven records"
    emoji = "2"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            hs = m["home_stats"]
            if not hs.get("found"):
                continue
            p = m["home_win_p"]
            odds = prob_to_odds(p)
            if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                continue
            home_wr = hs.get("home_win_rate", 0.33)
            form = hs.get("recent_form", 0.33)
            score = (home_wr * 0.5 + form * 0.3 + m["confidence"] * 0.2)
            if score > best_score:
                best_score = score
                best = Bet(m, "result", "Home Win", odds, p)
        return best


class GoalRush(PredictionBot):
    """Focuses on Over 2.5 goals markets."""
    name = "Goal Rush"
    description = "Over 2.5 goals in high-scoring matchups"
    emoji = "3"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            p = m["over25_p"]
            odds = prob_to_odds(p)
            if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                continue
            exp_goals = m["expected_home_goals"] + m["expected_away_goals"]
            score = p * 0.5 + min(1.0, exp_goals / 4.0) * 0.3 + m["confidence"] * 0.2
            if score > best_score:
                best_score = score
                best = Bet(m, "over25", "Over 2.5", odds, p)
        return best


class BTTSSpecialist(PredictionBot):
    """Both Teams To Score market specialist."""
    name = "BTTS King"
    description = "Both Teams To Score - attacking matchups"
    emoji = "4"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            p = m["btts_p"]
            odds = prob_to_odds(p)
            if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                continue
            hs = m["home_stats"]
            as_ = m["away_stats"]
            attack_score = 0.5
            if hs.get("found") and as_.get("found"):
                attack_score = min(1.0, (hs["avg_scored"] + as_["avg_scored"]) / 3.0)
            score = p * 0.4 + attack_score * 0.4 + m["confidence"] * 0.2
            if score > best_score:
                best_score = score
                best = Bet(m, "btts", "BTTS Yes", odds, p)
        return best


class FormChaser(PredictionBot):
    """Picks teams on hot winning streaks — searches all markets for value."""
    name = "Form Chaser"
    description = "Teams on hot streaks with recent momentum"
    emoji = "5"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            hs = m["home_stats"]
            as_ = m["away_stats"]
            if not hs.get("found") or not as_.get("found"):
                continue

            h_form = hs.get("recent_form", 0.33)
            a_form = as_.get("recent_form", 0.33)
            form_diff = abs(h_form - a_form)
            best_form = max(h_form, a_form)

            # Try all markets
            candidates = []
            if h_form > a_form:
                candidates.append(("result", "Home Win", m["home_win_p"]))
            else:
                candidates.append(("result", "Away Win", m["away_win_p"]))

            # Hot teams score goals
            if best_form > 0.5:
                candidates.append(("over25", "Over 2.5", m["over25_p"]))
                candidates.append(("btts", "BTTS Yes", m["btts_p"]))
                candidates.append(("over15", "Over 1.5", m["over15_p"]))

            for market, sel, p in candidates:
                odds = prob_to_odds(p)
                if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                    continue
                score = best_form * 0.4 + form_diff * 0.3 + p * 0.15 + m["confidence"] * 0.15
                if score > best_score:
                    best_score = score
                    best = Bet(m, market, sel, odds, p)
        return best


class UnderdogSpotter(PredictionBot):
    """Picks away wins or goals markets when the away team looks strong."""
    name = "Underdog Spotter"
    description = "Away-team value across all markets"
    emoji = "6"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            as_ = m["away_stats"]
            if not as_.get("found"):
                continue
            away_wr = as_.get("away_win_rate", 0.33)
            form = as_.get("recent_form", 0.33)

            # Try away win first, then goals markets
            candidates = [
                ("result", "Away Win", m["away_win_p"]),
                ("over25", "Over 2.5", m["over25_p"]),
                ("btts", "BTTS Yes", m["btts_p"]),
            ]
            for market, sel, p in candidates:
                odds = prob_to_odds(p)
                if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                    continue
                # Stronger away teams that attack = more goals too
                score = away_wr * 0.3 + form * 0.3 + p * 0.2 + m["confidence"] * 0.2
                if score > best_score:
                    best_score = score
                    best = Bet(m, market, sel, odds, p)
        return best


class DrawMaster(PredictionBot):
    """Spots evenly-matched teams — bets Under 2.5 in tight games (low-scoring draws)."""
    name = "Draw Master"
    description = "Tight games - Under 2.5 goals in even matchups"
    emoji = "7"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            hs = m["home_stats"]
            as_ = m["away_stats"]

            # Core idea: evenly-matched teams play tight, low-scoring games
            evenness = 0.5
            if hs.get("found") and as_.get("found"):
                wr_diff = abs(hs["win_rate"] - as_["win_rate"])
                evenness = max(0, 1.0 - wr_diff * 2)
            else:
                continue

            # Under 2.5: probability is 1 - over25_p
            under25_p = 1.0 - m["over25_p"]
            under25_odds = prob_to_odds(under25_p)

            # Also check Over 1.5 as fallback
            candidates = []
            if ODDS_RANGE[0] <= under25_odds <= ODDS_RANGE[1]:
                candidates.append(("under25", "Under 2.5", under25_odds, under25_p))

            over15_p = m["over15_p"]
            over15_odds = prob_to_odds(over15_p)
            if ODDS_RANGE[0] <= over15_odds <= ODDS_RANGE[1]:
                candidates.append(("over15", "Over 1.5", over15_odds, over15_p))

            # Also try the result markets
            for sel, p_key in [("Home Win", "home_win_p"), ("Draw", "draw_p"), ("Away Win", "away_win_p")]:
                p = m[p_key]
                odds = prob_to_odds(p)
                if ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]:
                    candidates.append(("result", sel, odds, p))

            for market, sel, odds, p in candidates:
                score = evenness * 0.5 + p * 0.3 + m["confidence"] * 0.2
                if score > best_score:
                    best_score = score
                    best = Bet(m, market, sel, odds, p)
        return best


class H2HExpert(PredictionBot):
    """Prioritizes matches with strong H2H patterns."""
    name = "H2H Expert"
    description = "Head-to-head history patterns"
    emoji = "8"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            h2h = m["h2h"]
            if h2h.get("meetings", 0) < 3:
                continue
            meetings = h2h["meetings"]
            h2h_home_wr = h2h.get("home_win_rate", 0.33)
            h2h_draw_r = h2h.get("draw_rate", 0.33)
            h2h_away_wr = 1 - h2h_home_wr - h2h_draw_r

            # Find the dominant H2H outcome
            options = [
                ("Home Win", h2h_home_wr, m["home_win_p"]),
                ("Draw", h2h_draw_r, m["draw_p"]),
                ("Away Win", h2h_away_wr, m["away_win_p"]),
            ]
            for sel, h2h_rate, model_p in options:
                odds = prob_to_odds(model_p)
                if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                    continue
                # Both H2H and model should agree
                agreement = min(h2h_rate, model_p)
                score = agreement * 0.5 + min(1.0, meetings / 10) * 0.3 + m["confidence"] * 0.2
                if score > best_score:
                    best_score = score
                    best = Bet(m, "result", sel, odds, model_p)
        return best


class StatEdge(PredictionBot):
    """Uses the biggest gap between model probability and implied 2.0 odds (50%)."""
    name = "Stat Edge"
    description = "Biggest edge over implied odds"
    emoji = "9"

    def pick_bet(self, matches):
        best = None
        best_edge = -1
        for m in matches:
            if m["data_quality"] == "LOW":
                continue
            candidates = [
                ("result", "Home Win", m["home_win_p"]),
                ("result", "Draw", m["draw_p"]),
                ("result", "Away Win", m["away_win_p"]),
                ("over25", "Over 2.5", m["over25_p"]),
                ("btts", "BTTS Yes", m["btts_p"]),
            ]
            for market, sel, p in candidates:
                odds = prob_to_odds(p)
                if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                    continue
                implied_p = odds_to_prob(odds)
                edge = p - implied_p  # positive = we think it's more likely
                if edge > best_edge:
                    best_edge = edge
                    best = Bet(m, market, sel, odds, p)
        return best


class ComboKing(PredictionBot):
    """Picks where multiple signals agree — any 2+ out of result/goals/BTTS."""
    name = "Combo King"
    description = "Multi-signal convergence picks"
    emoji = "10"

    def pick_bet(self, matches):
        best = None
        best_score = -1
        for m in matches:
            if m["data_quality"] == "LOW":
                continue

            signals = 0
            total_conf = 0

            # Result signal: clear favorite
            result_p = max(m["home_win_p"], m["draw_p"], m["away_win_p"])
            if result_p > 0.40:
                signals += 1
                total_conf += result_p

            # O2.5 signal: leaning one way
            if m["over25_p"] > 0.50 or m["over25_p"] < 0.40:
                signals += 1
                total_conf += max(m["over25_p"], 1 - m["over25_p"])

            # BTTS signal
            if m["btts_p"] > 0.50 or m["btts_p"] < 0.40:
                signals += 1
                total_conf += max(m["btts_p"], 1 - m["btts_p"])

            # O1.5 signal
            if m["over15_p"] > 0.60:
                signals += 1
                total_conf += m["over15_p"]

            if signals < 2:
                continue

            # Pick best market within odds range
            candidates = [
                ("result", "Home Win", m["home_win_p"]),
                ("result", "Away Win", m["away_win_p"]),
                ("over25", "Over 2.5", m["over25_p"]),
                ("btts", "BTTS Yes", m["btts_p"]),
                ("over15", "Over 1.5", m["over15_p"]),
            ]

            for market, sel, p in candidates:
                odds = prob_to_odds(p)
                if not (ODDS_RANGE[0] <= odds <= ODDS_RANGE[1]):
                    continue
                avg_conf = total_conf / signals if signals else 0
                score = avg_conf * 0.4 + (signals / 4) * 0.3 + m["confidence"] * 0.3
                if score > best_score:
                    best_score = score
                    best = Bet(m, market, sel, odds, p)
        return best


# ---------------------------------------------------------------------------
# All bots registry
# ---------------------------------------------------------------------------
ALL_BOTS = [
    ValueHunter,
    HomeFortress,
    GoalRush,
    BTTSSpecialist,
    FormChaser,
    UnderdogSpotter,
    DrawMaster,
    H2HExpert,
    StatEdge,
    ComboKing,
]


# ---------------------------------------------------------------------------
# Hour grouping
# ---------------------------------------------------------------------------
def group_by_hour(matches: List[Dict]) -> Dict[str, List[Dict]]:
    """Group matches by their kick-off hour."""
    hours = defaultdict(list)
    for m in matches:
        ko = m.get("kick_off", "00:00")
        hour = ko.split(":")[0].zfill(2)
        hours[hour].append(m)
    return dict(sorted(hours.items()))


# ---------------------------------------------------------------------------
# Run simulation for one day
# ---------------------------------------------------------------------------
def run_day(date_str: str, stake: float = DEFAULT_STAKE, verbose: bool = True) -> Dict:
    """Run all 10 bots through a single day's matches."""

    if verbose:
        print(f"\n{'='*70}")
        print(f"  BETSIGHTLY BOT SYSTEM - {date_str}")
        print(f"  Stake: ${stake:.2f} | Target: {CHAIN_TARGET} wins in a row")
        print(f"  Odds range: {ODDS_RANGE[0]} - {ODDS_RANGE[1]}")
        print(f"{'='*70}")

    # 1. Fetch all matches for the day
    if verbose:
        print("\n[1/3] Fetching matches...")
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        print("  No fixtures found!")
        return {"date": date_str, "error": "no fixtures"}

    if verbose:
        print(f"  {len(fixtures)} total matches")

    # 2. Load historical data
    if verbose:
        print("\n[2/3] Loading historical data...")
    hist = load_historical()

    # 3. Generate predictions for ALL matches (finished + upcoming)
    if verbose:
        print("\n[3/3] Generating predictions for all matches...")
    team_cache = {}
    enriched = []

    for match in fixtures:
        home = match["match_hometeam_name"]
        away = match["match_awayteam_name"]

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
        em = enrich_match(match, pred, home_stats, away_stats, h2h)
        enriched.append(em)

    if verbose:
        print(f"  {len(enriched)} matches enriched with predictions")

    # Group by hour
    hourly = group_by_hour(enriched)
    hours_sorted = sorted(hourly.keys())

    if verbose:
        print(f"  Hours with matches: {', '.join(hours_sorted)}")

    # 4. Run each bot through the day hour by hour
    bots = [BotClass(stake) for BotClass in ALL_BOTS]

    if verbose:
        print(f"\n{'='*70}")
        print(f"  RUNNING 10 BOTS THROUGH {len(hours_sorted)} HOURS")
        print(f"{'='*70}")

    for hour in hours_sorted:
        hour_matches = hourly[hour]
        # Separate finished (have scores) from upcoming
        finished = [m for m in hour_matches if m["home_score"] != ""]
        upcoming = [m for m in hour_matches if m["home_score"] == ""]

        if verbose:
            print(f"\n  --- Hour {hour}:00 --- ({len(hour_matches)} matches: {len(finished)} finished, {len(upcoming)} upcoming)")

        for bot in bots:
            # Bot picks from ALL matches in this hour
            bet = bot.pick_bet(hour_matches)
            if bet is None:
                continue

            # Resolve against actual results if match is finished
            match_for_resolve = None
            for m in hour_matches:
                if m["match_id"] == bet.match_id:
                    match_for_resolve = m
                    break

            if match_for_resolve and match_for_resolve["home_score"] != "":
                bet.resolve(match_for_resolve)
            else:
                bet.result = "pending"

            entry = bot.place_bet(bet)

            if verbose and bet.result != "pending":
                status_icon = "+" if entry["status"] == "WIN" else "-"
                print(f"    [{status_icon}] Bot {bot.emoji:>2} {bot.name:<16} | "
                      f"{entry['selection']:<10} @ {entry['odds']:.2f} | "
                      f"Chain: {bot.chain_count}/{CHAIN_TARGET} | "
                      f"${bot.bankroll:.2f}")

    # 5. Summary
    if verbose:
        print(f"\n{'='*70}")
        print(f"  DAY SUMMARY - {date_str}")
        print(f"{'='*70}")
        print(f"  {'Bot':<20} {'W/L':<8} {'Win%':<7} {'Best':<6} {'Chains':<8} {'Bank':<10}")
        print(f"  {'-'*19} {'-'*7} {'-'*6} {'-'*5} {'-'*7} {'-'*9}")

        for bot in bots:
            s = bot.summary()
            wl = f"{s['wins']}/{s['losses']}"
            print(f"  {s['emoji']:>2} {s['bot']:<17} {wl:<8} {s['win_rate']:<6.1f}% "
                  f"{s['best_chain']:<6} {s['chains_completed']:<8} ${s['current_bankroll']:<9.2f}")

    # Build result
    result = {
        "date": date_str,
        "total_matches": len(enriched),
        "hours": len(hours_sorted),
        "bots": [bot.summary() for bot in bots],
    }

    return result


# ---------------------------------------------------------------------------
# Multi-day simulation
# ---------------------------------------------------------------------------
def run_simulation(days: int = 3, end_date: str = None, stake: float = DEFAULT_STAKE):
    """Simulate bots over multiple days to measure performance."""
    if end_date is None:
        end = datetime.now()
    else:
        end = datetime.strptime(end_date, "%Y-%m-%d")

    dates = [(end - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days)]
    dates.reverse()

    print(f"\n{'#'*70}")
    print(f"  MULTI-DAY SIMULATION: {dates[0]} to {dates[-1]}")
    print(f"  Bots: {len(ALL_BOTS)} | Stake: ${stake:.2f} | Chain target: {CHAIN_TARGET}")
    print(f"{'#'*70}")

    all_results = []
    for date_str in dates:
        result = run_day(date_str, stake, verbose=True)
        all_results.append(result)

    # Aggregate stats
    print(f"\n{'#'*70}")
    print(f"  AGGREGATE RESULTS ({len(dates)} days)")
    print(f"{'#'*70}")

    bot_totals = defaultdict(lambda: {"wins": 0, "losses": 0, "bets": 0, "best": 0, "chains": 0})
    for result in all_results:
        if "error" in result:
            continue
        for bot_data in result["bots"]:
            name = bot_data["bot"]
            bot_totals[name]["wins"] += bot_data["wins"]
            bot_totals[name]["losses"] += bot_data["losses"]
            bot_totals[name]["bets"] += bot_data["total_bets"]
            bot_totals[name]["best"] = max(bot_totals[name]["best"], bot_data["best_chain"])
            bot_totals[name]["chains"] += bot_data["chains_completed"]
            bot_totals[name]["emoji"] = bot_data["emoji"]

    print(f"\n  {'Bot':<20} {'Total':<7} {'W/L':<10} {'Win%':<7} {'Best':<6} {'Chains':<7}")
    print(f"  {'-'*19} {'-'*6} {'-'*9} {'-'*6} {'-'*5} {'-'*6}")

    for name, data in sorted(bot_totals.items(), key=lambda x: x[1]["wins"], reverse=True):
        wl = f"{data['wins']}/{data['losses']}"
        wr = data["wins"] / data["bets"] * 100 if data["bets"] else 0
        print(f"  {data['emoji']:>2} {name:<17} {data['bets']:<7} {wl:<10} {wr:<6.1f}% "
              f"{data['best']:<6} {data['chains']}")

    # Save results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"sim_{dates[0]}_to_{dates[-1]}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")

    return all_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="BetSightly 10-Bot System")
    parser.add_argument("--date", default=datetime.now().strftime("%Y-%m-%d"),
                        help="Date to run (YYYY-MM-DD)")
    parser.add_argument("--stake", type=float, default=DEFAULT_STAKE,
                        help="Starting stake per bot")
    parser.add_argument("--target-odds", type=float, default=TARGET_ODDS,
                        help="Target odds (default 2.0)")
    parser.add_argument("--simulate", action="store_true",
                        help="Run multi-day simulation")
    parser.add_argument("--days", type=int, default=3,
                        help="Number of days for simulation")
    args = parser.parse_args()

    if args.target_odds != TARGET_ODDS:
        pass  # could adjust ODDS_RANGE here in future

    if args.simulate:
        run_simulation(days=args.days, end_date=args.date, stake=args.stake)
    else:
        result = run_day(args.date, stake=args.stake, verbose=True)
        RESULTS_DIR.mkdir(exist_ok=True)
        out_file = RESULTS_DIR / f"bots_{args.date}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n  Results saved to {out_file}")


if __name__ == "__main__":
    main()
