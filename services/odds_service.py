"""
Odds Service — fetches real bookmaker odds from The Odds API.

Uses The Odds API (the-odds-api.com) free tier:
    - 500 credits/month
    - Each request = 1 credit per sport queried
    - Covers all major football leagues

Purpose: compare model probabilities against real market odds
to find VALUE bets (where our edge > bookmaker's implied probability).

Without real odds, we're just picking favourites. With real odds,
we can identify where the market is wrong and exploit it.
"""

import os
import json
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import requests
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")
ODDS_BASE_URL = "https://api.the-odds-api.com/v4"
ODDS_CACHE_DIR = Path("cache/odds")
ODDS_CACHE_TTL_HOURS = 3  # Odds change frequently, shorter cache

# Map our API-Football league IDs to The Odds API sport keys
LEAGUE_TO_SPORT_KEY = {
    39: "soccer_epl",                    # Premier League
    40: "soccer_efl_champ",              # Championship
    61: "soccer_france_ligue_one",       # Ligue 1
    62: "soccer_france_ligue_two",       # Ligue 2
    78: "soccer_germany_bundesliga",     # Bundesliga
    79: "soccer_germany_bundesliga2",    # 2. Bundesliga
    135: "soccer_italy_serie_a",         # Serie A
    136: "soccer_italy_serie_b",         # Serie B
    140: "soccer_spain_la_liga",         # La Liga
    141: "soccer_spain_segunda_division", # Segunda División
    94: "soccer_portugal_primeira_liga", # Primeira Liga
    88: "soccer_belgium_first_div",      # Jupiler Pro League
    144: "soccer_turkey_super_league",   # Süper Lig
    2: "soccer_uefa_champs_league",      # Champions League
    3: "soccer_uefa_europa_league",      # Europa League
}

# Reverse mapping: sport key -> league ID
SPORT_KEY_TO_LEAGUE = {v: k for k, v in LEAGUE_TO_SPORT_KEY.items()}


class OddsService:
    """Fetches real bookmaker odds and calculates value."""

    def __init__(self, api_key: str = None, cache_ttl_hours: float = ODDS_CACHE_TTL_HOURS):
        self.api_key = api_key or ODDS_API_KEY
        self.base_url = ODDS_BASE_URL
        self.cache_dir = ODDS_CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.remaining_credits = None

        if not self.api_key:
            logger.warning("ODDS_API_KEY not configured — real odds unavailable")

    # ------------------------------------------------------------------
    # Public: get odds for fixtures
    # ------------------------------------------------------------------

    def get_odds_for_fixtures(self, fixtures: List[Dict]) -> Dict[int, Dict]:
        """Get real bookmaker odds for a list of fixtures.

        Returns a dict mapping fixture_id -> odds data:
        {
            fixture_id: {
                "home_odds": 1.85,
                "draw_odds": 3.40,
                "away_odds": 4.20,
                "over_2_5_odds": 1.95,
                "under_2_5_odds": 1.88,
                "btts_yes_odds": 1.80,
                "btts_no_odds": 2.00,
                "bookmaker": "pinnacle",
                "implied_home_prob": 0.54,
                "implied_draw_prob": 0.29,
                "implied_away_prob": 0.24,
            }
        }
        """
        if not self.api_key:
            logger.warning("No ODDS_API_KEY — returning empty odds")
            return {}

        # Group fixtures by league to minimize API calls
        leagues_needed = set()
        for fx in fixtures:
            league_id = fx.get("league_id", 0)
            if league_id in LEAGUE_TO_SPORT_KEY:
                leagues_needed.add(league_id)

        # Fetch odds per league (each = 1 credit)
        all_odds_data = []
        for league_id in leagues_needed:
            sport_key = LEAGUE_TO_SPORT_KEY[league_id]
            odds = self._fetch_sport_odds(sport_key)
            if odds:
                all_odds_data.extend(odds)

        # Match odds to our fixtures by team name similarity
        odds_map = {}
        for fx in fixtures:
            fixture_id = fx.get("fixture_id")
            home = fx.get("home_team", "").lower()
            away = fx.get("away_team", "").lower()

            matched = self._match_fixture_to_odds(home, away, all_odds_data)
            if matched:
                odds_map[fixture_id] = matched

        logger.info(f"Matched odds for {len(odds_map)}/{len(fixtures)} fixtures")
        return odds_map

    def get_odds_for_league(self, league_id: int) -> List[Dict]:
        """Get all odds for a specific league."""
        sport_key = LEAGUE_TO_SPORT_KEY.get(league_id)
        if not sport_key:
            return []
        return self._fetch_sport_odds(sport_key) or []

    # ------------------------------------------------------------------
    # Value calculation
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_value(model_prob: float, bookmaker_odds: float) -> Dict[str, float]:
        """Calculate betting value.

        Value = Model Probability - Implied Probability from odds.
        Positive value = we think outcome is MORE likely than bookmaker does.

        Args:
            model_prob: Our model's estimated probability (0-1)
            bookmaker_odds: Real decimal odds from bookmaker

        Returns:
            {
                "implied_prob": 0.476,      # What bookmaker thinks
                "model_prob": 0.55,         # What we think
                "edge": 0.074,              # Our advantage (7.4%)
                "expected_value": 1.155,    # EV per £1 bet
                "is_value": True,           # Worth betting?
                "kelly_fraction": 0.086,    # Optimal bet size (Kelly)
            }
        """
        if bookmaker_odds <= 1.0:
            return {"implied_prob": 1.0, "model_prob": model_prob, "edge": -1.0,
                    "expected_value": 0.0, "is_value": False, "kelly_fraction": 0.0}

        implied_prob = 1.0 / bookmaker_odds
        edge = model_prob - implied_prob
        ev = model_prob * bookmaker_odds  # Expected return per £1
        is_value = edge > 0.03  # Require at least 3% edge to account for model uncertainty

        # Kelly criterion: optimal fraction of bankroll to bet
        # f = (bp - q) / b where b=odds-1, p=model_prob, q=1-model_prob
        b = bookmaker_odds - 1.0
        kelly = max(0, (b * model_prob - (1 - model_prob)) / b)

        return {
            "implied_prob": round(implied_prob, 4),
            "model_prob": round(model_prob, 4),
            "edge": round(edge, 4),
            "expected_value": round(ev, 4),
            "is_value": is_value,
            "kelly_fraction": round(min(kelly, 0.25), 4),  # Cap at 25%
        }

    @staticmethod
    def odds_to_probability(odds: float) -> float:
        """Convert decimal odds to implied probability."""
        if odds <= 0:
            return 0.0
        return 1.0 / odds

    @staticmethod
    def probability_to_odds(prob: float) -> float:
        """Convert probability to fair decimal odds."""
        if prob <= 0:
            return 100.0
        return round(1.0 / prob, 2)

    # ------------------------------------------------------------------
    # API calls with caching
    # ------------------------------------------------------------------

    def _fetch_sport_odds(self, sport_key: str) -> Optional[List[Dict]]:
        """Fetch odds for a sport. Cached for ODDS_CACHE_TTL_HOURS."""
        cache_key = f"odds_{sport_key}_{datetime.now().strftime('%Y-%m-%d')}"
        key_hash = hashlib.md5(cache_key.encode()).hexdigest()

        # Check cache
        cached = self._read_cache(key_hash)
        if cached is not None:
            return cached

        # Make API call
        try:
            # h2h = match winner, totals = over/under (2.5 default + alternate lines)
            url = f"{self.base_url}/sports/{sport_key}/odds/"
            params = {
                "apiKey": self.api_key,
                "regions": "uk,eu",  # UK and EU bookmakers for best football coverage
                "markets": "h2h,totals,alternate_totals",  # Match winner + all goal lines
                "oddsFormat": "decimal",
            }

            resp = requests.get(url, params=params, timeout=30)

            # Track remaining credits
            self.remaining_credits = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            logger.info(f"Odds API: {sport_key} — {self.remaining_credits} credits remaining (used: {used})")

            if resp.status_code == 401:
                logger.error("Odds API: invalid API key")
                return None
            elif resp.status_code == 429:
                logger.error("Odds API: rate limited (out of credits)")
                return None
            elif resp.status_code == 422:
                logger.warning(f"Odds API: sport {sport_key} not available")
                return None

            resp.raise_for_status()
            data = resp.json()

            # Cache the response
            self._write_cache(key_hash, data)

            return data

        except requests.exceptions.RequestException as e:
            logger.error(f"Odds API request failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return None

    # ------------------------------------------------------------------
    # Match fixtures to odds by team name
    # ------------------------------------------------------------------

    def _match_fixture_to_odds(self, home: str, away: str, odds_data: List[Dict]) -> Optional[Dict]:
        """Match our fixture to odds data using fuzzy team name matching."""
        best_match = None
        best_score = 0

        for event in odds_data:
            odds_home = event.get("home_team", "").lower()
            odds_away = event.get("away_team", "").lower()

            # Score how well team names match
            score = self._name_similarity(home, odds_home) + self._name_similarity(away, odds_away)

            if score > best_score and score > 1.0:  # Require decent match on both teams
                best_score = score
                best_match = event

        if not best_match:
            return None

        return self._extract_odds(best_match)

    def _extract_odds(self, event: Dict) -> Dict:
        """Extract structured odds from an event."""
        result = {
            "home_team_odds_api": event.get("home_team", ""),
            "away_team_odds_api": event.get("away_team", ""),
            "commence_time": event.get("commence_time", ""),
            "home_odds": None,
            "draw_odds": None,
            "away_odds": None,
            "over_1_5_odds": None,
            "under_1_5_odds": None,
            "over_2_5_odds": None,
            "under_2_5_odds": None,
            "bookmaker": None,
        }

        bookmakers = event.get("bookmakers", [])
        if not bookmakers:
            return result

        # Prefer Pinnacle (sharpest odds), then bet365, then first available
        preferred = ["pinnacle", "bet365", "williamhill", "unibet", "betfair_ex_eu"]
        selected_bookie = None

        for pref in preferred:
            for bk in bookmakers:
                if bk.get("key", "") == pref:
                    selected_bookie = bk
                    break
            if selected_bookie:
                break

        if not selected_bookie:
            selected_bookie = bookmakers[0]

        result["bookmaker"] = selected_bookie.get("title", selected_bookie.get("key", "unknown"))

        # Extract h2h (match winner) odds
        for market in selected_bookie.get("markets", []):
            if market.get("key") == "h2h":
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").lower()
                    price = outcome.get("price", 0)
                    if name == event.get("home_team", "").lower():
                        result["home_odds"] = price
                    elif name == event.get("away_team", "").lower():
                        result["away_odds"] = price
                    elif name == "draw":
                        result["draw_odds"] = price

            # Extract totals (over/under) for multiple lines
            elif market.get("key") in ("totals", "alternate_totals"):
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name", "").lower()
                    point = outcome.get("point", 0)
                    price = outcome.get("price", 0)
                    if point == 1.5:
                        if name == "over" and result["over_1_5_odds"] is None:
                            result["over_1_5_odds"] = price
                        elif name == "under" and result["under_1_5_odds"] is None:
                            result["under_1_5_odds"] = price
                    elif point == 2.5:
                        if name == "over" and result["over_2_5_odds"] is None:
                            result["over_2_5_odds"] = price
                        elif name == "under" and result["under_2_5_odds"] is None:
                            result["under_2_5_odds"] = price

        # If over_1_5 not found in selected bookie, check all bookmakers
        if result["over_1_5_odds"] is None:
            for bk in bookmakers:
                if result["over_1_5_odds"] is not None:
                    break
                for market in bk.get("markets", []):
                    if market.get("key") in ("totals", "alternate_totals"):
                        for outcome in market.get("outcomes", []):
                            point = outcome.get("point", 0)
                            name = outcome.get("name", "").lower()
                            price = outcome.get("price", 0)
                            if point == 1.5 and name == "over" and price > 1.0:
                                result["over_1_5_odds"] = price
                                break
                            if point == 1.5 and name == "under" and price > 1.0:
                                result["under_1_5_odds"] = price

        # Calculate implied probabilities
        if result["home_odds"] and result["draw_odds"] and result["away_odds"]:
            total_implied = (1/result["home_odds"] + 1/result["draw_odds"] + 1/result["away_odds"])
            # Remove overround for fair probabilities
            result["implied_home_prob"] = round((1/result["home_odds"]) / total_implied, 4)
            result["implied_draw_prob"] = round((1/result["draw_odds"]) / total_implied, 4)
            result["implied_away_prob"] = round((1/result["away_odds"]) / total_implied, 4)
            result["overround"] = round(total_implied - 1.0, 4)  # Bookmaker margin

        return result

    @staticmethod
    def _name_similarity(name1: str, name2: str) -> float:
        """Simple fuzzy match score for team names.

        Handles cases like:
            "Borussia Dortmund" vs "Dortmund"
            "Manchester City" vs "Man City"
            "VfB Stuttgart" vs "Stuttgart"
        """
        if not name1 or not name2:
            return 0.0

        # Exact match
        if name1 == name2:
            return 1.0

        # One contains the other
        if name1 in name2 or name2 in name1:
            return 0.85

        # Split into words and check overlap
        words1 = set(name1.split())
        words2 = set(name2.split())

        # Remove common prefixes/suffixes that differ between APIs
        noise = {"fc", "cf", "sc", "ac", "as", "us", "ss", "1.", "sv", "vfb", "vfl",
                 "tsg", "rb", "rcd", "ud", "cd", "sd", "real", "racing", "sporting"}
        words1 = words1 - noise
        words2 = words2 - noise

        if not words1 or not words2:
            return 0.3

        overlap = words1 & words2
        if overlap:
            return len(overlap) / max(len(words1), len(words2))

        # Check if any word from one appears as substring in the other
        for w1 in words1:
            if len(w1) < 4:
                continue
            for w2 in words2:
                if w1 in w2 or w2 in w1:
                    return 0.6

        return 0.0

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _read_cache(self, key: str) -> Optional[List]:
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached["cached_at"])
            if datetime.now() - cached_at > self.cache_ttl:
                path.unlink(missing_ok=True)
                return None
            logger.info(f"Odds cache hit ({key[:8]}...)")
            return cached["data"]
        except Exception:
            return None

    def _write_cache(self, key: str, data: Any) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cached_at": datetime.now().isoformat(), "data": data}, f)
        except Exception as e:
            logger.debug(f"Odds cache write failed: {e}")

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def test_connection(self) -> bool:
        """Test the API connection and show remaining credits."""
        if not self.api_key:
            print("ODDS_API_KEY not set")
            return False
        try:
            url = f"{self.base_url}/sports/"
            resp = requests.get(url, params={"apiKey": self.api_key}, timeout=15)
            if resp.status_code == 401:
                print("Invalid API key")
                return False
            remaining = resp.headers.get("x-requests-remaining", "?")
            used = resp.headers.get("x-requests-used", "?")
            print(f"Odds API connected: {remaining} credits remaining ({used} used this month)")
            return True
        except Exception as e:
            print(f"Connection failed: {e}")
            return False


def get_odds_service() -> OddsService:
    return OddsService()
