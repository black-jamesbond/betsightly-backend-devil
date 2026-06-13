"""
API-Football (api-sports.io) integration service.

Uses the SAME data source as the training pipeline (fetch_history.py)
so team names align perfectly between training and live predictions.

Caching strategy (free tier = 100 requests/day):
    - Daily fixtures: cached per date, valid until end of that day.
      Calling get_daily_fixtures("2026-05-08") makes 1 API call,
      then every subsequent call that same day hits cache.
      Next day it auto-expires — new call fetches fresh fixtures.
    - Other endpoints: generic 6-hour TTL.
    - Stale cache files (older than 3 days) are auto-cleaned on init.

API docs: https://www.api-football.com/documentation-v3
"""

import os
import json
import hashlib
import requests
import logging
from datetime import datetime, timedelta, date as date_type
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

API_KEY = os.getenv("API_FOOTBALL_API_KEY", "")
BASE_URL = "https://v3.football.api-sports.io"
CACHE_DIR = Path("cache/api_football")
CACHE_TTL_HOURS = 6
STALE_CACHE_DAYS = 3  # auto-delete cache files older than this

TARGET_LEAGUE_IDS = {
    39, 40, 61, 62, 78, 79, 135, 136, 140, 141,
    94, 88, 144, 203, 2, 3,
}


class APIFootballService:
    """Service for fetching fixtures from API-Football (api-sports.io)."""

    def __init__(self, api_key: str = None, cache_ttl_hours: float = CACHE_TTL_HOURS):
        self.api_key = api_key or API_KEY
        self.base_url = BASE_URL
        self.headers = {"x-apisports-key": self.api_key}
        self.timeout = 30
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        self.cache_dir = CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.api_key:
            logger.warning("API_FOOTBALL_API_KEY not configured")

        # Auto-clean stale cache files on startup
        self._cleanup_stale_cache()

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _cache_key(self, endpoint: str, params: dict) -> str:
        raw = f"{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _daily_cache_key(self, target_date: str) -> str:
        """Deterministic cache key for a specific date's fixtures.

        Key includes the date itself, so each day gets its own file.
        """
        raw = f"fixtures:daily:{target_date}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _read_cache(self, key: str, ttl: timedelta = None) -> Optional[dict]:
        """Read from cache. Uses custom ttl if provided, else self.cache_ttl."""
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached_at = datetime.fromisoformat(cached["cached_at"])
            effective_ttl = ttl if ttl is not None else self.cache_ttl
            if datetime.now() - cached_at > effective_ttl:
                path.unlink(missing_ok=True)  # Delete expired cache
                return None
            logger.info(f"Cache hit ({key[:8]}...) — saved an API call")
            return cached["data"]
        except Exception:
            return None

    def _read_daily_cache(self, target_date: str) -> Optional[dict]:
        """Read daily fixture cache. Valid until end of the target date.

        If today is 2026-05-08 and we cached fixtures for 2026-05-08,
        the cache is valid all day. At midnight it expires automatically.
        Past dates stay cached for STALE_CACHE_DAYS before cleanup.
        """
        key = self._daily_cache_key(target_date)
        path = self.cache_dir / f"{key}.json"
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                cached = json.load(f)

            cached_date = cached.get("fixture_date", "")
            today = datetime.now().strftime("%Y-%m-%d")

            # If requesting today's fixtures and cache is from today -> valid
            if cached_date == target_date == today:
                logger.info(f"Daily cache hit for {target_date} — saved an API call")
                return cached["data"]

            # If requesting a future date and cache exists -> valid
            if cached_date == target_date and target_date > today:
                logger.info(f"Daily cache hit for future date {target_date}")
                return cached["data"]

            # Past date: cache is stale, let cleanup handle it
            return None

        except Exception:
            return None

    def _write_daily_cache(self, target_date: str, data: dict) -> None:
        """Write daily fixture cache with the target date embedded."""
        key = self._daily_cache_key(target_date)
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({
                    "cached_at": datetime.now().isoformat(),
                    "fixture_date": target_date,
                    "data": data,
                }, f)
        except Exception as e:
            logger.debug(f"Daily cache write failed: {e}")

    def _write_cache(self, key: str, data: dict) -> None:
        path = self.cache_dir / f"{key}.json"
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"cached_at": datetime.now().isoformat(), "data": data}, f)
        except Exception as e:
            logger.debug(f"Cache write failed: {e}")

    def _cleanup_stale_cache(self) -> int:
        """Remove cache files older than STALE_CACHE_DAYS. Runs on init."""
        cutoff = datetime.now() - timedelta(days=STALE_CACHE_DAYS)
        removed = 0
        try:
            for f in self.cache_dir.glob("*.json"):
                try:
                    with open(f, "r", encoding="utf-8") as fh:
                        cached = json.load(fh)
                    cached_at = datetime.fromisoformat(cached.get("cached_at", "2000-01-01"))
                    if cached_at < cutoff:
                        f.unlink()
                        removed += 1
                except Exception:
                    # Corrupt file — remove it
                    f.unlink(missing_ok=True)
                    removed += 1
            if removed:
                logger.info(f"Cleaned up {removed} stale cache files (older than {STALE_CACHE_DAYS} days)")
        except Exception as e:
            logger.debug(f"Cache cleanup error: {e}")
        return removed

    # ------------------------------------------------------------------
    # HTTP
    # ------------------------------------------------------------------

    def _get(self, endpoint: str, params: dict, use_cache: bool = True) -> dict:
        if use_cache:
            key = self._cache_key(endpoint, params)
            cached = self._read_cache(key)
            if cached is not None:
                return cached

        url = f"{self.base_url}/{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        errors = data.get("errors", {})
        if errors:
            logger.error(f"API-Football errors: {errors}")
            return {"response": []}

        if use_cache:
            self._write_cache(key, data)

        remaining = resp.headers.get("x-ratelimit-requests-remaining", "?")
        logger.info(f"API call: /{endpoint} — {remaining} requests remaining today")
        return data

    # ------------------------------------------------------------------
    # Public methods
    # ------------------------------------------------------------------

    def get_daily_fixtures(self, date: str = None) -> List[Dict[str, Any]]:
        """Fetch fixtures for a date. Uses date-aware cache.

        First call of the day makes 1 API request, all subsequent calls
        that same day use cache. Next day the cache auto-expires so
        fresh fixtures are fetched. Costs only 1 API call per day.
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"Getting fixtures for {date}")

        try:
            # Try date-aware cache first
            cached = self._read_daily_cache(date)
            if cached is not None:
                raw_fixtures = cached.get("response", [])
            else:
                # Cache miss — make API call
                data = self._get("fixtures", {"date": date}, use_cache=False)
                raw_fixtures = data.get("response", [])
                # Store in date-aware cache (not the generic one)
                self._write_daily_cache(date, data)

            fixtures = []
            for f in raw_fixtures:
                league_id = f.get("league", {}).get("id", 0)
                if league_id not in TARGET_LEAGUE_IDS:
                    continue
                parsed = self._parse_fixture(f)
                if parsed:
                    fixtures.append(parsed)

            logger.info(f"{len(fixtures)} fixtures from target leagues (of {len(raw_fixtures)} total)")
            return fixtures

        except requests.exceptions.RequestException as e:
            logger.error(f"Network error fetching fixtures: {e}")
            return []
        except Exception as e:
            logger.error(f"Error fetching fixtures: {e}")
            return []

    def get_fixtures_by_league(self, league_id: int, date: str = None) -> List[Dict[str, Any]]:
        params: dict = {"league": league_id, "season": datetime.now().year}
        if date:
            params["date"] = date

        try:
            data = self._get("fixtures", params)
            return [p for f in data.get("response", []) if (p := self._parse_fixture(f))]
        except Exception as e:
            logger.error(f"Error fetching league {league_id} fixtures: {e}")
            return []

    def get_live_fixtures(self) -> List[Dict[str, Any]]:
        try:
            data = self._get("fixtures", {"live": "all"}, use_cache=False)
            fixtures = []
            for f in data.get("response", []):
                league_id = f.get("league", {}).get("id", 0)
                if league_id not in TARGET_LEAGUE_IDS:
                    continue
                parsed = self._parse_fixture(f)
                if parsed:
                    fixtures.append(parsed)
            return fixtures
        except Exception as e:
            logger.error(f"Error fetching live fixtures: {e}")
            return []

    def get_historical_matches(self, from_date: str, to_date: str, league_id: int = None) -> List[Dict[str, Any]]:
        params: dict = {"from": from_date, "to": to_date, "status": "FT"}
        if league_id:
            params["league"] = league_id
            params["season"] = int(from_date[:4])

        try:
            data = self._get("fixtures", params)
            results = [p for f in data.get("response", []) if (p := self._parse_fixture(f))]
            logger.info(f"Retrieved {len(results)} historical matches")
            return results
        except Exception as e:
            logger.error(f"Error fetching historical matches: {e}")
            return []

    def get_leagues(self) -> List[Dict[str, Any]]:
        try:
            data = self._get("leagues", {})
            return data.get("response", [])
        except Exception as e:
            logger.error(f"Error fetching leagues: {e}")
            return []

    def clear_cache(self) -> int:
        """Remove all cached responses. Returns number of files removed."""
        count = 0
        for f in self.cache_dir.glob("*.json"):
            f.unlink()
            count += 1
        logger.info(f"Cleared {count} cached API responses")
        return count

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_fixture(self, f: dict) -> Optional[Dict[str, Any]]:
        """Convert raw API-Football fixture to standardised format.

        Team names match the training CSV because both come from the
        same v3.football.api-sports.io endpoint.
        """
        try:
            fixture_info = f.get("fixture", {})
            teams = f.get("teams", {})
            league = f.get("league", {})
            goals = f.get("goals", {})
            score = f.get("score", {})

            raw_date = fixture_info.get("date", "")
            status_obj = fixture_info.get("status", {})
            status_short = status_obj.get("short", "")
            status_long = status_obj.get("long", "")

            home_score = goals.get("home")
            away_score = goals.get("away")
            ft = score.get("fulltime", {})
            if ft.get("home") is not None:
                home_score = ft["home"]
                away_score = ft["away"]

            return {
                "fixture_id": fixture_info.get("id", 0),
                "date": raw_date,
                "league_id": league.get("id", 0),
                "league_name": league.get("name", "Unknown"),
                "country_name": league.get("country", ""),
                "season": league.get("season", ""),
                "round": league.get("round", ""),
                "home_team_id": teams.get("home", {}).get("id", 0),
                "home_team": teams.get("home", {}).get("name", "Unknown"),
                "away_team_id": teams.get("away", {}).get("id", 0),
                "away_team": teams.get("away", {}).get("name", "Unknown"),
                "home_team_logo": teams.get("home", {}).get("logo", ""),
                "away_team_logo": teams.get("away", {}).get("logo", ""),
                "league_logo": league.get("logo", ""),
                "status": status_short,
                "status_long": status_long,
                "home_score": home_score,
                "away_score": away_score,
                "home_odds": 0.0,
                "draw_odds": 0.0,
                "away_odds": 0.0,
            }
        except Exception as e:
            logger.warning(f"Error parsing fixture: {e}")
            return None

    def test_connection(self) -> bool:
        try:
            data = self._get("status", {}, use_cache=False)
            resp = data.get("response", {})
            account = resp.get("account", {})
            req_info = resp.get("requests", {})
            current = req_info.get("current", "?")
            limit = req_info.get("limit_day", "?")
            logger.info(f"API-Football connected: {account.get('firstname', '')} ({current}/{limit} requests today)")
            print(f"Connected: {current}/{limit} API calls used today")
            return True
        except Exception as e:
            logger.error(f"API-Football connection test failed: {e}")
            return False


def get_apifootball_service() -> APIFootballService:
    return APIFootballService()


if __name__ == "__main__":
    service = get_apifootball_service()
    print("Testing API-Football connection...")

    if not service.test_connection():
        print("Connection failed - check API_FOOTBALL_API_KEY")
    else:
        fixtures = service.get_daily_fixtures()
        print(f"\nFound {len(fixtures)} fixtures for today")
        ns = [fx for fx in fixtures if fx.get("status") in ("NS", "TBD")]
        print(f"Not started: {len(ns)}")
        for fx in fixtures[:10]:
            print(f"  {fx['home_team']} vs {fx['away_team']} ({fx['league_name']}) [{fx['status']}]")
