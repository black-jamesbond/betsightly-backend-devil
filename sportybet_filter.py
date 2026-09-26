"""
SportyBet game filter.

SportyBet (Nigeria) exposes the actual games it offers (team pairs) via its
public API. We fetch those and keep only the predictions whose match SportyBet
also lists — so the dashboard hides games you can't bet there. Matching is on
team names (normalized the same way as the prediction engine), which is far more
precise than league-name matching.

Two things this module is careful about, because both used to report a
confident, wrong "0 games on SportyBet":

1. SportyBet rejects requests that don't look like a browser. A bare
   ``User-Agent: Mozilla/5.0`` gets HTTP 403, so the headers below are what the
   site's own front-end sends. Without them every fetch failed silently.

2. A list that cannot be refreshed must never be silently substituted with an
   old one. Yesterday's list has different fixtures, so matching today's games
   against it marks real SportyBet games as unavailable — the error looks like
   a clean answer. ``load_offered_pairs`` therefore raises once the cache is
   too old to trust, and callers fail open (show everything) instead.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import difflib
from datetime import datetime, timezone
from pathlib import Path

from predict_today import normalize_name

log = logging.getLogger(__name__)

CACHE = Path(__file__).resolve().parent / "cache" / "sportybet_events.json"

BASE_URL = "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents"
QUERY = ("sportId=sr%3Asport%3A1&marketId=1&pageSize=100"
         "&pageNum={pg}&option=1&timeline=72")

# The site's own front-end headers. The Accept header is the one that matters:
# without it SportyBet answers 403 no matter what user-agent is sent.
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/122.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.sportybet.com/ng/sport/football",
}

MAX_PAGES = 12

# How long a cached list stays usable when the live fetch fails. The feed covers
# a rolling 72h window, so a few hours old is still correct; a day old is not
# (fixtures roll off and new ones appear).
CACHE_MAX_STALE_MIN = 180

# What the last load_offered_pairs() call actually used, for the dashboard to
# surface. status is "live", "stale" or "unavailable".
_SOURCE = {"status": "unknown", "games": 0, "age_min": None, "detail": ""}


class SportyBetUnavailable(RuntimeError):
    """SportyBet's game list could not be fetched, and no cache is trustworthy."""


def network_hint(exc) -> str:
    """Extra explanation when the failure is 'this process has no network'.

    WinError 10061 (connection refused) and friends surface as a bare URLError,
    which reads like a SportyBet outage. It usually means the process that is
    *running the server* was started somewhere without outbound network — a
    sandboxed/restricted shell, for instance — so the fix is to restart the
    dashboard normally.
    """
    text = str(exc)
    lowered = text.lower()
    looks_offline = (
        isinstance(exc, (urllib.error.URLError, ConnectionError, TimeoutError))
        or "10061" in text
        or "actively refused" in lowered
        or "timed out" in lowered
        or "getaddrinfo" in lowered
    )
    if looks_offline:
        return (" -- this process has no outbound network access (WinError 10061 "
                "means the connection was refused before leaving the machine). "
                "Start the dashboard from a normal terminal, not a restricted one.")
    return ""


def source_info() -> dict:
    """Details about the list behind the last load_offered_pairs() call."""
    return dict(_SOURCE)


def fetch_sportybet_events(save: bool = True) -> list:
    """Fetch SportyBet's offered soccer games (next 72h).

    Returns a list of {home, away, date}. Raises SportyBetUnavailable rather
    than returning an empty list, because an empty list is indistinguishable
    from "SportyBet is offering nothing" and callers act on that.
    """
    events: list = []
    pages_ok = 0

    for pg in range(1, MAX_PAGES + 1):
        url = f"{BASE_URL}?{QUERY.format(pg=pg)}"
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=25) as resp:
                payload = json.loads(resp.read().decode("utf-8", "replace"))
        except Exception as exc:
            if pages_ok == 0:
                raise SportyBetUnavailable(
                    f"request failed on page 1: {type(exc).__name__}: {exc}"
                    + network_hint(exc)) from exc
            # Later pages are best-effort: a partial list still beats nothing.
            log.warning("SportyBet page %s failed (%s); keeping the %s games "
                        "already fetched", pg, exc, len(events))
            break

        data = payload.get("data") or {}
        tournaments = data.get("tournaments") or []
        if not tournaments:
            break
        pages_ok += 1

        for t in tournaments:
            for e in t.get("events", []):
                start = e.get("estimateStartTime")
                date = ""
                if start:
                    try:
                        date = datetime.fromtimestamp(
                            int(start) / 1000, timezone.utc).strftime("%Y-%m-%d")
                    except (TypeError, ValueError, OSError):
                        pass
                events.append({
                    "home": e.get("homeTeamName", ""),
                    "away": e.get("awayTeamName", ""),
                    "date": date,
                    # Kept as well as the date so callers can group by *local*
                    # day: a 23:30 UTC kick-off is the next day for a UTC+1
                    # viewer, and the date string alone would put it in the
                    # wrong evening.
                    "start_ms": start,
                    # Everything the prediction pipeline needs is in the same
                    # payload, so the games we predict and the games you can bet
                    # on are the same list — not two feeds that have to agree.
                    "event_id": e.get("eventId", ""),
                    "league": t.get("name", ""),
                    "country": t.get("categoryName", ""),
                    "status": e.get("matchStatus", ""),
                    "score": e.get("setScore", ""),
                })

    if not events:
        raise SportyBetUnavailable("SportyBet returned no games")

    if save:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = CACHE.with_name(CACHE.name + ".tmp")
        tmp.write_text(json.dumps(events), encoding="utf-8")
        tmp.replace(CACHE)          # atomic, so a reader never sees a half-file

    log.info("SportyBet: fetched %s games over %s page(s)", len(events), pages_ok)
    _SOURCE.update(status="live", games=len(events), age_min=0.0, detail="")
    return events


def _cache_age_min():
    """Minutes since the cache was written, or None if there is no cache."""
    if not CACHE.exists():
        return None
    return (time.time() - CACHE.stat().st_mtime) / 60


def _read_cache() -> list:
    try:
        return json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SportyBetUnavailable(f"cached SportyBet list is unreadable: {exc}") from exc


def load_offered_pairs(max_age_min: int = 60,
                       max_stale_min: int = CACHE_MAX_STALE_MIN):
    """Return the normalized (pairs, teams) SportyBet is offering.

    Uses the cache while it is younger than ``max_age_min``, otherwise refetches.
    If the refetch fails the cache is used only while it is younger than
    ``max_stale_min``; past that this raises SportyBetUnavailable so the caller
    can fail open instead of hiding games that are genuinely offered.
    """
    age_min = _cache_age_min()
    status = "live"
    detail = ""

    if age_min is None or age_min > max_age_min:
        try:
            fetch_sportybet_events()
            age_min = 0.0
        except SportyBetUnavailable as exc:
            detail = str(exc)
            if age_min is None:
                _SOURCE.update(status="unavailable", games=0, age_min=None, detail=detail)
                raise
            if age_min > max_stale_min:
                hours = age_min / 60
                _SOURCE.update(status="unavailable", games=0, age_min=age_min, detail=detail)
                raise SportyBetUnavailable(
                    f"SportyBet list is {hours:.1f}h old and could not be refreshed "
                    f"({detail})") from exc
            log.warning("SportyBet unreachable (%s); falling back to the %.0f-minute-old "
                        "cache", detail, age_min)
            status = "stale"

    events = _read_cache()
    pairs, teams, unusable = set(), set(), 0
    for e in events:
        h, a = normalize_name(e.get("home", "")), normalize_name(e.get("away", ""))
        if h and a:
            pairs.add((h, a))
            teams.add(h)
            teams.add(a)
        else:
            unusable += 1

    _SOURCE.update(status=status, games=len(events), age_min=age_min, detail=detail)
    log.info("SportyBet: %s games from a %s source", len(events), status)
    if unusable:
        log.debug("SportyBet: %s cached entries had no usable team names", unusable)
    return pairs, teams


def is_offered(home: str, away: str, pairs, teams) -> bool:
    """True if SportyBet offers this fixture. See match_kind()."""
    return match_kind(home, away, pairs, teams) is not None


# Same team, different spelling. Club words (FC, SC, CD...) are already stripped
# by normalize_name(), so this only has to absorb suffix and word-order
# variants such as "manchester united w" vs "manchester united women".
_FUZZY_MIN_RATIO = 0.85
_FUZZY_MIN_LEN = 6


def _close(x: str, y: str) -> bool:
    if not x or not y:
        return False
    if x == y:
        return True
    if min(len(x), len(y)) < _FUZZY_MIN_LEN:
        return False
    return difflib.SequenceMatcher(None, x, y).ratio() >= _FUZZY_MIN_RATIO


def match_kind(home: str, away: str, pairs, teams):
    """How SportyBet offers this fixture: "exact", "fuzzy" or None.

    The fuzzy step requires a *pair* of offered teams to both be close to ours.
    The old rule only asked whether each team name appeared somewhere in the
    list, which flagged "Arsenal v Chelsea" as bettable whenever SportyBet
    happened to list an Arsenal and a Chelsea in unrelated games.
    """
    h, a = normalize_name(home), normalize_name(away)
    if (h, a) in pairs or (a, h) in pairs:
        return "exact"
    if not pairs or (h not in teams and a not in teams):
        return None                     # cheap exit: neither name appears at all
    for h2, a2 in pairs:
        if _close(h, h2) and _close(a, a2):
            return "fuzzy"
        if _close(h, a2) and _close(a, h2):
            return "fuzzy"
    return None


if __name__ == "__main__":
    import sys
    from collections import Counter

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    date = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")

    try:
        ev = fetch_sportybet_events()
        pairs, teams = load_offered_pairs(max_age_min=0)
    except SportyBetUnavailable as exc:
        print(f"FAILED: {exc}")
        sys.exit(1)

    today = sum(1 for e in ev if e["date"] == date)
    print(f"SportyBet offering {len(ev)} games over the next 72h ({today} of them on {date})")

    preds_path = Path(f"cache/apifootball/predictions_{date}.json")
    if not preds_path.exists():
        print(f"no predictions cached for {date}; nothing to match against")
        sys.exit(0)

    preds = json.loads(preds_path.read_text(encoding="utf-8"))
    matched = [(p, match_kind(p["home_team"], p["away_team"], pairs, teams)) for p in preds]
    kept = [p for p, kind in matched if kind]
    fuzzy = [(p, kind) for p, kind in matched if kind == "fuzzy"]
    print(f"\n{date}: {len(preds)} predictions -> {len(kept)} on SportyBet "
          f"({100 * len(kept) // max(1, len(preds))}%): "
          f"{len(kept) - len(fuzzy)} exact, {len(fuzzy)} name-fuzzy\n")

    for lg, n in Counter(f"{p.get('country')} - {p.get('league')}"
                         for p in kept).most_common(15):
        print(f"  {n:>3}  {lg}")

    if fuzzy:
        print("\nname-fuzzy matches (same fixture, different spelling):")
        for p, _ in fuzzy:
            print(f"   {p['home_team']} v {p['away_team']}")

    missing = today - len(kept)
    if missing > 0:
        print(f"\nnote: SportyBet lists {today} games today, {missing} of which are "
              f"not in our fixture list")
