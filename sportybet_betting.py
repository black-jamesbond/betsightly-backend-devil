"""
SportyBet real prices and booking codes.

Two jobs, both against SportyBet's own API:

1. Real odds. ``fetch_odds_table()`` reads the Over/Under 1.5 and 2.5 lines
   SportyBet is actually offering, so accumulator tickets can be priced at the
   odds you would really get instead of the model's fair odds (1/probability).
   Fair odds are systematically longer than book odds, so a "~5.0" ticket built
   on fair odds is not a 5.0 ticket at the counter.

2. Booking codes. ``create_booking_code()`` posts a ticket's selections to
   SportyBet's share endpoint, which returns a code you can type into their site
   to load the whole slip. Creating a code is anonymous (no login) and the same
   selections always return the same code, so they are cached locally.

3. Ids that outlive the board. A fixture disappears from the upcoming feed the
   moment it kicks off, which used to mean a ticket that had started could no
   longer be given a code at all. Two things fix that:

   * ``load_live_events()`` reads the *live* board, which lists in-play games
     with the same ``eventId`` they had before kick-off.
   * ``record_ids()`` remembers every id we have ever priced in
     ``cache/sportybet_ids.json``, so a game that has since finished can still
     be resolved from our own ledger.

   SportyBet's share endpoint happily stores selections for games in play, so
   with the ids in hand the code comes back as normal.

Selections need SportyBet's own identifiers, not ours: eventId
("sr:match:..."), marketId ("18" = Over/Under), specifier ("total=1.5") and
outcomeId ("12" = Over, "13" = Under). Those only exist in their feed, which is
why fixtures are matched on team names and the ids are carried alongside.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from sportybet_filter import (HEADERS, MAX_PAGES, SportyBetUnavailable,
                              _close, network_hint, normalize_name)

log = logging.getLogger(__name__)

ODDS_URL = "https://www.sportybet.com/api/ng/factsCenter/pcUpcomingEvents"
ODDS_QUERY = ("sportId=sr%3Asport%3A1&marketId=18&pageSize=100"
              "&pageNum={pg}&option=1&timeline=72")
LIVE_URL = "https://www.sportybet.com/api/ng/factsCenter/liveOrPrematchEvents"
LIVE_QUERY = ("sportId=sr%3Asport%3A1&marketId=18&pageSize=100"
              "&pageNum={pg}&option=1")
SHARE_URL = "https://www.sportybet.com/api/ng/orders/share"

# The share endpoint is a POST and needs a JSON content type + Origin: without
# them SportyBet answers "Sorry, something went wrong" instead of a code.
POST_HEADERS = {**HEADERS,
                "Content-Type": "application/json",
                "Origin": "https://www.sportybet.com"}

ODDS_CACHE = Path(__file__).resolve().parent / "cache" / "sportybet_odds.json"
LIVE_CACHE = Path(__file__).resolve().parent / "cache" / "sportybet_live.json"
ID_LEDGER = Path(__file__).resolve().parent / "cache" / "sportybet_ids.json"
CODE_CACHE = Path(__file__).resolve().parent / "cache" / "sportybet_codes.json"

# The Over/Under market id, and the two lines our tickets are allowed to use.
OU_MARKET_ID = "18"
LINES = {"over_1.5": "total=1.5", "over_2.5": "total=2.5"}

# Our leg labels -> the price key they were priced from. Used to pull ids back
# out of an event record for a leg that was built before the game kicked off.
LINE_FIELDS = {"Over 1.5": "over_1.5", "Over 2.5": "over_2.5"}

# The live board is polled often (ids vanish as games finish); the ledger is our
# own memory of what was on the board, so it only has to be pruned for size.
LIVE_REFRESH_MIN = 5
LIVE_MAX_STALE_MIN = 90
LEDGER_KEEP_DAYS = 4
LIVE_MAX_PAGES = 3

_SOURCE = {"status": "unknown", "events": 0, "age_min": None, "detail": ""}


def source_info() -> dict:
    return dict(_SOURCE)


def _get_json(url):
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=25) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _line_prices(markets):
    """Pull Over/Under 1.5 and 2.5 out of an event's market list."""
    out = {}
    for m in markets or []:
        spec = m.get("specifier")
        if m.get("id") != OU_MARKET_ID or spec not in LINES.values():
            continue
        key = next(k for k, v in LINES.items() if v == spec)
        over = under = None
        for o in m.get("outcomes") or []:
            desc = (o.get("desc") or "").lower()
            if desc.startswith("over"):
                over = o
            elif desc.startswith("under"):
                under = o
        if not over:
            continue
        out[key] = {
            "market_id": m.get("id"),
            "specifier": spec,
            "outcome_id": str(over.get("id")),
            "odds": float(over.get("odds") or 0),
            "under_outcome_id": str(under.get("id")) if under else None,
            "under_odds": float(under.get("odds") or 0) if under else None,
        }
    return out


def _parse_tournaments(tournaments, live: bool) -> list:
    """Events with Over/Under prices, flattened out of a tournament list."""
    out = []
    for t in tournaments or []:
        for e in t.get("events") or []:
            prices = _line_prices(e.get("markets"))
            if not prices or not e.get("eventId"):
                continue
            out.append({
                "event_id": e.get("eventId"),
                "home": e.get("homeTeamName", ""),
                "away": e.get("awayTeamName", ""),
                "start_ms": e.get("estimateStartTime"),
                "status": e.get("matchStatus", ""),
                "booking": e.get("bookingStatus", ""),
                "prices": prices,
                "live": bool(live),
                "score": e.get("setScore", "") or "",
                "tournament": t.get("name", ""),
            })
    return out


def fetch_odds_table(save: bool = True) -> list:
    """Every upcoming event with its Over 1.5 / 2.5 prices."""
    events, pages_ok = [], 0
    for pg in range(1, MAX_PAGES + 1):
        try:
            payload = _get_json(f"{ODDS_URL}?{ODDS_QUERY.format(pg=pg)}")
        except Exception as exc:
            if pages_ok == 0:
                raise SportyBetUnavailable(
                    f"odds request failed on page 1: {type(exc).__name__}: {exc}"
                    + network_hint(exc)) from exc
            log.warning("SportyBet odds page %s failed (%s); keeping %s events",
                        pg, exc, len(events))
            break
        tournaments = (payload.get("data") or {}).get("tournaments") or []
        if not tournaments:
            break
        pages_ok += 1
        events.extend(_parse_tournaments(tournaments, live=False))

    if not events:
        raise SportyBetUnavailable("SportyBet returned no Over/Under prices")

    if save:
        ODDS_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = ODDS_CACHE.with_name(ODDS_CACHE.name + ".tmp")
        tmp.write_text(json.dumps(events), encoding="utf-8")
        tmp.replace(ODDS_CACHE)
        record_ids(events)

    priced = sum(1 for e in events if "over_1.5" in e["prices"])
    log.info("SportyBet: %s events with Over/Under prices (%s with Over 1.5)",
             len(events), priced)
    _SOURCE.update(status="live", events=len(events), age_min=0.0, detail="")
    return events


def fetch_live_events(save: bool = True) -> list:
    """In-play events, with the same identifiers they had before kick-off.

    The payload is a flat list of tournaments (not the ``{"tournaments": ...}``
    shape the upcoming feed uses), and it repeats itself across pages, so events
    are de-duplicated by id.
    """
    events, seen, pages_ok = [], set(), 0
    for pg in range(1, LIVE_MAX_PAGES + 1):
        try:
            payload = _get_json(f"{LIVE_URL}?{LIVE_QUERY.format(pg=pg)}")
        except Exception as exc:
            if pages_ok == 0:
                raise SportyBetUnavailable(
                    f"live request failed on page 1: {type(exc).__name__}: {exc}"
                    + network_hint(exc)) from exc
            log.warning("SportyBet live page %s failed (%s); keeping %s events",
                        pg, exc, len(events))
            break
        data = payload.get("data") or []
        tournaments = data if isinstance(data, list) else (data.get("tournaments") or [])
        if not tournaments:
            break
        pages_ok += 1
        gained = 0
        for row in _parse_tournaments(tournaments, live=True):
            if row["event_id"] in seen:
                continue
            seen.add(row["event_id"])
            events.append(row)
            gained += 1
        if not gained:              # paging repeats the same list; stop early
            break

    if save:
        LIVE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        tmp = LIVE_CACHE.with_name(LIVE_CACHE.name + ".tmp")
        tmp.write_text(json.dumps(events), encoding="utf-8")
        tmp.replace(LIVE_CACHE)
        record_ids(events)
    return events


def load_live_events(max_age_min: int = LIVE_REFRESH_MIN,
                     max_stale_min: int = LIVE_MAX_STALE_MIN) -> list:
    """Cached live board, refreshed every few minutes."""
    age = _cache_age_min(LIVE_CACHE)
    if age is None or age > max_age_min:
        try:
            return fetch_live_events()
        except SportyBetUnavailable as exc:
            if age is None:
                raise
            if age > max_stale_min:
                raise
            log.warning("SportyBet live board unreachable (%s); using the "
                        "%.0f-minute-old one", exc, age)
    try:
        return json.loads(LIVE_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SportyBetUnavailable(f"cached SportyBet live board unreadable: {exc}")


def live_events_safe(max_age_min: int = LIVE_REFRESH_MIN) -> list:
    """load_live_events() that degrades to the cache and then to nothing."""
    try:
        return load_live_events(max_age_min=max_age_min)
    except SportyBetUnavailable as exc:
        log.warning("SportyBet live board unavailable: %s", exc)
        return []


# --- our own record of the ids we have seen ---------------------------------
# The upcoming feed drops a fixture as soon as it kicks off, and the live feed
# drops it again when it finishes, so a ticket built this morning would lose the
# ids it needs to be re-booked by the evening. Everything we price is therefore
# written to a ledger, keyed by the fixture's normalized team names.

def _ledger_key(home: str, away: str) -> str:
    return f"{normalize_name(home)}|{normalize_name(away)}"


def load_id_ledger() -> dict:
    try:
        data = json.loads(ID_LEDGER.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def record_ids(events, day: str | None = None) -> int:
    """Remember each event's identifiers so they survive kick-off.

    Returns the number of ledger entries changed (0 means nothing to write).
    """
    if not events:
        return 0
    day = day or datetime.now().strftime("%Y-%m-%d")
    ledger = load_id_ledger()
    changed = 0
    for e in events:
        key = _ledger_key(e.get("home", ""), e.get("away", ""))
        if key == "|" or not e.get("event_id"):
            continue
        row = {"event_id": e["event_id"], "home": e.get("home", ""),
               "away": e.get("away", ""), "prices": e.get("prices") or {},
               "start_ms": e.get("start_ms"), "saved": day}
        if ledger.get(key) != row:
            ledger[key] = row
            changed += 1

    # Keep the file small: ids are only useful for tickets from the last few days.
    cutoff = (datetime.now() - timedelta(days=LEDGER_KEEP_DAYS)).strftime("%Y-%m-%d")
    for key in [k for k, v in ledger.items() if str(v.get("saved", "")) < cutoff]:
        del ledger[key]
        changed += 1

    if changed:
        try:
            ID_LEDGER.parent.mkdir(parents=True, exist_ok=True)
            tmp = ID_LEDGER.with_name(ID_LEDGER.name + ".tmp")
            tmp.write_text(json.dumps(ledger), encoding="utf-8")
            tmp.replace(ID_LEDGER)
        except OSError as exc:
            log.warning("could not persist the SportyBet id ledger: %s", exc)
            return 0
    return changed


def ledger_events() -> list:
    """The ledger in the same shape as a feed event, for id lookups."""
    out = []
    for v in load_id_ledger().values():
        if not v.get("event_id") or not v.get("prices"):
            continue
        out.append({"event_id": v["event_id"], "home": v.get("home", ""),
                    "away": v.get("away", ""), "prices": v["prices"],
                    "start_ms": v.get("start_ms"), "status": "", "booking": "",
                    "live": False, "score": "", "tournament": ""})
    return out


_ID_INDEX_CACHE = {"at": 0.0, "index": None}
ID_INDEX_TTL_S = 30


def load_id_index(board: list | None = None) -> dict:
    """Index covering every fixture we can still produce a booking code for.

    The upcoming board is the authority (exact names, current prices); the live
    board adds games in play; the ledger adds games we have already seen and
    priced, even after they have left both feeds. Sources are merged in that
    order, and ``build_index`` keeps the first record per fixture.

    Callers that pass their own freshly loaded ``board`` get a one-off index;
    the default form is memoised briefly, because the dashboard asks for it on
    every predictions poll and the ledger has to be read from disk each time.
    """
    if board is None:
        cached = _ID_INDEX_CACHE["index"]
        if cached is not None and (time.time() - _ID_INDEX_CACHE["at"]) < ID_INDEX_TTL_S:
            return cached
    events = list(board or [])
    if board is None:
        try:
            events.extend(load_odds_table())
        except SportyBetUnavailable as exc:
            log.warning("SportyBet odds unavailable for id lookups: %s", exc)
    events.extend(live_events_safe())
    events.extend(ledger_events())
    index = build_index(events)
    if board is None:
        _ID_INDEX_CACHE.update(at=time.time(), index=index)
    return index


def snapshot_ids() -> dict:
    """Read the board and the live board, remembering their identifiers.

    Called whenever the day's fixtures are (re)generated, so the ledger is
    filled while the games are still listed — the point of the ledger is to
    already hold an id by the time a fixture kicks off and disappears.
    """
    out = {"board": 0, "live": 0, "ledger": 0}
    try:
        out["board"] = len(load_odds_table())
    except SportyBetUnavailable as exc:
        log.warning("SportyBet board unavailable while snapshotting ids: %s", exc)
    try:
        out["live"] = len(load_live_events())
    except SportyBetUnavailable as exc:
        log.warning("SportyBet live board unavailable while snapshotting ids: %s", exc)
    out["ledger"] = len(load_id_ledger())
    return out


def selection_ids(event: dict | None, field: str) -> dict | None:
    """{event_id, market_id, specifier, outcome_id, odds} for one Over/Under line."""
    if not event:
        return None
    price = (event.get("prices") or {}).get(field)
    if not price:
        return None
    return {"event_id": event["event_id"], "market_id": price["market_id"],
            "specifier": price["specifier"], "outcome_id": price["outcome_id"],
            "odds": price["odds"]}


def _cache_age_min(path: Path):
    if not path.exists():
        return None
    return (time.time() - path.stat().st_mtime) / 60


def load_odds_table(max_age_min: int = 60,
                    max_stale_min: int = 180) -> list:
    """Cached odds table, refreshed when older than ``max_age_min``.

    Same rule as the fixture list: a stale table is used only within
    ``max_stale_min``, otherwise this raises so callers can fall back to fair
    odds instead of pricing tickets against hours-old lines.
    """
    age = _cache_age_min(ODDS_CACHE)
    status, detail = "live", ""
    if age is None or age > max_age_min:
        try:
            fetch_odds_table()
            age = 0.0
        except SportyBetUnavailable as exc:
            detail = str(exc)
            if age is None:
                _SOURCE.update(status="unavailable", detail=detail)
                raise
            if age > max_stale_min:
                _SOURCE.update(status="unavailable", detail=detail, age_min=age)
                raise SportyBetUnavailable(
                    f"SportyBet odds are {age / 60:.1f}h old and could not be "
                    f"refreshed ({detail})") from exc
            log.warning("SportyBet odds unreachable (%s); using the %.0f-minute-old "
                        "table", detail, age)
            status = "stale"
    try:
        events = json.loads(ODDS_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        _SOURCE.update(status="unavailable", detail=str(exc))
        raise SportyBetUnavailable(f"cached SportyBet odds unreadable: {exc}") from exc
    # Every look at the board is a chance to remember the fixtures on it: the
    # last read before kick-off is what lets the ticket be booked afterwards.
    record_ids(events)
    _SOURCE.update(status=status, events=len(events), age_min=age, detail=detail)
    return events


def build_index(events) -> dict:
    """{normalized (home, away): event} for exact lookups."""
    idx = {}
    for e in events:
        h, a = normalize_name(e.get("home", "")), normalize_name(e.get("away", ""))
        if h and a:
            idx.setdefault((h, a), e)
    return idx


def find_event(index, home: str, away: str):
    """(event, "exact"|"fuzzy") for a fixture, or (None, None).

    Fuzzy is the same pair-level rule the SportyBet fixture filter uses, so a
    "UE Sant Andreu" / "Sant Andreu" spelling difference doesn't cost the price.
    """
    h, a = normalize_name(home), normalize_name(away)
    if (h, a) in index:
        return index[(h, a)], "exact"
    if (a, h) in index:
        return index[(a, h)], "exact"
    for (h2, a2), e in index.items():
        if _close(h, h2) and _close(a, a2):
            return e, "fuzzy"
    return None, None


def create_booking_code(selections: list) -> dict:
    """Create (or fetch) the SportyBet booking code for a set of selections.

    ``selections`` is a list of {event_id, market_id, specifier, outcome_id}.
    Returns {share_code, share_url, deadline}. Selections must be in SportyBet's
    own identifiers — see the module docstring.
    """
    if not selections:
        raise SportyBetUnavailable("no selections to book")

    payload = {"selections": [
        {"eventId": s["event_id"], "marketId": str(s["market_id"]),
         "specifier": s.get("specifier", ""), "outcomeId": str(s["outcome_id"])}
        for s in selections
    ]}
    body = json.dumps(payload).encode()
    req = urllib.request.Request(SHARE_URL, headers=POST_HEADERS, data=body,
                                 method="POST")
    try:
        with urllib.request.urlopen(req, timeout=25) as r:
            resp = json.loads(r.read().decode("utf-8", "replace"))
    except urllib.error.HTTPError as exc:
        raise SportyBetUnavailable(
            f"booking request failed: HTTP {exc.code}" + network_hint(exc)) from exc
    except Exception as exc:
        raise SportyBetUnavailable(
            f"booking request failed: {type(exc).__name__}: {exc}"
            + network_hint(exc)) from exc

    data = resp.get("data") or {}
    code = data.get("shareCode")
    if resp.get("bizCode") != 10000 or not code:
        raise SportyBetUnavailable(
            f"SportyBet rejected the slip: {resp.get('message') or resp}")
    return {
        "share_code": code,
        "share_url": data.get("shareURL") or f"https://www.sportybet.com/ng/?shareCode={code}",
        "deadline": data.get("deadline"),
    }


def fingerprint(selections) -> str:
    """Stable id for a set of selections, used to detect a changed ticket."""
    parts = sorted(f"{s['event_id']}|{s['market_id']}|{s.get('specifier','')}|{s['outcome_id']}"
                   for s in selections)
    return "&".join(parts)


def codes_cache() -> dict:
    try:
        return json.loads(CODE_CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def booking_code_for(selections: list, use_cache: bool = True) -> dict:
    """create_booking_code() with a local cache, so repeat lookups are free."""
    key = fingerprint(selections)
    cache = codes_cache() if use_cache else {}
    if key in cache:
        return {**cache[key], "cached": True}

    result = create_booking_code(selections)
    cache[key] = result
    CODE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    try:
        CODE_CACHE.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    except OSError as exc:
        log.warning("could not persist booking code: %s", exc)
    return {**result, "cached": False}


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    table = fetch_odds_table()
    print(f"\n{len(table)} events with Over/Under prices")
    priced = [e for e in table if "over_1.5" in e["prices"]]
    print(f"{len(priced)} offer Over 1.5\n")
    for e in priced[:8]:
        p15, p25 = e["prices"].get("over_1.5"), e["prices"].get("over_2.5")
        print(f"  {e['home']} v {e['away']}")
        if p15:
            print(f"     O1.5 {p15['odds']:<6} (id {p15['outcome_id']})  "
                  f"U1.5 {p15['under_odds']}")
        if p25:
            print(f"     O2.5 {p25['odds']:<6} (id {p25['outcome_id']})")

    if "--book" in sys.argv:
        sample = [{"event_id": priced[0]["event_id"], "market_id": OU_MARKET_ID,
                   "specifier": "total=1.5",
                   "outcome_id": priced[0]["prices"]["over_1.5"]["outcome_id"]}]
        print("\ntest booking:", booking_code_for(sample, use_cache=False))
