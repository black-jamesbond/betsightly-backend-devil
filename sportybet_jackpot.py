"""
SportyBet jackpot fixtures ("Sporty 12").

The jackpot page (sportybet.com/ng/jackpot) is a client-rendered app: the twelve
fixtures arrive over XHR and are injected into the DOM, so fetching the page
server-side yields an empty shell with no matches in it. There is therefore no
HTML to parse here — the round's fixtures are kept as a snapshot in
``cache/sportybet_jackpot.json`` and served from there.

The snapshot below was read straight off SportyBet's own jackpot page (round
20260927) and holds: the round number, the prize, and each game's kick-off
(their site shows GMT+01:00, which is local time here) plus the 1 / X / 2 odds.

To roll it forward when a new round opens, replace the cache:

    python sportybet_jackpot.py show                 # what is cached now
    python sportybet_jackpot.py set new_round.json   # load a fresh snapshot

If SportyBet's jackpot XHR endpoint is ever pinned down, add it in
``refresh()`` — everything else (API route, UI, model picks) already works off
whatever this module returns.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

CACHE = Path(__file__).resolve().parent / "cache" / "sportybet_jackpot.json"


def load_jackpot() -> dict:
    """The cached jackpot round: {round, prize, games, fetched, source}."""
    try:
        data = json.loads(CACHE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"round": "", "prize": "", "games": [], "fetched": "",
                "source": "no snapshot cached"}
    data.setdefault("games", [])
    return data


def snapshot_freshness(payload: dict, now: datetime | None = None) -> dict:
    """Do not present an old manual jackpot round as today's bookable fixture list."""
    now = now or datetime.now(ZoneInfo("Africa/Lagos"))
    fetched = str(payload.get("fetched") or "").strip()
    if not fetched:
        return {"fresh": False, "reason": "snapshot has no retrieval time"}
    try:
        stamped = datetime.fromisoformat(fetched)
        if stamped.tzinfo is None:
            stamped = stamped.replace(tzinfo=ZoneInfo("Africa/Lagos"))
        age = (now.astimezone(timezone.utc) - stamped.astimezone(timezone.utc)).total_seconds()
    except (ValueError, TypeError):
        return {"fresh": False, "reason": "invalid snapshot timestamp"}
    if age < -3600 or age > 7 * 86400:
        return {"fresh": False, "reason": "jackpot snapshot is outdated"}
    if not isinstance(payload.get("games"), list) or len(payload["games"]) != 12:
        return {"fresh": False, "reason": "jackpot must contain exactly 12 fixtures"}
    return {"fresh": True, "reason": "", "age_hours": round(max(0, age) / 3600, 1)}


def games() -> list:
    """Just the fixtures, in jackpot order."""
    return load_jackpot().get("games", [])


def save_snapshot(payload: dict) -> Path:
    """Write a fresh snapshot (round, prize, games)."""
    payload = dict(payload)
    if not payload.get("round") or not isinstance(payload.get("games"), list) or len(payload["games"]) != 12:
        raise ValueError("A jackpot round must have an ID and exactly 12 games")
    seen = set()
    for game in payload["games"]:
        identity = (str(game.get("home") or "").strip(), str(game.get("away") or "").strip())
        if not all(identity) or identity in seen:
            raise ValueError("Jackpot game teams missing or duplicated")
        seen.add(identity)
        prices = game.get("odds") or []
        try:
            if len(prices) != 3 or any(not math.isfinite(float(x)) or float(x) <= 1.0 for x in prices):
                raise ValueError("Jackpot game requires three valid 1X2 prices")
        except (TypeError, ValueError):
            raise ValueError("Jackpot game requires three valid 1X2 prices")
    payload.setdefault("fetched", datetime.now(ZoneInfo("Africa/Lagos")).isoformat())
    payload.setdefault("source", "manually verified snapshot")
    from services.weekly_fixture_collector import _atomic_json
    _atomic_json(CACHE, payload)
    return CACHE


def refresh() -> dict:
    """Placeholder for automatic refresh.

    Deliberately does not invent an endpoint: guessing one would fail silently
    and look like "no games this week". Until the real XHR path is known the
    cached snapshot is the source of truth.
    """
    raise NotImplementedError(
        "SportyBet's jackpot fixtures load over XHR from their app bundle; the "
        "endpoint is not known yet, so refresh() is not wired up. Use "
        "save_snapshot() to load a new round instead."
    )


if __name__ == "__main__":
    import sys

    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "show":
        j = load_jackpot()
        print(f"Round {j.get('round') or '?'} · {j.get('prize') or '?'} "
              f"· {len(j.get('games', []))} games · fetched {j.get('fetched') or '?'}")
        for g in j.get("games", []):
            odds = g.get("odds") or []
            print(f"  {g.get('n'):>2}. {g.get('date','')} {g.get('time',''):<5} "
                  f"{g.get('home','')} v {g.get('away','')}"
                  + (f"   {odds[0]} / {odds[1]} / {odds[2]}" if len(odds) == 3 else ""))
    elif cmd == "set" and len(sys.argv) > 2:
        payload = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        print("saved:", save_snapshot(payload))
    else:
        print(__doc__)
