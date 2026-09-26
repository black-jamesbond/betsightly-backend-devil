"""
odds-api.io client — live pre-match odds.

Flow (per their docs):
  GET /v3/events?sport=football&apiKey=KEY        -> events (id, home, away, date)
  GET /v3/odds?eventId={id}&apiKey=KEY            -> bookmakers -> markets -> "ML"

We fetch events for a date, pull each event's odds, average the "ML"
(home/draw/away) decimal odds across bookmakers, de-vig them into fair
probabilities, and key the result by normalized (home, away) so the prediction
pipeline can match it to apifootball fixtures.

Activation: set the ODDS_API_IO_KEY environment variable to your key. With no
key set, every function is a safe no-op and the pipeline is unchanged.

    setx ODDS_API_IO_KEY "your_key_here"      (Windows, new shell after)
"""
from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.parse
from datetime import datetime
from pathlib import Path

from predict_today import normalize_name

BASE = "https://api.odds-api.io/v3"
CACHE = Path("cache/oddsapi")
KEY = os.environ.get("ODDS_API_IO_KEY", "").strip()


def enabled():
    return bool(KEY)


def _get(path, params, retries=3):
    params = {**params, "apiKey": KEY}
    url = f"{BASE}/{path}?{urllib.parse.urlencode(params)}"
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < retries - 1:
                time.sleep(2 ** attempt)  # exponential backoff on rate limit
                continue
            raise
    return None


def devig(oh, od, oa):
    """De-vig 1X2 decimal odds into fair probabilities."""
    try:
        rh, rd, ra = 1.0 / oh, 1.0 / od, 1.0 / oa
    except ZeroDivisionError:
        return None
    s = rh + rd + ra
    if s <= 0:
        return None
    return rh / s, rd / s, ra / s


def fetch_events(date_str=None):
    """List football events. Optionally filter to a date (YYYY-MM-DD)."""
    if not enabled():
        return []
    data = _get("events", {"sport": "football"})
    events = data if isinstance(data, list) else data.get("events", []) if data else []
    if date_str:
        events = [e for e in events if str(e.get("date", "")).startswith(date_str)]
    return events


def fetch_event_odds(event_id, bookmakers=None):
    """Odds for one event. `bookmakers` is an optional comma-separated string."""
    if not enabled():
        return None
    params = {"eventId": event_id}
    if bookmakers:
        params["bookmakers"] = bookmakers
    return _get("odds", params)


def consensus_1x2(odds_resp):
    """Average the ML (match-result) home/draw/away decimal odds across all
    bookmakers in the response, then de-vig into fair probabilities."""
    if not odds_resp:
        return None
    books = odds_resp.get("bookmakers", {}) if isinstance(odds_resp, dict) else {}
    hs, ds, as_ = [], [], []
    for _bk, markets in books.items():
        if not isinstance(markets, list):
            continue
        for m in markets:
            if str(m.get("name", "")).upper() in ("ML", "1X2", "MATCH RESULT"):
                try:
                    hs.append(float(m["odds"]["home"]))
                    ds.append(float(m["odds"]["draw"]))
                    as_.append(float(m["odds"]["away"]))
                except (KeyError, TypeError, ValueError):
                    continue
    if not hs:
        return None
    oh = sum(hs) / len(hs); od = sum(ds) / len(ds); oa = sum(as_) / len(as_)
    probs = devig(oh, od, oa)
    if not probs:
        return None
    return {"home": probs[0], "draw": probs[1], "away": probs[2],
            "odds": {"home": round(oh, 2), "draw": round(od, 2), "away": round(oa, 2)},
            "n_books": len(hs)}


def build_odds_map(date_str, max_events=200, bookmakers=None, use_cache=True):
    """Return {(home_norm, away_norm): consensus_1x2_dict} for a date.

    Cached to cache/oddsapi/odds_{date}.json so a day is fetched once."""
    if not enabled():
        return {}
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"odds_{date_str}.json"
    if use_cache and cache_file.exists():
        raw = json.load(open(cache_file, encoding="utf-8"))
        return {tuple(k.split("|")): v for k, v in raw.items()}

    events = fetch_events(date_str)[:max_events]
    out = {}
    for e in events:
        eid = e.get("id")
        home, away = e.get("home"), e.get("away")
        if not (eid and home and away):
            continue
        cons = consensus_1x2(fetch_event_odds(eid, bookmakers))
        if cons:
            out[(normalize_name(home), normalize_name(away))] = cons

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump({"|".join(k): v for k, v in out.items()}, f, indent=2)
    return out


def apply_market(pred, entry):
    """Overwrite a prediction's 1X2 with de-vigged market probabilities."""
    out = dict(pred)
    h, d, a = entry["home"], entry["draw"], entry["away"]
    out["home_win"] = round(h * 100, 1)
    out["draw"] = round(d * 100, 1)
    out["away_win"] = round(a * 100, 1)
    best = max((("Home Win", h), ("Draw", d), ("Away Win", a)), key=lambda t: t[1])
    out["prediction"] = best[0]
    out["confidence"] = round(best[1] * 100, 1)
    out["engine"] = "market"
    out["market_odds"] = entry["odds"]
    return out


if __name__ == "__main__":
    # Self-test: de-vig math (works without a key), plus a live probe if key set.
    p = devig(2.10, 3.40, 3.60)
    print("de-vig(2.10, 3.40, 3.60) ->", tuple(round(x, 3) for x in p),
          "sum", round(sum(p), 6))
    print("ODDS_API_IO_KEY set:", enabled())
    if enabled():
        today = datetime.now().strftime("%Y-%m-%d")
        evs = fetch_events(today)
        print(f"events today: {len(evs)}")
        if evs:
            print("sample:", {k: evs[0].get(k) for k in ("id", "home", "away", "date")})
