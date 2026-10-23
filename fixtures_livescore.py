"""
Free fallback fixture source — LiveScore public CDN.

The primary feed (apiv3.apifootball.com) stopped serving fixtures because the
account is unpaid: every request now returns
``{"error": 404, "message": "Please make the payment for your account!"}``.
The second configured provider (api-sports.io) is suspended too, so
``predict_today.py``, ``bot_system.py`` and ``dashboard/app.py`` all reported
"no fixtures" from late August 2026 onward.

This module pulls the same day's schedule from LiveScore's public scoreboard
feed and maps it into the apifootball event schema, so the rest of the pipeline
is unchanged. It is a drop-in fallback, not a replacement: the moment the paid
feed works again, the primary path is used and this is never called.

Schema produced per match (exactly the keys the pipeline reads):

    match_id, match_date, match_time, match_status,
    match_hometeam_name, match_awayteam_name,
    match_hometeam_score, match_awayteam_score,
    match_hometeam_ft_score, match_awayteam_ft_score,
    league_name, country_name, source

Times are UTC ("YYYY-MM-DD" / "HH:MM"), matching what apifootball returns.

Usage:
    from fixtures_livescore import fetch_day, write_events_cache
    matches = fetch_day("2026-09-23")
    write_events_cache("2026-09-23", matches)
"""
from __future__ import annotations

import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

CDN = ("https://prod-cdn-public-api.livescore.com/v1/api/app/date/"
       "soccer/{ymd}/0")
CACHE_DIR = Path("cache/livescore")

_HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/122.0 Safari/537.36"),
    "Accept": "application/json",
}

# LiveScore status -> the wording the rest of the project expects.
_CANCELLED = {"Canc.", "Postp.", "Abandoned", "Susp.", "Awarded"}
_FINISHED = {"FT", "AET", "Pen.", "FT Pen.", "After ET", "Finished"}


def _map_status(eps: str) -> str:
    """Map a LiveScore status code to apifootball-style wording."""
    eps = (eps or "").strip()
    if eps in _CANCELLED:
        return "Cancelled"
    if eps in _FINISHED:
        return "Finished"
    if eps and eps != "NS":
        return "In Play"          # e.g. "84'", "45+3'", "HT"
    return "Not Started"


def status_and_minute(eps: str) -> tuple[str, str]:
    """(status, clock) for a LiveScore code.

    Keeps the clock the feed gives us — "84'", "45+3'", "HT" — which is the
    live detail worth showing next to a score. Returns ("", "") for not started.
    """
    eps = (eps or "").strip()
    if eps in _FINISHED:
        return "Finished", ""
    if eps in _CANCELLED:
        return "Cancelled", ""
    if eps == "HT":
        return "Half Time", "HT"
    if eps and eps != "NS":
        return "In Play", eps
    return "Not Started", ""


def _event_live(ev: dict) -> dict:
    """Live fields for one feed event: status, clock, scores, kick-off."""
    status, minute = status_and_minute(ev.get("Eps"))
    hs, as_ = _score(ev, 1), _score(ev, 2)
    h_ht = "" if ev.get("Trh1") in (None, "") else str(ev.get("Trh1"))
    a_ht = "" if ev.get("Trh2") in (None, "") else str(ev.get("Trh2"))
    date_str, time_str = _kickoff(ev.get("Esd"))
    return {
        "status": status,
        "raw_status": str(ev.get("Eps") or "").strip(),
        "minute": minute,
        "score": f"{hs}-{as_}" if hs != "" and as_ != "" else "",
        "home_score": hs,
        "away_score": as_,
        "ht_score": f"{h_ht}-{a_ht}" if h_ht != "" and a_ht != "" else "",
        "kick_off": time_str,
        "date": date_str,
        "event_id": str(ev.get("Eid") or ""),
    }


def _kickoff(esd) -> tuple[str, str]:
    """LiveScore sends Esd as an int/str YYYYMMDDHHMMSS in UTC."""
    raw = str(esd or "")
    if len(raw) < 12:
        return "", ""
    return f"{raw[0:4]}-{raw[4:6]}-{raw[6:8]}", f"{raw[8:10]}:{raw[10:12]}"


def to_local_hhmm(value: str, date_str: str = "") -> str:
    """Convert a stored UTC kick-off time to the machine's local time.

    Everything we persist keeps UTC — that is what the feed sends, and it keeps
    cached files unambiguous — so the shift happens only where a human reads a
    time. Pass the match date when you have it (DST-correct elsewhere; Lagos has
    no DST so it is a no-op here).
    """
    raw = str(value or "").strip()
    if ":" not in raw:
        return value
    try:
        hh, mm = raw.split(":")[0], raw.split(":")[1]
        if date_str:
            dt = datetime.strptime(f"{date_str} {hh}:{mm}", "%Y-%m-%d %H:%M")
        else:
            now = datetime.now()
            dt = now.replace(hour=int(hh), minute=int(mm), second=0, microsecond=0)
        dt = dt.replace(tzinfo=timezone.utc).astimezone()
        local = dt.strftime("%H:%M")
        # A late UTC kick-off can land after midnight locally (23:30Z -> 00:30).
        # Say so, otherwise a game later tonight reads as one from this morning
        # that should have finished by now.
        if local[:2] < str(hh).zfill(2):
            return f"{local} (+1d)"
        return local
    except (ValueError, TypeError):
        return value


def _score(event: dict, side: str) -> str:
    v = event.get(f"Tr{side}")
    return "" if v in (None, "") else str(v)


def to_apifootball(payload: dict) -> list:
    """Convert a LiveScore daily payload into apifootball-shaped fixtures."""
    out = []
    for stage in payload.get("Stages") or []:
        country = stage.get("Cnm") or ""
        league = stage.get("Snm") or ""
        comp = stage.get("CompN") or league
        for ev in stage.get("Events") or []:
            t1 = (ev.get("T1") or [{}])[0]
            t2 = (ev.get("T2") or [{}])[0]
            home, away = t1.get("Nm"), t2.get("Nm")
            if not home or not away:
                continue
            date_str, time_str = _kickoff(ev.get("Esd"))
            raw_phase = str(ev.get("Eps") or "")
            status = _map_status(raw_phase)
            h_score, a_score = _score(ev, 1), _score(ev, 2)
            # Half-time scores: only present once a match has kicked off.
            h_ht = "" if ev.get("Trh1") in (None, "") else str(ev.get("Trh1"))
            a_ht = "" if ev.get("Trh2") in (None, "") else str(ev.get("Trh2"))
            out.append({
                "match_id": str(ev.get("Eid") or ""),
                "match_date": date_str,
                "match_time": time_str,
                "match_status": status,
                "match_raw_status": raw_phase,
                "match_hometeam_name": home,
                "match_awayteam_name": away,
                "match_hometeam_score": h_score,
                "match_awayteam_score": a_score,
                # Extra time/penalties scores are not 90-minute FT scores.
                "match_hometeam_ft_score": h_score if raw_phase == "FT" else "",
                "match_awayteam_ft_score": a_score if raw_phase == "FT" else "",
                "match_hometeam_ht_score": h_ht,
                "match_awayteam_ht_score": a_ht,
                "league_name": league,
                "country_name": country,
                "league_competition": comp,
                "source": "livescore",
            })
    # Chronological, and unique by match id (the CDN can repeat an event
    # across stages for cup competitions).
    dedup = {}
    for m in out:
        dedup.setdefault(m["match_id"] or f"{m['match_hometeam_name']}|{m['match_awayteam_name']}", m)
    return sorted(dedup.values(), key=lambda m: (m["match_date"], m["match_time"]))


def fetch_day(date_str: str, timeout: int = 25, use_cache: bool = True) -> list:
    """Return one day's fixtures mapped into the apifootball schema.

    Raw JSON is cached per date under cache/livescore/ so repeated runs during
    the day do not re-hit the CDN.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    raw_file = CACHE_DIR / f"livescore_{date_str}.json"

    if use_cache and raw_file.exists():
        try:
            payload = json.loads(raw_file.read_text(encoding="utf-8"))
            return to_apifootball(payload)
        except (json.JSONDecodeError, OSError):
            pass  # fall through and re-fetch

    ymd = date_str.replace("-", "")
    req = urllib.request.Request(CDN.format(ymd=ymd), headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    payload = json.loads(body)
    raw_file.write_text(body, encoding="utf-8")
    return to_apifootball(payload)


def fetch_live(date_str: str, timeout: int = 15, allow_cache_fallback: bool = True) -> dict:
    """Fresh status + score for every match on a date, keyed by event id.

    Deliberately bypasses the on-disk cache — this is the "what is happening
    right now" call, so it always hits the feed first. If the feed is
    unreachable it falls back to the last raw payload we saved, tagging each
    record ``stale`` so the UI can say so instead of showing nothing. Raises only
    when there is neither a live response nor a cached payload.
    """
    ymd = date_str.replace("-", "")
    payload, stale = None, False
    try:
        req = urllib.request.Request(CDN.format(ymd=ymd), headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode("utf-8", "replace"))
    except Exception:
        raw_file = CACHE_DIR / f"livescore_{date_str}.json"
        if not (allow_cache_fallback and raw_file.exists()):
            raise
        payload = json.loads(raw_file.read_text(encoding="utf-8"))
        stale = True

    out = {}
    for stage in payload.get("Stages") or []:
        for ev in stage.get("Events") or []:
            eid = str(ev.get("Eid") or "")
            if not eid:
                continue
            rec = _event_live(ev)
            rec["league"] = stage.get("Snm") or ""
            rec["country"] = stage.get("Cnm") or ""
            if stale:
                rec["stale"] = True
            out[eid] = rec
    return out


def write_events_cache(date_str: str, matches: list) -> Path:
    """Write fixtures to the shared cache the CLI + dashboard both read."""
    cache_dir = Path("cache/apifootball")
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"events_{date_str}.json"
    dest.write_text(json.dumps(matches, ensure_ascii=False, indent=2),
                    encoding="utf-8")
    return dest


if __name__ == "__main__":
    import sys
    from datetime import datetime

    day = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    matches = fetch_day(day)
    print(f"{len(matches)} fixtures for {day} (source: LiveScore)")
    leagues = {}
    for m in matches:
        leagues.setdefault(f"{m['league_name']} ({m['country_name']})", []).append(m)
    for name, ms in sorted(leagues.items()):
        print(f"\n{name}  [{len(ms)}]")
        for m in ms:
            print(f"  {m['match_time']}  {m['match_status']:<12} "
                  f"{m['match_hometeam_name']} v {m['match_awayteam_name']}")
