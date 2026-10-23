"""
BetSightly Dashboard — visual overview of predictions, model coverage, and bot strategies.

Usage:
    python dashboard/app.py [--port 5050]
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import difflib
import time
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

from flask import Flask, render_template, jsonify, make_response, request

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

# Kick-off times are stored in UTC (that is what the feed sends, and it keeps the
# cached files unambiguous); this shifts them to local time for display only.
try:
    from fixtures_livescore import to_local_hhmm
except Exception:            # keep the dashboard usable if that module is missing
    def to_local_hhmm(value, *_args, **_kwargs):
        return value

app = Flask(__name__)

# Browser-visible endpoints that create expensive jobs or SportyBet share
# codes require a long operator key (never put it in a URL or HTML template).
# Read-only predictions/results/dashboard remain public.
from services.dashboard_security import (WindowLimiter, needs_operator_auth,
                                         valid_operator_token)
_operator_limiter = WindowLimiter(max_events=8, window_sec=60)


@app.before_request
def protect_operator_actions():
    if not needs_operator_auth(request.path, request.method, request.args):
        return None
    token = os.environ.get("BETSIGHTLY_DASHBOARD_ADMIN_TOKEN", "")
    if len(token) < 32:
        return jsonify({"error": "Admin operations disabled: configure a strong operator token"}), 503
    provided = request.headers.get("X-Betsightly-Admin-Token", "")
    if not valid_operator_token(provided, token):
        return jsonify({"error": "Operator authorization required"}), 401
    client = request.remote_addr or "unknown"
    if not _operator_limiter.allowed(client + ":" + request.path):
        return jsonify({"error": "Too many operator requests"}), 429
    return None


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "same-origin"
    return response

# Reload templates from disk on every render. Without this a running server
# keeps the template it read at startup, so UI edits look like they "didn't
# apply" until the process is restarted. Static files (theme.css) are always
# read fresh, which is why only template edits ever seemed to go missing.
app.config["TEMPLATES_AUTO_RELOAD"] = True


# --- stale-code detection ----------------------------------------------------
# A Flask process loads its Python once, at startup, so a long-running server can
# silently serve older logic than what is on disk — the cause of the "my changes
# didn't apply" rounds.
#
# Only the modules this process actually imports are listed, because only those
# need a restart to change:
#   * index.html and theme.css are read per request (a refresh is enough),
#   * predict_today.py and bot_system.py run as subprocesses (always current).
# Flagging those would be a false alarm on every UI tweak.
_WATCHED_FILES = [
    Path(__file__),
    ROOT / "fixtures_livescore.py",
    ROOT / "build_accas.py",
    ROOT / "sportybet_filter.py",
    ROOT / "sportybet_betting.py",
]

# The page itself, watched separately from the Python modules: a template edit
# only needs a *refresh*, not a restart, but a tab left open keeps rendering the
# copy it was loaded with. The hash below is handed to the browser and compared
# with the file on disk, so an open tab can tell that it is out of date.
TEMPLATE_FILE = ROOT / "dashboard" / "templates" / "index.html"
STATIC_DIR = ROOT / "dashboard" / "static"


def code_fingerprint():
    """Short hash of the watched files as they are on disk right now."""
    import hashlib
    h = hashlib.sha1()
    for p in _WATCHED_FILES:
        try:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{p.name}:missing".encode())
    return h.hexdigest()[:12]


def page_fingerprint():
    """Short hash of the page and its stylesheet as they are on disk now.

    Changes to these need a refresh rather than a restart — the dashboard's
    long-standing "my edits didn't show up" trap was an open tab quietly
    re-rendering the document it was loaded with.
    """
    import hashlib
    h = hashlib.sha1()
    for p in (TEMPLATE_FILE, STATIC_DIR / "theme.css"):
        try:
            st = p.stat()
            h.update(f"{p.name}:{st.st_size}:{int(st.st_mtime)}".encode())
        except OSError:
            h.update(f"{p.name}:missing".encode())
    return h.hexdigest()[:8]


_CODE_AT_START = code_fingerprint()
_STARTED_AT = datetime.now()

# --- Paths ---------------------------------------------------------------
CACHE_DIR = ROOT / "cache" / "apifootball"
API_FOOTBALL_DATA = ROOT / "data" / "api-football" / "matches.csv"
GITHUB_DATA = ROOT / "data" / "github-football" / "Matches.csv"
APIFOOTBALL_HISTORY = ROOT / "data" / "apifootball" / "history.csv"
CHECKPOINT = ROOT / "data" / "apifootball" / "fetch_checkpoint.json"
MODEL_DIR = ROOT / "models" / "api_football"
BOT_RESULTS_DIR = ROOT / "bot_results"
# From the environment (see .env, which is gitignored) — never hardcoded here,
# this file is in a public repository.
API_KEY = os.environ.get("APIFOOTBALL_API_KEY", "")

# --- Caches ---------------------------------------------------------------
_hist_cache = {"df": None, "ts": 0}
_fixtures_cache = {}


def _norm(name: str) -> str:
    n = str(name).strip().lower()
    for suffix in [" fc", " sc", " cf", " ac", " fk", " sk", " if", " bk"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    n = re.sub(r"['\-\.]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def load_historical() -> pd.DataFrame | None:
    if _hist_cache["df"] is not None:
        return _hist_cache["df"]

    frames = []

    if GITHUB_DATA.exists():
        df = pd.read_csv(GITHUB_DATA, low_memory=False)
        df = df.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "MatchDate": "date", "FTHome": "home_score", "FTAway": "away_score",
            "Division": "league_name",
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
        df["source"] = "github"
        if "country" not in df.columns:
            df["country"] = ""
        frames.append(df)

    if API_FOOTBALL_DATA.exists():
        df2 = pd.read_csv(API_FOOTBALL_DATA, low_memory=False)
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2["home_score"] = pd.to_numeric(df2["home_score"], errors="coerce")
        df2["away_score"] = pd.to_numeric(df2["away_score"], errors="coerce")
        df2 = df2.dropna(subset=["date", "home_score", "away_score"])
        df2["source"] = "api-football"
        if "country" not in df2.columns:
            df2["country"] = ""
        frames.append(df2)

    if APIFOOTBALL_HISTORY.exists():
        df3 = pd.read_csv(APIFOOTBALL_HISTORY, low_memory=False)
        df3["date"] = pd.to_datetime(df3["date"], errors="coerce")
        df3["home_score"] = pd.to_numeric(df3["home_score"], errors="coerce")
        df3["away_score"] = pd.to_numeric(df3["away_score"], errors="coerce")
        df3 = df3.dropna(subset=["date", "home_score", "away_score"])
        df3["source"] = "apifootball-history"
        if "country" not in df3.columns:
            df3["country"] = ""
        frames.append(df3)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined["_home_norm"] = combined["home_team"].apply(_norm)
    combined["_away_norm"] = combined["away_team"].apply(_norm)
    _hist_cache["df"] = combined
    return combined


def fetch_fixtures(date_str: str, refresh: bool = False) -> list:
    # `refresh` bypasses both the in-memory and on-disk caches so a Load/Refresh
    # re-pulls the current fixture list + statuses from the API.
    if not refresh and date_str in _fixtures_cache:
        return _fixtures_cache[date_str]

    cache_file = CACHE_DIR / f"events_{date_str}.json"
    if not refresh and cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        _fixtures_cache[date_str] = data
        return data

    import urllib.request
    url = (
        f"https://apiv3.apifootball.com/?action=get_events"
        f"&from={date_str}&to={date_str}&APIkey={API_KEY}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode()
        data = json.loads(raw)
        if isinstance(data, dict) and "error" in data:
            return []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(raw)
        _fixtures_cache[date_str] = data
        return data
    except Exception:
        return []


def find_team_stats(team_name: str, hist: pd.DataFrame) -> dict:
    norm = _norm(team_name)
    home_mask = hist["_home_norm"] == norm
    away_mask = hist["_away_norm"] == norm
    if home_mask.sum() + away_mask.sum() == 0:
        home_mask = hist["_home_norm"].str.contains(norm, na=False, regex=False)
        away_mask = hist["_away_norm"].str.contains(norm, na=False, regex=False)
    total = home_mask.sum() + away_mask.sum()
    if total == 0:
        return {"found": False, "matches": 0}

    hg = hist[home_mask].tail(20)
    ag = hist[away_mask].tail(20)
    hw = (hg["home_score"] > hg["away_score"]).sum()
    hd = (hg["home_score"] == hg["away_score"]).sum()
    aw = (ag["away_score"] > ag["home_score"]).sum()
    ad = (ag["home_score"] == ag["away_score"]).sum()
    tg = len(hg) + len(ag)
    tw = hw + aw
    td = hd + ad

    gs_h = hg["home_score"].sum() if len(hg) else 0
    gc_h = hg["away_score"].sum() if len(hg) else 0
    gs_a = ag["away_score"].sum() if len(ag) else 0
    gc_a = ag["home_score"].sum() if len(ag) else 0

    avg_s = (gs_h + gs_a) / tg if tg else 1.2
    avg_c = (gc_h + gc_a) / tg if tg else 1.2

    hwr = hw / len(hg) if len(hg) else 0.4
    awr = aw / len(ag) if len(ag) else 0.3

    h_btts = ((hg["home_score"] > 0) & (hg["away_score"] > 0)).sum()
    a_btts = ((ag["home_score"] > 0) & (ag["away_score"] > 0)).sum()
    btts_r = (h_btts + a_btts) / tg if tg else 0.5

    h_o25 = ((hg["home_score"] + hg["away_score"]) > 2.5).sum()
    a_o25 = ((ag["home_score"] + ag["away_score"]) > 2.5).sum()
    o25_r = (h_o25 + a_o25) / tg if tg else 0.45

    return {
        "found": True, "matches": total,
        "win_rate": tw / tg if tg else 0.33,
        "draw_rate": td / tg if tg else 0.33,
        "loss_rate": (tg - tw - td) / tg if tg else 0.33,
        "home_win_rate": hwr, "away_win_rate": awr,
        "avg_scored": avg_s, "avg_conceded": avg_c,
        "btts_rate": btts_r, "over25_rate": o25_r,
    }


def predict_match(home: str, away: str, hs: dict, as_: dict) -> dict:
    HOME_ADV = 0.08
    if hs["found"] and as_["found"]:
        hp = 0.5 * hs["home_win_rate"] + 0.5 * hs["win_rate"]
        ap = 0.5 * as_["away_win_rate"] + 0.5 * as_["win_rate"]
        dp = (hs["draw_rate"] + as_["draw_rate"]) / 2
        ehg = (hs["avg_scored"] + as_["avg_conceded"]) / 2
        eag = (as_["avg_scored"] + hs["avg_conceded"]) / 2
        conf = min(0.95, 0.4 + 0.02 * min(hs["matches"], as_["matches"]))
    elif hs["found"]:
        hp, dp, ap = hs["home_win_rate"], hs["draw_rate"], hs["loss_rate"]
        ehg, eag = hs["avg_scored"], hs["avg_conceded"]
        conf = 0.35
    elif as_["found"]:
        ap, dp, hp = as_["away_win_rate"], as_["draw_rate"], as_["loss_rate"]
        ehg, eag = as_["avg_conceded"], as_["avg_scored"]
        conf = 0.35
    else:
        hp, dp, ap = 0.42, 0.27, 0.31
        ehg, eag = 1.3, 1.0
        conf = 0.15

    hp += HOME_ADV
    ap -= HOME_ADV * 0.5
    dp -= HOME_ADV * 0.5
    t = hp + dp + ap
    if t <= 0:
        t = 1
    hp, dp, ap = hp / t, dp / t, ap / t
    hp = max(0.05, min(0.90, hp))
    dp = max(0.05, min(0.50, dp))
    ap = max(0.05, min(0.90, ap))
    t = hp + dp + ap
    hp, dp, ap = hp / t, dp / t, ap / t

    et = ehg + eag
    o15 = 1 - math.exp(-et) * (1 + et)
    o25 = 1 - math.exp(-et) * (1 + et + et ** 2 / 2)
    if hs["found"] and as_["found"]:
        o25 = 0.6 * o25 + 0.4 * (hs["over25_rate"] + as_["over25_rate"]) / 2
    o25 = max(0.1, min(0.9, o25))

    if hs["found"] and as_["found"]:
        btts = (hs["btts_rate"] + as_["btts_rate"]) / 2
    else:
        btts = 0.48
    btts = max(0.15, min(0.85, btts))

    rm = {"Home Win": hp, "Draw": dp, "Away Win": ap}
    best = max(rm, key=rm.get)
    dq = "HIGH" if (hs["found"] and as_["found"] and min(hs["matches"], as_["matches"]) >= 10) \
        else "MEDIUM" if (hs["found"] or as_["found"]) else "LOW"

    return {
        "home_win": round(hp * 100, 1), "draw": round(dp * 100, 1),
        "away_win": round(ap * 100, 1), "prediction": best,
        "over_2.5": round(o25 * 100, 1), "btts": round(btts * 100, 1),
        "confidence": round(conf * 100, 1), "data_quality": dq,
        "exp_home": round(ehg, 2), "exp_away": round(eag, 2),
    }


# =========================================================================
# ROUTES
# =========================================================================

@app.route("/")
def index():
    # The hash identifies this exact build of the page; checkVersion() compares
    # it with the file on disk so an open tab can refresh itself when the UI
    # changes instead of quietly showing yesterday's wording.
    resp = make_response(render_template("index.html",
                                         page_hash=page_fingerprint()))
    resp.headers["Cache-Control"] = "no-store, must-revalidate, no-cache, max-age=0"
    return resp


# A game is settled once it reaches any of these — includes cup ties decided in
# extra time or penalties, which carry a final score but aren't plain "Finished".
_FINISHED_STATUSES = {
    "Finished", "After Pen.", "After ET", "AET", "FT", "Pen.",
    "After Penalties", "After Extra Time", "Awarded",
}

_scores_mem = {}  # date_str -> (fetch_time, scores) — 60s TTL for today's scores
_scores_refreshing = set()  # dates with an in-flight background score refresh
_gen_jobs = {}  # date_str -> {"status","step","count","error"} for Fetch & Predict


def fetch_final_scores(date_str, refresh=False):
    """Fetch FINAL scores for a date, bypassing the pre-match events cache
    (which is captured before kickoff and has no scores). Past-date results are
    cached to results_{date}.json; today's scores are memory-cached for 60s so
    concurrent tab loads share one API call instead of racing the rate limit.
    `refresh=True` bypasses both caches to pull the very latest scores."""
    import time as _time, threading
    mem = _scores_mem.get(date_str)
    ttl = 8 if refresh else 60
    if mem and (_time.time() - mem[0]) < ttl:
        return _with_live_finals(date_str, mem[1])

    results_cache = CACHE_DIR / f"results_{date_str}.json"

    # The apifootball payload is slow/flaky, so NEVER block a page load on it.
    # Serve the best snapshot we have (memory, else the on-disk results file)
    # instantly, and refresh in the background so the next Load is current.
    snapshot = mem[1] if mem else None
    if snapshot is None and results_cache.exists():
        try:
            with open(results_cache, "r", encoding="utf-8") as f:
                snapshot = json.load(f)
                _scores_mem[date_str] = (_time.time() - ttl, snapshot)  # mark stale
        except Exception:
            snapshot = None

    # Always refresh in the background; never block the page on the slow API.
    if date_str not in _scores_refreshing:
        _scores_refreshing.add(date_str)
        def _bg():
            try:
                _do_fetch_scores(date_str)
            finally:
                _scores_refreshing.discard(date_str)
        threading.Thread(target=_bg, daemon=True).start()

    # Serve the best snapshot we have right now (possibly empty on a cold start;
    # the background fetch will populate it for the next Load within seconds).
    return _with_live_finals(date_str, snapshot if snapshot is not None else {})


def _with_live_finals(date_str, scores):
    """Fold finished matches from the live feed into a scores dict.

    apifootball (the historical source of results) is unpaid, so on its own it
    returns nothing and every game looks unfinished forever — the hourly view
    showed morning kick-offs as "not started" at lunchtime for exactly this
    reason. The live feed knows which matches have ended, so finished ones are
    merged in here and every consumer (hourly view, accumulators, bots, the
    predictions tab) inherits correct results from one place.

    Only FINISHED matches are merged: an in-play score is not a final score and
    must never settle a bet.
    """
    live = live_map(date_str)
    if not live:
        return scores or {}
    out = dict(scores or {})
    for mid, rec in live.items():
        if rec.get("status") == "Finished" and rec.get("score"):
            out[str(mid)] = {"hs": rec.get("home_score") or 0,
                             "as": rec.get("away_score") or 0}
    return out


def _do_fetch_scores(date_str):
    """Blocking fetch of final scores from the API; updates the memory cache."""
    import urllib.request, time as _time
    today = datetime.now().strftime("%Y-%m-%d")
    results_cache = CACHE_DIR / f"results_{date_str}.json"
    url = (f"https://apiv3.apifootball.com/?action=get_events"
           f"&from={date_str}&to={date_str}&APIkey={API_KEY}")
    data = None
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception:
            data = None
    if not isinstance(data, list):
        mem = _scores_mem.get(date_str)
        if mem:
            return mem[1]
        if results_cache.exists():
            try:
                with open(results_cache, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    # MERGE with everything we already knew: a finished game must never revert
    # to "pending" just because a later (flaky) API pull omitted it.
    scores = {}
    prev = _scores_mem.get(date_str)
    if prev:
        scores.update(prev[1])
    elif results_cache.exists():
        try:
            with open(results_cache, "r", encoding="utf-8") as f:
                scores.update(json.load(f))
        except Exception:
            pass

    for m in data:
        mid = str(m.get("match_id", ""))
        if m.get("match_status") not in _FINISHED_STATUSES:
            continue
        # Settle on the 90-minute full-time score (not extra time / penalties).
        hs = m.get("match_hometeam_ft_score") or m.get("match_hometeam_score", "")
        aws = m.get("match_awayteam_ft_score") or m.get("match_awayteam_score", "")
        if hs != "":
            scores[mid] = {"hs": hs, "as": aws}  # add/update, never remove

    try:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(results_cache, "w", encoding="utf-8") as f:
            json.dump(scores, f)
    except Exception:
        pass
    _scores_mem[date_str] = (_time.time(), scores)
    return scores


def enrich_finished(p, s):
    """Attach final score + correctness fields to a cached prediction."""
    hs = int(s["hs"] or 0)
    aws = int(s["as"] or 0)
    p["status"] = "finished"
    p["home_score"] = s["hs"]
    p["away_score"] = s["as"]
    actual = "Home Win" if hs > aws else "Away Win" if aws > hs else "Draw"
    p["actual_result"] = actual
    p["result_correct"] = p.get("prediction") == actual
    p["actual_btts"] = hs > 0 and aws > 0
    p["actual_goals"] = hs + aws


# --- Live status (LiveScore) -------------------------------------------------
# Short-lived snapshot of what is happening right now, per match id. Refreshed
# in the background so a page load never blocks on the feed, and kept in memory
# so a polling dashboard cannot hammer it.
_live_mem = {}
_live_refreshing = set()
LIVE_TTL = 25          # seconds a live snapshot is treated as current


def live_map(date_str, force=False, wait=0.0):
    """{match_id: {status, minute, score, ht_score, ...}} for a date.

    Returns the last snapshot while a refresh is in flight, and keeps the
    previous one if the feed is unreachable — a dashboard should show slightly
    stale live data rather than an empty column.

    `wait` seconds are spent waiting for the *first* snapshot when we have none
    yet, so a cold page load shows live statuses instead of an empty column; it
    never blocks once a snapshot exists.
    """
    import time as _time, threading
    mem = _live_mem.get(date_str)
    if mem and not force and (_time.time() - mem[0]) < LIVE_TTL:
        return mem[1]

    if date_str not in _live_refreshing:
        _live_refreshing.add(date_str)

        def _bg():
            try:
                import fixtures_livescore as fxl
                data = fxl.fetch_live(date_str)
                if data:
                    _live_mem[date_str] = (_time.time(), data)
            except Exception as exc:
                app.logger.warning("live status unavailable (%s); keeping last known", exc)
            finally:
                _live_refreshing.discard(date_str)

        threading.Thread(target=_bg, daemon=True).start()

    if wait and date_str not in _live_mem:
        deadline = _time.time() + wait
        while date_str not in _live_mem and _time.time() < deadline:
            _time.sleep(0.1)
        mem = _live_mem.get(date_str)

    return mem[1] if mem else {}


@app.route("/api/live")
def api_live():
    """Current status and score for a date's matches, for live detail on demand."""
    import time as _time
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    force = request.args.get("force") in ("1", "true", "yes")
    # Wait briefly for a fresh snapshot: on a cold start there is nothing to show
    # yet, and on a poll the snapshot we hold is already past its TTL, so without
    # this every poll would serve the *previous* one and lag a full interval.
    matches = live_map(date_str, force=force, wait=6.0)
    snap = _live_mem.get(date_str)
    live_n = sum(1 for m in matches.values()
                 if m.get("status") in ("In Play", "Half Time"))
    done_n = sum(1 for m in matches.values() if m.get("status") == "Finished")
    return jsonify({
        "date": date_str,
        "updated": datetime.fromtimestamp(snap[0]).strftime("%H:%M:%S") if snap else None,
        "age_s": round(_time.time() - snap[0]) if snap else None,
        "count": len(matches), "live": live_n, "finished": done_n,
        "matches": matches,
    })


_net_status = {"ok": None, "checked": 0.0, "error": ""}


def network_status(max_age=120):
    """Can this process reach the internet? Cached briefly.

    Booking codes, live scores and SportyBet prices all need outbound access,
    and a dashboard started from a restricted shell has none. The only symptom
    used to be a connection error at click time — far too late — so it is
    checked here and surfaced in the UI instead.
    """
    import socket, time as _time
    now = _time.time()
    if _net_status["ok"] is not None and (now - _net_status["checked"]) < max_age:
        return _net_status
    try:
        socket.create_connection(("www.sportybet.com", 443), timeout=5).close()
        _net_status.update(ok=True, checked=now, error="")
    except Exception as exc:
        _net_status.update(ok=False, checked=now,
                           error=f"{type(exc).__name__}: {exc}")
    return _net_status


@app.route("/api/version")
def api_version():
    """Is this process running the same code that is on disk?

    The UI polls this and warns when a restart is needed, so a server stuck on an
    older build can never quietly serve yesterday's behaviour again.
    """
    on_disk = code_fingerprint()
    port = request.host.split(":")[-1] if ":" in request.host else "5050"
    return jsonify({
        "running": _CODE_AT_START,
        "on_disk": on_disk,
        "stale": on_disk != _CODE_AT_START,
        # What the page on disk looks like now; a tab compares this with the
        # copy it was loaded with and refreshes itself when they differ.
        "page": page_fingerprint(),
        "started": _STARTED_AT.strftime("%Y-%m-%d %H:%M:%S"),
        "uptime_min": round((datetime.now() - _STARTED_AT).total_seconds() / 60, 1),
        "reload_enabled": bool(app.config.get("RUNNING_WITH_RELOADER")),
        "network": network_status(),
        "restart_cmd": f'cd "{ROOT}" && python dashboard\\app.py --port {port}',
    })


# --- jackpot -----------------------------------------------------------------
# The Sporty 12 jackpot is a 12-leg parlay, so the section is only ever as good
# as the engine behind it. These picks run through the same calibrated hybrid
# Dixon-Coles engine as the Predictions tab, and `pick_conf` is the picked
# outcome's own probability. The parlay maths is reported with them: a ticket's
# chance is the product of its legs, however good the model is.

_PICK_INDEX = {"Home Win": 0, "Draw": 1, "Away Win": 2}
_jackpot_hist = {"at": 0.0, "df": None}
_jackpot_cache = {}          # round -> (time, payload)
# A sweep of 12 calibrated reads needs the 670K-row history, which costs ~30s on
# a cold process. The result is keyed to the snapshot it came from and kept on
# disk, so a restart or a fresh browser never pays for it twice.
JACKPOT_READS = ROOT / "cache" / "jackpot_reads.json"


def load_jackpot_reads(round_id, snapshot_fetched):
    try:
        with open(JACKPOT_READS, "r", encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if str(saved.get("round")) != str(round_id):
        return None
    if str(saved.get("snapshot", "")) != str(snapshot_fetched):
        return None
    if (time.time() - float(saved.get("at") or 0)) > 12 * 3600:
        return None
    return saved.get("payload")


def save_jackpot_reads(round_id, snapshot_fetched, payload):
    try:
        JACKPOT_READS.parent.mkdir(parents=True, exist_ok=True)
        tmp = JACKPOT_READS.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"round": round_id, "snapshot": snapshot_fetched,
                       "at": time.time(), "payload": payload}, f, ensure_ascii=False)
        tmp.replace(JACKPOT_READS)
    except OSError as exc:
        app.logger.warning("could not save the jackpot reads: %s", exc)


def _calibrated_history():
    """predict_today's history frame, loaded once per process (670K rows)."""
    # Six hours: the CSVs behind it are rewritten by the data pipeline, not by the
    # minute, and reloading 670K rows costs seconds.
    if _jackpot_hist["df"] is None or (time.time() - _jackpot_hist["at"]) > 6 * 3600:
        import predict_today as pt
        _jackpot_hist.update(at=time.time(), df=pt.load_historical())
    return _jackpot_hist["df"]


def _calibrated_read(home, away, hist):
    """One calibrated read: probabilities, pick, data quality, engine used."""
    import predict_today as pt
    hs = pt.find_team_stats(home, hist)
    as_ = pt.find_team_stats(away, hist)
    pred = pt.predict_match(home, away, hs, as_, pt.get_h2h(home, away, hist))
    engine = "built-in"
    if getattr(pt, "_HYBRID", None) is not None:
        pred = pt._HYBRID.apply(pred, pt.normalize_name(home), pt.normalize_name(away))
        engine = "hybrid-dc"
    elif getattr(pt, "_CALIBRATOR", None) is not None:
        pred = pt._CALIBRATOR.apply(pred)
        engine = "calibrated"
    probs = [float(pred.get("home_win") or 0), float(pred.get("draw") or 0),
             float(pred.get("away_win") or 0)]
    best = max(range(3), key=lambda i: probs[i])
    return {
        "probs": [round(x, 1) for x in probs],
        "home_win": round(probs[0], 1),
        "draw": round(probs[1], 1),
        "away_win": round(probs[2], 1),
        "pick": ["Home Win", "Draw", "Away Win"][best],
        # The picked outcome's own probability — not an internal confidence score.
        "pick_conf": round(probs[best], 1),
        "data_quality": pred.get("data_quality"),
        "engine": engine,
        # The engine hands back numpy scalars, which Flask's JSON encoder
        # refuses; everything crossing the wire is cast here.
        "over_2.5": float(pred["over_2.5"]) if pred.get("over_2.5") is not None else None,
        "btts": float(pred["btts"]) if pred.get("btts") is not None else None,
        "history": {"home": int(hs.get("matches", 0) or 0),
                    "away": int(as_.get("matches", 0) or 0)},
    }


def _combo_stats(rows):
    """The parlay maths: chance of all 12, of 11, of 10, and the market's view.

    Legs are treated as independent, which is the standard (and slightly
    optimistic) reading for a jackpot. The point of showing it is honesty: the
    chance of landing twelve legs is small even when every leg is a good bet.
    """
    picks = [r.get("pick_conf", 0) / 100.0 for r in rows if r.get("pick_conf")]
    if not picks:
        return {}
    p_all = math.prod(picks)
    miss1 = sum(p_all * (1 - p) / p for p in picks)          # exactly one wrong
    miss2 = sum(p_all * ((1 - picks[i]) * (1 - picks[j])) / (picks[i] * picks[j])
                for i in range(len(picks)) for j in range(i + 1, len(picks)))
    favourites = [max(r["implied"]) / 100.0 for r in rows if r.get("implied")]
    on_our_picks = [r["implied"][_PICK_INDEX[r["pick"]]] / 100.0
                    for r in rows if r.get("implied") and r.get("pick") in _PICK_INDEX]
    return {
        "legs": len(picks),
        "all": round(p_all * 100, 4),
        "one_in": int(round(1 / p_all)) if p_all else None,
        "at_least_11": round((p_all + miss1) * 100, 4),
        "at_least_10": round((p_all + miss1 + miss2) * 100, 4),
        "market_favourites": round(math.prod(favourites) * 100, 4) if favourites else None,
        "market_on_our_picks": round(math.prod(on_our_picks) * 100, 4) if on_our_picks else None,
        "mean_pick_prob": round(sum(picks) / len(picks) * 100, 1),
    }


@app.route("/api/jackpot")
def api_jackpot():
    """SportyBet's Sporty 12 jackpot, read through the calibrated engine.

    Fixtures come from the cached snapshot (see sportybet_jackpot.py) because the
    jackpot page is client-rendered and cannot be scraped server-side.

    The picks used to come from the dashboard's built-in fallback engine, which
    saturates its confidence at 95% and leans to the home side — so every
    percentage here was noise. They now come from the same hybrid Dixon-Coles
    engine as the Predictions tab, and `pick_conf` is the pick's own
    probability. Because a jackpot is a 12-leg parlay, the chance of landing all
    of them is reported alongside, not hidden.
    """
    import sportybet_jackpot as jp

    data = jp.load_jackpot()
    games = data.get("games", [])
    round_id = str(data.get("round") or "current")
    hit = _jackpot_cache.get(round_id)
    # A jackpot round is fixed for days and the fixtures come from a snapshot, so
    # the reads only change when the data behind them does. First call in a
    # process pays for the history load; after that it is instant.
    if hit and (time.time() - hit[0]) < 6 * 3600:
        return jsonify(hit[1])
    saved = load_jackpot_reads(round_id, data.get("fetched", ""))
    if saved:
        _jackpot_cache[round_id] = (time.time(), saved)
        return jsonify(saved)

    hist = _calibrated_history()
    out = []
    for g in games:
        row = dict(g)
        odds = g.get("odds") or []
        if len(odds) == 3 and all(o for o in odds):
            # De-vig the market: 1/odds normalised back to 100%. That is the
            # bookmaker's own view, and the yardstick for "edge" below.
            inv = [1.0 / float(o) for o in odds]
            total = sum(inv)
            row["implied"] = [round(100 * x / total, 1) for x in inv]

        if hist is not None:
            row.update(_calibrated_read(g.get("home", ""), g.get("away", ""), hist))
            if row.get("implied") and row.get("pick"):
                i = _PICK_INDEX[row["pick"]]
                row["edge"] = round(row["probs"][i] - row["implied"][i], 1)

        out.append(row)

    payload = {
        "round": data.get("round", ""),
        "prize": data.get("prize", ""),
        "games_to_predict": data.get("games_to_predict", 12),
        "prizes_note": data.get("prizes_note", ""),
        "fetched": data.get("fetched", ""),
        "source": data.get("source", ""),
        "games": out,
        "combo": _combo_stats(out),
        "engine": out[0].get("engine", "") if out else "",
    }
    _jackpot_cache[round_id] = (time.time(), payload)
    save_jackpot_reads(round_id, data.get("fetched", ""), payload)
    return jsonify(payload)


@app.route("/api/generate")
def api_generate():
    """Build the day's card from SportyBet's own games, then run the bots.

    The games come from SportyBet (see predict_sportybet.py), so every prediction
    on the card is one you can actually back and the odds behind it are theirs;
    the free fixture feed is still pulled for live scores and results, and stays
    as the fallback if their list cannot be read. Runs in a background thread —
    poll /api/generate-status for progress.
    """
    import subprocess as _sp, threading, time as _time
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    source = request.args.get("source", "sportybet")
    job = _gen_jobs.get(date_str)
    if job and job.get("status") == "running":
        return jsonify({"status": "running", "step": job.get("step"), "date": date_str})

    _gen_jobs[date_str] = {"status": "running", "step": "pulling SportyBet's games",
                           "started": _time.time()}

    def _run():
        try:
            # The fixture feed is what live scores and results are keyed on, so
            # let the new pipeline pull it fresh rather than reuse a day-old copy.
            try:
                (CACHE_DIR / f"events_{date_str}.json").unlink()
            except Exception:
                pass
            _fixtures_cache.pop(date_str, None)
            _scores_mem.pop(date_str, None)

            py = sys.executable
            if source == "fixtures":
                cmd = [py, "predict_today.py", "--date", date_str]
            else:
                cmd = [py, "predict_sportybet.py", "--date", date_str]

            _gen_jobs[date_str]["step"] = "predicting the card"
            p1 = _sp.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=3600)
            if p1.returncode != 0 and source != "fixtures":
                # Their list can be empty (nothing inside the rolling window) or
                # unreachable; the fixture feed still fills the card.
                _gen_jobs[date_str]["step"] = "SportyBet unavailable — using the fixture feed"
                p1 = _sp.run([py, "predict_today.py", "--date", date_str],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
            if p1.returncode != 0:
                _gen_jobs[date_str] = {"status": "error", "step": "predictions",
                                       "error": (p1.stderr or p1.stdout or "")[-400:]}
                return

            _gen_jobs[date_str]["step"] = "running bots"
            # The bots are a leftover from the old pipeline and scale badly with
            # a big card; the predictions are already written, so a slow (or
            # failed) simulation must never leave the job stuck on "running".
            try:
                p2 = _sp.run([py, "bot_system.py", "--date", date_str],
                             cwd=str(ROOT), capture_output=True, text=True, timeout=600)
                bots_ok = p2.returncode == 0
            except _sp.TimeoutExpired:
                bots_ok = False
                _gen_jobs[date_str]["step"] = "bots timed out (card is ready)"

            n = 0
            pf = CACHE_DIR / f"predictions_{date_str}.json"
            if pf.exists():
                try:
                    with open(pf, "r", encoding="utf-8") as f:
                        n = len(json.load(f))
                except Exception:
                    pass
            _scores_mem.pop(date_str, None)  # let the tabs re-pull fresh scores
            _gen_jobs[date_str] = {"status": "done", "step": "done", "count": n,
                                   "bots_ok": bots_ok}
        except _sp.TimeoutExpired:
            _gen_jobs[date_str] = {"status": "error", "error": "timed out (>30 min)"}
        except Exception as e:
            _gen_jobs[date_str] = {"status": "error", "error": str(e)}

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"status": "started", "date": date_str})


@app.route("/api/generate-status")
def api_generate_status():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    return jsonify(_gen_jobs.get(date_str, {"status": "idle"}))


@app.after_request
def _no_store(resp):
    # Never let the browser serve a stale response — this covers both API data
    # AND the HTML page itself, so a refresh always gets the latest dashboard
    # code (otherwise new buttons/toggles won't appear after an update).
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


def annotate_sportybet(preds):
    """Tag each prediction with whether SportyBet offers that game.

    Returns a status dict for the UI so it can tell "no games on SportyBet"
    apart from "couldn't read SportyBet's list" — those looked identical when
    the report was just a count of zero.

    Fail-open: if the list can't be trusted, every game is marked available so
    the filter never hides games that are genuinely offered.
    """
    def mark_all_available():
        for p in preds:
            p["sportybet"] = True

    try:
        import sportybet_filter as sb
        pairs, teams = sb.load_offered_pairs(max_age_min=60)
        info = sb.source_info()
        if not pairs:
            mark_all_available()
            return {"status": "unavailable", "reason": "SportyBet returned no games"}

        fuzzy = 0
        for p in preds:
            kind = sb.match_kind(p.get("home_team", ""), p.get("away_team", ""),
                                 pairs, teams)
            p["sportybet"] = kind is not None
            if kind == "fuzzy":
                fuzzy += 1
        return {
            "status": info.get("status", "live"),      # "live" or "stale"
            "offered": sum(1 for p in preds if p["sportybet"]),
            "list_games": info.get("games"),
            "age_min": info.get("age_min"),
            "name_fuzzy": fuzzy,
        }
    except Exception as exc:
        app.logger.warning("SportyBet list unavailable (%s); showing all games", exc)
        mark_all_available()
        return {"status": "unavailable", "reason": str(exc)}


def _started_match_ids(date_str):
    """Ids of matches that have already kicked off today.

    Live feed first, fixtures cache as a backup. Used to keep played games on the
    Build board, so a ticket built earlier settles instead of disappearing the
    moment one of its legs kicks off.
    """
    started = set()
    try:
        live = live_map(date_str)
        started |= {str(mid) for mid, rec in live.items()
                    if rec.get("status") in ("In Play", "Half Time", "Finished")}
    except Exception:
        pass
    if not started:
        try:
            import build_accas as ba
            open_ids = ba.kickable_ids(date_str)        # ids NOT yet started
            ev = CACHE_DIR / f"events_{date_str}.json"
            if open_ids is not None and ev.exists():
                fixtures = json.loads(ev.read_text(encoding="utf-8"))
                started |= {str(m.get("match_id", "")) for m in fixtures
                            if str(m.get("match_id", "")) not in open_ids}
        except Exception:
            pass
    return started


def tag_pricable(preds, field="over_1.5"):
    """Mark each prediction with whether SportyBet prices that line.

    Being on SportyBet's game list is not the same as being usable: a booking
    code needs the actual Over/Under price, and a fixture can be offered without
    the 1.5 line. The panels use this to decide whether offering a "Booking
    code" button for a group would be a promise it can keep.
    """
    try:
        import sportybet_betting as sbb
        index = sbb.load_id_index()
    except Exception as exc:
        app.logger.warning("SportyBet prices unavailable for tagging: %s", exc)
        for p in preds:
            # An offered fixture is NOT proof its actual market can be priced.
            p["sb_price"] = False
        return
    for p in preds:
        event, _kind = sbb.find_event(index, p.get("home_team", ""),
                                      p.get("away_team", ""))
        p["sb_price"] = bool(((event or {}).get("prices") or {}).get(field))


def _acca_inputs(date_str, sportybet_only=True, market=None):
    """Everything /api/build and /api/build/code need to build the same tickets.

    Both endpoints funnel through here so the booking code is always created
    from exactly the ticket the dashboard displayed.

    With sportybet_only the legs are priced at SportyBet's real Over/Under lines
    and a leg only qualifies if SportyBet is offering that line — otherwise the
    ticket could not be placed. If the odds feed is down this falls back to fair
    odds rather than returning nothing.

    `market` narrows which lines the tickets may use: "over_1.5" keeps every leg
    on the Over/Under 1.5 line, anything else uses the default 1.5 + 2.5 mix.
    """
    import build_accas as ba
    markets = ba.resolve_markets(market)

    pred_file = CACHE_DIR / f"predictions_{date_str}.json"
    if not pred_file.exists():
        return {"error": "no predictions for this date"}
    with open(pred_file, "r", encoding="utf-8") as f:
        preds = json.load(f)
    total = len(preds)

    sportybet_status = None
    odds_lookup, odds_status = None, {"status": "off"}
    id_lookup, id_status = None, {"status": "off"}
    started_ids = _started_match_ids(date_str)
    played_preds = []
    if sportybet_only:
        sportybet_status = annotate_sportybet(preds)
        # Split the day in two: what SportyBet is offering now (placeable) and
        # what has already kicked off (settled or in play). They get their own
        # sections, so won/busted tickets stay visible instead of vanishing the
        # moment their legs start.
        played_preds = [p for p in preds if str(p.get("match_id", "")) in started_ids]
        preds = [p for p in preds if p.get("sportybet")]
        board = []
        try:
            import sportybet_betting as sbb
            board = sbb.load_odds_table()
            index = sbb.build_index(board)
            odds_status = sbb.source_info()

            def odds_lookup(home, away, _index=index, _sbb=sbb):
                return _sbb.find_event(_index, home, away)
        except Exception as exc:
            app.logger.warning("SportyBet odds unavailable (%s); ticket booking disabled", exc)
            odds_status = {"status": "unavailable", "reason": str(exc)}
        # Booking-code ids for games the board has dropped: the live board still
        # lists in-play fixtures under the same event id, and anything we priced
        # earlier today is remembered in the id ledger. Without this a ticket
        # stops being bookable the moment its first game kicks off.
        try:
            import sportybet_betting as sbb
            id_index = sbb.load_id_index(board)
            id_status = {"status": "ready", "fixtures": len(id_index)}

            def id_lookup(home, away, _index=id_index, _sbb=sbb):
                return _sbb.find_event(_index, home, away)
        except Exception as exc:
            app.logger.warning("SportyBet ids unavailable (%s)", exc)
            id_status = {"status": "unavailable", "reason": str(exc)}

    # Build from ALL usable matches (not just upcoming) so a ticket can be tracked
    # to a result as its games finish. In SportyBet mode the feed only lists
    # upcoming games, so the pool is naturally limited to what can still be backed.
    # Never turn fair model odds into purportedly placeable bookmaker legs.
    if sportybet_only and odds_lookup is None:
        pool = []
    else:
        pool = ba.build_pool(preds, None, odds_lookup=odds_lookup, markets=markets,
                             started_ids=started_ids, id_lookup=id_lookup)
    # Already-played games are priced at fair odds (their board prices are gone),
    # but they still carry SportyBet ids when we know them, so their tickets keep
    # a working booking code.
    played_pool = ba.build_pool(played_preds, None, markets=markets,
                                started_ids=started_ids,
                                id_lookup=id_lookup) if played_preds else []
    # Legs carry UTC kick-offs; show them in local time (and use local hours for
    # the "early momo" 00:00-12:00 window below).
    for leg in pool + played_pool:
        leg["kick_off"] = to_local_hhmm(leg.get("kick_off"), date_str)
    # "Early momo" pool: only games kicking off 00:00–11:59 (HH:MM strings sort).
    early_pool = [c for c in pool if "00:00" <= (c.get("kick_off") or "99") < "12:00"]

    return {"ba": ba, "pool": pool, "early": early_pool, "total": total,
            "played": played_pool,
            "id_lookup": id_lookup, "id_status": id_status,
            "sportybet": sportybet_status, "sportybet_odds": odds_status,
            "real_odds": odds_lookup is not None, "market": market or "",
            "markets": [m[1] for m in markets]}


@app.route("/api/predictions")
def api_predictions():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    refresh = request.args.get("refresh") in ("1", "true", "yes")

    pred_cache = CACHE_DIR / f"predictions_{date_str}.json"
    if pred_cache.exists():
        with open(pred_cache, "r", encoding="utf-8") as f:
            cached = json.load(f)
        live = live_map(date_str, force=refresh, wait=6.0)
        scores = fetch_final_scores(date_str, refresh=refresh)
        for p in cached:
            mid = str(p.get("match_id", ""))
            lv = live.get(mid) or {}
            lstat = lv.get("status") or ""
            # Display times in the viewer's zone; the cache stays UTC.
            p["kick_off"] = to_local_hhmm(p.get("kick_off"), date_str)
            p["minute"] = ""
            p["live_status"] = lstat
            if lstat in ("In Play", "Half Time"):
                # In play: show the clock and running score; the pick is unresolved.
                p["status"] = "live"
                p["minute"] = lv.get("minute") or ("HT" if lstat == "Half Time" else "")
                p["home_score"] = lv.get("home_score")
                p["away_score"] = lv.get("away_score")
                p["live_score"] = lv.get("score") or ""
                p["ht_score"] = lv.get("ht_score") or ""
            elif lstat == "Finished" and (lv.get("score") or ""):
                enrich_finished(p, {"hs": lv.get("home_score") or 0,
                                    "as": lv.get("away_score") or 0})
                p["ht_score"] = lv.get("ht_score") or ""
            else:
                s = scores.get(mid)
                if s:
                    enrich_finished(p, s)
                else:
                    p["status"] = "upcoming"
        sportybet = annotate_sportybet(cached)
        # And which of them we can actually price a code for.
        tag_pricable(cached)
        snap = _live_mem.get(date_str)
        return jsonify({"date": date_str, "predictions": cached, "source": "cache",
                        "sportybet": sportybet,
                        "live_updated": datetime.fromtimestamp(snap[0]).strftime("%H:%M:%S") if snap else None})

    # No prediction cache for this date. Do NOT compute live here — that loads
    # 670K rows and runs per-match lookups, which freezes the page on a cold
    # date (e.g. right after midnight rolls over). Return fast and let the user
    # press "Fetch & Predict" to generate them in the background.
    return jsonify({"date": date_str, "predictions": [], "source": "none",
                    "error": "No predictions generated for this date yet — "
                             "press “Fetch & Predict” to build them."})


@app.route("/api/coverage")
def api_coverage():
    hist = load_historical()
    if hist is None:
        return jsonify({"leagues": [], "total_matches": 0})

    ln = hist.get("league_name")
    if ln is None:
        return jsonify({"leagues": [], "total_matches": len(hist)})

    grouped = hist.groupby("league_name").agg(
        matches=("date", "count"),
        teams=("home_team", "nunique"),
        first=("date", "min"),
        last=("date", "max"),
    ).reset_index()
    grouped = grouped.sort_values("matches", ascending=False)

    leagues = []
    for _, r in grouped.iterrows():
        leagues.append({
            "league": str(r["league_name"]),
            "matches": int(r["matches"]),
            "teams": int(r["teams"]),
            "first": str(r["first"])[:10],
            "last": str(r["last"])[:10],
        })

    sources = hist["source"].value_counts().to_dict() if "source" in hist.columns else {}
    return jsonify({
        "leagues": leagues,
        "total_matches": len(hist),
        "total_leagues": len(leagues),
        "sources": {str(k): int(v) for k, v in sources.items()},
    })


@app.route("/api/fetch-status")
def api_fetch_status():
    if not CHECKPOINT.exists():
        return jsonify({"status": "not_started"})
    with open(CHECKPOINT, "r", encoding="utf-8") as f:
        cp = json.load(f)

    history_rows = 0
    if APIFOOTBALL_HISTORY.exists():
        with open(APIFOOTBALL_HISTORY, "r", encoding="utf-8") as f:
            history_rows = sum(1 for _ in f) - 1

    return jsonify({
        "status": "in_progress" if len(cp["done_chunks"]) < 244 else "complete",
        "chunks_done": len(cp["done_chunks"]),
        "chunks_total": 244,
        "rows_written": cp.get("rows_written", 0),
        "history_file_rows": history_rows,
        "last_chunk": cp["done_chunks"][-1] if cp["done_chunks"] else None,
    })


@app.route("/api/model-info")
def api_model_info():
    meta_path = MODEL_DIR / "meta.json"
    weights_path = MODEL_DIR / "model_weights.json"
    elo_path = MODEL_DIR / "elo_ratings.json"

    info = {"models": [], "elo_teams": 0, "total_size_mb": 0}

    if meta_path.exists():
        with open(meta_path, "r") as f:
            meta = json.load(f)
        info["feature_columns"] = meta.get("feature_columns", [])
        info["model_names"] = meta.get("models", [])

    if weights_path.exists():
        with open(weights_path, "r") as f:
            weights = json.load(f)
        for name, acc in weights.items():
            info["models"].append({"name": name, "accuracy": round(acc, 4)})

    if elo_path.exists():
        with open(elo_path, "r") as f:
            elo = json.load(f)
        info["elo_teams"] = len(elo)

    total = sum(f.stat().st_size for f in MODEL_DIR.glob("*.joblib")) if MODEL_DIR.exists() else 0
    info["total_size_mb"] = round(total / 1024 / 1024, 1)

    return jsonify(info)


@app.route("/api/hourly-breakdown")
def api_hourly_breakdown():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    refresh = request.args.get("refresh") in ("1", "true", "yes")

    # Predictions from the cache (calibrated hybrid), keyed by match_id
    preds_by_id = {}
    pred_cache = CACHE_DIR / f"predictions_{date_str}.json"
    if pred_cache.exists():
        try:
            with open(pred_cache, "r", encoding="utf-8") as f:
                for p in json.load(f):
                    preds_by_id[str(p.get("match_id", ""))] = p
            # Tag the day's predictions with SportyBet membership once, so each
            # hour row can say whether a booking code is even possible for it.
            annotate_sportybet(list(preds_by_id.values()))
            tag_pricable(list(preds_by_id.values()))
        except Exception:
            pass

    # The card is SportyBet's own games, so the hours come from the predictions.
    # Falling back to the free fixture feed when a date has no predictions keeps
    # old dates viewable (and the tab useful before the first run of the day).
    if preds_by_id:
        entries = [{
            "match_id": str(p.get("match_id", "")),
            "home": p.get("home_team", ""),
            "away": p.get("away_team", ""),
            "league": p.get("league", ""),
            "country": p.get("country", ""),
            "utc": p.get("kick_off", "00:00"),
            "pred": p,
            "fixture_status": "",
            "sb_status": p.get("sb_status", ""),
            "tracked": bool(p.get("live_tracked")),
        } for p in preds_by_id.values()]
    else:
        # Keep the cached fixture LIST (it rarely changes intraday and re-pulling
        # the full payload would freeze the Load); freshness is the scores below.
        fixtures = fetch_fixtures(date_str)
        if not fixtures:
            return jsonify({"hours": {}})
        entries = [{
            "match_id": str(m.get("match_id", "")),
            "home": m.get("match_hometeam_name", ""),
            "away": m.get("match_awayteam_name", ""),
            "league": m.get("league_name", ""),
            "country": m.get("country_name", ""),
            "utc": m.get("match_time", "00:00"),
            "pred": {},
            "fixture_status": m.get("match_status", ""),
            "sb_status": "",
            "tracked": True,
        } for m in fixtures]

    # Live status first (so a snapshot exists), then finals merged from it.
    live = live_map(date_str, force=refresh, wait=6.0)
    scores = fetch_final_scores(date_str, refresh=refresh)  # includes live finals

    by_hour = defaultdict(list)
    for entry in entries:
        # The stored time is UTC; group and display by local hour so a 10:00 UTC
        # kick-off lands under 11:00 for a UTC+1 viewer.
        t = to_local_hhmm(entry["utc"], date_str)
        hour = t[:2] if len(t) >= 2 else "00"
        mid = entry["match_id"]
        pred = entry["pred"]
        lv = live.get(mid) or {}
        lstat = lv.get("status") or ""
        # A match counts as finished only on a final score. In-play games report
        # their running score and clock, and are never settled as a result.
        if lstat in ("In Play", "Half Time"):
            status = "In Play"
            minute = lv.get("minute") or ("HT" if lstat == "Half Time" else "")
            hs, aws = lv.get("home_score", ""), lv.get("away_score", "")
            final = None
        elif lstat == "Finished" and lv.get("score"):
            status, minute = "Finished", ""
            hs, aws = lv.get("home_score", ""), lv.get("away_score", "")
            final = {"hs": hs or 0, "as": aws or 0}
        else:
            s = scores.get(mid)
            minute = ""
            if s:
                status, hs, aws, final = "Finished", s["hs"], s["as"], s
            else:
                hs = aws = ""
                final = None
                # Say which kind of silence this is: SportyBet's own status when
                # we have it, "not started" for a game a live feed can follow,
                # and "no live scores" for the long tail no feed reaches.
                status = (entry["sb_status"] or entry["fixture_status"]
                          or ("Not Started" if entry["tracked"] else "No live scores"))
        # Hourly view tracks the Over/Under 1.5 market — both the current
        # (raw) probability and the calibrated one (over_1.5v).
        def ou15(prob):
            if prob is None:
                return "", None
            over = prob >= 50
            return ("Over 1.5" if over else "Under 1.5"), round(prob if over else 100 - prob, 1)

        pick_label, pick_conf = ou15(pred.get("over_1.5"))
        pick_label_v, pick_conf_v = ou15(pred.get("over_1.5v"))
        row = {
            # Carried so a whole hour (or a set of rows) can be turned into a
            # SportyBet booking code from the UI.
            "match_id": mid,
            "home": entry["home"],
            "away": entry["away"],
            "league": entry["league"],
            "country": entry["country"],
            "time": t,
            "prediction": pick_label,
            "confidence": pick_conf,
            "prediction_v": pick_label_v,
            "confidence_v": pick_conf_v,
            "sportybet": bool(pred.get("sportybet")),
            "sb_price": bool(pred.get("sb_price")),
            "status": status,
            "minute": minute,
            "home_score": hs,
            "away_score": aws,
            "ht_score": lv.get("ht_score", ""),
        }
        if final:
            fh = int(final["hs"] or 0)
            fa = int(final["as"] or 0)
            actual_over = (fh + fa) > 1.5
            if pick_label:
                row["result_correct"] = (pick_label == "Over 1.5") == actual_over
            if pick_label_v:
                row["result_correct_v"] = (pick_label_v == "Over 1.5") == actual_over
        by_hour[hour].append(row)

    return jsonify({"date": date_str, "hours": dict(sorted(by_hour.items())),
                    "total": len(entries),
                    "source": "sportybet" if preds_by_id else "fixtures"})


def settle_selection(selection, hs, aws):
    """True if the bet selection won given the final score."""
    total = hs + aws
    rules = {
        "Home Win": hs > aws,
        "Away Win": aws > hs,
        "Draw": hs == aws,
        "Over 1.5": total > 1.5, "Under 1.5": total < 1.5,
        "Over 2.5": total > 2.5, "Under 2.5": total < 2.5,
        "Over 3.5": total > 3.5, "Under 3.5": total < 3.5,
        "BTTS Yes": hs > 0 and aws > 0,
        "BTTS No": hs == 0 or aws == 0,
    }
    return rules.get(selection)


def settle_pending_bets(data, date_str, refresh=False):
    """Settle any PENDING bets in a bots_{date}.json payload against live final
    scores, so the dashboard shows results without waiting for the nightly
    bot_system re-run."""
    scores = fetch_final_scores(date_str, refresh=refresh)
    if not scores:
        return data
    for bot in data.get("bots", []):
        for h in bot.get("history", []):
            if h.get("status") != "PENDING":
                continue
            s = scores.get(str(h.get("match_id", "")))
            if not s:
                continue
            won = settle_selection(h.get("selection", ""), int(s["hs"] or 0), int(s["as"] or 0))
            if won is None:
                continue
            h["status"] = "WIN" if won else "LOSS"
            h["result"] = "win" if won else "loss"
            h["final_score"] = f"{s['hs']}-{s['as']}"
    return data


def _build_snapshot_path(date_str):
    return CACHE_DIR / f"build_{date_str}.json"


def load_build_snapshot(date_str):
    """The frozen ticket sets for a date, keyed by mode: {"modes": {mode: [...]}}."""
    try:
        with open(_build_snapshot_path(date_str), "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_build_snapshot(date_str, snapshot):
    path = _build_snapshot_path(date_str)
    tmp = path.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, ensure_ascii=False)
    tmp.replace(path)
    return path


@app.route("/api/build")
def api_build():
    """Accumulator tickets built from the calibrated predictions."""
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    refresh = request.args.get("refresh") in ("1", "true", "yes")
    # Default: only games SportyBet offers, so the tickets are actually placeable.
    # Pass all=1 to include every match.
    sportybet_only = request.args.get("all") not in ("1", "true", "yes")
    # market=over_1.5 keeps every leg on the Over/Under 1.5 line.
    market = request.args.get("market", "")
    try:
        targets = [float(t) for t in request.args.get("targets", "2,5,10,100").split(",")]
    except ValueError:
        targets = [2.0, 5.0, 10.0, 100.0]

    try:
        import build_accas as ba
    except Exception as e:
        return jsonify({"error": f"builder unavailable: {e}", "tickets": []})

    inputs = _acca_inputs(date_str, sportybet_only, market)
    if "error" in inputs:
        return jsonify({"date": date_str, "tickets": [], "pool": 0,
                        "error": inputs["error"]})
    pool, early_pool = inputs["pool"], inputs["early"]
    played_pool = inputs.get("played", [])
    id_lookup = inputs.get("id_lookup")
    started_ids = _started_match_ids(date_str)
    total_preds = inputs["total"]
    sportybet_status = inputs["sportybet"]
    # Live state first, and wait for it: fetch_final_scores() folds finished
    # results out of this same snapshot, so on a cold start the other order left
    # the first request believing nothing had finished yet.
    live = live_map(date_str, force=refresh, wait=6.0)
    scores = fetch_final_scores(date_str, refresh=refresh)

    def settle_and_pack(a, t, label, section):
        won = lost = pending = 0
        for leg in a["legs"]:
            mid = str(leg.get("match_id", ""))
            s = scores.get(mid)
            lv = live.get(mid) or {}
            lstat = lv.get("status") or ""
            leg["live_status"] = lstat
            leg["minute"] = lv.get("minute") or ""
            leg["live_score"] = lv.get("score") or ""
            leg["ht_score"] = lv.get("ht_score") or ""
            if not s:
                leg["result"] = "pending"
                pending += 1
                continue
            hs, aws = int(s["hs"] or 0), int(s["as"] or 0)
            ok = settle_selection(leg["selection"], hs, aws)
            leg["result"] = "win" if ok else "loss"
            leg["final_score"] = f"{s['hs']}-{s['as']}"
            won, lost = (won + 1, lost) if ok else (won, lost + 1)
        status = "busted" if lost else ("landed" if pending == 0 else "live")
        # A leg frozen without ids — built before its game kicked off, or before
        # this snapshot was written — can still be resolved from the live board
        # or the id ledger, which is what keeps a started ticket bookable.
        missing = []
        for leg in a["legs"]:
            if leg.get("sb"):
                continue
            if id_lookup is not None:
                home, away = ba.leg_teams(leg)
                event, _kind = id_lookup(home, away) if home else (None, None)
                # A leg in "Played today" has kicked off even if the feed has no
                # status for it, so a later meeting of the same clubs is refused.
                started = section == "played" or bool(
                    leg.get("live_status") or leg.get("final_score"))
                ids = ba.leg_ids_for(event, leg, started=started)
                if ids:
                    leg["sb"] = ids
                    continue
            missing.append(leg.get("match", ""))
        # A ticket is only bookable if every leg carries SportyBet identifiers.
        bookable = bool(a["legs"]) and all(
            leg.get("sb") and str(leg.get("match_id", "")) not in started_ids
            for leg in a["legs"]
        )
        fingerprint = None
        if bookable:
            import sportybet_betting as sbb
            fingerprint = sbb.fingerprint([leg["sb"] for leg in a["legs"]])
        return {"target": t, "label": label, "section": section, "reachable": True,
                "odds": a["odds"], "probability": round(a["prob"] * 100, 2),
                "n_legs": len(a["legs"]), "won": won, "lost": lost,
                "pending": pending, "status": status, "legs": a["legs"],
                "bookable": bookable, "fingerprint": fingerprint,
                "missing_sb": missing}

    def build_section(src_pool, section, seed_shift):
        """Unsettled tickets for a section, plus a placeholder for an empty tier."""
        out = []
        for t in targets:
            # A fresh set per TIER: no option repeats inside a tier, but different
            # tiers may share one. A single page-wide set starved them — a ~5
            # ticket needs ~4 unique options, ~10 needs ~7, ~100 needs ~13, so
            # with a ~15-option pool the ~2 tier consumed everything and the
            # higher tiers could only ever read "none reachable".
            used_selections = set()
            # 10 distinct tickets per odds tier, seeded so they persist across refreshes.
            sets = ba.build_ticket_set(src_pool, t, n=10,
                                       seed=int(t * 100) + seed_shift,
                                       used_selections=used_selections)
            if not sets:
                out.append({"target": t, "reachable": False, "legs": [], "section": section})
                continue
            for v, a in enumerate(sets):
                out.append({"target": t, "label": f"Ticket {v + 1}", "section": section,
                            "reachable": True, "legs": a["legs"],
                            "odds": a["odds"], "prob": a["prob"]})
        return out

    # The ticket set for a day is frozen the first time it is built.
    #
    # Tickets come from a pool that only shrinks: once a game kicks off it leaves
    # SportyBet's board and its price is gone, so a set built at 10am cannot be
    # rebuilt at 5pm. Freezing is what lets won and busted slips stay on the page
    # — each request re-settles the same tickets against the latest results
    # instead of quietly dropping the ones whose games have finished.
    mode = ("all" if not sportybet_only else "sportybet") + (f"_{market}" if market else "")
    rebuild = request.args.get("rebuild") in ("1", "true", "yes")
    snapshot = load_build_snapshot(date_str)
    raw_tickets = None if rebuild else snapshot.get("modes", {}).get(mode)
    frozen = raw_tickets is not None

    if raw_tickets is None:
        raw_tickets = (build_section(pool, "all", 0)
                       + build_section(early_pool, "early", 1)
                       + build_section(played_pool, "played", 2))
        keep = [t for t in raw_tickets if t.get("legs")]
        if keep:
            snapshot.setdefault("modes", {})[mode] = keep
            snapshot["built"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            try:
                save_build_snapshot(date_str, snapshot)
            except OSError as exc:
                app.logger.warning("could not save the build snapshot: %s", exc)
    elif played_pool and not any(t.get("section") == "played" for t in raw_tickets):
        # A set frozen before this section existed: add the already-played tickets
        # once, then they are part of the frozen set like everything else.
        raw_tickets = raw_tickets + build_section(played_pool, "played", 2)
        snapshot.setdefault("modes", {})[mode] = [t for t in raw_tickets if t.get("legs")]
        try:
            save_build_snapshot(date_str, snapshot)
        except OSError as exc:
            app.logger.warning("could not save the build snapshot: %s", exc)

    tickets = [t if not t.get("legs")
               else settle_and_pack(t, t["target"], t["label"], t["section"])
               for t in raw_tickets]

    return jsonify({"date": date_str, "pool": len(pool),
                    "early_pool": len(early_pool), "tickets": tickets,
                    "played_pool": len(played_pool),
                    "sportybet_only": sportybet_only, "total_matches": total_preds,
                    "sportybet": sportybet_status,
                    "real_odds": inputs["real_odds"],
                    "sportybet_odds": inputs["sportybet_odds"],
                    "market": inputs["market"], "markets": inputs["markets"],
                    "frozen": frozen, "built": snapshot.get("built", "")})


# --- booking codes for a whole competition / hour ---------------------------
# The predictions and hourly panels are both Over/Under 1.5 panels: a match's
# pick is Over 1.5 when the model rates it 50%+ for goals, Under 1.5 otherwise.
# So a group's booking code is that same line across every match in the group
# that SportyBet actually prices — one slip per competition, or per hour.

_GROUP_MAX_LEGS = 20        # a slip longer than this is refused by the book


@app.route("/api/group-code")
def api_group_code():
    """Booking code for one competition or one hour of the panels.

    The caller passes the match ids it rendered, so the slip is exactly the
    group on screen — no re-grouping, no drift. Games SportyBet doesn't price,
    or that have already kicked off, are reported back rather than silently
    dropped.
    """
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    variant = "v" if request.args.get("variant") == "v" else ""
    field = "over_1.5v" if variant else "over_1.5"
    ids = [i for i in (request.args.get("ids") or "").replace(" ", "").split(",") if i]
    if not ids:
        return jsonify({"error": "no matches given"}), 400
    if len(ids) > _GROUP_MAX_LEGS or len(set(ids)) != len(ids):
        return jsonify({"error": "Too many games or repeated fixture IDs"}), 400

    try:
        import build_accas as ba
        import sportybet_betting as sbb
    except Exception as exc:
        return jsonify({"error": f"SportyBet client unavailable: {exc}"}), 500

    pred_file = CACHE_DIR / f"predictions_{date_str}.json"
    if not pred_file.exists():
        return jsonify({"error": "no predictions for this date"}), 400
    with open(pred_file, "r", encoding="utf-8") as f:
        by_id = {str(p.get("match_id", "")): p for p in json.load(f)}

    try:
        id_index = sbb.load_id_index()
    except Exception as exc:
        app.logger.warning("SportyBet ids unavailable for a group code: %s", exc)
        id_index = {}

    started = _started_match_ids(date_str)
    selections, legs, skipped = [], [], []

    for mid in ids:
        p = by_id.get(mid)
        if not p:
            skipped.append({"match": f"match {mid}",
                            "reason": "not in the prediction cache"})
            continue
        name = f"{p.get('home_team', '')} vs {p.get('away_team', '')}"
        prob = p.get(field)
        if prob is None:
            skipped.append({"match": name, "reason": "no model probability"})
            continue
        if mid in started:
            skipped.append({"match": name, "reason": "already kicked off"})
            continue

        side = "over" if prob >= 50 else "under"
        leg = {"match_id": mid, "match": name,
               "home": p.get("home_team", ""), "away": p.get("away_team", ""),
               "match_date": p.get("match_date", ""),
               "league": p.get("league", ""), "country": p.get("country", ""),
               "kick_off": to_local_hhmm(p.get("kick_off"), date_str),
               "kick_off_utc": p.get("kick_off", ""),
               # The panels only ever use the 1.5 line; the side is what changes.
               "field": "over_1.5",
               "selection": "Over 1.5" if side == "over" else "Under 1.5",
               "prob": round(prob if side == "over" else 100 - prob, 1)}

        event, _kind = sbb.find_event(id_index, p.get("home_team", ""),
                                      p.get("away_team", ""))
        ids_sb = ba.leg_ids_for(event, leg, side=side)
        if not ids_sb:
            if not event:
                reason = "SportyBet doesn't list this game"
            elif not (event.get("prices") or {}).get("over_1.5"):
                reason = "no Over/Under 1.5 price on SportyBet"
            else:
                reason = "couldn't match it to the SportyBet fixture"
            skipped.append({"match": name, "reason": reason})
            continue
        if len(selections) >= _GROUP_MAX_LEGS:
            skipped.append({"match": name,
                            "reason": f"slip is full ({_GROUP_MAX_LEGS} legs max)"})
            continue
        leg["odds"] = ids_sb["odds"]
        selections.append(ids_sb)
        legs.append(leg)

    # Never silently truncate a group and call its code the requested ticket.
    if skipped or len(selections) != len(ids):
        return jsonify({
            "error": "Exact group unavailable: some games are missing, started or unpriced",
            "skipped": skipped, "requested": len(ids), "resolved": len(selections),
        }), 409

    try:
        from services.bookmaker_verification import verify_and_reprice
        board = sbb.fetch_odds_table(save=True)  # fresh bookmaker board; fail closed
        quote = verify_and_reprice(selections, board, expected_legs=legs)
        for leg, q in zip(legs, quote["selections"]):
            leg["odds"] = q["odds"]
        code = sbb.booking_code_for(
            quote["selections"], quote_signature=quote["quote_signature"],
            deadline_ms=quote["deadline_ms"],
        )
    except Exception as exc:
        return jsonify({"error": str(exc), "legs": legs}), 409

    return jsonify({
        "date": date_str, "n_legs": len(legs), "odds": quote["odds"],
        "share_code": code["share_code"], "share_url": code["share_url"],
        "cached": code.get("cached", False), "variant": variant,
        "verification": code.get("verification", "issued_unverified"),
        "tracking_id": code["tracking_id"],
        "quote_checked_at_ms": quote["checked_at_ms"],
        "legs": legs, "skipped": [],
    })


@app.route("/api/booking/status")
def api_booking_status():
    """Read-only tracking for an issued code, including automatic renewals.

    No claim of a full SportyBet ticket readback is made. A cached code that
    crossed kickoff is marked expired. Upstream failures cannot be 'bookable'.
    """
    import hashlib
    import re
    import sportybet_betting as sbb
    from services.bookmaker_verification import ensure_not_expired
    ticket_id = (request.args.get("ticket_id") or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{24}", ticket_id):
        return jsonify({"error": "invalid booking tracking ID"}), 400
    saved = None
    identity = None
    for key, record in sbb.codes_cache().items():
        if hashlib.sha256(key.encode("utf-8")).hexdigest()[:24] == ticket_id:
            saved, identity = record, key
            break
    if not saved:
        return jsonify({"error": "booking code not found"}), 404
    status = saved.get("verification", "issued_unverified") if ensure_not_expired(saved) else "expired"
    recent_check = None
    watch_path = ROOT / "cache" / "booking_monitor_status.json"
    if watch_path.exists():
        try:
            with open(watch_path, "r", encoding="utf-8") as f:
                watch = json.load(f)
            record = (watch.get("tickets") or {}).get(identity)
            if record:
                recent_check = record.get("quote_checked_at_ms")
                if record.get("status") in ("unavailable", "error"):
                    status = record["status"]
        except (ValueError, OSError, TypeError):
            pass
    return jsonify({
        "ticket_id": ticket_id, "status": status,
        "share_code": saved.get("share_code") if status in ("readback_verified", "issued_unverified") else None,
        "share_url": saved.get("share_url") if status in ("readback_verified", "issued_unverified") else None,
        "updated_at_ms": saved.get("issued_at_ms"),
        "quote_checked_at_ms": recent_check,
        "deadline_ms": saved.get("deadline_ms"),
        "verification": saved.get("verification", "issued_unverified"),
    })


@app.route("/api/build/code")
def api_build_code():
    """Booking code for one ticket, created through SportyBet's share endpoint.

    The ticket is read back out of the same frozen set /api/build serves, so the
    code is always for the slip that was on screen even after its games have
    kicked off and left the board. `expect` is that ticket's fingerprint: if the
    slip changed under us we refuse rather than hand back a code for a different
    one. Legs that are still short of identifiers get another chance here, from
    the live board and the id ledger.
    """
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))
    section = request.args.get("section", "all")
    market = request.args.get("market", "")
    expect = request.args.get("expect", "")
    # Which table the ticket came from — /api/build defaults to SportyBet-only.
    all_games = request.args.get("all") in ("1", "true", "yes")
    try:
        target = float(request.args.get("target", "2"))
        index = int(request.args.get("index", "0"))
    except ValueError:
        return jsonify({"error": "target and index must be numbers"}), 400

    try:
        import sportybet_betting as sbb
    except Exception as exc:
        return jsonify({"error": f"SportyBet client unavailable: {exc}"}), 500

    inputs = _acca_inputs(date_str, sportybet_only=True, market=market)
    if "error" in inputs:
        return jsonify({"error": inputs["error"]}), 400

    ba = inputs["ba"]
    id_lookup = inputs.get("id_lookup")

    # The frozen set first: it is the slip the dashboard is showing.
    mode = ("all" if all_games else "sportybet") + (f"_{market}" if market else "")
    frozen = load_build_snapshot(date_str).get("modes", {}).get(mode) or []
    tickets = [t for t in frozen
               if (t.get("section") or "all") == section
               and abs(float(t.get("target", 0)) - target) < 1e-6
               and t.get("legs")]
    if not tickets:
        # No frozen set for this table (or an older one): fall back to rebuilding
        # the same seeded tickets /api/build would have produced.
        src = inputs["pool"] if section == "all" else (
            inputs["early"] if section == "early" else inputs["played"])
        seed_shift = {"all": 0, "early": 1, "played": 2}.get(section, 0)
        tickets = ba.build_ticket_set(src, target, n=10,
                                     seed=int(target * 100) + seed_shift)
    if index < 0 or index >= len(tickets):
        return jsonify({"error": f"no ticket {index} at ~{target:g} odds"}), 404

    legs = [dict(leg) for leg in tickets[index]["legs"]]
    for leg in legs:
        if leg.get("sb"):
            continue
        home, away = ba.leg_teams(leg)
        event, _kind = id_lookup(home, away) if (id_lookup and home) else (None, None)
        ids = ba.leg_ids_for(event, leg, started=section == "played")
        if ids:
            leg["sb"] = ids
    if not all(leg.get("sb") for leg in legs):
        stuck = [leg.get("match", "") for leg in legs if not leg.get("sb")]
        return jsonify({
            "error": "SportyBet has no price for " + ", ".join(stuck)
                     + " — it only issues a code for games on its own board.",
            "missing": stuck}), 409

    selections = [leg["sb"] for leg in legs]
    fingerprint = sbb.fingerprint(selections)
    if expect and expect != fingerprint:
        return jsonify({"error": "this ticket changed while you were looking at it "
                                 "(the odds moved) — reload and try again"}), 409

    try:
        from services.bookmaker_verification import verify_and_reprice
        # A frozen display ticket is not proof that its odds remain available.
        # Require a fresh upcoming board and exact event/market/fixture match.
        board = sbb.fetch_odds_table(save=True)
        for leg in legs:
            leg.setdefault("home", ba.leg_teams(leg)[0])
            leg.setdefault("away", ba.leg_teams(leg)[1])
        quote = verify_and_reprice(selections, board, expected_legs=legs)
        for leg, q in zip(legs, quote["selections"]):
            leg["odds"] = q["odds"]
        code = sbb.booking_code_for(
            quote["selections"], quote_signature=quote["quote_signature"],
            deadline_ms=quote["deadline_ms"],
        )
    except Exception as exc:
        return jsonify({"error": str(exc),
                        "message": "Ticket must be repriced on the active bookmaker board"}), 409

    return jsonify({
        "date": date_str, "section": section, "target": target, "index": index,
        "n_legs": len(legs), "odds": quote["odds"],
        "fingerprint": fingerprint, "cached": code.get("cached", False),
        "verification": code.get("verification", "issued_unverified"),
        "tracking_id": code["tracking_id"],
        "quote_checked_at_ms": quote["checked_at_ms"],
        "share_code": code["share_code"], "share_url": code["share_url"],
        "legs": [{"match": leg["match"], "selection": leg["selection"],
                  "odds": leg["odds"], "kick_off": leg["kick_off"]}
                 for leg in legs],
    })


@app.route("/api/settlement-report")
def api_settlement_report():
    """Read persisted final settlement totals without network calls or side effects."""
    import sqlite3
    from datetime import date as _date
    from services.settlement_service import init, settlement_report
    date_str = request.args.get("date") or datetime.now().strftime("%Y-%m-%d")
    try:
        _date.fromisoformat(date_str)
    except ValueError:
        return jsonify({"error": "date must be YYYY-MM-DD"}), 400
    db_path = ROOT / "data" / "fixtures" / "fixture_store.sqlite3"
    if not db_path.exists():
        return jsonify({"date": date_str, "status": "not_configured",
                        "message": "settlement database has not been initialized"}), 503
    try:
        with sqlite3.connect(db_path, timeout=5) as db:
            db.row_factory = sqlite3.Row
            init(db)
            result = settlement_report(db, date_str)
            latest = db.execute(
                "SELECT status,completed_utc,error FROM settlement_runs "
                "WHERE match_date=? ORDER BY id DESC LIMIT 1",
                (date_str,),
            ).fetchone()
            result["job"] = dict(latest) if latest else None
        return jsonify(result)
    except Exception as exc:
        app.logger.exception("Settlement report failed: %s", exc)
        return jsonify({"error": "settlement reporting unavailable"}), 503


@app.route("/api/bots")
def api_bots():
    """Return bot simulation results."""
    date_str = request.args.get("date")
    mode = request.args.get("mode", "latest")  # "latest", "day", "sim"
    refresh = request.args.get("refresh") in ("1", "true", "yes")

    if mode == "day" and date_str:
        result_file = BOT_RESULTS_DIR / f"bots_{date_str}.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return jsonify(settle_pending_bets(data, date_str, refresh=refresh))
        return jsonify({"error": "no results", "date": date_str})

    # Find all simulation results
    sim_files = sorted(BOT_RESULTS_DIR.glob("sim_*.json")) if BOT_RESULTS_DIR.exists() else []
    day_files = sorted(BOT_RESULTS_DIR.glob("bots_*.json")) if BOT_RESULTS_DIR.exists() else []

    if not sim_files and not day_files:
        return jsonify({"error": "no results", "available": []})

    # Latest simulation
    if sim_files:
        with open(sim_files[-1], "r", encoding="utf-8") as f:
            sim_data = json.load(f)

        # Aggregate across days
        bot_totals = {}
        for day_result in sim_data:
            if "error" in day_result:
                continue
            for bot in day_result.get("bots", []):
                name = bot["bot"]
                if name not in bot_totals:
                    bot_totals[name] = {
                        "bot": name, "emoji": bot["emoji"],
                        "description": bot["description"],
                        "wins": 0, "losses": 0, "total_bets": 0,
                        "best_chain": 0, "chains_completed": 0,
                        "daily": [],
                    }
                bt = bot_totals[name]
                bt["wins"] += bot["wins"]
                bt["losses"] += bot["losses"]
                bt["total_bets"] += bot["total_bets"]
                bt["best_chain"] = max(bt["best_chain"], bot["best_chain"])
                bt["chains_completed"] += bot["chains_completed"]
                bt["daily"].append({
                    "date": day_result["date"],
                    "wins": bot["wins"],
                    "losses": bot["losses"],
                    "best_chain": bot["best_chain"],
                    "win_rate": bot["win_rate"],
                    "history": bot.get("history", []),
                })

        for bt in bot_totals.values():
            bt["win_rate"] = round(bt["wins"] / bt["total_bets"] * 100, 1) if bt["total_bets"] else 0

        dates = [d["date"] for d in sim_data if "error" not in d]

        # Merge any single-day files not already in the sim
        for df in day_files:
            fname = df.stem  # e.g. bots_2026-06-14
            day_date = fname.replace("bots_", "")
            if day_date not in dates:
                with open(df, "r", encoding="utf-8") as f2:
                    day_data = json.load(f2)
                if isinstance(day_data, dict) and "bots" in day_data:
                    # Settle pending bets against live scores and refresh counts
                    day_data = settle_pending_bets(day_data, day_date)
                    for bot in day_data["bots"]:
                        hist = bot.get("history", [])
                        if hist:
                            bot["wins"] = sum(1 for h in hist if h.get("status") == "WIN")
                            bot["losses"] = sum(1 for h in hist if h.get("status") == "LOSS")
                            settled = bot["wins"] + bot["losses"]
                            bot["win_rate"] = round(bot["wins"] / settled * 100, 1) if settled else 0
                    dates.append(day_date)
                    for bot in day_data["bots"]:
                        name = bot["bot"]
                        if name not in bot_totals:
                            bot_totals[name] = {
                                "bot": name, "emoji": bot.get("emoji", ""),
                                "description": bot.get("description", ""),
                                "wins": 0, "losses": 0, "total_bets": 0,
                                "best_chain": 0, "chains_completed": 0,
                                "daily": [],
                            }
                        bt = bot_totals[name]
                        bt["wins"] += bot.get("wins", 0)
                        bt["losses"] += bot.get("losses", 0)
                        bt["total_bets"] += bot.get("total_bets", 0)
                        bt["best_chain"] = max(bt["best_chain"], bot.get("best_chain", 0))
                        bt["chains_completed"] += bot.get("chains_completed", 0)
                        bt["daily"].append({
                            "date": day_date,
                            "wins": bot.get("wins", 0),
                            "losses": bot.get("losses", 0),
                            "best_chain": bot.get("best_chain", 0),
                            "win_rate": bot.get("win_rate", 0),
                            "history": bot.get("history", []),
                        })
                    for bt in bot_totals.values():
                        bt["win_rate"] = round(bt["wins"] / bt["total_bets"] * 100, 1) if bt["total_bets"] else 0

        dates.sort()
        return jsonify({
            "type": "simulation",
            "dates": dates,
            "days": len(dates),
            "bots": sorted(bot_totals.values(), key=lambda x: -x["win_rate"]),
        })

    # Single day only
    with open(day_files[-1], "r", encoding="utf-8") as f:
        return jsonify(json.load(f))


@app.route("/api/bots/run", methods=["POST"])
def api_bots_run():
    """Trigger a bot simulation run."""
    import subprocess
    date_str = request.json.get("date") if request.json else None
    days = request.json.get("days", 1) if request.json else 1

    cmd = [sys.executable, str(ROOT / "bot_system.py")]
    if days > 1:
        cmd += ["--simulate", "--days", str(days)]
    if date_str:
        cmd += ["--date", date_str]

    try:
        proc = subprocess.Popen(cmd, cwd=str(ROOT),
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return jsonify({"status": "started", "pid": proc.pid})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/api/bots/available")
def api_bots_available():
    """List available bot result files."""
    files = []
    if BOT_RESULTS_DIR.exists():
        for f in sorted(BOT_RESULTS_DIR.glob("*.json")):
            files.append({
                "name": f.name,
                "type": "simulation" if f.name.startswith("sim_") else "day",
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
            })
    return jsonify({"files": files})


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-reload", action="store_true",
                        help="disable the auto-reloader (use while a long job runs, "
                             "so a file change cannot interrupt it)")
    args = parser.parse_args()

    # Auto-reload: the dev server restarts itself when a watched .py file changes,
    # so code edits take effect without a manual restart. Templates are covered by
    # TEMPLATES_AUTO_RELOAD and static files are always read fresh, so this closes
    # every path by which "the changes didn't apply".
    use_reload = not args.no_reload
    app.config["RUNNING_WITH_RELOADER"] = use_reload
    print(f"BetSightly Dashboard: http://localhost:{args.port}")
    print("  auto-reload " + ("ON - edits apply without restarting" if use_reload
                              else "OFF - restart manually after code changes"))
    app.run(host="0.0.0.0", port=args.port, debug=args.debug, use_reloader=use_reload)
