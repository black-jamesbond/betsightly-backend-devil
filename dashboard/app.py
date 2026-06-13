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
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path

from flask import Flask, render_template, jsonify, request

# Project root
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np

app = Flask(__name__)

# --- Paths ---------------------------------------------------------------
CACHE_DIR = ROOT / "cache" / "apifootball"
API_FOOTBALL_DATA = ROOT / "data" / "api-football" / "matches.csv"
GITHUB_DATA = ROOT / "data" / "github-football" / "Matches.csv"
APIFOOTBALL_HISTORY = ROOT / "data" / "apifootball" / "history.csv"
CHECKPOINT = ROOT / "data" / "apifootball" / "fetch_checkpoint.json"
MODEL_DIR = ROOT / "models" / "api_football"
BOT_RESULTS_DIR = ROOT / "bot_results"
API_KEY = "7233da77b5d53606a174f2759f0d947beb18aa17c00bf350d852f35b80aa7455"

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


def fetch_fixtures(date_str: str) -> list:
    if date_str in _fixtures_cache:
        return _fixtures_cache[date_str]

    cache_file = CACHE_DIR / f"events_{date_str}.json"
    if cache_file.exists():
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
    return render_template("index.html")


@app.route("/api/predictions")
def api_predictions():
    date_str = request.args.get("date", datetime.now().strftime("%Y-%m-%d"))

    pred_cache = CACHE_DIR / f"predictions_{date_str}.json"
    if pred_cache.exists():
        with open(pred_cache, "r", encoding="utf-8") as f:
            predictions = json.load(f)
        return jsonify({"date": date_str, "predictions": predictions, "source": "cache"})

    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        return jsonify({"date": date_str, "predictions": [], "source": "none"})

    upcoming = [m for m in fixtures if m.get("match_hometeam_score", "") == ""]
    finished = [m for m in fixtures if m.get("match_hometeam_score", "") != ""
                and m.get("match_status") == "Finished"]

    hist = load_historical()
    tc = {}
    predictions = []

    for m in upcoming:
        home = m["match_hometeam_name"]
        away = m["match_awayteam_name"]
        if hist is not None:
            if home not in tc:
                tc[home] = find_team_stats(home, hist)
            if away not in tc:
                tc[away] = find_team_stats(away, hist)
            pred = predict_match(home, away, tc[home], tc[away])
        else:
            pred = predict_match(home, away, {"found": False, "matches": 0},
                                 {"found": False, "matches": 0})
        pred["home_team"] = home
        pred["away_team"] = away
        pred["league"] = m.get("league_name", "")
        pred["country"] = m.get("country_name", "")
        pred["kick_off"] = m.get("match_time", "")
        pred["match_id"] = m.get("match_id", "")
        pred["status"] = "upcoming"
        predictions.append(pred)

    for m in finished:
        predictions.append({
            "home_team": m["match_hometeam_name"],
            "away_team": m["match_awayteam_name"],
            "league": m.get("league_name", ""),
            "country": m.get("country_name", ""),
            "kick_off": m.get("match_time", ""),
            "match_id": m.get("match_id", ""),
            "status": "finished",
            "home_score": m.get("match_hometeam_score", ""),
            "away_score": m.get("match_awayteam_score", ""),
        })

    predictions.sort(key=lambda x: x.get("kick_off", ""))
    return jsonify({"date": date_str, "predictions": predictions,
                    "total_fixtures": len(fixtures), "source": "live"})


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
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        return jsonify({"hours": {}})

    by_hour = defaultdict(list)
    for m in fixtures:
        t = m.get("match_time", "00:00")
        hour = t[:2] if len(t) >= 2 else "00"
        by_hour[hour].append({
            "home": m["match_hometeam_name"],
            "away": m["match_awayteam_name"],
            "league": m.get("league_name", ""),
            "country": m.get("country_name", ""),
            "time": t,
            "status": m.get("match_status", ""),
            "home_score": m.get("match_hometeam_score", ""),
            "away_score": m.get("match_awayteam_score", ""),
        })

    return jsonify({"date": date_str, "hours": dict(sorted(by_hour.items())),
                    "total": len(fixtures)})


@app.route("/api/bots")
def api_bots():
    """Return bot simulation results."""
    date_str = request.args.get("date")
    mode = request.args.get("mode", "latest")  # "latest", "day", "sim"

    if mode == "day" and date_str:
        result_file = BOT_RESULTS_DIR / f"bots_{date_str}.json"
        if result_file.exists():
            with open(result_file, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
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
        return jsonify({
            "type": "simulation",
            "dates": dates,
            "days": len(dates),
            "bots": sorted(bot_totals.values(), key=lambda x: -x["win_rate"]),
            "raw": sim_data,
        })

    # Single day
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
    args = parser.parse_args()
    print(f"BetSightly Dashboard: http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=args.debug)
