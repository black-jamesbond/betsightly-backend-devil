"""
Evaluate prediction accuracy against actual match results.

Fetches finished matches from apifootball.com for recent days,
runs predictions on each, and compares against actual outcomes.

Usage:
    python evaluate_accuracy.py [--days 7] [--date YYYY-MM-DD]
"""
from __future__ import annotations

import json
import math
import re
import sys
import argparse
from collections import defaultdict
from datetime import datetime, date, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

API_KEY = "7233da77b5d53606a174f2759f0d947beb18aa17c00bf350d852f35b80aa7455"
CACHE_DIR = ROOT / "cache" / "apifootball"
GITHUB_DATA = ROOT / "data" / "github-football" / "Matches.csv"
API_FOOTBALL_DATA = ROOT / "data" / "api-football" / "matches.csv"
APIFOOTBALL_HISTORY = ROOT / "data" / "apifootball" / "history.csv"
MODEL_DIR = ROOT / "models" / "api_football"

RESULTS_DIR = ROOT / "evaluation"


def _norm(name: str) -> str:
    n = str(name).strip().lower()
    for suffix in [" fc", " sc", " cf", " ac", " fk", " sk", " if", " bk"]:
        if n.endswith(suffix):
            n = n[: -len(suffix)].strip()
    n = re.sub(r"['\-\.]", " ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def fetch_day(date_str: str) -> list:
    cache_file = CACHE_DIR / f"events_{date_str}.json"
    if cache_file.exists():
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    import urllib.request
    url = (
        f"https://apiv3.apifootball.com/?action=get_events"
        f"&from={date_str}&to={date_str}&APIkey={API_KEY}"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode()
        data = json.loads(raw)
        if isinstance(data, dict):
            return []
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(cache_file, "w", encoding="utf-8") as f:
            f.write(raw)
        return data
    except Exception as e:
        print(f"  Error fetching {date_str}: {e}")
        return []


def load_historical():
    frames = []
    if GITHUB_DATA.exists():
        df = pd.read_csv(GITHUB_DATA, low_memory=False)
        df = df.rename(columns={
            "HomeTeam": "home_team", "AwayTeam": "away_team",
            "MatchDate": "date", "FTHome": "home_score", "FTAway": "away_score",
        })
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
        df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
        df = df.dropna(subset=["date", "home_score", "away_score", "home_team", "away_team"])
        df["source"] = "github"
        frames.append(df)
        print(f"  GitHub: {len(df):,} matches")

    if API_FOOTBALL_DATA.exists():
        df2 = pd.read_csv(API_FOOTBALL_DATA, low_memory=False)
        df2["date"] = pd.to_datetime(df2["date"], errors="coerce")
        df2["home_score"] = pd.to_numeric(df2["home_score"], errors="coerce")
        df2["away_score"] = pd.to_numeric(df2["away_score"], errors="coerce")
        df2 = df2.dropna(subset=["date", "home_score", "away_score"])
        df2["source"] = "api-football"
        frames.append(df2)
        print(f"  API-Football: {len(df2):,} matches")

    if APIFOOTBALL_HISTORY.exists():
        df3 = pd.read_csv(APIFOOTBALL_HISTORY, low_memory=False)
        df3["date"] = pd.to_datetime(df3["date"], errors="coerce")
        df3["home_score"] = pd.to_numeric(df3["home_score"], errors="coerce")
        df3["away_score"] = pd.to_numeric(df3["away_score"], errors="coerce")
        df3 = df3.dropna(subset=["date", "home_score", "away_score"])
        df3["source"] = "apifootball"
        frames.append(df3)
        print(f"  apifootball history: {len(df3):,} matches")

    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    combined["_home_norm"] = combined["home_team"].apply(_norm)
    combined["_away_norm"] = combined["away_team"].apply(_norm)
    print(f"  Total: {len(combined):,} matches loaded")
    return combined


def find_team_stats(team_name, hist, cutoff_date=None):
    norm = _norm(team_name)
    h = hist
    if cutoff_date is not None:
        h = hist[hist["date"] < cutoff_date]

    home_mask = h["_home_norm"] == norm
    away_mask = h["_away_norm"] == norm
    if home_mask.sum() + away_mask.sum() == 0:
        home_mask = h["_home_norm"].str.contains(norm, na=False, regex=False)
        away_mask = h["_away_norm"].str.contains(norm, na=False, regex=False)
    total = home_mask.sum() + away_mask.sum()
    if total == 0:
        return {"found": False, "matches": 0}

    hg = h[home_mask].tail(30)
    ag = h[away_mask].tail(30)

    # Recency weights
    def _weights(n):
        if n <= 1:
            return np.ones(n)
        return np.linspace(0.3, 1.0, n)
    hw_w = _weights(len(hg))
    aw_w = _weights(len(ag))

    h_win_m = (hg["home_score"].values > hg["away_score"].values).astype(float)
    h_draw_m = (hg["home_score"].values == hg["away_score"].values).astype(float)
    h_loss_m = (hg["home_score"].values < hg["away_score"].values).astype(float)
    a_win_m = (ag["away_score"].values > ag["home_score"].values).astype(float)
    a_draw_m = (ag["home_score"].values == ag["away_score"].values).astype(float)
    a_loss_m = (ag["away_score"].values < ag["home_score"].values).astype(float)

    wt_h = hw_w.sum()
    wt_a = aw_w.sum()
    wt = wt_h + wt_a

    tw = (h_win_m * hw_w).sum() + (a_win_m * aw_w).sum()
    td = (h_draw_m * hw_w).sum() + (a_draw_m * aw_w).sum()
    tl = (h_loss_m * hw_w).sum() + (a_loss_m * aw_w).sum()

    tg = len(hg) + len(ag)
    gs_h = hg["home_score"].sum() if len(hg) else 0
    gc_h = hg["away_score"].sum() if len(hg) else 0
    gs_a = ag["away_score"].sum() if len(ag) else 0
    gc_a = ag["home_score"].sum() if len(ag) else 0
    avg_s = (gs_h + gs_a) / tg if tg else 1.2
    avg_c = (gc_h + gc_a) / tg if tg else 1.2

    hwr = (h_win_m * hw_w).sum() / wt_h if wt_h > 0 else 0.33
    awr = (a_win_m * aw_w).sum() / wt_a if wt_a > 0 else 0.33

    h_btts = ((hg["home_score"] > 0) & (hg["away_score"] > 0)).sum()
    a_btts = ((ag["home_score"] > 0) & (ag["away_score"] > 0)).sum()
    btts_r = (h_btts + a_btts) / tg if tg else 0.5

    h_o25 = ((hg["home_score"] + hg["away_score"]) > 2.5).sum()
    a_o25 = ((ag["home_score"] + ag["away_score"]) > 2.5).sum()
    o25_r = (h_o25 + a_o25) / tg if tg else 0.45

    # Recent form (last 5)
    rh = hg.tail(5)
    ra = ag.tail(5)
    rn = len(rh) + len(ra)
    if rn > 0:
        rw = ((rh["home_score"] > rh["away_score"]).sum() +
              (ra["away_score"] > ra["home_score"]).sum())
        rd_r = ((rh["home_score"] == rh["away_score"]).sum() +
                (ra["home_score"] == ra["away_score"]).sum())
        recent_form = (rw * 3 + rd_r) / (rn * 3)
    else:
        recent_form = 0.33

    return {
        "found": True, "matches": total,
        "win_rate": tw / wt if wt > 0 else 0.33,
        "draw_rate": td / wt if wt > 0 else 0.33,
        "loss_rate": tl / wt if wt > 0 else 0.33,
        "home_win_rate": hwr, "away_win_rate": awr,
        "avg_scored": avg_s, "avg_conceded": avg_c,
        "btts_rate": btts_r, "over25_rate": o25_r,
        "recent_form": recent_form,
    }


BASE_HOME, BASE_DRAW, BASE_AWAY = 0.36, 0.28, 0.36
BASE_O25, BASE_BTTS = 0.47, 0.47
HOME_ADV = 0.03


def predict_match(home_stats, away_stats):
    both = home_stats["found"] and away_stats["found"]
    either = home_stats["found"] or away_stats["found"]

    if both:
        h_n = min(home_stats["matches"], 50)
        a_n = min(away_stats["matches"], 50)
        h_s = h_n / (h_n + 15)
        a_s = a_n / (a_n + 15)

        h_atk = h_s * home_stats["avg_scored"] + (1 - h_s) * 1.2
        h_def = h_s * home_stats["avg_conceded"] + (1 - h_s) * 1.2
        a_atk = a_s * away_stats["avg_scored"] + (1 - a_s) * 1.2
        a_def = a_s * away_stats["avg_conceded"] + (1 - a_s) * 1.2

        ehg = max(0.3, min(3.5, (h_atk * a_def) / 1.2))
        eag = max(0.3, min(3.5, (a_atk * h_def) / 1.2))

        sd = (home_stats["win_rate"] - home_stats["loss_rate"]) - \
             (away_stats["win_rate"] - away_stats["loss_rate"])
        fd = home_stats.get("recent_form", 0.33) - away_stats.get("recent_form", 0.33)
        cd = 0.6 * sd + 0.4 * fd

        hp = BASE_HOME + cd * 0.3
        ap = BASE_AWAY - cd * 0.3
        closeness = 1.0 - min(1.0, abs(cd))
        team_draw_avg = (home_stats["draw_rate"] + away_stats["draw_rate"]) / 2
        dp = 0.5 * (BASE_DRAW * (0.6 + 0.8 * closeness)) + 0.5 * team_draw_avg

        min_m = min(h_n, a_n)
        conf = min(0.7, 0.2 + 0.01 * min_m) + min(0.3, abs(hp - ap) * 0.5)
    elif home_stats["found"]:
        s = min(home_stats["matches"], 50) / (min(home_stats["matches"], 50) + 15)
        hp = s * home_stats["home_win_rate"] + (1 - s) * BASE_HOME
        dp = s * home_stats["draw_rate"] + (1 - s) * BASE_DRAW
        ap = 1 - hp - dp
        ehg = s * home_stats["avg_scored"] + (1 - s) * 1.2
        eag = s * home_stats["avg_conceded"] + (1 - s) * 1.2
        conf = 0.25
    elif away_stats["found"]:
        s = min(away_stats["matches"], 50) / (min(away_stats["matches"], 50) + 15)
        ap = s * away_stats["away_win_rate"] + (1 - s) * BASE_AWAY
        dp = s * away_stats["draw_rate"] + (1 - s) * BASE_DRAW
        hp = 1 - ap - dp
        ehg = s * away_stats["avg_conceded"] + (1 - s) * 1.2
        eag = s * away_stats["avg_scored"] + (1 - s) * 1.2
        conf = 0.25
    else:
        hp, dp, ap = BASE_HOME, BASE_DRAW, BASE_AWAY
        ehg, eag = 1.2, 1.0
        conf = 0.10

    hp += HOME_ADV; ap -= HOME_ADV * 0.6; dp -= HOME_ADV * 0.4
    hp = max(0.08, hp); dp = max(0.12, dp); ap = max(0.08, ap)
    t = hp + dp + ap
    hp, dp, ap = hp / t, dp / t, ap / t

    et = (ehg + eag) * 0.92
    o15 = 1 - math.exp(-et) * (1 + et)
    o25 = 1 - math.exp(-et) * (1 + et + et ** 2 / 2)
    if both:
        o25 = 0.4 * o25 + 0.3 * (home_stats["over25_rate"] + away_stats["over25_rate"]) / 2 + 0.3 * BASE_O25
    else:
        o25 = 0.5 * o25 + 0.5 * BASE_O25
    o25 = max(0.1, min(0.85, o25))

    if both:
        bh = (home_stats["btts_rate"] + away_stats["btts_rate"]) / 2
        sf = min(1.0, (home_stats["avg_scored"] * away_stats["avg_scored"]) / 1.44)
        btts = 0.4 * bh + 0.3 * sf * 0.6 + 0.3 * BASE_BTTS
    else:
        btts = BASE_BTTS
    btts = max(0.15, min(0.80, btts))

    margin = abs(hp - ap)
    if margin < 0.06 and dp > 0.22:
        best = "Draw"
    elif margin < 0.04:
        best = "Draw"
    else:
        rm = {"Home": hp, "Draw": dp, "Away": ap}
        best = max(rm, key=rm.get)
    dq = "HIGH" if (both and min(home_stats["matches"], away_stats["matches"]) >= 10) \
        else "MEDIUM" if either else "LOW"
    conf = max(0.05, min(0.95, conf))

    return {
        "home_win_p": hp, "draw_p": dp, "away_win_p": ap,
        "prediction": best,
        "over25_p": o25, "btts_p": btts,
        "confidence": conf, "data_quality": dq,
        "exp_home_goals": ehg, "exp_away_goals": eag,
    }


def evaluate_ml_models(matches, hist):
    """Try to load and evaluate the 24 trained ML models."""
    meta_path = MODEL_DIR / "meta.json"
    if not meta_path.exists():
        return {}

    try:
        import joblib
    except ImportError:
        print("  joblib not installed, skipping ML model evaluation")
        return {}

    with open(meta_path, "r") as f:
        meta = json.load(f)

    feature_cols = meta.get("feature_columns", [])
    model_names = meta.get("models", [])
    result_classes = meta.get("result_classes", ["away", "draw", "home"])

    results = {}
    for mname in model_names:
        mpath = MODEL_DIR / f"{mname}.joblib"
        if not mpath.exists():
            continue
        try:
            model = joblib.load(mpath)
            results[mname] = {"correct": 0, "total": 0, "predictions": []}
        except Exception as e:
            print(f"  Failed to load {mname}: {e}")

    if not results:
        return {}

    print(f"\n  Evaluating {len(results)} ML models...")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=3, help="Number of recent days to evaluate")
    parser.add_argument("--date", default=None, help="Specific date YYYY-MM-DD")
    args = parser.parse_args()

    print("=" * 70)
    print("  BETSIGHTLY MODEL ACCURACY EVALUATION")
    print("=" * 70)

    # Determine date range
    if args.date:
        dates = [args.date]
    else:
        today = date.today()
        dates = [(today - timedelta(days=i)).isoformat() for i in range(args.days)]

    print(f"\n  Evaluating: {', '.join(dates)}")

    # Load historical data
    print("\n[1/4] Loading historical data...")
    hist = load_historical()
    if hist is None:
        print("ERROR: No historical data found!")
        return

    # Fetch finished matches
    print("\n[2/4] Fetching match results...")
    all_finished = []
    for d in dates:
        print(f"  {d}...", end=" ", flush=True)
        fixtures = fetch_day(d)
        finished = [
            m for m in fixtures
            if m.get("match_status") == "Finished"
            and m.get("match_hometeam_score", "") != ""
            and m.get("match_awayteam_score", "") != ""
        ]
        print(f"{len(finished)} finished matches")
        for m in finished:
            try:
                hs = int(m["match_hometeam_score"])
                as_ = int(m["match_awayteam_score"])
            except (ValueError, KeyError):
                continue
            actual_result = "Home" if hs > as_ else "Away" if as_ > hs else "Draw"
            total_goals = hs + as_
            all_finished.append({
                "date": d,
                "home_team": m["match_hometeam_name"],
                "away_team": m["match_awayteam_name"],
                "league": m.get("league_name", ""),
                "country": m.get("country_name", ""),
                "home_score": hs,
                "away_score": as_,
                "actual_result": actual_result,
                "actual_total_goals": total_goals,
                "actual_over25": total_goals > 2.5,
                "actual_over15": total_goals > 1.5,
                "actual_btts": hs > 0 and as_ > 0,
            })

    print(f"\n  Total finished matches to evaluate: {len(all_finished)}")
    if not all_finished:
        print("No finished matches found!")
        return

    # Generate predictions and compare
    print("\n[3/4] Running predictions and comparing...")
    team_cache = {}
    results = {
        "match_result": {"correct": 0, "total": 0},
        "over_2.5": {"correct": 0, "total": 0},
        "over_1.5": {"correct": 0, "total": 0},
        "btts": {"correct": 0, "total": 0},
    }

    # Breakdown by confidence level
    by_confidence = {
        "high": {"result_correct": 0, "result_total": 0, "o25_correct": 0, "o25_total": 0,
                 "btts_correct": 0, "btts_total": 0},
        "medium": {"result_correct": 0, "result_total": 0, "o25_correct": 0, "o25_total": 0,
                   "btts_correct": 0, "btts_total": 0},
        "low": {"result_correct": 0, "result_total": 0, "o25_correct": 0, "o25_total": 0,
                "btts_correct": 0, "btts_total": 0},
    }

    # Breakdown by data quality
    by_quality = {
        "HIGH": {"result_correct": 0, "result_total": 0},
        "MEDIUM": {"result_correct": 0, "result_total": 0},
        "LOW": {"result_correct": 0, "result_total": 0},
    }

    # Breakdown by prediction type
    by_pred_type = {
        "Home": {"correct": 0, "total": 0},
        "Draw": {"correct": 0, "total": 0},
        "Away": {"correct": 0, "total": 0},
    }

    # Per-league accuracy
    by_league = defaultdict(lambda: {"correct": 0, "total": 0})

    match_details = []

    for i, m in enumerate(all_finished):
        home = m["home_team"]
        away = m["away_team"]

        if home not in team_cache:
            team_cache[home] = find_team_stats(home, hist)
        if away not in team_cache:
            team_cache[away] = find_team_stats(away, hist)

        pred = predict_match(team_cache[home], team_cache[away])

        # Match result accuracy
        result_correct = pred["prediction"] == m["actual_result"]
        results["match_result"]["correct"] += int(result_correct)
        results["match_result"]["total"] += 1

        # Over 2.5 accuracy
        pred_o25 = pred["over25_p"] > 0.5
        o25_correct = pred_o25 == m["actual_over25"]
        results["over_2.5"]["correct"] += int(o25_correct)
        results["over_2.5"]["total"] += 1

        # Over 1.5
        ehg = pred["exp_home_goals"]
        eag = pred["exp_away_goals"]
        et = ehg + eag
        o15_p = 1 - math.exp(-et) * (1 + et)
        pred_o15 = o15_p > 0.5
        o15_correct = pred_o15 == m["actual_over15"]
        results["over_1.5"]["correct"] += int(o15_correct)
        results["over_1.5"]["total"] += 1

        # BTTS accuracy
        pred_btts = pred["btts_p"] > 0.5
        btts_correct = pred_btts == m["actual_btts"]
        results["btts"]["correct"] += int(btts_correct)
        results["btts"]["total"] += 1

        # Confidence breakdown
        conf = pred["confidence"]
        if conf >= 0.6:
            bucket = "high"
        elif conf >= 0.4:
            bucket = "medium"
        else:
            bucket = "low"
        by_confidence[bucket]["result_correct"] += int(result_correct)
        by_confidence[bucket]["result_total"] += 1
        by_confidence[bucket]["o25_correct"] += int(o25_correct)
        by_confidence[bucket]["o25_total"] += 1
        by_confidence[bucket]["btts_correct"] += int(btts_correct)
        by_confidence[bucket]["btts_total"] += 1

        # Data quality breakdown
        dq = pred["data_quality"]
        by_quality[dq]["result_correct"] += int(result_correct)
        by_quality[dq]["result_total"] += 1

        # Prediction type breakdown
        by_pred_type[pred["prediction"]]["correct"] += int(result_correct)
        by_pred_type[pred["prediction"]]["total"] += 1

        # League breakdown
        lg = f"{m['league']} ({m['country']})"
        by_league[lg]["correct"] += int(result_correct)
        by_league[lg]["total"] += 1

        match_details.append({
            "date": m["date"],
            "home_team": home,
            "away_team": away,
            "league": m["league"],
            "country": m["country"],
            "actual_score": f"{m['home_score']}-{m['away_score']}",
            "actual_result": m["actual_result"],
            "predicted_result": pred["prediction"],
            "result_correct": result_correct,
            "home_win_p": round(pred["home_win_p"] * 100, 1),
            "draw_p": round(pred["draw_p"] * 100, 1),
            "away_win_p": round(pred["away_win_p"] * 100, 1),
            "confidence": round(conf * 100, 1),
            "data_quality": dq,
            "over25_p": round(pred["over25_p"] * 100, 1),
            "actual_over25": m["actual_over25"],
            "over25_correct": o25_correct,
            "btts_p": round(pred["btts_p"] * 100, 1),
            "actual_btts": m["actual_btts"],
            "btts_correct": btts_correct,
        })

        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{len(all_finished)} evaluated")

    # Print results
    print(f"\n{'=' * 70}")
    print(f"  ACCURACY RESULTS ({len(all_finished)} matches)")
    print(f"{'=' * 70}")

    print(f"\n  OVERALL ACCURACY:")
    print(f"  {'-' * 50}")
    for key, val in results.items():
        pct = val["correct"] / val["total"] * 100 if val["total"] > 0 else 0
        print(f"  {key:<20} {val['correct']:>5}/{val['total']:<5}  = {pct:>5.1f}%")

    print(f"\n  BY CONFIDENCE LEVEL:")
    print(f"  {'-' * 50}")
    for bucket in ["high", "medium", "low"]:
        b = by_confidence[bucket]
        if b["result_total"] == 0:
            continue
        r_pct = b["result_correct"] / b["result_total"] * 100
        o_pct = b["o25_correct"] / b["o25_total"] * 100 if b["o25_total"] > 0 else 0
        bt_pct = b["btts_correct"] / b["btts_total"] * 100 if b["btts_total"] > 0 else 0
        print(f"  {bucket.upper():<10} ({b['result_total']:>4} matches)  "
              f"Result: {r_pct:5.1f}%  |  O2.5: {o_pct:5.1f}%  |  BTTS: {bt_pct:5.1f}%")

    print(f"\n  BY DATA QUALITY:")
    print(f"  {'-' * 50}")
    for dq in ["HIGH", "MEDIUM", "LOW"]:
        b = by_quality[dq]
        if b["result_total"] == 0:
            continue
        pct = b["result_correct"] / b["result_total"] * 100
        print(f"  {dq:<10} ({b['result_total']:>4} matches)  Result: {pct:5.1f}%")

    print(f"\n  BY PREDICTION TYPE:")
    print(f"  {'-' * 50}")
    for pt in ["Home", "Draw", "Away"]:
        b = by_pred_type[pt]
        if b["total"] == 0:
            continue
        pct = b["correct"] / b["total"] * 100
        print(f"  {pt:<10} ({b['total']:>4} predictions)  Accuracy: {pct:5.1f}%")

    # Top and bottom leagues
    league_acc = [(lg, d["correct"] / d["total"] * 100, d["total"])
                  for lg, d in by_league.items() if d["total"] >= 5]
    league_acc.sort(key=lambda x: -x[1])

    if league_acc:
        print(f"\n  TOP 15 LEAGUES (>=5 matches):")
        print(f"  {'-' * 50}")
        for lg, acc, cnt in league_acc[:15]:
            safe = lg.encode("ascii", "replace").decode()
            print(f"  {safe:<40} {acc:5.1f}%  ({cnt} matches)")

        print(f"\n  BOTTOM 15 LEAGUES (>=5 matches):")
        print(f"  {'-' * 50}")
        for lg, acc, cnt in league_acc[-15:]:
            safe = lg.encode("ascii", "replace").decode()
            print(f"  {safe:<40} {acc:5.1f}%  ({cnt} matches)")

    # Save detailed results
    RESULTS_DIR.mkdir(exist_ok=True)
    out_file = RESULTS_DIR / f"eval_{dates[0]}_to_{dates[-1]}.json"
    eval_summary = {
        "dates": dates,
        "total_matches": len(all_finished),
        "overall": {k: {"correct": v["correct"], "total": v["total"],
                        "accuracy": round(v["correct"] / v["total"] * 100, 2) if v["total"] > 0 else 0}
                    for k, v in results.items()},
        "by_confidence": {
            bucket: {
                "matches": b["result_total"],
                "result_accuracy": round(b["result_correct"] / b["result_total"] * 100, 2) if b["result_total"] > 0 else 0,
                "over25_accuracy": round(b["o25_correct"] / b["o25_total"] * 100, 2) if b["o25_total"] > 0 else 0,
                "btts_accuracy": round(b["btts_correct"] / b["btts_total"] * 100, 2) if b["btts_total"] > 0 else 0,
            } for bucket, b in by_confidence.items()
        },
        "by_quality": {
            dq: {
                "matches": b["result_total"],
                "result_accuracy": round(b["result_correct"] / b["result_total"] * 100, 2) if b["result_total"] > 0 else 0,
            } for dq, b in by_quality.items()
        },
        "by_prediction_type": {
            pt: {
                "predictions": b["total"],
                "accuracy": round(b["correct"] / b["total"] * 100, 2) if b["total"] > 0 else 0,
            } for pt, b in by_pred_type.items()
        },
        "league_accuracy": [
            {"league": lg, "accuracy": round(acc, 2), "matches": cnt}
            for lg, acc, cnt in league_acc
        ],
        "match_details": match_details,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  Detailed results saved to {out_file}")

    # Also save as CSV for easy viewing
    csv_file = RESULTS_DIR / f"eval_{dates[0]}_to_{dates[-1]}.csv"
    pd.DataFrame(match_details).to_csv(csv_file, index=False)
    print(f"  Match-by-match CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
