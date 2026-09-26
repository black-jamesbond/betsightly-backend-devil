"""
Calibration Harness — measures how well BetSightly's predicted probabilities
match reality, WITHOUT look-ahead bias.

For every test match on date D, team stats are computed using ONLY matches that
finished before D (strict cutoff, same-day excluded). We then call the real
prediction engine (predict_today.predict_match) and compare its probabilities to
what actually happened.

Outputs (per market: 1X2, Over 2.5, BTTS):
  - Brier score       (0 = perfect, lower is better)
  - Log loss          (lower is better)
  - ECE               (expected calibration error, 0 = perfectly calibrated)
  - Reliability table  (predicted prob vs actual frequency, 10 bins)
  - Skill vs base-rate baseline

Also evaluates the CONFIDENCE heuristic: does higher confidence => higher
top-pick accuracy?

Usage:
    python calibration_harness.py [--years 3] [--min-games 3] [--out report.json]
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import predict_today as pt


# ----------------------------------------------------------------------------
# As-of team stats — faithful reimplementation of predict_today.find_team_stats,
# but restricted to matches strictly before a cutoff date. Validated against the
# original in validate_fidelity().
# ----------------------------------------------------------------------------
LAST_N = 30


def build_indices(hist: pd.DataFrame):
    """Per-team chronological indices for fast as-of lookups."""
    for c in ("home_elo", "away_elo"):
        if c not in hist.columns:
            hist[c] = np.nan

    dates = hist["date"].values.astype("datetime64[ns]")
    hn = hist["_home_norm"].values
    an = hist["_away_norm"].values
    hs = hist["home_score"].values.astype(float)
    as_ = hist["away_score"].values.astype(float)
    helo = hist["home_elo"].values.astype(float)
    aelo = hist["away_elo"].values.astype(float)

    home_dates = defaultdict(list)
    home_recs = defaultdict(list)   # (hs, as, home_elo)
    away_dates = defaultdict(list)
    away_recs = defaultdict(list)   # (hs, as, away_elo)
    h2h_dates = defaultdict(list)
    h2h_recs = defaultdict(list)    # (home_norm, hs, as)

    n = len(hist)
    for i in range(n):
        d = dates[i]
        h, a = hn[i], an[i]
        sh, sa = hs[i], as_[i]
        home_dates[h].append(d)
        home_recs[h].append((sh, sa, helo[i]))
        away_dates[a].append(d)
        away_recs[a].append((sh, sa, aelo[i]))
        key = (h, a) if h <= a else (a, h)
        h2h_dates[key].append(d)
        h2h_recs[key].append((h, sh, sa))

    return {
        "home_dates": home_dates, "home_recs": home_recs,
        "away_dates": away_dates, "away_recs": away_recs,
        "h2h_dates": h2h_dates, "h2h_recs": h2h_recs,
    }


def _weights(n):
    if n <= 1:
        return np.ones(n)
    return np.linspace(0.3, 1.0, n)


def team_stats_asof(norm, cutoff, idx):
    """Replicates find_team_stats using only games before `cutoff`."""
    hd = idx["home_dates"].get(norm)
    ad = idx["away_dates"].get(norm)
    home_slice = []
    away_slice = []
    j_home = j_away = 0
    if hd:
        j_home = bisect.bisect_left(hd, cutoff)
        home_slice = idx["home_recs"][norm][max(0, j_home - LAST_N):j_home]
    if ad:
        j_away = bisect.bisect_left(ad, cutoff)
        away_slice = idx["away_recs"][norm][max(0, j_away - LAST_N):j_away]

    if len(home_slice) + len(away_slice) == 0:
        return {"found": False, "matches": 0}

    # `matches` = lifetime appearances up to (but not including) cutoff — matches
    # the engine's find_team_stats semantics (which counts all, not just last N).
    total = j_home + j_away

    hg = np.array(home_slice, dtype=float) if home_slice else np.empty((0, 3))
    ag = np.array(away_slice, dtype=float) if away_slice else np.empty((0, 3))

    hw = _weights(len(hg))
    aw = _weights(len(ag))

    # home appearances: col0 = team goals (hs), col1 = opp goals (as)
    h_win = (hg[:, 0] > hg[:, 1]).astype(float) if len(hg) else np.array([])
    h_draw = (hg[:, 0] == hg[:, 1]).astype(float) if len(hg) else np.array([])
    h_loss = (hg[:, 0] < hg[:, 1]).astype(float) if len(hg) else np.array([])
    # away appearances: col0 = opp goals (hs), col1 = team goals (as)
    a_win = (ag[:, 1] > ag[:, 0]).astype(float) if len(ag) else np.array([])
    a_draw = (ag[:, 0] == ag[:, 1]).astype(float) if len(ag) else np.array([])
    a_loss = (ag[:, 1] < ag[:, 0]).astype(float) if len(ag) else np.array([])

    w_hw = (h_win * hw).sum()
    w_hd = (h_draw * hw).sum()
    w_hl = (h_loss * hw).sum()
    w_aw = (a_win * aw).sum()
    w_ad = (a_draw * aw).sum()
    w_al = (a_loss * aw).sum()

    w_th = hw.sum()
    w_ta = aw.sum()
    w_t = w_th + w_ta

    tw = w_hw + w_aw
    td = w_hd + w_ad
    tl = w_hl + w_al

    tg = len(hg) + len(ag)
    gs_h = hg[:, 0].sum() if len(hg) else 0.0
    gc_h = hg[:, 1].sum() if len(hg) else 0.0
    gs_a = ag[:, 1].sum() if len(ag) else 0.0
    gc_a = ag[:, 0].sum() if len(ag) else 0.0
    avg_scored = (gs_h + gs_a) / tg if tg else 1.2
    avg_conceded = (gc_h + gc_a) / tg if tg else 1.2

    home_win_rate = w_hw / w_th if w_th > 0 else 0.33
    away_win_rate = w_aw / w_ta if w_ta > 0 else 0.33

    # ELO: home_elo from home apps + away_elo from away apps, last 5 of concat
    elo = None
    elo_vals = []
    if len(hg):
        elo_vals += [v for v in hg[:, 2] if not math.isnan(v)]
    if len(ag):
        elo_vals += [v for v in ag[:, 2] if not math.isnan(v)]
    if elo_vals:
        elo = float(np.mean(elo_vals[-5:])) if len(elo_vals) >= 5 else float(np.mean(elo_vals))

    h_btts = ((hg[:, 0] > 0) & (hg[:, 1] > 0)).sum() if len(hg) else 0
    a_btts = ((ag[:, 0] > 0) & (ag[:, 1] > 0)).sum() if len(ag) else 0
    btts_rate = (h_btts + a_btts) / tg if tg else 0.5

    h_o25 = ((hg[:, 0] + hg[:, 1]) > 2.5).sum() if len(hg) else 0
    a_o25 = ((ag[:, 0] + ag[:, 1]) > 2.5).sum() if len(ag) else 0
    over25_rate = (h_o25 + a_o25) / tg if tg else 0.45

    rh = hg[-5:] if len(hg) else hg
    ra = ag[-5:] if len(ag) else ag
    rn = len(rh) + len(ra)
    if rn > 0:
        rw = ((rh[:, 0] > rh[:, 1]).sum() if len(rh) else 0) + \
             ((ra[:, 1] > ra[:, 0]).sum() if len(ra) else 0)
        rd = ((rh[:, 0] == rh[:, 1]).sum() if len(rh) else 0) + \
             ((ra[:, 0] == ra[:, 1]).sum() if len(ra) else 0)
        recent_form = (rw * 3 + rd) / (rn * 3)
    else:
        recent_form = 0.33

    return {
        "found": True, "matches": total,
        "win_rate": tw / w_t if w_t > 0 else 0.33,
        "draw_rate": td / w_t if w_t > 0 else 0.33,
        "loss_rate": tl / w_t if w_t > 0 else 0.33,
        "home_win_rate": home_win_rate, "away_win_rate": away_win_rate,
        "avg_scored": avg_scored, "avg_conceded": avg_conceded,
        "elo": elo, "btts_rate": btts_rate, "over25_rate": over25_rate,
        "recent_form": recent_form,
    }


def h2h_asof(hn_, an_, cutoff, idx):
    key = (hn_, an_) if hn_ <= an_ else (an_, hn_)
    hd = idx["h2h_dates"].get(key)
    if not hd:
        return {"meetings": 0}
    j = bisect.bisect_left(hd, cutoff)
    recs = idx["h2h_recs"][key][:j]
    if not recs:
        return {"meetings": 0}
    home_wins = draws = 0
    tot_goals = 0.0
    for home_norm, sh, sa in recs:
        tot_goals += sh + sa
        if home_norm == hn_:  # perspective team was home
            if sh > sa:
                home_wins += 1
            elif sh == sa:
                draws += 1
        else:  # perspective team was away
            if sa > sh:
                home_wins += 1
            elif sh == sa:
                draws += 1
    m = len(recs)
    return {
        "meetings": m,
        "home_win_rate": home_wins / m,
        "draw_rate": draws / m,
        "avg_goals": tot_goals / m,
    }


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def brier_binary(p, y):
    p = np.asarray(p); y = np.asarray(y)
    return float(np.mean((p - y) ** 2))


def logloss_binary(p, y, eps=1e-15):
    p = np.clip(np.asarray(p), eps, 1 - eps); y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_multiclass(P, Y):
    # P: (n,3), Y: (n,) class idx
    P = np.asarray(P)
    onehot = np.zeros_like(P)
    onehot[np.arange(len(Y)), Y] = 1
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def logloss_multiclass(P, Y, eps=1e-15):
    P = np.clip(np.asarray(P), eps, 1)
    return float(-np.mean(np.log(P[np.arange(len(Y)), Y])))


def reliability(p, y, bins=10):
    p = np.asarray(p); y = np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    ece = 0.0
    n = len(p)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        if i == bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        c = int(m.sum())
        if c == 0:
            out.append({"lo": round(lo, 2), "hi": round(hi, 2),
                        "pred": None, "actual": None, "count": 0})
            continue
        mean_pred = float(p[m].mean())
        mean_act = float(y[m].mean())
        out.append({"lo": round(lo, 2), "hi": round(hi, 2),
                    "pred": round(mean_pred, 4), "actual": round(mean_act, 4),
                    "count": c})
        ece += (c / n) * abs(mean_pred - mean_act)
    return out, float(ece)


# ----------------------------------------------------------------------------
# Fidelity check: as-of stats with a far-future cutoff should reproduce the
# engine's own find_team_stats on the full history (sanity that we didn't drift).
# ----------------------------------------------------------------------------
def validate_fidelity(hist, idx, n_checks=8):
    far = np.datetime64("2100-01-01")
    teams = list(idx["home_dates"].keys())[:200]
    checked = mismatches = 0
    for t in teams:
        if checked >= n_checks:
            break
        orig = pt.find_team_stats(t, hist, last_n=LAST_N)
        if not orig.get("found"):
            continue
        mine = team_stats_asof(t, far, idx)
        checked += 1
        for k in ("matches", "win_rate", "draw_rate", "avg_scored", "over25_rate"):
            ov, mv = orig.get(k), mine.get(k)
            if ov is None or mv is None:
                continue
            if abs(float(ov) - float(mv)) > 1e-6:
                mismatches += 1
                print(f"    [fidelity] {t}: {k} orig={ov:.5f} mine={mv:.5f}")
    print(f"  Fidelity check: {checked} teams, {mismatches} field mismatches")
    return mismatches == 0


# ----------------------------------------------------------------------------
# Main backtest
# ----------------------------------------------------------------------------
def generate_pairs(years, min_games, max_matches=None, apply_calibration=False, verbose=True):
    """Run the engine over historical matches with no look-ahead and return the
    (predicted probability, actual outcome) pairs for every market.

    Set apply_calibration=True to measure the engine AFTER the fitted calibration
    layer is applied (used to prove the harness improvement)."""
    if verbose:
        print("Loading historical data (same as prediction engine)...")
    hist = pt.load_historical()
    if hist is None:
        raise RuntimeError("No historical data.")
    hist = hist.sort_values("date").reset_index(drop=True)

    if verbose:
        print("Building as-of indices...")
    idx = build_indices(hist)
    if verbose:
        print("Validating as-of fidelity vs engine...")
        validate_fidelity(hist, idx)

    cal = None
    if apply_calibration:
        from calibration import Calibrator
        cal = Calibrator.load()

    max_date = hist["date"].max()
    cutoff_start = max_date - pd.Timedelta(days=365 * years)
    test = hist[hist["date"] >= cutoff_start]
    if verbose:
        print(f"\nTest window: {cutoff_start.date()} -> {max_date.date()}  "
              f"({len(test):,} candidate matches)")

    dates = test["date"].values.astype("datetime64[ns]")
    hn = test["_home_norm"].values
    an = test["_away_norm"].values
    ht = test["home_team"].values
    at = test["away_team"].values
    hs = test["home_score"].values.astype(int)
    as_ = test["away_score"].values.astype(int)

    P1x2, Y1x2, CONF, TOP_OK, DQ, MDATES = [], [], [], [], [], []
    o15_p, o15_y = [], []
    o25_p, o25_y = [], []
    btts_p, btts_y = [], []

    n = len(test)
    step = max(1, n // 20)
    used = 0
    for i in range(n):
        if max_matches and used >= max_matches:
            break
        d = dates[i]
        h_stats = team_stats_asof(hn[i], d, idx)
        a_stats = team_stats_asof(an[i], d, idx)
        if not (h_stats.get("found") and a_stats.get("found")):
            continue
        if min(h_stats["matches"], a_stats["matches"]) < min_games:
            continue
        h2h = h2h_asof(hn[i], an[i], d, idx)

        pred = pt.predict_match(ht[i], at[i], h_stats, a_stats, h2h)
        if cal is not None:
            pred = cal.apply(pred)

        ph = pred["home_win"] / 100.0
        pd_ = pred["draw"] / 100.0
        pa = pred["away_win"] / 100.0
        if hs[i] > as_[i]:
            ycls = 0
        elif hs[i] == as_[i]:
            ycls = 1
        else:
            ycls = 2
        P1x2.append([ph, pd_, pa])
        Y1x2.append(ycls)
        CONF.append(pred["confidence"] / 100.0)
        top = {"Home Win": 0, "Draw": 1, "Away Win": 2}[pred["prediction"]]
        TOP_OK.append(1 if top == ycls else 0)
        DQ.append(pred["data_quality"])
        MDATES.append(d)

        o15_p.append(pred["over_1.5"] / 100.0)
        o15_y.append(1 if (hs[i] + as_[i]) > 1.5 else 0)
        o25_p.append(pred["over_2.5"] / 100.0)
        o25_y.append(1 if (hs[i] + as_[i]) > 2.5 else 0)
        btts_p.append(pred["btts"] / 100.0)
        btts_y.append(1 if (hs[i] > 0 and as_[i] > 0) else 0)

        used += 1
        if verbose and used % step == 0:
            print(f"  processed {used:,} ...")

    if verbose:
        print(f"\nEvaluated {used:,} matches (both teams had >= {min_games} prior games)\n")

    return {
        "n": used,
        "test_window": f"{cutoff_start.date()} to {max_date.date()}",
        "P1x2": np.array(P1x2), "Y1x2": np.array(Y1x2),
        "CONF": np.array(CONF), "TOP_OK": np.array(TOP_OK),
        "DQ": np.array(DQ), "dates": np.array(MDATES),
        "o15_p": np.array(o15_p), "o15_y": np.array(o15_y),
        "o25_p": np.array(o25_p), "o25_y": np.array(o25_y),
        "btts_p": np.array(btts_p), "btts_y": np.array(btts_y),
    }


def run(years, min_games, out_path, max_matches=None, apply_calibration=False):
    data = generate_pairs(years, min_games, max_matches, apply_calibration)
    used = data["n"]
    P1x2 = data["P1x2"]; Y1x2 = data["Y1x2"]
    CONF = data["CONF"]; TOP_OK = data["TOP_OK"]; DQ = data["DQ"]
    o15_p, o15_y = data["o15_p"], data["o15_y"]
    o25_p, o25_y = data["o25_p"], data["o25_y"]
    btts_p, btts_y = data["btts_p"], data["btts_y"]
    cutoff_label = data["test_window"]
    report = {"n_matches": used, "test_window": cutoff_label,
              "min_games": min_games, "calibrated": apply_calibration, "markets": {}}

    # --- 1X2 ---
    rel_h, ece_h = reliability(P1x2[:, 0], (Y1x2 == 0).astype(int))
    rel_d, ece_d = reliability(P1x2[:, 1], (Y1x2 == 1).astype(int))
    rel_a, ece_a = reliability(P1x2[:, 2], (Y1x2 == 2).astype(int))
    base = np.tile([pt.BASE_HOME, pt.BASE_DRAW, pt.BASE_AWAY], (len(Y1x2), 1))
    report["markets"]["1X2"] = {
        "brier": round(brier_multiclass(P1x2, Y1x2), 4),
        "brier_baseline": round(brier_multiclass(base, Y1x2), 4),
        "logloss": round(logloss_multiclass(P1x2, Y1x2), 4),
        "logloss_baseline": round(logloss_multiclass(base, Y1x2), 4),
        "ece_home": round(ece_h, 4), "ece_draw": round(ece_d, 4), "ece_away": round(ece_a, 4),
        "reliability_home": rel_h, "reliability_draw": rel_d, "reliability_away": rel_a,
    }

    # --- Over 1.5 ---
    rel_o15, ece_o15 = reliability(o15_p, o15_y)
    report["markets"]["Over1.5"] = {
        "brier": round(brier_binary(o15_p, o15_y), 4),
        "logloss": round(logloss_binary(o15_p, o15_y), 4),
        "ece": round(ece_o15, 4),
        "actual_rate": round(float(np.mean(o15_y)), 4),
        "under_pick_rate": round(float(np.mean(o15_p < 0.5)), 4),
        "reliability": rel_o15,
    }

    # --- Over 2.5 ---
    rel_o, ece_o = reliability(o25_p, o25_y)
    base_o = np.full(len(o25_y), pt.BASE_O25)
    report["markets"]["Over2.5"] = {
        "brier": round(brier_binary(o25_p, o25_y), 4),
        "brier_baseline": round(brier_binary(base_o, o25_y), 4),
        "logloss": round(logloss_binary(o25_p, o25_y), 4),
        "logloss_baseline": round(logloss_binary(base_o, o25_y), 4),
        "ece": round(ece_o, 4),
        "actual_rate": round(float(np.mean(o25_y)), 4),
        "reliability": rel_o,
    }

    # --- BTTS ---
    rel_b, ece_b = reliability(btts_p, btts_y)
    base_b = np.full(len(btts_y), pt.BASE_BTTS)
    report["markets"]["BTTS"] = {
        "brier": round(brier_binary(btts_p, btts_y), 4),
        "brier_baseline": round(brier_binary(base_b, btts_y), 4),
        "logloss": round(logloss_binary(btts_p, btts_y), 4),
        "logloss_baseline": round(logloss_binary(base_b, btts_y), 4),
        "ece": round(ece_b, 4),
        "actual_rate": round(float(np.mean(btts_y)), 4),
        "reliability": rel_b,
    }

    # --- Confidence heuristic: does higher conf => higher top-pick accuracy? ---
    CONF = np.array(CONF); TOP_OK = np.array(TOP_OK)
    conf_bins = []
    for lo in np.arange(0.0, 1.0, 0.1):
        hi = lo + 0.1
        m = (CONF >= lo) & (CONF < hi) if hi < 1.0 else (CONF >= lo) & (CONF <= hi)
        c = int(m.sum())
        conf_bins.append({
            "lo": round(float(lo), 1), "hi": round(float(hi), 1),
            "accuracy": round(float(TOP_OK[m].mean()), 4) if c else None,
            "count": c,
        })
    report["confidence_vs_accuracy"] = conf_bins
    report["overall_top_pick_accuracy"] = round(float(TOP_OK.mean()), 4)

    # --- Data-quality segmentation ---
    DQ = np.array(DQ)
    seg = {}
    for q in ("HIGH", "MEDIUM", "LOW"):
        m = DQ == q
        if m.sum():
            seg[q] = {"count": int(m.sum()),
                      "top_pick_accuracy": round(float(TOP_OK[m].mean()), 4)}
    report["by_data_quality"] = seg

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print_summary(report)
    print(f"\nFull report -> {out_path}")
    return report


def print_summary(r):
    print("=" * 64)
    print(f"  CALIBRATION REPORT  ({r['n_matches']:,} matches, {r['test_window']})")
    print("=" * 64)
    for mk, d in r["markets"].items():
        print(f"\n  {mk}")
        print(f"    Brier    : {d['brier']:.4f}   (baseline {d['brier_baseline']:.4f})")
        print(f"    Log loss : {d['logloss']:.4f}   (baseline {d['logloss_baseline']:.4f})")
        if "ece" in d:
            print(f"    ECE      : {d['ece']:.4f}   (0 = perfectly calibrated)")
            print(f"    Reliability (pred -> actual):")
            for b in d["reliability"]:
                if b["count"]:
                    bar = "#" * int(b["count"] / max(1, r['n_matches']) * 200)
                    print(f"      {b['lo']:.1f}-{b['hi']:.1f}: pred {b['pred']:.3f} "
                          f"actual {b['actual']:.3f}  n={b['count']:<6} {bar}")
        else:
            print(f"    ECE home/draw/away: {d['ece_home']:.4f} / "
                  f"{d['ece_draw']:.4f} / {d['ece_away']:.4f}")

    print(f"\n  CONFIDENCE vs TOP-PICK ACCURACY "
          f"(overall {r['overall_top_pick_accuracy']:.1%})")
    for b in r["confidence_vs_accuracy"]:
        if b["count"]:
            print(f"    conf {b['lo']:.1f}-{b['hi']:.1f}: "
                  f"accuracy {b['accuracy']:.1%}  n={b['count']:,}")
    print(f"\n  BY DATA QUALITY:")
    for q, s in r["by_data_quality"].items():
        print(f"    {q:7s}: {s['top_pick_accuracy']:.1%} top-pick acc  (n={s['count']:,})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--min-games", type=int, default=3)
    ap.add_argument("--max-matches", type=int, default=None)
    ap.add_argument("--calibrated", action="store_true",
                    help="Apply the fitted calibration layer before scoring")
    ap.add_argument("--out", default="calibration_report.json")
    a = ap.parse_args()
    run(a.years, a.min_games, a.out, a.max_matches, a.calibrated)
