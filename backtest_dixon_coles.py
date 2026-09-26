"""
Walk-forward A/B: Dixon-Coles vs the current heuristic engine.

Honest, no look-ahead:
  - Test window = last N years.
  - Step through it in quarters. At each quarter, fit Dixon-Coles on matches
    strictly BEFORE the quarter start, then predict every match in the quarter.
  - For each match we ALSO compute the current engine's prediction using the
    same as-of team stats (from calibration_harness).
  - Only matches where BOTH engines produce a prediction are scored, so the
    comparison is apples-to-apples on an identical match set.

Scores each engine per market with Brier / log loss / ECE, plus top-pick
accuracy, and prints a head-to-head.

Usage:
    python backtest_dixon_coles.py [--years 3] [--min-games 5] [--refit-days 91]
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import predict_today as pt
import calibration_harness as ch
from dixon_coles import DixonColes


def market_metrics(p, y, multiclass=False):
    if multiclass:
        P, Y = np.asarray(p), np.asarray(y)
        ece = float(np.mean([ch.reliability(P[:, k], (Y == k).astype(int))[1] for k in range(3)]))
        return {"brier": round(ch.brier_multiclass(P, Y), 4),
                "logloss": round(ch.logloss_multiclass(P, Y), 4), "ece": round(ece, 4)}
    p, y = np.asarray(p), np.asarray(y)
    _, ece = ch.reliability(p, y)
    return {"brier": round(ch.brier_binary(p, y), 4),
            "logloss": round(ch.logloss_binary(p, y), 4), "ece": round(ece, 4)}


def run(years, min_games, refit_days, out_path):
    print("Loading historical data...")
    hist = pt.load_historical().sort_values("date").reset_index(drop=True)
    print("Building as-of indices (current engine)...")
    idx = ch.build_indices(hist)

    # Base the window on league-labeled data only (Dixon-Coles needs a league;
    # unlabeled api-football rows would otherwise inflate the end date).
    eligible = hist[hist["league"].notna()]
    max_date = eligible["date"].max()
    start = max_date - pd.Timedelta(days=365 * years)
    test = eligible[eligible["date"] >= start].reset_index(drop=True)
    # itertuples mangles underscore-prefixed columns; rename for attribute access
    test = test.rename(columns={"_home_norm": "home_norm", "_away_norm": "away_norm"})
    print(f"Test window: {start.date()} -> {max_date.date()} ({len(test):,} candidate matches)\n")

    # accumulators, aligned (only when BOTH engines predict)
    cur_1x2, cur_o25, cur_btts = [], [], []
    dc_1x2, dc_o25, dc_btts = [], [], []
    Ycls, Yo25, Ybtts = [], [], []
    cur_top_ok, dc_top_ok = [], []
    MDATES = []
    ODDS = []  # real market [OddHome, OddDraw, OddAway] per scored match (NaN if none)

    period_start = pd.Timestamp(start)
    n_scored = 0
    while period_start < max_date:
        period_end = period_start + pd.Timedelta(days=refit_days)
        chunk = test[(test["date"] >= period_start) & (test["date"] < period_end)]
        if len(chunk) == 0:
            period_start = period_end
            continue

        dc = DixonColes.fit(hist, as_of=np.datetime64(period_start))

        for row in chunk.itertuples(index=False):
            hn = row.home_norm; an = row.away_norm
            league = row.league
            d = np.datetime64(row.date)

            dc_pred = dc.predict(hn, an, league)
            if dc_pred is None:
                continue
            h_stats = ch.team_stats_asof(hn, d, idx)
            a_stats = ch.team_stats_asof(an, d, idx)
            if not (h_stats.get("found") and a_stats.get("found")):
                continue
            if min(h_stats["matches"], a_stats["matches"]) < min_games:
                continue
            h2h = ch.h2h_asof(hn, an, d, idx)
            cur_pred = pt.predict_match(row.home_team, row.away_team, h_stats, a_stats, h2h)

            hs, as_ = int(row.home_score), int(row.away_score)
            ycls = 0 if hs > as_ else (1 if hs == as_ else 2)
            yo = 1 if (hs + as_) > 2.5 else 0
            yb = 1 if (hs > 0 and as_ > 0) else 0

            cur_1x2.append([cur_pred["home_win"] / 100, cur_pred["draw"] / 100, cur_pred["away_win"] / 100])
            dc_1x2.append([dc_pred["home_win"] / 100, dc_pred["draw"] / 100, dc_pred["away_win"] / 100])
            cur_o25.append(cur_pred["over_2.5"] / 100); dc_o25.append(dc_pred["over_2.5"] / 100)
            cur_btts.append(cur_pred["btts"] / 100); dc_btts.append(dc_pred["btts"] / 100)
            Ycls.append(ycls); Yo25.append(yo); Ybtts.append(yb)
            cur_top_ok.append(1 if {"Home Win": 0, "Draw": 1, "Away Win": 2}[cur_pred["prediction"]] == ycls else 0)
            dc_top_ok.append(1 if {"Home Win": 0, "Draw": 1, "Away Win": 2}[dc_pred["prediction"]] == ycls else 0)
            MDATES.append(np.datetime64(row.date))
            ODDS.append([getattr(row, "OddHome", np.nan),
                         getattr(row, "OddDraw", np.nan),
                         getattr(row, "OddAway", np.nan)])
            n_scored += 1

        print(f"  {period_start.date()} .. {period_end.date()}: "
              f"fit {dc.meta['n_leagues']} leagues, total scored {n_scored:,}")
        period_start = period_end

    print(f"\nScored {n_scored:,} matches on both engines.\n")
    if n_scored == 0:
        print("No matches scored — check the test window overlaps league-labeled data.")
        return

    Ycls = np.array(Ycls)
    report = {"n_matches": n_scored, "test_window": f"{start.date()} to {max_date.date()}",
              "engines": {}}
    for name, P1, o, b in [("current", cur_1x2, cur_o25, cur_btts),
                           ("dixon_coles", dc_1x2, dc_o25, dc_btts)]:
        report["engines"][name] = {
            "1X2": market_metrics(P1, Ycls, multiclass=True),
            "Over2.5": market_metrics(o, Yo25),
            "BTTS": market_metrics(b, Ybtts),
            "top_pick_accuracy": round(float(np.mean(cur_top_ok if name == "current" else dc_top_ok)), 4),
        }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Persist aligned per-match arrays so a DC-specific calibrator can be fit and
    # a calibrated-vs-calibrated comparison can be run without another backtest.
    npz_path = out_path.replace(".json", "_arrays.npz")
    np.savez_compressed(
        npz_path,
        cur_1x2=np.array(cur_1x2), dc_1x2=np.array(dc_1x2),
        cur_o25=np.array(cur_o25), dc_o25=np.array(dc_o25),
        cur_btts=np.array(cur_btts), dc_btts=np.array(dc_btts),
        Ycls=np.array(Ycls), Yo25=np.array(Yo25), Ybtts=np.array(Ybtts),
        dates=np.array(MDATES), odds=np.array(ODDS, dtype=float),
    )
    print_headtohead(report)
    print(f"\nReport -> {out_path}\nArrays -> {npz_path}")
    return report


def print_headtohead(r):
    cur, dc = r["engines"]["current"], r["engines"]["dixon_coles"]
    print("=" * 70)
    print(f"  HEAD-TO-HEAD  ({r['n_matches']:,} matches, {r['test_window']})")
    print(f"  {'':22s}{'current':>14s}{'dixon-coles':>14s}{'winner':>10s}")
    print("=" * 70)

    def line(label, a, b, lower_better=True):
        win = "DC" if (b < a) == lower_better else "current"
        if abs(a - b) < 1e-4:
            win = "tie"
        print(f"  {label:22s}{a:>14.4f}{b:>14.4f}{win:>10s}")

    for mkt in ["1X2", "Over2.5", "BTTS"]:
        print(f"\n  {mkt}")
        line("  Brier", cur[mkt]["brier"], dc[mkt]["brier"])
        line("  Log loss", cur[mkt]["logloss"], dc[mkt]["logloss"])
        line("  ECE", cur[mkt]["ece"], dc[mkt]["ece"])
    print()
    line("Top-pick accuracy", cur["top_pick_accuracy"], dc["top_pick_accuracy"], lower_better=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--min-games", type=int, default=5)
    ap.add_argument("--refit-days", type=int, default=91)
    ap.add_argument("--out", default="dc_ab_report.json")
    a = ap.parse_args()
    run(a.years, a.min_games, a.refit_days, a.out)
