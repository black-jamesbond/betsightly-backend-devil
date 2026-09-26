"""
Positive-EV betting simulation — the rebuilt bot logic, tested on real odds.

A bet is +EV when  p_model * decimal_odds - 1 > threshold.  This only makes money
if p_model is better calibrated than the market on the bets it selects. We test
that honestly on leak-free Dixon-Coles probabilities vs real bookmaker odds.

Pipeline:
  - Load per-match arrays from the DC walk-forward backtest (dc probs, real 1X2
    odds, actual result, dates).
  - Temporal split: fit an isotonic calibrator on the older 60%, test on the
    newer 40% (out-of-sample — no leakage).
  - For every outcome of every test match, compute EV vs the market and flat-bet
    the ones above the threshold. Report yield (profit / staked).

Controls:
  - "bet all model picks" (no EV filter)   -> baseline
  - "market as model"                       -> must yield ~= -overround (sanity)

Usage:
    python ev_backtest.py [--npz dc_ab_report_arrays.npz]
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.isotonic import IsotonicRegression


def calibrate_fit(P, Y):
    """Fit one isotonic map per outcome (home/draw/away)."""
    maps = []
    for k in range(3):
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
        iso.fit(P[:, k], (Y == k).astype(float))
        maps.append(iso)
    return maps


def calibrate_apply(maps, P):
    C = np.column_stack([maps[k].predict(P[:, k]) for k in range(3)])
    C = np.clip(C, 1e-4, None)
    return C / C.sum(axis=1, keepdims=True)


def simulate(prob, odds, y, threshold):
    """Flat-stake 1 unit on every (match, outcome) with EV > threshold."""
    ev = prob * odds - 1.0                       # (n,3)
    bet = ev > threshold
    valid = np.isfinite(odds) & (odds > 1.0)
    bet = bet & valid
    n = int(bet.sum())
    if n == 0:
        return {"bets": 0, "yield": None, "profit": 0.0, "hit_rate": None}
    win = np.zeros_like(odds)
    for k in range(3):
        win[:, k] = (y == k).astype(float)
    profit = np.where(win == 1, odds - 1.0, -1.0)
    tot = profit[bet].sum()
    return {"bets": n, "yield": round(float(tot / n) * 100, 2),
            "profit": round(float(tot), 1),
            "hit_rate": round(float(win[bet].mean()) * 100, 1),
            "avg_odds": round(float(odds[bet].mean()), 2)}


def main(npz):
    d = np.load(npz, allow_pickle=True)
    P = d["dc_1x2"]; odds = d["odds"]; y = d["Ycls"].astype(int)
    dates = d["dates"]
    # Keep only matches that actually have market odds, then order by date.
    keep = np.isfinite(odds[:, 0]) & (odds[:, 0] > 1.0)
    P, odds, y, dates = P[keep], odds[keep], y[keep], dates[keep]
    order = np.argsort(dates)
    P, odds, y = P[order], odds[order], y[order]

    cut = int(len(y) * 0.60)
    maps = calibrate_fit(P[:cut], y[:cut])
    Pte = calibrate_apply(maps, P[cut:])
    ote, yte = odds[cut:], y[cut:]
    print(f"Odds-covered matches: {len(y):,}  |  calibrate on {cut:,} older, "
          f"test +EV on {len(yte):,} newer\n")

    # de-vigged market probs (for the sanity control)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / ote
    mkt = inv / inv.sum(axis=1, keepdims=True)

    print("=" * 60)
    print("  POSITIVE-EV FILTER (Dixon-Coles calibrated vs market)")
    print("=" * 60)
    print(f"  {'EV threshold':<16}{'bets':>8}{'hit%':>8}{'avg odds':>10}{'yield%':>10}")
    for t in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20]:
        r = simulate(Pte, ote, yte, t)
        if r["bets"]:
            print(f"  EV > {int(t*100):>3d}%{'':<7}{r['bets']:>8,}{r['hit_rate']:>8.1f}"
                  f"{r['avg_odds']:>10.2f}{r['yield']:>10.2f}")
        else:
            print(f"  EV > {int(t*100):>3d}%{'':<7}{'0':>8}")

    print("\n  CONTROLS")
    # bet every model argmax pick, no EV filter (stake 1 on argmax)
    pick = Pte.argmax(1)
    valid = np.isfinite(ote[:, 0])
    prof = np.where(pick == yte, ote[np.arange(len(pick)), pick] - 1, -1.0)
    m = valid
    print(f"    bet all model picks   : {int(m.sum()):>7,} bets, "
          f"yield {prof[m].mean()*100:>6.2f}%")
    # market as model (sanity: de-vigged market probs are never +EV vs the vig)
    rm = simulate(mkt, ote, yte, 0.0)
    if rm["bets"]:
        print(f"    market-as-model EV>0  : {rm['bets']:>7,} bets, yield {rm['yield']:>6.2f}%")
    else:
        print(f"    market-as-model EV>0  :       0 bets  (correct: vig makes market never +EV)")
    ov = (1 / ote).sum(axis=1)
    print(f"    mean bookmaker overround: {np.nanmean(ov):.3f}  "
          f"(vig ~ {(np.nanmean(ov)-1)*100:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="dc_ab_report_arrays.npz")
    a = ap.parse_args()
    main(a.npz)
