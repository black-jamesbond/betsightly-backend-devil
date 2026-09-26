"""
Fit the calibration layer and prove it improves calibration on held-out data.

Method (honest, no leakage):
  1. Generate engine predictions over historical matches (no look-ahead).
  2. Temporal split: fit calibrators on the OLDER 70% of matches, validate on
     the NEWER 30%. Calibrating and testing on the same rows would overfit.
  3. Fit isotonic maps on the fit split.
  4. Report ECE / Brier BEFORE vs AFTER on the validation split.
  5. Save calibrators only if they help.

Usage:
    python fit_calibration.py [--years 3] [--min-games 5]
"""
from __future__ import annotations

import argparse

import numpy as np

import calibration_harness as ch
from calibration import Calibrator


def split_by_date(pairs, frac=0.70):
    order = np.argsort(pairs["dates"])
    cut = int(len(order) * frac)
    fit_idx, val_idx = order[:cut], order[cut:]

    def take(idx):
        return {
            "P1x2": pairs["P1x2"][idx], "Y1x2": pairs["Y1x2"][idx],
            "o15_p": pairs["o15_p"][idx], "o15_y": pairs["o15_y"][idx],
            "o25_p": pairs["o25_p"][idx], "o25_y": pairs["o25_y"][idx],
            "btts_p": pairs["btts_p"][idx], "btts_y": pairs["btts_y"][idx],
            "dates": pairs["dates"][idx],
        }
    return take(fit_idx), take(val_idx)


def market_metrics(p, y, multiclass=False):
    if multiclass:
        P, Y = p, y
        ece = np.mean([ch.reliability(P[:, k], (Y == k).astype(int))[1] for k in range(3)])
        return {"brier": ch.brier_multiclass(P, Y), "logloss": ch.logloss_multiclass(P, Y),
                "ece": float(ece)}
    _, ece = ch.reliability(p, y)
    return {"brier": ch.brier_binary(p, y), "logloss": ch.logloss_binary(p, y), "ece": ece}


def apply_to_val(cal, val):
    """Apply calibrator to raw validation probabilities (vectorized)."""
    P = val["P1x2"]
    H = np.array([cal._map("home", v) for v in P[:, 0]])
    D = np.array([cal._map("draw", v) for v in P[:, 1]])
    A = np.array([cal._map("away", v) for v in P[:, 2]])
    H = np.clip(H, 1e-4, None); D = np.clip(D, 1e-4, None); A = np.clip(A, 1e-4, None)
    S = H + D + A
    Pc = np.column_stack([H / S, D / S, A / S])
    o15c = np.array([cal._map("over15", v) for v in val["o15_p"]])
    o25c = np.array([cal._map("over25", v) for v in val["o25_p"]])
    bttsc = np.array([cal._map("btts", v) for v in val["btts_p"]])
    return Pc, o15c, o25c, bttsc


def fmt(before, after):
    def arrow(b, a):
        d = a - b
        sign = "v" if d < 0 else "^"
        return f"{b:.4f} -> {a:.4f}  ({sign}{abs(d):.4f})"
    return arrow(before, after)


def main(years, min_games):
    print("Generating engine predictions (no look-ahead)...")
    pairs = ch.generate_pairs(years, min_games, verbose=True)

    fit, val = split_by_date(pairs, 0.70)
    print(f"Fit split:  {fit['P1x2'].shape[0]:,} matches (older)")
    print(f"Val split:  {val['P1x2'].shape[0]:,} matches (newer)\n")

    cal = Calibrator.fit(fit, meta={
        "years": years, "min_games": min_games,
        "fit_n": int(fit["P1x2"].shape[0]), "val_n": int(val["P1x2"].shape[0]),
        "test_window": pairs["test_window"],
    })

    # BEFORE (raw) on validation
    b_1x2 = market_metrics(val["P1x2"], val["Y1x2"], multiclass=True)
    b_o15 = market_metrics(val["o15_p"], val["o15_y"])
    b_o25 = market_metrics(val["o25_p"], val["o25_y"])
    b_btts = market_metrics(val["btts_p"], val["btts_y"])

    # AFTER (calibrated) on validation
    Pc, o15c, o25c, bttsc = apply_to_val(cal, val)
    a_1x2 = market_metrics(Pc, val["Y1x2"], multiclass=True)
    a_o15 = market_metrics(o15c, val["o15_y"])
    a_o25 = market_metrics(o25c, val["o25_y"])
    a_btts = market_metrics(bttsc, val["btts_y"])

    # How often each version would pick Under 1.5 (prob < 0.5)
    under_raw = float(np.mean(val["o15_p"] < 0.5))
    under_cal = float(np.mean(o15c < 0.5))

    print("=" * 66)
    print("  VALIDATION (held-out newer 30%) — BEFORE -> AFTER calibration")
    print("=" * 66)
    for name, b, a in [("1X2", b_1x2, a_1x2), ("Over 1.5", b_o15, a_o15),
                       ("Over 2.5", b_o25, a_o25), ("BTTS", b_btts, a_btts)]:
        print(f"\n  {name}")
        print(f"    ECE     : {fmt(b['ece'], a['ece'])}")
        print(f"    Brier   : {fmt(b['brier'], a['brier'])}")
        print(f"    Log loss: {fmt(b['logloss'], a['logloss'])}")
    print(f"\n  Under 1.5 pick rate: {under_raw:.1%} (raw) -> {under_cal:.1%} (calibrated)  "
          f"| actual Under 1.5 rate: {1-float(np.mean(val['o15_y'])):.1%}")

    # Decide whether to save: save if mean ECE improves
    before_ece = np.mean([b_1x2["ece"], b_o15["ece"], b_o25["ece"], b_btts["ece"]])
    after_ece = np.mean([a_1x2["ece"], a_o15["ece"], a_o25["ece"], a_btts["ece"]])
    print("\n" + "=" * 66)
    print(f"  Mean ECE: {before_ece:.4f} -> {after_ece:.4f}")

    if after_ece < before_ece:
        cal.save()
        print(f"  Calibration IMPROVES calibration. Saved -> {cal.__class__.__name__} "
              f"({cal.save.__defaults__[0]})")
    else:
        print("  Calibration did NOT improve held-out ECE. Not saving.")
    print("=" * 66)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--min-games", type=int, default=5)
    a = ap.parse_args()
    main(a.years, a.min_games)
