"""
Calibrated head-to-head: current engine vs Dixon-Coles, BOTH calibrated.

Uses the aligned per-match arrays saved by backtest_dixon_coles.py. Fits a
separate isotonic calibrator for each engine on the OLDER 70% of matches and
scores both on the NEWER 30% — so the comparison is calibrated-vs-calibrated on
identical held-out matches. Saves the Dixon-Coles calibrator for production.

Usage:
    python compare_calibrated.py [--npz dc_ab_report_arrays.npz]
"""
from __future__ import annotations

import argparse

import numpy as np

import calibration_harness as ch
from calibration import Calibrator


def metrics(P1x2, o25, btts, Ycls, Yo25, Ybtts):
    ece_1x2 = np.mean([ch.reliability(P1x2[:, k], (Ycls == k).astype(int))[1] for k in range(3)])
    return {
        "1X2": {"brier": ch.brier_multiclass(P1x2, Ycls),
                "logloss": ch.logloss_multiclass(P1x2, Ycls), "ece": float(ece_1x2)},
        "Over2.5": {"brier": ch.brier_binary(o25, Yo25),
                    "logloss": ch.logloss_binary(o25, Yo25), "ece": ch.reliability(o25, Yo25)[1]},
        "BTTS": {"brier": ch.brier_binary(btts, Ybtts),
                 "logloss": ch.logloss_binary(btts, Ybtts), "ece": ch.reliability(btts, Ybtts)[1]},
    }


def calibrate_arrays(cal, P1x2, o25, btts):
    H = np.array([cal._map("home", v) for v in P1x2[:, 0]])
    D = np.array([cal._map("draw", v) for v in P1x2[:, 1]])
    A = np.array([cal._map("away", v) for v in P1x2[:, 2]])
    H = np.clip(H, 1e-4, None); D = np.clip(D, 1e-4, None); A = np.clip(A, 1e-4, None)
    S = H + D + A
    Pc = np.column_stack([H / S, D / S, A / S])
    o = np.array([cal._map("over25", v) for v in o25])
    b = np.array([cal._map("btts", v) for v in btts])
    return Pc, o, b


def make_pairs(P1x2, o25, btts, Ycls, Yo25, Ybtts):
    return {"P1x2": P1x2, "Y1x2": Ycls, "o25_p": o25, "o25_y": Yo25,
            "btts_p": btts, "btts_y": Ybtts}


def main(npz_path):
    d = np.load(npz_path)
    dates = d["dates"]
    order = np.argsort(dates)
    cut = int(len(order) * 0.70)
    fit_i, val_i = order[:cut], order[cut:]
    print(f"Loaded {len(order):,} matches. Fit {len(fit_i):,} (older) / val {len(val_i):,} (newer)\n")

    results = {}
    for eng in ("cur", "dc"):
        P = d[f"{eng}_1x2"]; O = d[f"{eng}_o25"]; B = d[f"{eng}_btts"]
        Ycls, Yo, Yb = d["Ycls"], d["Yo25"], d["Ybtts"]

        cal = Calibrator.fit(make_pairs(P[fit_i], O[fit_i], B[fit_i],
                                        Ycls[fit_i], Yo[fit_i], Yb[fit_i]),
                             meta={"engine": eng})
        # raw + calibrated metrics on validation
        raw = metrics(P[val_i], O[val_i], B[val_i], Ycls[val_i], Yo[val_i], Yb[val_i])
        Pc, Oc, Bc = calibrate_arrays(cal, P[val_i], O[val_i], B[val_i])
        cald = metrics(Pc, Oc, Bc, Ycls[val_i], Yo[val_i], Yb[val_i])
        results[eng] = {"raw": raw, "cal": cald, "cal_obj": cal}

    # Save the DC calibrator for production use
    results["dc"]["cal_obj"].save("models/calibration/calibrators_dc.json")

    # ---- report ----
    print("=" * 74)
    print("  CALIBRATED HEAD-TO-HEAD (held-out newer 30%)")
    print("  current (calibrated)   vs   dixon-coles (calibrated)")
    print("=" * 74)
    cur, dc = results["cur"]["cal"], results["dc"]["cal"]
    for mkt in ("1X2", "Over2.5", "BTTS"):
        print(f"\n  {mkt:>8s} {'metric':<10}{'current':>12}{'dixon-coles':>14}{'winner':>9}")
        for k in ("brier", "logloss", "ece"):
            a, b = cur[mkt][k], dc[mkt][k]
            win = "DC" if b < a - 1e-4 else ("current" if a < b - 1e-4 else "tie")
            print(f"           {k:<10}{a:>12.4f}{b:>14.4f}{win:>9}")

    # aggregate: mean Brier across markets (equal weight)
    cur_brier = np.mean([cur[m]["brier"] for m in ("1X2", "Over2.5", "BTTS")])
    dc_brier = np.mean([dc[m]["brier"] for m in ("1X2", "Over2.5", "BTTS")])
    print("\n" + "=" * 74)
    print(f"  Mean Brier (calibrated):  current {cur_brier:.4f}   dixon-coles {dc_brier:.4f}")
    verdict = "DIXON-COLES" if dc_brier < cur_brier else "CURRENT"
    print(f"  VERDICT: {verdict} is the better base engine (lower mean Brier).")
    print("  DC calibrator saved -> models/calibration/calibrators_dc.json")
    print("=" * 74)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="dc_ab_report_arrays.npz")
    a = ap.parse_args()
    main(a.npz)
