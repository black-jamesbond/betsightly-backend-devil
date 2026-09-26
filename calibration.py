"""
Calibration layer — maps the engine's raw probabilities onto calibrated ones
learned from backtest results, so that a stated "70%" actually wins ~70%.

Uses isotonic regression (monotonic, non-parametric) fitted per market:
  - Over 2.5   : single map
  - BTTS       : single map
  - 1X2        : three one-vs-rest maps (home / draw / away), then renormalized

After calibration, `confidence` is redefined as the calibrated probability of the
selected 1X2 outcome — i.e. a real win-probability, not a heuristic score.

Maps are stored as isotonic breakpoints (x, y) and applied with linear
interpolation (np.interp), which reproduces sklearn's clipped isotonic predict.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CALIB_PATH = Path(__file__).resolve().parent / "models" / "calibration" / "calibrators.json"

# Draw-aware 1X2 selection. Pure argmax never picks Draw (a draw is rarely the
# single most-likely outcome), so the model predicted 0 draws vs a ~19% real
# rate. This rule picks Draw on genuinely balanced matches — recovering a
# sensible share of draws at ~break-even accuracy. Tunable:
#   raise DRAW_MARGIN / lower DRAW_FLOOR  -> more draws, slightly less accuracy.
DRAW_MARGIN = 0.10   # home/away probs within this => "balanced" match
DRAW_FLOOR = 0.28    # ...and draw prob at least this => predict Draw


def pick_1x2(h, d, a):
    """Return (selection, probability) for the 1X2 pick, draw-aware."""
    top = max(h, d, a)
    if abs(h - a) <= DRAW_MARGIN and d >= DRAW_FLOOR:
        return "Draw", d
    if d >= top:
        return "Draw", d
    return ("Home Win", h) if h >= a else ("Away Win", a)


def _fit_isotonic(x, y):
    """Fit isotonic regression and return (x_breakpoints, y_breakpoints)."""
    from sklearn.isotonic import IsotonicRegression
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(x, y)
    xt = np.asarray(iso.X_thresholds_, dtype=float)
    yt = np.asarray(iso.y_thresholds_, dtype=float)
    return xt.tolist(), yt.tolist()


class Calibrator:
    def __init__(self, maps=None, meta=None):
        # maps[name] = {"x": [...], "y": [...]}
        self.maps = maps or {}
        self.meta = meta or {}

    # -- fitting -----------------------------------------------------------
    @classmethod
    def fit(cls, pairs, meta=None):
        """Fit calibrators from a dict of (prob, outcome) arrays produced by
        calibration_harness.generate_pairs()."""
        P = pairs["P1x2"]
        Y = pairs["Y1x2"]
        maps = {
            "home": dict(zip(("x", "y"), _fit_isotonic(P[:, 0], (Y == 0).astype(int)))),
            "draw": dict(zip(("x", "y"), _fit_isotonic(P[:, 1], (Y == 1).astype(int)))),
            "away": dict(zip(("x", "y"), _fit_isotonic(P[:, 2], (Y == 2).astype(int)))),
            "over25": dict(zip(("x", "y"), _fit_isotonic(pairs["o25_p"], pairs["o25_y"]))),
            "btts": dict(zip(("x", "y"), _fit_isotonic(pairs["btts_p"], pairs["btts_y"]))),
        }
        if "o15_p" in pairs:
            maps["over15"] = dict(zip(("x", "y"), _fit_isotonic(pairs["o15_p"], pairs["o15_y"])))
        return cls(maps, meta or {})

    # -- applying ----------------------------------------------------------
    def _map(self, name, p):
        m = self.maps.get(name)
        if not m or not m["x"]:
            return p
        return float(np.interp(p, m["x"], m["y"]))

    def calibrate_1x2(self, ph, pd_, pa):
        h = self._map("home", ph)
        d = self._map("draw", pd_)
        a = self._map("away", pa)
        h = max(1e-4, h); d = max(1e-4, d); a = max(1e-4, a)
        s = h + d + a
        return h / s, d / s, a / s

    def apply(self, pred):
        """Take a predict_match() output dict and return a calibrated copy.
        Redefines `prediction` (argmax of calibrated 1X2) and `confidence`
        (calibrated probability of that pick)."""
        out = dict(pred)
        ph, pd_, pa = pred["home_win"] / 100.0, pred["draw"] / 100.0, pred["away_win"] / 100.0
        h, d, a = self.calibrate_1x2(ph, pd_, pa)
        out["home_win"] = round(h * 100, 1)
        out["draw"] = round(d * 100, 1)
        out["away_win"] = round(a * 100, 1)

        best_name, best_p = pick_1x2(h, d, a)
        out["prediction"] = best_name
        out["confidence"] = round(best_p * 100, 1)

        out["over_2.5"] = round(self._map("over25", pred["over_2.5"] / 100.0) * 100, 1)
        out["btts"] = round(self._map("btts", pred["btts"] / 100.0) * 100, 1)
        # Calibrated Over 1.5 as a NEW field; raw over_1.5 is left untouched.
        if "over_1.5" in pred:
            out["over_1.5v"] = round(self._map("over15", pred["over_1.5"] / 100.0) * 100, 1)
        out["calibrated"] = True
        return out

    # -- persistence -------------------------------------------------------
    def save(self, path=CALIB_PATH):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"maps": self.maps, "meta": self.meta}, f, indent=2)

    @classmethod
    def load(cls, path=CALIB_PATH):
        path = Path(path)
        if not path.exists():
            return cls()  # identity — no-op until fitted
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(d.get("maps", {}), d.get("meta", {}))

    def is_fitted(self):
        return bool(self.maps)
