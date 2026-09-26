"""
Hybrid predictor — the shipped decision from the Dixon-Coles A/B.

On 33k leak-free backtested matches, Dixon-Coles was clearly better at match
result (1X2) prediction (+1.85 pts accuracy, lower Brier/log-loss/ECE) while the
current engine's Over 2.5 stayed better calibrated. So we take the best of each:

    1X2       -> Dixon-Coles (calibrated with the DC calibrator), when it covers
                 both teams; otherwise the current engine (current calibrator)
    Over 2.5  -> current engine (current calibrator)
    BTTS      -> current engine (current calibrator)

Everything is calibrated, so `confidence` remains a real win-probability.
Falls back gracefully: if the DC model or a calibrator is missing, it behaves
like the plain current-engine + calibration path.
"""
from __future__ import annotations

from pathlib import Path

from calibration import Calibrator
from dixon_coles import DixonColes

CUR_CAL = Path("models/calibration/calibrators.json")
DC_CAL = Path("models/calibration/calibrators_dc.json")
DC_MODEL = Path("models/dixon_coles/model.json")


class HybridPredictor:
    def __init__(self):
        self.cur_cal = Calibrator.load(CUR_CAL)
        self.dc_cal = Calibrator.load(DC_CAL)
        self.dc = DixonColes.load(DC_MODEL) if DC_MODEL.exists() else None

    @property
    def ready(self):
        return self.cur_cal.is_fitted()

    def apply(self, pred, home_norm=None, away_norm=None):
        """Calibrate `pred` (a predict_match output), swapping in Dixon-Coles for
        1X2 when it covers the match. home_norm/away_norm are normalized team
        names (predict_today.normalize_name)."""
        out = dict(pred)

        # Over 2.5 and BTTS: current engine, current calibrator
        out["over_2.5"] = round(self.cur_cal._map("over25", pred["over_2.5"] / 100.0) * 100, 1)
        out["btts"] = round(self.cur_cal._map("btts", pred["btts"] / 100.0) * 100, 1)
        # Calibrated Over 1.5 as a NEW field; raw over_1.5 passes through unchanged.
        if "over_1.5" in pred:
            out["over_1.5v"] = round(self.cur_cal._map("over15", pred["over_1.5"] / 100.0) * 100, 1)

        # 1X2: Dixon-Coles when available, else current
        dc_pred = None
        if self.dc is not None and home_norm and away_norm:
            dc_pred = self.dc.predict_any(home_norm, away_norm)

        if dc_pred is not None:
            h, d, a = self.dc_cal.calibrate_1x2(
                dc_pred["home_win"] / 100.0, dc_pred["draw"] / 100.0, dc_pred["away_win"] / 100.0)
            out["expected_home_goals"] = dc_pred["expected_home_goals"]
            out["expected_away_goals"] = dc_pred["expected_away_goals"]
            source = "hybrid-dc"
        else:
            h, d, a = self.cur_cal.calibrate_1x2(
                pred["home_win"] / 100.0, pred["draw"] / 100.0, pred["away_win"] / 100.0)
            source = "current"

        out["home_win"] = round(h * 100, 1)
        out["draw"] = round(d * 100, 1)
        out["away_win"] = round(a * 100, 1)
        from calibration import pick_1x2
        best = pick_1x2(h, d, a)
        out["prediction"] = best[0]
        out["confidence"] = round(best[1] * 100, 1)
        out["engine"] = source
        out["calibrated"] = True
        return out
