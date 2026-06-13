"""
Risk Scorer — rates how safe a prediction is for accumulator inclusion.

Risk score [0, 1]:  0 = perfectly safe,  1 = maximum risk.

Designed for football prediction where match-result models output
3-class probabilities (home/draw/away).  A 45% confidence with full
model consensus and a big ELO gap IS safe — the scorer handles this.

Factors (weighted):
    1. Confidence         (0.30) — higher confidence = safer
    2. Model agreement    (0.35) — full consensus = much safer
    3. Estimated odds     (0.15) — very high odds = riskier
    4. ELO gap            (0.20) — big gap = safer for favourites
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

SAFE_CONFIDENCE_THRESHOLD = 0.35
SAFE_RISK_THRESHOLD = 0.55


class RiskScorer:
    def __init__(
        self,
        confidence_weight: float = 0.30,
        agreement_weight: float = 0.35,
        odds_weight: float = 0.15,
        elo_weight: float = 0.20,
    ):
        total = confidence_weight + agreement_weight + odds_weight + elo_weight
        self.w_conf = confidence_weight / total
        self.w_agree = agreement_weight / total
        self.w_odds = odds_weight / total
        self.w_elo = elo_weight / total

    def score(self, prediction: Dict[str, Any]) -> float:
        risk = (
            self.w_conf  * self._conf_risk(prediction.get("confidence", 0.5))
            + self.w_agree * self._agree_risk(prediction.get("model_disagreement", 0.5))
            + self.w_odds  * self._odds_risk(prediction.get("estimated_odds", 2.0))
            + self.w_elo   * self._elo_risk(prediction.get("elo_gap", 0.0))
        )
        return round(float(min(max(risk, 0.0), 1.0)), 4)

    def is_safe(
        self,
        prediction: Dict[str, Any],
        risk_threshold: float = SAFE_RISK_THRESHOLD,
        confidence_threshold: float = SAFE_CONFIDENCE_THRESHOLD,
    ) -> bool:
        if prediction.get("confidence", 0.0) < confidence_threshold:
            return False
        return self.score(prediction) <= risk_threshold

    def score_and_annotate(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
        risk = self.score(prediction)
        safe = (
            risk <= SAFE_RISK_THRESHOLD
            and prediction.get("confidence", 0) >= SAFE_CONFIDENCE_THRESHOLD
        )
        return {**prediction, "risk_score": risk, "is_safe": safe, "risk_level": self._label(risk)}

    def filter_safe(
        self,
        predictions: List[Dict[str, Any]],
        risk_threshold: float = SAFE_RISK_THRESHOLD,
        confidence_threshold: float = SAFE_CONFIDENCE_THRESHOLD,
    ) -> List[Dict[str, Any]]:
        scored = [self.score_and_annotate(p) for p in predictions]
        safe = [
            p for p in scored
            if p["risk_score"] <= risk_threshold
            and p.get("confidence", 0) >= confidence_threshold
        ]
        return sorted(safe, key=lambda x: x["risk_score"])

    # ------------------------------------------------------------------
    # Component risk functions — all return [0, 1]
    # ------------------------------------------------------------------

    def _conf_risk(self, confidence: float) -> float:
        """Map confidence to risk.

        Calibrated for 3-class football predictions where 33% is random
        and 50%+ is already a strong signal.
        """
        c = min(max(confidence, 0.0), 1.0)
        if c >= 0.70:
            return 0.05
        elif c >= 0.55:
            return 0.10 + (0.70 - c) / 0.15 * 0.15
        elif c >= 0.45:
            return 0.25 + (0.55 - c) / 0.10 * 0.20
        elif c >= 0.35:
            return 0.45 + (0.45 - c) / 0.10 * 0.20
        else:
            return 0.65 + (0.35 - c) / 0.35 * 0.35

    def _agree_risk(self, disagreement: float) -> float:
        """Full consensus (0.0) = very safe, total split (1.0) = risky."""
        d = min(max(disagreement, 0.0), 1.0)
        if d <= 0.05:
            return 0.0
        elif d <= 0.20:
            return d * 1.5
        else:
            return 0.30 + (d - 0.20) * 0.875

    def _odds_risk(self, odds: float) -> float:
        if odds <= 0:
            return 1.0
        if odds < 1.10:
            return 0.05
        elif odds <= 1.50:
            return 0.05 + (odds - 1.10) / 0.40 * 0.15
        elif odds <= 2.00:
            return 0.20 + (odds - 1.50) / 0.50 * 0.25
        elif odds <= 3.00:
            return 0.45 + (odds - 2.00) * 0.25
        else:
            return min(0.70 + (odds - 3.0) * 0.05, 1.0)

    def _elo_risk(self, elo_gap: float) -> float:
        """Big ELO gap = low risk (the favourite is much stronger)."""
        gap = abs(elo_gap)
        if gap >= 300:
            return 0.05
        elif gap >= 150:
            return 0.15 + (300 - gap) / 150 * 0.15
        elif gap >= 50:
            return 0.30 + (150 - gap) / 100 * 0.25
        else:
            return 0.55 + (50 - gap) / 50 * 0.25

    @staticmethod
    def _label(risk: float) -> str:
        if risk <= 0.20:
            return "VERY_LOW"
        elif risk <= 0.35:
            return "LOW"
        elif risk <= 0.50:
            return "MEDIUM"
        elif risk <= 0.70:
            return "HIGH"
        else:
            return "VERY_HIGH"
