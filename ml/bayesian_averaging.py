"""
Bayesian Model Averaging (BMA) for the prediction ensemble.
Weights each model by its posterior accuracy (beta-binomial).
Models that are right more often get higher weight in the combined prediction.
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class BayesianModelAverager:
    """
    Beta-binomial BMA:
        alpha_i += 1 on correct, beta_i += 1 on incorrect
        weight_i = alpha_i / (alpha_i + beta_i)
    """
    def __init__(self, prior_alpha=2.0, prior_beta=2.0):
        self.prior_alpha = prior_alpha
        self.prior_beta = prior_beta
        self._alpha: Dict[str, float] = {}
        self._beta: Dict[str, float] = {}

    def _init(self, name):
        if name not in self._alpha:
            self._alpha[name] = self.prior_alpha
            self._beta[name] = self.prior_beta

    def update_performance(self, model_name, correct):
        self._init(model_name)
        if correct: self._alpha[model_name] += 1.0
        else: self._beta[model_name] += 1.0

    def get_weight(self, model_name):
        self._init(model_name)
        a = self._alpha[model_name]; b = self._beta[model_name]
        return a / (a + b)

    def get_all_weights(self):
        return {n: self.get_weight(n) for n in self._alpha}

    def combine(self, predictions):
        if not predictions: return {}
        weights = [self.get_weight(p.get("model_name", "unknown")) for p in predictions]
        total_w = sum(weights) or 1.0
        nw = [w / total_w for w in weights]
        combined: Dict[str, float] = {}
        for pred, w in zip(predictions, nw):
            probs = pred.get("probabilities", {})
            if not probs:
                outcome = pred.get("prediction", "home_win"); conf = pred.get("confidence", 0.5)
                others = [k for k in ["home_win", "draw", "away_win"] if k != outcome]
                probs = {outcome: conf, **{k: (1 - conf) / 2 for k in others}}
            for outcome, p in probs.items():
                combined[outcome] = combined.get(outcome, 0.0) + w * p
        total_p = sum(combined.values()) or 1.0
        combined = {k: v / total_p for k, v in combined.items()}
        best = max(combined, key=combined.get)
        preds = [p.get("prediction", "") for p in predictions]
        if len(preds) > 1:
            majority = max(set(preds), key=preds.count)
            disagreement = 1.0 - preds.count(majority) / len(preds)
        else:
            disagreement = 0.0
        return {
            "prediction": best, "confidence": round(combined[best], 4),
            "probabilities": {k: round(v, 4) for k, v in combined.items()},
            "model_disagreement": round(disagreement, 4), "models_combined": len(predictions),
            "model_weights": {p.get("model_name", f"m{i}"): round(nw[i], 4) for i, p in enumerate(predictions)},
            "model_name": "bayesian_model_average", "model_type": "ensemble",
        }

    def save(self, path):
        import json, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"alpha": self._alpha, "beta": self._beta,
                       "prior_alpha": self.prior_alpha, "prior_beta": self.prior_beta}, f, indent=2)

    def load(self, path):
        import json
        with open(path) as f:
            d = json.load(f)
        self._alpha = d.get("alpha", {}); self._beta = d.get("beta", {})
        self.prior_alpha = d.get("prior_alpha", self.prior_alpha)
        self.prior_beta = d.get("prior_beta", self.prior_beta)

    def summary(self):
        rows = []
        for name in self._alpha:
            a = self._alpha[name]; b = self._beta[name]
            n = a + b - self.prior_alpha - self.prior_beta
            rows.append({"model": name, "weight": round(self.get_weight(name), 4),
                         "accuracy_estimate": round((a - self.prior_alpha) / max(n, 1), 4),
                         "predictions_recorded": int(n)})
        return sorted(rows, key=lambda x: x["weight"], reverse=True)
