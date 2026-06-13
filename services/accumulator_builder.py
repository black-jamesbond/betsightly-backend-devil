"""
Accumulator Builder — Value-Based Selection with Real Odds.

Two modes:
  1. VALUE MODE (when real odds available):
     - Picks bets where model probability > bookmaker implied probability
     - Requires minimum 3% edge over the market
     - Sorts by expected value (EV), not raw confidence
     - Uses REAL bookmaker odds for accumulator totals

  2. FALLBACK MODE (no real odds):
     - Original confidence-based selection
     - Quality gates: 45%+ for match result, 60%+ for binary
     - Uses estimated odds (less accurate)

Quality gates always apply in both modes:
  - Match result (3-class): >= 45% confidence + ALL models agree
  - Over/Under (binary):    >= 60% confidence + XGB & LGBM agree
  - BTTS (binary):          >= 60% confidence + XGB & LGBM agree
"""
from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

try:
    from services.risk_scorer import RiskScorer, SAFE_RISK_THRESHOLD
except ImportError:
    from risk_scorer import RiskScorer, SAFE_RISK_THRESHOLD

try:
    from services.odds_service import OddsService
except ImportError:
    try:
        from odds_service import OddsService
    except ImportError:
        OddsService = None

# Confidence floors per prediction type
MIN_CONFIDENCE_MATCH_RESULT = 0.45   # 3-class (random = 33%)
MIN_CONFIDENCE_BINARY = 0.60         # binary  (random = 50%)

# Value thresholds
MIN_EDGE = 0.03        # Minimum 3% edge over bookmaker to qualify
MIN_EV = 1.03          # Minimum expected value (£1.03 return per £1 bet)


class AccumulatorBuilder:

    TARGET_ODDS = {
        "2_odds":     {"min": 1.80, "max": 2.50},
        "5_odds":     {"min": 4.50, "max": 6.00},
        "10_odds":    {"min": 8.00, "max": 15.00},
        "over_1_5":   {"min": 2.50, "max": 8.00, "filter": "over_1_5"},
        "rollover":   {"min": 2.00, "max": 3.00},
    }
    MAX_GAMES = 15

    def __init__(self, risk_threshold: float = SAFE_RISK_THRESHOLD):
        self.risk_scorer = RiskScorer()
        self.risk_threshold = risk_threshold

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build_accumulators(
        self,
        predictions: List[Dict[str, Any]],
        odds_map: Optional[Dict[int, Dict]] = None,
    ) -> Dict[str, Any]:
        """Build accumulators from predictions.

        Args:
            predictions: List of prediction results per fixture
            odds_map: Optional dict mapping fixture_id -> real bookmaker odds.
                      If provided, uses value-based selection.
                      If None, falls back to confidence-based selection.
        """
        try:
            use_value = odds_map is not None and len(odds_map) > 0
            strategy = "value_based" if use_value else "confidence_based"

            # Extract best selection per fixture (for general categories)
            selections = self._extract_best_selections(predictions, odds_map)
            safe_pool = [s for s in selections if s.get("is_safe", False)]

            # Also extract ALL qualifying predictions (for filtered categories like over_1_5)
            all_candidates = self._extract_all_selections(predictions, odds_map)
            safe_all = [s for s in all_candidates if s.get("is_safe", False)]

            # In value mode, further filter to only value bets
            if use_value:
                value_pool = [s for s in safe_pool if s.get("is_value", False)]
                value_all = [s for s in safe_all if s.get("is_value", False)]
                logger.info(
                    "Accumulator [VALUE]: %d fixtures -> %d quality -> %d safe -> %d value bets",
                    len(predictions), len(selections), len(safe_pool), len(value_pool),
                )
                build_pool = value_pool
                build_pool_all = value_all
            else:
                logger.info(
                    "Accumulator [FALLBACK]: %d fixtures -> %d quality picks -> %d safe",
                    len(predictions), len(selections), len(safe_pool),
                )
                build_pool = safe_pool
                build_pool_all = safe_all
                value_pool = []

            # Build accumulators — use filtered pool for categories with filters
            accumulators = {}
            for cat, target in self.TARGET_ODDS.items():
                if target.get("filter"):
                    # Filtered categories (over_1_5) use ALL candidates for that type.
                    # If value mode but no real odds for this market, allow confidence-based.
                    filtered_pool = [
                        s for s in safe_all
                        if s.get("prediction_value") == target["filter"]
                    ]
                    # Use value bets if they have odds, else confidence-based
                    value_filtered = [s for s in filtered_pool if s.get("is_value", False)]
                    if value_filtered:
                        accumulators[cat] = self._build_category(value_filtered, cat, target, True)
                    elif filtered_pool:
                        # Confidence-based fallback for this category
                        accumulators[cat] = self._build_category(filtered_pool, cat, target, False)
                    else:
                        accumulators[cat] = self._build_category([], cat, target, use_value)
                else:
                    accumulators[cat] = self._build_category(build_pool, cat, target, use_value)

            has_selections = build_pool or any(
                a.get("selected") for a in accumulators.values()
            )

            if not has_selections:
                reason = "No value bets found (model edge < 3%)" if use_value else "No confident + safe selections found"
                return {
                    "status": "no_selections",
                    "strategy": strategy,
                    "total_games_analyzed": len(predictions),
                    "quality_selections": len(selections),
                    "safe_selections": len(safe_pool),
                    "value_selections": len(value_pool) if use_value else None,
                    "accumulators": self._empty_accumulators(reason),
                    "summary": self._summary({}, strategy),
                }

            return {
                "status": "success",
                "strategy": strategy,
                "total_games_analyzed": len(predictions),
                "quality_selections": len(selections),
                "safe_selections": len(safe_pool),
                "value_selections": len(value_pool) if use_value else None,
                "accumulators": accumulators,
                "summary": self._summary(accumulators, strategy),
            }
        except Exception as e:
            logger.error("AccumulatorBuilder error: %s", e, exc_info=True)
            return {
                "status": "error", "message": str(e),
                "accumulators": self._empty_accumulators("Internal error"),
            }

    # ------------------------------------------------------------------
    # Selection extraction — quality-gated per prediction type
    # ------------------------------------------------------------------

    def _extract_best_selections(
        self,
        predictions: List[Dict[str, Any]],
        odds_map: Optional[Dict[int, Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """For each fixture, find the best prediction.

        When odds_map is provided, also calculates value metrics.
        """
        selections: List[Dict[str, Any]] = []

        for pred in predictions:
            fi = pred.get("fixture_info", {})
            ml = pred.get("ml_predictions", {})
            fixture_disagreement = pred.get("model_disagreement", 0.5)
            fixture_elo_gap = pred.get("elo_gap", 0.0)
            fixture_id = fi.get("fixture_id")

            # Get real odds for this fixture if available
            fixture_odds = odds_map.get(fixture_id, {}) if odds_map else {}

            candidates = []

            # --- Evaluate match result predictions ---
            mr_candidate = self._evaluate_match_result(ml, fi, fixture_disagreement, fixture_elo_gap)
            if mr_candidate:
                self._attach_value(mr_candidate, fixture_odds)
                candidates.append(mr_candidate)

            # --- Evaluate over 1.5 predictions ---
            o15_candidate = self._evaluate_binary_pair(
                ml, fi, fixture_disagreement, fixture_elo_gap,
                key_pattern="over_1_5",
                pos_label="over_1_5", neg_label="under_1_5",
                pos_readable="Over 1.5 goals", neg_readable="Under 1.5 goals",
            )
            if o15_candidate:
                self._attach_value(o15_candidate, fixture_odds)
                candidates.append(o15_candidate)

            # --- Evaluate over 2.5 predictions ---
            ou_candidate = self._evaluate_binary_pair(
                ml, fi, fixture_disagreement, fixture_elo_gap,
                key_pattern="over_2_5",
                pos_label="over_2_5", neg_label="under_2_5",
                pos_readable="Over 2.5 goals", neg_readable="Under 2.5 goals",
            )
            if ou_candidate:
                self._attach_value(ou_candidate, fixture_odds)
                candidates.append(ou_candidate)

            # --- Evaluate BTTS predictions ---
            btts_candidate = self._evaluate_binary_pair(
                ml, fi, fixture_disagreement, fixture_elo_gap,
                key_pattern="btts",
                pos_label="yes", neg_label="no",
                pos_readable="BTTS Yes", neg_readable="BTTS No",
            )
            if btts_candidate:
                self._attach_value(btts_candidate, fixture_odds)
                candidates.append(btts_candidate)

            if not candidates:
                continue

            # Pick strategy: value-first if odds available, else confidence
            if odds_map and fixture_odds:
                # Prefer the candidate with the highest edge (value)
                value_candidates = [c for c in candidates if c.get("edge", -1) > 0]
                if value_candidates:
                    best = max(value_candidates, key=lambda c: c.get("edge", 0))
                else:
                    best = max(candidates, key=lambda c: c["confidence"])
            else:
                best = max(candidates, key=lambda c: c["confidence"])

            scored = self.risk_scorer.score_and_annotate(best)
            selections.append(scored)

        # Sort: by edge (value) if available, else by risk
        if odds_map:
            return sorted(selections, key=lambda x: x.get("edge", -1), reverse=True)
        return sorted(selections, key=lambda x: x.get("risk_score", 1.0))

    def _extract_all_selections(
        self,
        predictions: List[Dict[str, Any]],
        odds_map: Optional[Dict[int, Dict]] = None,
    ) -> List[Dict[str, Any]]:
        """Extract ALL qualifying predictions across all fixtures (not just best).

        Used for filtered categories (e.g., over_1_5) that need to pick from all
        over_1_5 predictions, not just the ones that won the per-fixture comparison.
        """
        all_selections: List[Dict[str, Any]] = []

        for pred in predictions:
            fi = pred.get("fixture_info", {})
            ml = pred.get("ml_predictions", {})
            fixture_disagreement = pred.get("model_disagreement", 0.5)
            fixture_elo_gap = pred.get("elo_gap", 0.0)
            fixture_id = fi.get("fixture_id")
            fixture_odds = odds_map.get(fixture_id, {}) if odds_map else {}

            # Evaluate over 1.5
            o15 = self._evaluate_binary_pair(
                ml, fi, fixture_disagreement, fixture_elo_gap,
                key_pattern="over_1_5",
                pos_label="over_1_5", neg_label="under_1_5",
                pos_readable="Over 1.5 goals", neg_readable="Under 1.5 goals",
            )
            if o15:
                self._attach_value(o15, fixture_odds)
                scored = self.risk_scorer.score_and_annotate(o15)
                all_selections.append(scored)

            # Evaluate over 2.5
            ou = self._evaluate_binary_pair(
                ml, fi, fixture_disagreement, fixture_elo_gap,
                key_pattern="over_2_5",
                pos_label="over_2_5", neg_label="under_2_5",
                pos_readable="Over 2.5 goals", neg_readable="Under 2.5 goals",
            )
            if ou:
                self._attach_value(ou, fixture_odds)
                scored = self.risk_scorer.score_and_annotate(ou)
                all_selections.append(scored)

            # Evaluate BTTS
            btts = self._evaluate_binary_pair(
                ml, fi, fixture_disagreement, fixture_elo_gap,
                key_pattern="btts",
                pos_label="yes", neg_label="no",
                pos_readable="BTTS Yes", neg_readable="BTTS No",
            )
            if btts:
                self._attach_value(btts, fixture_odds)
                scored = self.risk_scorer.score_and_annotate(btts)
                all_selections.append(scored)

            # Evaluate match result
            mr = self._evaluate_match_result(ml, fi, fixture_disagreement, fixture_elo_gap)
            if mr:
                self._attach_value(mr, fixture_odds)
                scored = self.risk_scorer.score_and_annotate(mr)
                all_selections.append(scored)

        # Sort by edge if value mode, else by risk
        if odds_map:
            return sorted(all_selections, key=lambda x: x.get("edge", -1), reverse=True)
        return sorted(all_selections, key=lambda x: x.get("risk_score", 1.0))

    def _attach_value(self, candidate: Dict, fixture_odds: Dict) -> None:
        """Attach value metrics to a candidate using real odds."""
        if not fixture_odds:
            candidate["real_odds"] = None
            candidate["edge"] = -1.0
            candidate["expected_value"] = 0.0
            candidate["is_value"] = False
            return

        # Determine which real odds to use based on prediction
        pred_value = candidate.get("prediction_value", "")
        pred_type = candidate.get("prediction_type", "")
        real_odds = None

        if pred_value == "home_win":
            real_odds = fixture_odds.get("home_odds")
        elif pred_value == "away_win":
            real_odds = fixture_odds.get("away_odds")
        elif pred_value == "draw":
            real_odds = fixture_odds.get("draw_odds")
        elif pred_value == "over_1_5":
            real_odds = fixture_odds.get("over_1_5_odds")
        elif pred_value == "under_1_5":
            real_odds = fixture_odds.get("under_1_5_odds")
        elif pred_value == "over_2_5":
            real_odds = fixture_odds.get("over_2_5_odds")
        elif pred_value == "under_2_5":
            real_odds = fixture_odds.get("under_2_5_odds")
        elif pred_value == "yes":  # BTTS yes
            real_odds = fixture_odds.get("btts_yes_odds")
        elif pred_value == "no":  # BTTS no
            real_odds = fixture_odds.get("btts_no_odds")

        if real_odds is None or real_odds <= 1.0:
            candidate["real_odds"] = None
            candidate["edge"] = -1.0
            candidate["expected_value"] = 0.0
            candidate["is_value"] = False
            return

        # Calculate value
        model_prob = candidate.get("confidence", 0.5)
        implied_prob = 1.0 / real_odds
        edge = model_prob - implied_prob
        ev = model_prob * real_odds

        candidate["real_odds"] = real_odds
        candidate["implied_prob"] = round(implied_prob, 4)
        candidate["edge"] = round(edge, 4)
        candidate["expected_value"] = round(ev, 4)
        candidate["is_value"] = edge >= MIN_EDGE and ev >= MIN_EV
        candidate["bookmaker"] = fixture_odds.get("bookmaker", "unknown")

        # Use REAL odds instead of fake estimated odds
        candidate["estimated_odds"] = real_odds

    def _evaluate_match_result(
        self, ml: dict, fi: dict, disagreement: float, elo_gap: float
    ) -> Optional[Dict[str, Any]]:
        """Check if match-result models produce a confident, unanimous pick."""
        mr_models = {}
        for key, pd in ml.items():
            if "match_result" in key or key in ("elo_rating", "dixon_coles"):
                prediction = pd.get("prediction", "")
                if prediction in ("home_win", "draw", "away_win"):
                    mr_models[key] = {
                        "prediction": prediction,
                        "confidence": pd.get("confidence", 0),
                    }

        if len(mr_models) < 2:
            return None

        # Check unanimity — ALL match-result models must agree
        predictions_set = set(m["prediction"] for m in mr_models.values())
        if len(predictions_set) != 1:
            return None  # Models disagree — skip

        agreed_prediction = predictions_set.pop()

        # Use the highest confidence among agreeing models
        best_conf = max(m["confidence"] for m in mr_models.values())

        if best_conf < MIN_CONFIDENCE_MATCH_RESULT:
            return None  # Not confident enough

        # Average confidence across all agreeing models
        avg_conf = sum(m["confidence"] for m in mr_models.values()) / len(mr_models)

        return self._make_candidate(
            fi, agreed_prediction, self._fmt_readable(agreed_prediction),
            best_conf, avg_conf, 0.0, elo_gap, "match_result",
            models_agreed=len(mr_models),
        )

    def _evaluate_binary_pair(
        self, ml: dict, fi: dict, disagreement: float, elo_gap: float,
        key_pattern: str, pos_label: str, neg_label: str,
        pos_readable: str, neg_readable: str,
    ) -> Optional[Dict[str, Any]]:
        """Check if XGB and LGBM agree on a binary prediction with high confidence."""
        xgb_pred = None
        lgbm_pred = None

        for key, pd in ml.items():
            if key_pattern not in key:
                continue
            prediction = pd.get("prediction", "")
            conf = pd.get("confidence", 0)
            if "xgb" in key:
                xgb_pred = {"prediction": prediction, "confidence": conf}
            elif "lgbm" in key:
                lgbm_pred = {"prediction": prediction, "confidence": conf}

        if xgb_pred is None or lgbm_pred is None:
            return None

        # Both must agree
        if xgb_pred["prediction"] != lgbm_pred["prediction"]:
            return None

        agreed = xgb_pred["prediction"]
        best_conf = max(xgb_pred["confidence"], lgbm_pred["confidence"])
        avg_conf = (xgb_pred["confidence"] + lgbm_pred["confidence"]) / 2

        if best_conf < MIN_CONFIDENCE_BINARY:
            return None  # Not confident enough

        readable = pos_readable if agreed == pos_label else neg_readable

        return self._make_candidate(
            fi, agreed, readable, best_conf, avg_conf, 0.0, elo_gap,
            key_pattern, models_agreed=2,
        )

    def _make_candidate(
        self, fi: dict, prediction: str, readable: str,
        best_conf: float, avg_conf: float,
        disagreement: float, elo_gap: float, pred_type: str,
        models_agreed: int = 0,
    ) -> Dict[str, Any]:
        return {
            "fixture_id": fi.get("fixture_id"),
            "home_team": fi.get("home_team", ""),
            "away_team": fi.get("away_team", ""),
            "league": fi.get("league", ""),
            "date": fi.get("date", ""),
            "prediction_type": pred_type,
            "prediction_value": prediction,
            "readable_prediction": readable,
            "confidence": best_conf,
            "average_confidence": avg_conf,
            "estimated_odds": self._conf_to_odds(best_conf, pred_type),  # Overridden if real odds
            "model_disagreement": disagreement,
            "elo_gap": elo_gap,
            "models_agreed": models_agreed,
            "home_team_logo": fi.get("home_team_logo", ""),
            "away_team_logo": fi.get("away_team_logo", ""),
            "league_logo": fi.get("league_logo", ""),
        }

    # ------------------------------------------------------------------
    # Greedy accumulator building
    # ------------------------------------------------------------------

    def _build_category(self, pool, category, target, use_value: bool = False):
        min_odds, max_odds = target["min"], target["max"]
        pred_filter = target.get("filter")

        if not pool:
            reason = f"No {pred_filter} predictions available" if pred_filter else "No qualifying games available"
            return self._no(category, min_odds, max_odds, reason)

        chosen = []
        running_odds = 1.0
        used = set()

        # Sort: by edge (value mode) or by risk (fallback mode)
        if use_value:
            sorted_pool = sorted(pool, key=lambda g: g.get("edge", 0), reverse=True)
        else:
            sorted_pool = sorted(pool, key=lambda g: g.get("risk_score", 1.0))

        # Pass 1: strict — stay within max_odds
        for game in sorted_pool:
            if len(chosen) >= self.MAX_GAMES:
                break
            fid = game.get("fixture_id")
            if fid in used:
                continue
            game_odds = game.get("real_odds") or game.get("estimated_odds", 1.5)
            projected = running_odds * game_odds
            if projected <= max_odds:
                chosen.append(game)
                running_odds = projected
                if fid is not None:
                    used.add(fid)
                if running_odds >= min_odds:
                    break

        # Pass 2: allow up to 15% overshoot if needed
        if running_odds < min_odds:
            for game in sorted_pool:
                if len(chosen) >= self.MAX_GAMES:
                    break
                fid = game.get("fixture_id")
                if fid in used:
                    continue
                game_odds = game.get("real_odds") or game.get("estimated_odds", 1.5)
                projected = running_odds * game_odds
                if projected <= max_odds * 1.15:
                    chosen.append(game)
                    running_odds = projected
                    if fid is not None:
                        used.add(fid)
                    if running_odds >= min_odds:
                        break

        if not chosen or running_odds < min_odds * 0.80:
            return self._no(
                category, min_odds, max_odds,
                "Not enough qualifying bets to reach %.2f odds (best: %.2f)" % (min_odds, running_odds),
            )

        avg_conf = sum(g["confidence"] for g in chosen) / len(chosen)
        avg_risk = sum(g.get("risk_score", 0) for g in chosen) / len(chosen)
        avg_edge = sum(g.get("edge", 0) for g in chosen) / len(chosen) if use_value else None
        avg_ev = sum(g.get("expected_value", 0) for g in chosen) / len(chosen) if use_value else None

        result = {
            "selected": True,
            "category": category,
            "games": [self._fmt_game(g, use_value) for g in chosen],
            "total_odds": round(running_odds, 3),
            "target_range": "%.2f-%.2f" % (min_odds, max_odds),
            "num_games": len(chosen),
            "average_confidence": round(avg_conf, 4),
            "average_risk_score": round(avg_risk, 4),
            "risk_level": self._risk_label(avg_risk),
            "recommendation": "INCLUDE",
            "strategy": "value_based" if use_value else "confidence_based",
        }

        if use_value:
            result["average_edge"] = round(avg_edge, 4) if avg_edge else 0
            result["average_ev"] = round(avg_ev, 4) if avg_ev else 0
            result["value_rating"] = self._value_rating(avg_edge)

        return result

    # ------------------------------------------------------------------
    # Confidence → Odds mapping (fallback when no real odds)
    # ------------------------------------------------------------------

    def _conf_to_odds(self, confidence: float, pred_type: str = "") -> float:
        """Map confidence to estimated decimal odds (used only as fallback)."""
        c = confidence

        if pred_type == "match_result":
            if c >= 0.75: return 1.15
            elif c >= 0.65: return 1.25
            elif c >= 0.55: return 1.40
            elif c >= 0.50: return 1.55
            elif c >= 0.45: return 1.75
            else: return 2.10
        else:
            if c >= 0.80: return 1.15
            elif c >= 0.75: return 1.25
            elif c >= 0.70: return 1.35
            elif c >= 0.65: return 1.45
            elif c >= 0.60: return 1.55
            else: return 1.80

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def _fmt_readable(self, value: str) -> str:
        m = {
            "home_win": "Home Win", "away_win": "Away Win", "draw": "Draw",
            "yes": "BTTS Yes", "no": "BTTS No",
            "over_1_5": "Over 1.5 goals", "under_1_5": "Under 1.5 goals",
            "over_2_5": "Over 2.5 goals", "under_2_5": "Under 2.5 goals",
        }
        return m.get(value, value.replace("_", " ").title())

    def _fmt_game(self, g, include_value: bool = False):
        result = {
            "fixture_id": g.get("fixture_id"),
            "home_team": g.get("home_team"),
            "away_team": g.get("away_team"),
            "league": g.get("league"),
            "date": g.get("date"),
            "prediction": g.get("readable_prediction", g.get("prediction_value", "")),
            "prediction_type": g.get("prediction_type"),
            "confidence": round(g.get("confidence", 0), 4),
            "average_confidence": round(g.get("average_confidence", 0), 4),
            "odds": g.get("real_odds") or g.get("estimated_odds"),
            "risk_score": g.get("risk_score"),
            "risk_level": g.get("risk_level"),
            "models_agreed": g.get("models_agreed", 0),
            "home_team_logo": g.get("home_team_logo", ""),
            "away_team_logo": g.get("away_team_logo", ""),
            "league_logo": g.get("league_logo", ""),
        }

        if include_value:
            result["real_odds"] = g.get("real_odds")
            result["edge"] = g.get("edge", 0)
            result["expected_value"] = g.get("expected_value", 0)
            result["implied_prob"] = g.get("implied_prob", 0)
            result["bookmaker"] = g.get("bookmaker", "")
            result["value_label"] = self._edge_label(g.get("edge", 0))

        return result

    @staticmethod
    def _risk_label(r):
        if r <= 0.20: return "VERY_LOW"
        elif r <= 0.30: return "LOW"
        elif r <= 0.45: return "MEDIUM"
        else: return "HIGH"

    @staticmethod
    def _value_rating(avg_edge):
        if avg_edge is None:
            return "N/A"
        if avg_edge >= 0.10: return "EXCELLENT"
        elif avg_edge >= 0.07: return "STRONG"
        elif avg_edge >= 0.05: return "GOOD"
        elif avg_edge >= 0.03: return "FAIR"
        else: return "MARGINAL"

    @staticmethod
    def _edge_label(edge):
        if edge >= 0.10: return "STRONG VALUE"
        elif edge >= 0.05: return "GOOD VALUE"
        elif edge >= 0.03: return "VALUE"
        else: return "NO VALUE"

    def _no(self, cat, mn, mx, reason):
        return {"selected": False, "category": cat,
                "target_range": "%.2f-%.2f" % (mn, mx), "reason": reason,
                "recommendation": "EXCLUDE"}

    def _empty_accumulators(self, reason):
        return {cat: self._no(cat, t["min"], t["max"], reason)
                for cat, t in self.TARGET_ODDS.items()}

    def _summary(self, accumulators, strategy="confidence_based"):
        sel = [a for a in accumulators.values() if a.get("selected", False)]
        return {
            "categories_with_accumulators": len(sel),
            "total_categories": len(self.TARGET_ODDS),
            "total_games_in_accumulators": sum(a.get("num_games", 0) for a in sel),
            "success_rate": "%.1f%%" % (len(sel) / max(len(self.TARGET_ODDS), 1) * 100),
            "strategy": strategy,
        }


def format_accumulator_for_display(result):
    lines = ["Accumulator Result (%s) [strategy: %s]" % (
        result.get("status", "unknown"), result.get("strategy", "unknown")
    )]
    for cat, acc in result.get("accumulators", {}).items():
        if acc.get("selected"):
            edge_info = ""
            if acc.get("average_edge"):
                edge_info = f", edge: {acc['average_edge']*100:.1f}%"
            lines.append(
                "  %s: %d games @ %.2f odds (avg conf: %.1f%%, risk: %s%s)"
                % (cat, acc["num_games"], acc["total_odds"],
                   acc["average_confidence"] * 100, acc["risk_level"], edge_info)
            )
        else:
            lines.append("  %s: NOT SELECTED - %s" % (cat, acc.get("reason", "")))
    return "\n".join(lines)
