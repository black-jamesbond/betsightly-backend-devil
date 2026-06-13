"""
ELO Rating System for football teams.
Dynamic team strength that updates after every match.
Used as standalone predictor and as feature input to other models.
A large ELO gap = low-risk prediction for the favourite.
"""
from __future__ import annotations
import logging
from typing import Dict, Any, List, Tuple

logger = logging.getLogger(__name__)

DEFAULT_ELO = 1500.0
K_FACTOR = 32.0
HOME_ADVANTAGE = 100.0


class EloRatingSystem:
    def __init__(self, k_factor=K_FACTOR, home_advantage=HOME_ADVANTAGE, default_rating=DEFAULT_ELO):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.default_rating = default_rating
        self.ratings: Dict[str, float] = {}
        self.match_count: Dict[str, int] = {}

    def get_rating(self, team):
        return self.ratings.get(team, self.default_rating)

    def _expected(self, ra, rb):
        return 1.0 / (1.0 + 10.0 ** ((rb - ra) / 400.0))

    def _result(self, hg, ag):
        if hg > ag: return 1.0, 0.0
        elif hg == ag: return 0.5, 0.5
        else: return 0.0, 1.0

    def _goal_index(self, hg, ag):
        diff = abs(hg - ag)
        if diff <= 1: return 1.0
        elif diff == 2: return 1.5
        else: return 1.75 + (diff - 3) * 0.05

    def update(self, home_team, away_team, home_goals, away_goals):
        rh = self.get_rating(home_team) + self.home_advantage
        ra = self.get_rating(away_team)
        eh = self._expected(rh, ra); ea = 1.0 - eh
        sh, sa = self._result(home_goals, away_goals)
        gi = self._goal_index(home_goals, away_goals)
        new_h = self.get_rating(home_team) + self.k_factor * gi * (sh - eh)
        new_a = self.get_rating(away_team) + self.k_factor * gi * (sa - ea)
        self.ratings[home_team] = new_h; self.ratings[away_team] = new_a
        self.match_count[home_team] = self.match_count.get(home_team, 0) + 1
        self.match_count[away_team] = self.match_count.get(away_team, 0) + 1
        return new_h, new_a

    def train(self, matches):
        from datetime import datetime
        def parse(m):
            try: return datetime.fromisoformat(str(m.get("date","")).split("T")[0])
            except: return datetime.min
        for m in sorted(matches, key=parse):
            try: self.update(m["home_team"], m["away_team"], int(m["home_goals"]), int(m["away_goals"]))
            except (KeyError, ValueError): continue
        logger.info(f"ELO trained on {len(matches)} matches, {len(self.ratings)} teams")

    def predict_win_probability(self, home_team, away_team):
        rh = self.get_rating(home_team) + self.home_advantage
        ra = self.get_rating(away_team)
        eh = self._expected(rh, ra); ea = self._expected(ra, rh)
        draw = max(0.05, 0.30 - abs(rh - ra) * 0.0003)
        rem = 1.0 - draw
        hw = eh * rem; aw = ea * rem; tot = hw + draw + aw
        return {"home_win": round(hw/tot, 4), "draw": round(draw/tot, 4), "away_win": round(aw/tot, 4)}

    def predict(self, home_team, away_team):
        probs = self.predict_win_probability(home_team, away_team)
        best = max(probs, key=probs.get)
        return {
            "prediction": best, "confidence": round(probs[best], 4), "probabilities": probs,
            "home_elo": round(self.get_rating(home_team), 1), "away_elo": round(self.get_rating(away_team), 1),
            "elo_gap": round(abs(self.get_rating(home_team) - self.get_rating(away_team)), 1),
            "model_name": "elo_rating", "model_type": "elo",
        }

    def get_features(self, home_team, away_team):
        probs = self.predict_win_probability(home_team, away_team)
        rh = self.get_rating(home_team); ra = self.get_rating(away_team)
        return {
            "home_elo": rh, "away_elo": ra, "elo_diff": rh - ra,
            "elo_diff_with_home_adv": (rh + self.home_advantage) - ra,
            "elo_ratio": rh / max(ra, 1.0),
            "elo_home_win_prob": probs["home_win"], "elo_draw_prob": probs["draw"],
            "elo_away_win_prob": probs["away_win"],
            "home_matches_played": float(self.match_count.get(home_team, 0)),
            "away_matches_played": float(self.match_count.get(away_team, 0)),
        }

    def save(self, path):
        import json, os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump({"ratings": self.ratings, "match_count": self.match_count,
                       "k_factor": self.k_factor, "home_advantage": self.home_advantage,
                       "default_rating": self.default_rating}, f)

    def load(self, path):
        import json
        with open(path) as f:
            d = json.load(f)
        self.ratings = {k: float(v) for k, v in d["ratings"].items()}
        self.match_count = {k: int(v) for k, v in d.get("match_count", {}).items()}
        self.k_factor = d.get("k_factor", self.k_factor)
        self.home_advantage = d.get("home_advantage", self.home_advantage)
        self.default_rating = d.get("default_rating", self.default_rating)
