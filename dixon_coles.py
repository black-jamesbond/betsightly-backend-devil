"""
Dixon-Coles goal model — a proper probabilistic base engine for football.

Instead of blending summary win rates, we estimate latent attack and defense
ratings for every team by maximum likelihood over historical results, with:

  - a home-advantage term (per league)
  - the Dixon-Coles low-score correlation correction (rho)
  - time-decay weighting so recent form dominates (xi, in 1/days)
  - ridge shrinkage toward league-average (attack=defense=0), which handles
    sparse teams by partial-pooling them toward the mean

For a match, the fitted ratings give expected goal rates (lambda, mu); the full
scoreline probability matrix P(i,j) follows, and every market is derived from
that ONE joint distribution — so 1X2, Over 2.5 and BTTS are mutually consistent.

Models are fit per league (each league gets its own home advantage and scoring
baseline, which also normalizes across leagues of different scoring levels).

    Home goals ~ Poisson(lambda),  lambda = exp(atk_home - def_away + gamma)
    Away goals ~ Poisson(mu),      mu     = exp(atk_away - def_home)
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

MAX_GOALS = 10               # scoreline matrix truncation
DEFAULT_XI = math.log(2) / 180.0   # time-decay: ~180-day half-life
DEFAULT_REG = 1.0            # ridge strength (shrinkage toward league mean)


def _tau(hg, ag, lam, mu, rho):
    """Dixon-Coles low-score dependency correction (vectorized)."""
    tau = np.ones_like(lam)
    m00 = (hg == 0) & (ag == 0)
    m01 = (hg == 0) & (ag == 1)
    m10 = (hg == 1) & (ag == 0)
    m11 = (hg == 1) & (ag == 1)
    tau[m00] = 1.0 - lam[m00] * mu[m00] * rho
    tau[m01] = 1.0 + lam[m01] * rho
    tau[m10] = 1.0 + mu[m10] * rho
    tau[m11] = 1.0 - rho
    return tau


class LeagueModel:
    """Fitted Dixon-Coles ratings for a single league."""

    def __init__(self, teams, attack, defense, gamma, rho, last_date=None):
        self.teams = list(teams)
        self.index = {t: i for i, t in enumerate(self.teams)}
        self.attack = np.asarray(attack, dtype=float)
        self.defense = np.asarray(defense, dtype=float)
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.last_date = last_date  # ISO string of the most recent training match

    def has(self, team):
        return team in self.index

    def rates(self, home, away):
        i, j = self.index[home], self.index[away]
        lam = math.exp(self.attack[i] - self.defense[j] + self.gamma)
        mu = math.exp(self.attack[j] - self.defense[i])
        return lam, mu

    def score_matrix(self, home, away):
        lam, mu = self.rates(home, away)
        ks = np.arange(0, MAX_GOALS + 1)
        # Poisson pmfs
        ph = np.exp(ks * math.log(lam) - lam - gammaln(ks + 1))
        pa = np.exp(ks * math.log(mu) - mu - gammaln(ks + 1))
        M = np.outer(ph, pa)  # M[i,j] = P(home=i, away=j)
        # DC correction on the 2x2 low-score corner
        M[0, 0] *= 1.0 - lam * mu * self.rho
        M[0, 1] *= 1.0 + lam * self.rho
        M[1, 0] *= 1.0 + mu * self.rho
        M[1, 1] *= 1.0 - self.rho
        M = np.clip(M, 0, None)
        M /= M.sum()
        return M, lam, mu

    def predict(self, home, away):
        """Return a prediction dict matching predict_today.predict_match keys."""
        M, lam, mu = self.score_matrix(home, away)
        idx = np.arange(0, MAX_GOALS + 1)
        HI, AI = np.meshgrid(idx, idx, indexing="ij")
        home_win = M[HI > AI].sum()
        draw = M[HI == AI].sum()
        away_win = M[HI < AI].sum()
        total = HI + AI
        over15 = M[total >= 2].sum()
        over25 = M[total >= 3].sum()
        over35 = M[total >= 4].sum()
        btts = M[(HI >= 1) & (AI >= 1)].sum()

        best = max((("Home Win", home_win), ("Draw", draw), ("Away Win", away_win)),
                   key=lambda t: t[1])
        return {
            "home_win": round(home_win * 100, 1),
            "draw": round(draw * 100, 1),
            "away_win": round(away_win * 100, 1),
            "prediction": best[0],
            "confidence": round(best[1] * 100, 1),
            "over_1.5": round(over15 * 100, 1),
            "over_2.5": round(over25 * 100, 1),
            "over_3.5": round(over35 * 100, 1),
            "btts": round(btts * 100, 1),
            "expected_home_goals": round(lam, 2),
            "expected_away_goals": round(mu, 2),
            "data_quality": "HIGH",
            "h2h_meetings": 0,
            "engine": "dixon_coles",
        }

    def to_dict(self):
        return {"teams": self.teams, "attack": self.attack.tolist(),
                "defense": self.defense.tolist(), "gamma": self.gamma, "rho": self.rho,
                "last_date": self.last_date}

    @classmethod
    def from_dict(cls, d):
        return cls(d["teams"], d["attack"], d["defense"], d["gamma"], d["rho"],
                   d.get("last_date"))


def fit_league(home_idx, away_idx, home_goals, away_goals, weights, n_teams,
               reg=DEFAULT_REG):
    """MLE fit of one league's ratings. Arrays are integer team indices / goals."""
    hi = np.asarray(home_idx); ai = np.asarray(away_idx)
    hg = np.asarray(home_goals, dtype=float); ag = np.asarray(away_goals, dtype=float)
    w = np.asarray(weights, dtype=float)

    def nll(params):
        att = params[:n_teams]
        dfn = params[n_teams:2 * n_teams]
        gamma = params[2 * n_teams]
        rho = params[2 * n_teams + 1]
        lam = np.exp(att[hi] - dfn[ai] + gamma)
        mu = np.exp(att[ai] - dfn[hi])
        ll = (hg * np.log(lam) - lam - gammaln(hg + 1)
              + ag * np.log(mu) - mu - gammaln(ag + 1))
        tau = np.clip(_tau(hg, ag, lam, mu, rho), 1e-12, None)
        ll = ll + np.log(tau)
        return -np.sum(w * ll) + reg * (np.sum(att ** 2) + np.sum(dfn ** 2))

    x0 = np.zeros(2 * n_teams + 2)
    x0[2 * n_teams] = 0.25  # home advantage prior
    bounds = [(-3, 3)] * (2 * n_teams) + [(-1.0, 1.0), (-0.4, 0.4)]
    res = minimize(nll, x0, method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": 200})
    p = res.x
    att = p[:n_teams] - p[:n_teams].mean()  # center for identifiability
    dfn = p[n_teams:2 * n_teams] - p[n_teams:2 * n_teams].mean()
    return att, dfn, p[2 * n_teams], p[2 * n_teams + 1]


class DixonColes:
    """Manager holding one LeagueModel per league."""

    def __init__(self, leagues=None, meta=None):
        self.leagues = leagues or {}   # league_name -> LeagueModel
        self.meta = meta or {}

    def predict(self, home_norm, away_norm, league):
        m = self.leagues.get(league)
        if m is None or not (m.has(home_norm) and m.has(away_norm)):
            return None
        return m.predict(home_norm, away_norm)

    def predict_any(self, home_norm, away_norm):
        """Find the league model containing BOTH teams and predict. When several
        cover them (e.g. a stale GitHub model and a fresh api-football one), use
        the one with the most recent training data."""
        candidates = [m for m in self.leagues.values()
                      if m.has(home_norm) and m.has(away_norm)]
        if not candidates:
            return None
        best = max(candidates, key=lambda m: (m.last_date or ""))
        return best.predict(home_norm, away_norm)

    @classmethod
    def fit(cls, df, as_of=None, xi=DEFAULT_XI, reg=DEFAULT_REG,
            min_matches=60, min_team_games=4, lookback_days=1095):
        """Fit per-league models using matches strictly before `as_of`.

        df needs columns: league, _home_norm, _away_norm, home_score,
        away_score, date. Time-decay weight = exp(-xi * age_days). Matches older
        than `lookback_days` before as_of are dropped (their decayed weight is
        negligible) — this bounds fit cost with no meaningful accuracy loss."""
        if as_of is not None:
            df = df[df["date"] < as_of]
            if lookback_days:
                lo = np.datetime64(as_of) - np.timedelta64(int(lookback_days), "D")
                df = df[df["date"] >= lo]
        df = df.dropna(subset=["league", "_home_norm", "_away_norm",
                               "home_score", "away_score", "date"])
        leagues = {}
        ref = as_of if as_of is not None else df["date"].max()
        ref = np.datetime64(ref)

        for lg, g in df.groupby("league"):
            if len(g) < min_matches:
                continue
            hn = g["_home_norm"].values
            an = g["_away_norm"].values
            # count games per team; drop teams with too few
            from collections import Counter
            cnt = Counter()
            for t in hn:
                cnt[t] += 1
            for t in an:
                cnt[t] += 1
            teams = [t for t, c in cnt.items() if c >= min_team_games]
            if len(teams) < 4:
                continue
            tset = set(teams)
            keep = np.array([(h in tset and a in tset) for h, a in zip(hn, an)])
            if keep.sum() < min_matches:
                continue
            hn, an = hn[keep], an[keep]
            hg = g["home_score"].values[keep].astype(float)
            ag = g["away_score"].values[keep].astype(float)
            dates = g["date"].values[keep].astype("datetime64[D]")
            age_days = (ref.astype("datetime64[D]") - dates).astype("timedelta64[D]").astype(float)
            age_days = np.clip(age_days, 0, None)
            w = np.exp(-xi * age_days)

            index = {t: i for i, t in enumerate(teams)}
            hi = np.array([index[t] for t in hn])
            ai = np.array([index[t] for t in an])
            att, dfn, gamma, rho = fit_league(hi, ai, hg, ag, w, len(teams), reg)
            last_date = str(np.max(dates))
            leagues[lg] = LeagueModel(teams, att, dfn, gamma, rho, last_date)

        return cls(leagues, meta={"as_of": str(as_of) if as_of else None,
                                  "xi": xi, "reg": reg, "n_leagues": len(leagues)})

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"meta": self.meta,
                "leagues": {lg: m.to_dict() for lg, m in self.leagues.items()}}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        leagues = {lg: LeagueModel.from_dict(md) for lg, md in d["leagues"].items()}
        return cls(leagues, d.get("meta", {}))
