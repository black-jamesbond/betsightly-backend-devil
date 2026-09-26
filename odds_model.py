"""
Odds-as-a-feature model (Tier 3).

The bookmaker market is the strongest single predictor in football. This trains a
gradient-boosted classifier on the GitHub dataset, which already carries odds +
ELO + recent form per match, and answers three questions honestly:

  1. How good is the pure MARKET baseline (de-vigged odds)?
  2. Does a model WITH odds beat the market (i.e. is there exploitable edge)?
  3. How much do odds add vs a model with only ELO + form (no odds)?

Temporal split (older 70% train, newer 30% test), no leakage. Metrics: accuracy,
multiclass Brier, log loss. Saves the fitted model for optional live use.

Usage:
    python odds_model.py
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss

GITHUB = Path("data/github-football/Matches.csv")
OUT = Path("models/odds/odds_model.pkl")

RESULT_MAP = {"H": 0, "D": 1, "A": 2}


def devig(oh, od, oa):
    """De-vig 1X2 odds into implied probabilities."""
    with np.errstate(divide="ignore", invalid="ignore"):
        rh, rd, ra = 1.0 / oh, 1.0 / od, 1.0 / oa
    s = rh + rd + ra
    return np.column_stack([rh / s, rd / s, ra / s])


def brier_multiclass(P, y):
    onehot = np.zeros_like(P)
    onehot[np.arange(len(y)), y] = 1
    return float(np.mean(np.sum((P - onehot) ** 2, axis=1)))


def build(df):
    df = df.copy()
    df["MatchDate"] = pd.to_datetime(df["MatchDate"], errors="coerce")
    df = df.dropna(subset=["MatchDate", "FTResult", "OddHome", "OddDraw", "OddAway"])
    df = df[df["FTResult"].isin(RESULT_MAP)]
    df = df.sort_values("MatchDate").reset_index(drop=True)

    oh = df["OddHome"].astype(float).values
    od = df["OddDraw"].astype(float).values
    oa = df["OddAway"].astype(float).values
    mkt = devig(oh, od, oa)

    def col(name):
        return pd.to_numeric(df[name], errors="coerce").values if name in df.columns else np.full(len(df), np.nan)

    he, ae = col("HomeElo"), col("AwayElo")
    elo_diff = he - ae
    elo_exp = 1.0 / (1.0 + 10 ** (-elo_diff / 400.0))

    feats_market = {
        "mkt_home": mkt[:, 0], "mkt_draw": mkt[:, 1], "mkt_away": mkt[:, 2],
        "odd_home": oh, "odd_draw": od, "odd_away": oa,
    }
    feats_form = {
        "home_elo": he, "away_elo": ae, "elo_diff": elo_diff, "elo_exp": elo_exp,
        "form3_home": col("Form3Home"), "form5_home": col("Form5Home"),
        "form3_away": col("Form3Away"), "form5_away": col("Form5Away"),
    }
    X_all = pd.DataFrame({**feats_market, **feats_form})
    X_noodds = pd.DataFrame(feats_form)
    y = df["FTResult"].map(RESULT_MAP).values
    return X_all, X_noodds, y, mkt, df["MatchDate"].values


def evaluate(name, P, y):
    acc = float(np.mean(P.argmax(1) == y))
    return {"model": name, "accuracy": round(acc, 4),
            "brier": round(brier_multiclass(P, y), 4),
            "logloss": round(log_loss(y, P, labels=[0, 1, 2]), 4)}


def main():
    print("Loading GitHub dataset...")
    df = pd.read_csv(GITHUB, low_memory=False)
    X_all, X_noodds, y, mkt, dates = build(df)
    n = len(y)
    cut = int(n * 0.70)
    print(f"{n:,} matches with odds. Train {cut:,} / Test {n - cut:,} "
          f"(split {pd.Timestamp(dates[cut]).date()})\n")

    tr = slice(0, cut); te = slice(cut, n)

    results = []
    # 1. Market baseline
    results.append(evaluate("market (de-vigged odds)", mkt[te], y[te]))

    # 2. GBM with odds
    t0 = time.time()
    m_all = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                           max_depth=6, l2_regularization=1.0,
                                           random_state=0)
    m_all.fit(X_all.iloc[tr], y[tr])
    P_all = m_all.predict_proba(X_all.iloc[te])
    results.append(evaluate("GBM + odds + elo + form", P_all, y[te]))
    print(f"  trained odds model in {time.time()-t0:.0f}s")

    # 3. GBM without odds
    m_no = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05,
                                          max_depth=6, l2_regularization=1.0,
                                          random_state=0)
    m_no.fit(X_noodds.iloc[tr], y[tr])
    P_no = m_no.predict_proba(X_noodds.iloc[te])
    results.append(evaluate("GBM elo + form (no odds)", P_no, y[te]))

    print("\n" + "=" * 68)
    print(f"  {'model':<30}{'acc':>8}{'brier':>10}{'logloss':>10}")
    print("=" * 68)
    for r in results:
        print(f"  {r['model']:<30}{r['accuracy']:>8.3f}{r['brier']:>10.4f}{r['logloss']:>10.4f}")
    print("=" * 68)

    # Value angle: how often does the model disagree with the market, and is it right?
    market_pick = mkt[te].argmax(1)
    model_pick = P_all.argmax(1)
    disagree = market_pick != model_pick
    if disagree.sum():
        acc_dis = float(np.mean(model_pick[disagree] == y[te][disagree]))
        acc_mkt_dis = float(np.mean(market_pick[disagree] == y[te][disagree]))
        print(f"\n  Model disagrees with market on {disagree.sum():,}/{len(model_pick):,} "
              f"({100*disagree.mean():.1f}%)")
        print(f"    On those, model right {acc_dis:.1%} vs market {acc_mkt_dis:.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(OUT, "wb") as f:
        pickle.dump({"model": m_all, "features": list(X_all.columns)}, f)
    print(f"\n  Saved odds model -> {OUT}")


if __name__ == "__main__":
    main()
