#!/usr/bin/env python3
"""
Quick training script for Over 1.5 goals models (XGB + LGBM).

Uses the same 19-feature vector as the existing over_2_5 models.
This is just like over_2_5, but the target is: total_goals > 1 (i.e., 2+ goals).
"""
import sys
import os
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.chdir(str(Path(__file__).parent.parent))

MODELS_DIR = Path("models/api_football")
MATCHES_CSV = Path("data/api-football/matches.csv")


def compute_features(home, away, date, df_past, league_tier=1):
    """Compute 19-feature vector (same as retrain_models.py)."""
    def team_stats(team, n):
        games = df_past[(df_past["home_team"] == team) | (df_past["away_team"] == team)].tail(n)
        if len(games) == 0:
            return {"win_rate": 0.5, "draw_rate": 0.25, "goals_scored": 1.2, "goals_conceded": 1.2}
        wins = draws = gs = gc = 0
        for _, r in games.iterrows():
            if r["home_team"] == team:
                s, c = r["home_score"], r["away_score"]
            else:
                s, c = r["away_score"], r["home_score"]
            gs += s; gc += c
            if s > c: wins += 1
            elif s == c: draws += 1
        n_g = len(games)
        return {"win_rate": wins/n_g, "draw_rate": draws/n_g,
                "goals_scored": gs/n_g, "goals_conceded": gc/n_g}

    def home_stats(team, n):
        games = df_past[df_past["home_team"] == team].tail(n)
        if len(games) == 0:
            return {"win_rate": 0.5, "goals_scored": 1.5}
        wins = sum(1 for _, r in games.iterrows() if r["home_score"] > r["away_score"])
        goals = sum(r["home_score"] for _, r in games.iterrows())
        return {"win_rate": wins/len(games), "goals_scored": goals/len(games)}

    def away_stats(team, n):
        games = df_past[df_past["away_team"] == team].tail(n)
        if len(games) == 0:
            return {"win_rate": 0.35, "goals_scored": 1.1}
        wins = sum(1 for _, r in games.iterrows() if r["away_score"] > r["home_score"])
        goals = sum(r["away_score"] for _, r in games.iterrows())
        return {"win_rate": wins/len(games), "goals_scored": goals/len(games)}

    def h2h_stats(home_t, away_t, n):
        h2h = df_past[
            ((df_past["home_team"] == home_t) & (df_past["away_team"] == away_t)) |
            ((df_past["home_team"] == away_t) & (df_past["away_team"] == home_t))
        ].tail(n)
        if len(h2h) == 0:
            return {"home_win_rate": 0.45, "avg_goals": 2.5, "btts_rate": 0.5, "n": 0}
        hw = btts = tg = 0
        for _, r in h2h.iterrows():
            hg = r["home_score"] if r["home_team"] == home_t else r["away_score"]
            ag = r["away_score"] if r["home_team"] == home_t else r["home_score"]
            tg += hg + ag
            if hg > ag: hw += 1
            if hg > 0 and ag > 0: btts += 1
        n_g = len(h2h)
        return {"home_win_rate": hw/n_g, "avg_goals": tg/n_g, "btts_rate": btts/n_g, "n": n_g}

    h5  = team_stats(home, 5);  h10 = team_stats(home, 10)
    hh5 = home_stats(home, 5)
    a5  = team_stats(away, 5);  a10 = team_stats(away, 10)
    aa5 = away_stats(away, 5)
    h2h = h2h_stats(home, away, 10)

    return [
        h5["win_rate"], h10["win_rate"], h5["draw_rate"],
        h5["goals_scored"], h5["goals_conceded"],
        hh5["win_rate"], hh5["goals_scored"],
        a5["win_rate"], a10["win_rate"], a5["draw_rate"],
        a5["goals_scored"], a5["goals_conceded"],
        aa5["win_rate"], aa5["goals_scored"],
        h2h["home_win_rate"], h2h["avg_goals"], h2h["btts_rate"],
        min(h2h["n"], 10) / 10,
        league_tier / 2,
    ]


def main():
    print("Training Over 1.5 Goals models...")
    print(f"Data: {MATCHES_CSV}")

    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df):,} matches")

    # Use last 4000 matches for training (faster, still representative)
    df = df.tail(4000).reset_index(drop=True)
    print(f"Using last {len(df):,} matches for training")

    X, y_over15 = [], []
    skipped = 0

    for idx, row in df.iterrows():
        if idx % 1000 == 0:
            print(f"  {idx:,}/{len(df):,} processed...")

        home = row["home_team"]
        away = row["away_team"]
        date = row["date"]
        tier = int(row.get("league_tier", 1)) if "league_tier" in df.columns else 1

        df_past = df[df["date"] < date]
        home_count = ((df_past["home_team"] == home) | (df_past["away_team"] == home)).sum()
        away_count = ((df_past["home_team"] == away) | (df_past["away_team"] == away)).sum()

        if home_count < 5 or away_count < 5:
            skipped += 1
            continue

        feats = compute_features(home, away, date, df_past, tier)
        X.append(feats)

        hg, ag = row["home_score"], row["away_score"]
        y_over15.append(1 if hg + ag > 1 else 0)  # 2+ goals

    X = np.array(X)
    y_over15 = np.array(y_over15)
    print(f"\nDataset: {len(X):,} samples (skipped {skipped:,})")
    print(f"Over 1.5 rate: {y_over15.mean():.1%}")

    # Train/test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_over15, test_size=0.15, random_state=42, stratify=y_over15
    )

    # XGBoost
    print("\nTraining XGBoost over_1_5...")
    from xgboost import XGBClassifier
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric="logloss", verbosity=0,
    )
    xgb.fit(X_train, y_train)
    from sklearn.metrics import accuracy_score
    xgb_acc = accuracy_score(y_test, xgb.predict(X_test))
    print(f"  XGBoost accuracy: {xgb_acc:.3f}")

    # LightGBM
    print("Training LightGBM over_1_5...")
    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.05,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    lgbm.fit(X_train, y_train)
    lgbm_acc = accuracy_score(y_test, lgbm.predict(X_test))
    print(f"  LightGBM accuracy: {lgbm_acc:.3f}")

    # Save models
    xgb_path = MODELS_DIR / "over_1_5_xgb.joblib"
    lgbm_path = MODELS_DIR / "over_1_5_lgbm.joblib"
    joblib.dump(xgb, xgb_path)
    joblib.dump(lgbm, lgbm_path)
    print(f"\nSaved: {xgb_path}")
    print(f"Saved: {lgbm_path}")

    # Update meta.json
    meta_path = MODELS_DIR / "meta.json"
    with open(meta_path) as f:
        meta = json.load(f)

    if "over_1_5_xgb" not in meta["models"]:
        meta["models"].append("over_1_5_xgb")
    if "over_1_5_lgbm" not in meta["models"]:
        meta["models"].append("over_1_5_lgbm")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Updated: {meta_path}")

    print("\nDone! Over 1.5 models ready.")


if __name__ == "__main__":
    main()
