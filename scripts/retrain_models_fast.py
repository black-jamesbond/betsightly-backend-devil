"""
Fast retrain of ALL ML models — incremental feature engineering.

Produces features IDENTICAL to scripts/retrain_models.py but computes them
in a single chronological pass (O(n) instead of O(n^2)), so it handles
100K+ matches in seconds instead of hours.

Same outputs as the original:
    models/api_football/<target>_<model_type>.joblib
    models/api_football/meta.json
    models/api_football/model_weights.json

Usage:
    python scripts/retrain_models_fast.py
"""

import json
import sys
from collections import defaultdict, deque
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

MATCHES_CSV = Path("data/api-football/matches.csv")
MODELS_DIR  = Path("models/api_football")
FORM_WINDOW = 5
H2H_WINDOW  = 10

FEATURE_COLUMNS = [
    "home_win_rate_5", "home_win_rate_10", "home_draw_rate_5",
    "home_goals_scored_5", "home_goals_conceded_5",
    "home_home_win_rate_5", "home_home_goals_5",
    "away_win_rate_5", "away_win_rate_10", "away_draw_rate_5",
    "away_goals_scored_5", "away_goals_conceded_5",
    "away_away_win_rate_5", "away_away_goals_5",
    "h2h_home_win_rate", "h2h_avg_goals", "h2h_btts_rate", "h2h_meetings",
    "league_tier",
]


# ---------------------------------------------------------------------------
# Incremental feature state
# ---------------------------------------------------------------------------

class State:
    """Rolling per-team and head-to-head state, updated chronologically."""

    def __init__(self):
        # (goals_scored, goals_conceded) tuples, newest last
        self.games = defaultdict(lambda: deque(maxlen=10))     # any venue
        self.home_games = defaultdict(lambda: deque(maxlen=5)) # as home
        self.away_games = defaultdict(lambda: deque(maxlen=5)) # as away
        self.count = defaultdict(int)                          # total games seen
        # h2h key: tuple(sorted([a, b])) -> deque of (home_team, hg, ag)
        self.h2h = defaultdict(lambda: deque(maxlen=H2H_WINDOW))

    # -- mirrors _team_stats(df_past, team, n) -----------------------------
    def team_stats(self, team, n):
        games = list(self.games[team])[-n:]
        if not games:
            return {"win_rate": 0.5, "draw_rate": 0.25,
                    "goals_scored": 1.2, "goals_conceded": 1.2}
        wins = sum(1 for gs, gc in games if gs > gc)
        draws = sum(1 for gs, gc in games if gs == gc)
        return {
            "win_rate": wins / len(games),
            "draw_rate": draws / len(games),
            "goals_scored": sum(gs for gs, _ in games) / len(games),
            "goals_conceded": sum(gc for _, gc in games) / len(games),
        }

    # -- mirrors _team_home_stats ------------------------------------------
    def home_stats(self, team):
        games = list(self.home_games[team])
        if not games:
            return {"win_rate": 0.5, "goals_scored": 1.5}
        wins = sum(1 for hg, ag in games if hg > ag)
        return {"win_rate": wins / len(games),
                "goals_scored": sum(hg for hg, _ in games) / len(games)}

    # -- mirrors _team_away_stats ------------------------------------------
    def away_stats(self, team):
        games = list(self.away_games[team])
        if not games:
            return {"win_rate": 0.35, "goals_scored": 1.1}
        wins = sum(1 for ag, hg in games if ag > hg)
        return {"win_rate": wins / len(games),
                "goals_scored": sum(ag for ag, _ in games) / len(games)}

    # -- mirrors _h2h_stats --------------------------------------------------
    def h2h_stats(self, home, away):
        meetings = self.h2h[tuple(sorted((home, away)))]
        if not meetings:
            return {"home_win_rate": 0.45, "avg_goals": 2.5,
                    "btts_rate": 0.5, "n": 0}
        home_wins = btts = total_goals = 0
        for h_team, hg, ag in meetings:
            # express from the CURRENT home team's perspective
            if h_team == home:
                g1, g2 = hg, ag
            else:
                g1, g2 = ag, hg
            total_goals += g1 + g2
            if g1 > g2:
                home_wins += 1
            if g1 > 0 and g2 > 0:
                btts += 1
        n = len(meetings)
        return {"home_win_rate": home_wins / n, "avg_goals": total_goals / n,
                "btts_rate": btts / n, "n": n}

    def features(self, home, away, tier):
        h5 = self.team_stats(home, 5)
        h10 = self.team_stats(home, 10)
        hh5 = self.home_stats(home)
        a5 = self.team_stats(away, 5)
        a10 = self.team_stats(away, 10)
        aa5 = self.away_stats(away)
        h2h = self.h2h_stats(home, away)
        return [
            h5["win_rate"], h10["win_rate"], h5["draw_rate"],
            h5["goals_scored"], h5["goals_conceded"],
            hh5["win_rate"], hh5["goals_scored"],
            a5["win_rate"], a10["win_rate"], a5["draw_rate"],
            a5["goals_scored"], a5["goals_conceded"],
            aa5["win_rate"], aa5["goals_scored"],
            h2h["home_win_rate"], h2h["avg_goals"], h2h["btts_rate"],
            min(h2h["n"], 10) / 10,
            tier / 2,
        ]

    def update(self, home, away, hg, ag):
        self.games[home].append((hg, ag))
        self.games[away].append((ag, hg))
        self.home_games[home].append((hg, ag))
        self.away_games[away].append((ag, hg))
        self.count[home] += 1
        self.count[away] += 1
        self.h2h[tuple(sorted((home, away)))].append((home, hg, ag))


# ---------------------------------------------------------------------------
# Build dataset in one chronological pass
# ---------------------------------------------------------------------------

def build_dataset(df: pd.DataFrame):
    print("Building feature matrix (single pass)...")
    state = State()
    X, y_result, y_over15, y_over25, y_btts = [], [], [], [], []
    skipped = 0

    # Group by date so same-day matches don't see each other (matches the
    # original's strict df["date"] < date filter).
    for date, day_df in df.groupby("date", sort=True):
        rows = list(day_df.itertuples(index=False))
        feats_today = []
        for r in rows:
            home, away = r.home_team, r.away_team
            if state.count[home] < 5 or state.count[away] < 5:
                feats_today.append(None)
                skipped += 1
                continue
            tier = int(r.league_tier) if pd.notna(r.league_tier) else 1
            feats_today.append(state.features(home, away, tier))

        for r, feats in zip(rows, feats_today):
            hg, ag = int(r.home_score), int(r.away_score)
            if feats is not None:
                X.append(feats)
                y_result.append(2 if hg > ag else (1 if hg == ag else 0))
                y_over15.append(1 if hg + ag > 1 else 0)
                y_over25.append(1 if hg + ag > 2 else 0)
                y_btts.append(1 if hg > 0 and ag > 0 else 0)
            state.update(r.home_team, r.away_team, hg, ag)

    print(f"  Built {len(X):,} samples (skipped {skipped:,} with <5 games history)")
    return (np.array(X), np.array(y_result), np.array(y_over15),
            np.array(y_over25), np.array(y_btts))


# ---------------------------------------------------------------------------
# Model trainers (same params as retrain_models.py; RF/ET trimmed to 300
# trees to keep file sizes manageable at 100K samples)
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train):
    from xgboost import XGBClassifier
    n_classes = len(np.unique(y_train))
    return XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob" if n_classes > 2 else "binary:logistic",
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        use_label_encoder=False, random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)


def train_lightgbm(X_train, y_train):
    import lightgbm as lgb
    n_classes = len(np.unique(y_train))
    return lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        objective="multiclass" if n_classes > 2 else "binary",
        num_class=n_classes if n_classes > 2 else None,
        random_state=42, n_jobs=-1, verbose=-1,
    ).fit(X_train, y_train)


def train_catboost(X_train, y_train):
    from catboost import CatBoostClassifier
    n_classes = len(np.unique(y_train))
    return CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05,
        loss_function="MultiClass" if n_classes > 2 else "Logloss",
        random_seed=42, verbose=0,
    ).fit(X_train, y_train)


def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)


def train_extra_trees(X_train, y_train):
    from sklearn.ensemble import ExtraTreesClassifier
    return ExtraTreesClassifier(
        n_estimators=300, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    ).fit(X_train, y_train)


def train_neural_network(X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32), activation="relu", solver="adam",
        learning_rate="adaptive", learning_rate_init=0.001, max_iter=500,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
        random_state=42, verbose=False,
    ).fit(X_scaled, y_train)
    return (model, scaler)


ALL_TRAINERS = [
    ("xgb", train_xgboost, False),
    ("lgbm", train_lightgbm, False),
    ("catboost", train_catboost, False),
    ("rf", train_random_forest, False),
    ("et", train_extra_trees, False),
    ("nn", train_neural_network, True),
]


def evaluate(model, X_test, y_test, name, is_nn=False):
    from sklearn.metrics import accuracy_score, log_loss
    if is_nn:
        m, scaler = model
        X_eval = scaler.transform(X_test)
        preds, proba = m.predict(X_eval), m.predict_proba(X_eval)
    else:
        preds, proba = model.predict(X_test), model.predict_proba(X_test)
    acc = accuracy_score(y_test, preds)
    try:
        print(f"    {name:20s}: accuracy = {acc:.3f}  |  log_loss = {log_loss(y_test, proba):.4f}")
    except Exception:
        print(f"    {name:20s}: accuracy = {acc:.3f}")
    return acc


def train_and_save(X, y, label, out_dir):
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y)
    print(f"\n{'='*60}\n  TARGET: {label}  ({len(np.unique(y))} classes, "
          f"{len(X_train):,} train / {len(X_test):,} test)\n{'='*60}")
    accuracies = {}
    for tag, fn, is_nn in ALL_TRAINERS:
        try:
            model = fn(X_train, y_train)
            acc = evaluate(model, X_test, y_test, tag, is_nn)
            accuracies[f"{label}_{tag}"] = round(acc, 4)
            path = out_dir / f"{label}_{tag}.joblib"
            joblib.dump(model, path, compress=3)
            print(f"    Saved -> {path}  ({path.stat().st_size/1024/1024:.1f} MB)")
        except ImportError:
            print(f"    {tag} not installed -- skipping")
        except Exception as e:
            print(f"    {tag} FAILED: {e}")
    return accuracies


def main():
    print(f"Loading {MATCHES_CSV} ...")
    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} matches across {df['league_name'].nunique()} league names, "
          f"{df['date'].min().date()} - {df['date'].max().date()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    X, y_result, y_over15, y_over25, y_btts = build_dataset(df)

    print(f"\nDataset summary:")
    print(f"  Samples   : {len(X):,}")
    print(f"  Home wins : {(y_result == 2).mean():.1%} | Draws: {(y_result == 1).mean():.1%} "
          f"| Away wins: {(y_result == 0).mean():.1%}")
    print(f"  Over 1.5  : {y_over15.mean():.1%} | Over 2.5: {y_over25.mean():.1%} "
          f"| BTTS: {y_btts.mean():.1%}")

    all_acc = {}
    for label, y in [("match_result", y_result), ("over_1_5", y_over15),
                     ("over_2_5", y_over25), ("btts", y_btts)]:
        all_acc.update(train_and_save(X, y, label, MODELS_DIR))

    # Statistical models
    print(f"\n{'='*60}\n  STATISTICAL MODELS\n{'='*60}")
    matches_for_stat = [
        {"home_team": r.home_team, "away_team": r.away_team,
         "home_goals": int(r.home_score), "away_goals": int(r.away_score),
         "date": str(r.date)}
        for r in df.itertuples(index=False)
    ]
    try:
        from ml.elo_model import EloRatingSystem
        elo = EloRatingSystem()
        elo.train(matches_for_stat)
        elo.save(str(MODELS_DIR / "elo_ratings.json"))
        print(f"  ELO ratings saved ({len(elo.ratings)} teams)")
    except Exception as e:
        print(f"  ELO training failed: {e}")
    try:
        from ml.dixon_coles_model import DixonColesModel
        dc = DixonColesModel()
        dc.train(matches_for_stat[-2000:])
        dc.save(str(MODELS_DIR / "dixon_coles.json"))
        print(f"  Dixon-Coles saved ({len(dc.teams)} teams)")
    except Exception as e:
        print(f"  Dixon-Coles training failed: {e}")

    with open(MODELS_DIR / "model_weights.json", "w") as f:
        json.dump(all_acc, f, indent=2)

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "form_window": FORM_WINDOW,
        "h2h_window": H2H_WINDOW,
        "trained_at": datetime.utcnow().isoformat(),
        "n_samples": len(X),
        "models": sorted(all_acc.keys()),
        "model_types": ["xgb", "lgbm", "catboost", "rf", "et", "nn"],
        "targets": ["match_result", "over_1_5", "over_2_5", "btts"],
        "result_classes": {0: "Away Win", 1: "Draw", 2: "Home Win"},
    }
    with open(MODELS_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*60}\n  FINAL ACCURACY SUMMARY\n{'='*60}")
    for target in ["match_result", "over_1_5", "over_2_5", "btts"]:
        t_acc = {k: v for k, v in all_acc.items() if k.startswith(target)}
        if t_acc:
            best = max(t_acc, key=t_acc.get)
            print(f"  Best {target:14s}: {best} ({t_acc[best]:.3f})")
    total_mb = sum(f.stat().st_size for f in MODELS_DIR.glob("*.joblib")) / 1024 / 1024
    print(f"\n  {len(all_acc)} models, {total_mb:.1f} MB -> {MODELS_DIR}/")


if __name__ == "__main__":
    main()
