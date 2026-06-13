"""
Retrain ALL ML models on API-Football historical data.

Reads data/api-football/matches.csv (produced by fetch_history.py),
engineers features, and trains a full ensemble of models:

  - XGBoost          (gradient boosting — strong baseline)
  - LightGBM         (gradient boosting — fast, handles categoricals)
  - CatBoost         (gradient boosting — robust to overfitting)
  - Random Forest    (bagging — diverse errors from boosters)
  - Extra Trees      (bagging — even more randomness than RF)
  - Neural Network   (MLP — completely different architecture)

Each model is trained on 4 prediction targets:
  match_result, over_1_5, over_2_5, btts

Total: up to 24 models, all saved to models/api_football/.
The prediction service loads them all and uses weighted ensemble voting.

Usage:
    py scripts/retrain_models.py

Outputs:
    models/api_football/<target>_<model_type>.joblib
    models/api_football/meta.json
    models/api_football/model_weights.json
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Ensure project root is on sys.path so `ml.*` imports work when run from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
MATCHES_CSV  = Path("data/api-football/matches.csv")
MODELS_DIR   = Path("models/api_football")
FORM_WINDOW  = 5    # last N games for form features
H2H_WINDOW   = 10   # last N head-to-head meetings

# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    print(f"Loading {MATCHES_CSV} ...")
    df = pd.read_csv(MATCHES_CSV, low_memory=False)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"  {len(df):,} finished matches across {df['league_name'].nunique()} leagues")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _team_stats(df_past: pd.DataFrame, team: str, n: int) -> dict:
    """Rolling form stats for a team from their last N games."""
    games = df_past[
        (df_past["home_team"] == team) | (df_past["away_team"] == team)
    ].tail(n)

    if len(games) == 0:
        return {
            "win_rate": 0.5, "draw_rate": 0.25,
            "goals_scored": 1.2, "goals_conceded": 1.2,
            "n_games": 0,
        }

    wins = draws = goals_scored = goals_conceded = 0
    for _, r in games.iterrows():
        if r["home_team"] == team:
            gs, gc = r["home_score"], r["away_score"]
        else:
            gs, gc = r["away_score"], r["home_score"]
        goals_scored    += gs
        goals_conceded  += gc
        if gs > gc:
            wins  += 1
        elif gs == gc:
            draws += 1

    n_games = len(games)
    return {
        "win_rate":        wins  / n_games,
        "draw_rate":       draws / n_games,
        "goals_scored":    goals_scored  / n_games,
        "goals_conceded":  goals_conceded / n_games,
        "n_games":         n_games,
    }


def _team_home_stats(df_past: pd.DataFrame, team: str, n: int) -> dict:
    games = df_past[df_past["home_team"] == team].tail(n)
    if len(games) == 0:
        return {"win_rate": 0.5, "goals_scored": 1.5}
    wins = sum(1 for _, r in games.iterrows() if r["home_score"] > r["away_score"])
    goals = sum(r["home_score"] for _, r in games.iterrows())
    n_g = len(games)
    return {"win_rate": wins / n_g, "goals_scored": goals / n_g}


def _team_away_stats(df_past: pd.DataFrame, team: str, n: int) -> dict:
    games = df_past[df_past["away_team"] == team].tail(n)
    if len(games) == 0:
        return {"win_rate": 0.35, "goals_scored": 1.1}
    wins = sum(1 for _, r in games.iterrows() if r["away_score"] > r["home_score"])
    goals = sum(r["away_score"] for _, r in games.iterrows())
    n_g = len(games)
    return {"win_rate": wins / n_g, "goals_scored": goals / n_g}


def _h2h_stats(df_past: pd.DataFrame, home: str, away: str, n: int) -> dict:
    h2h = df_past[
        ((df_past["home_team"] == home) & (df_past["away_team"] == away)) |
        ((df_past["home_team"] == away) & (df_past["away_team"] == home))
    ].tail(n)

    if len(h2h) == 0:
        return {"home_win_rate": 0.45, "avg_goals": 2.5, "btts_rate": 0.5, "n": 0}

    home_wins = btts = total_goals = 0
    for _, r in h2h.iterrows():
        if r["home_team"] == home:
            hg, ag = r["home_score"], r["away_score"]
        else:
            hg, ag = r["away_score"], r["home_score"]
        total_goals += hg + ag
        if hg > ag:
            home_wins += 1
        if hg > 0 and ag > 0:
            btts += 1

    n_g = len(h2h)
    return {
        "home_win_rate": home_wins / n_g,
        "avg_goals":     total_goals / n_g,
        "btts_rate":     btts / n_g,
        "n":             n_g,
    }


FEATURE_COLUMNS = [
    "home_win_rate_5",
    "home_win_rate_10",
    "home_draw_rate_5",
    "home_goals_scored_5",
    "home_goals_conceded_5",
    "home_home_win_rate_5",
    "home_home_goals_5",
    "away_win_rate_5",
    "away_win_rate_10",
    "away_draw_rate_5",
    "away_goals_scored_5",
    "away_goals_conceded_5",
    "away_away_win_rate_5",
    "away_away_goals_5",
    "h2h_home_win_rate",
    "h2h_avg_goals",
    "h2h_btts_rate",
    "h2h_meetings",
    "league_tier",
]


def compute_features(home: str, away: str, date, df_past: pd.DataFrame,
                     league_tier: int = 1) -> list:
    """Build feature vector -- must mirror advanced_prediction_service._compute_api_features()."""
    h5  = _team_stats(df_past, home, 5)
    h10 = _team_stats(df_past, home, 10)
    hh5 = _team_home_stats(df_past, home, 5)
    a5  = _team_stats(df_past, away, 5)
    a10 = _team_stats(df_past, away, 10)
    aa5 = _team_away_stats(df_past, away, 5)
    h2h = _h2h_stats(df_past, home, away, H2H_WINDOW)

    return [
        h5["win_rate"],
        h10["win_rate"],
        h5["draw_rate"],
        h5["goals_scored"],
        h5["goals_conceded"],
        hh5["win_rate"],
        hh5["goals_scored"],
        a5["win_rate"],
        a10["win_rate"],
        a5["draw_rate"],
        a5["goals_scored"],
        a5["goals_conceded"],
        aa5["win_rate"],
        aa5["goals_scored"],
        h2h["home_win_rate"],
        h2h["avg_goals"],
        h2h["btts_rate"],
        min(h2h["n"], 10) / 10,   # normalised 0-1
        league_tier / 2,           # normalised 0-1 (tier 0=Europe, 1=top, 2=second)
    ]


# ---------------------------------------------------------------------------
# Build dataset
# ---------------------------------------------------------------------------

def build_dataset(df: pd.DataFrame) -> tuple:
    """
    Compute features + targets for every match in df (using only past data).
    Skips teams with fewer than 5 historical games.
    Returns X (numpy), y_result, y_over15, y_over25, y_btts.
    """
    print("Building feature matrix (this takes a few minutes)...")

    X, y_result, y_over15, y_over25, y_btts = [], [], [], [], []
    skipped = 0

    for idx, row in df.iterrows():
        if idx % 5000 == 0 and idx > 0:
            print(f"  {idx:,}/{len(df):,} matches processed...")

        home  = row["home_team"]
        away  = row["away_team"]
        date  = row["date"]
        tier  = int(row.get("league_tier", 1))

        df_past = df[df["date"] < date]

        # Skip if either team has fewer than 5 historical games
        home_count = ((df_past["home_team"] == home) | (df_past["away_team"] == home)).sum()
        away_count = ((df_past["home_team"] == away) | (df_past["away_team"] == away)).sum()
        if home_count < 5 or away_count < 5:
            skipped += 1
            continue

        feats = compute_features(home, away, date, df_past, tier)
        X.append(feats)

        # Targets
        hg, ag = row["home_score"], row["away_score"]
        y_result.append(2 if hg > ag else (1 if hg == ag else 0))
        y_over15.append(1 if hg + ag > 1 else 0)   # 2+ goals total
        y_over25.append(1 if hg + ag > 2 else 0)   # 3+ goals total
        y_btts.append(1 if hg > 0 and ag > 0 else 0)

    print(f"  Built {len(X):,} training samples (skipped {skipped:,} with insufficient history)")
    return np.array(X), np.array(y_result), np.array(y_over15), np.array(y_over25), np.array(y_btts)


# ---------------------------------------------------------------------------
# Model trainers
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train, label: str):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        print("  xgboost not installed -- skipping")
        return None

    n_classes = len(np.unique(y_train))
    objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train, label: str):
    try:
        import lightgbm as lgb
    except ImportError:
        print("  lightgbm not installed -- skipping")
        return None

    n_classes = len(np.unique(y_train))
    objective = "multiclass" if n_classes > 2 else "binary"

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective=objective,
        num_class=n_classes if n_classes > 2 else None,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train, label: str):
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        print("  catboost not installed -- skipping")
        return None

    n_classes = len(np.unique(y_train))
    loss = "MultiClass" if n_classes > 2 else "Logloss"

    model = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function=loss,
        random_seed=42,
        verbose=0,
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, label: str):
    """Random Forest — bagging ensemble, different error profile from boosters."""
    from sklearn.ensemble import RandomForestClassifier

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_extra_trees(X_train, y_train, label: str):
    """Extra Trees — more randomized splits than RF, often better generalization."""
    from sklearn.ensemble import ExtraTreesClassifier

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_neural_network(X_train, y_train, label: str):
    """MLP Neural Network — completely different architecture from tree-based models."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler

    # Neural nets need scaled features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate="adaptive",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42,
        verbose=False,
    )
    model.fit(X_scaled, y_train)

    # Return both model and scaler (needed at prediction time)
    return (model, scaler)


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, X_test, y_test, name: str, is_nn: bool = False):
    """Evaluate a model and return accuracy."""
    from sklearn.metrics import accuracy_score, log_loss

    if is_nn:
        model_obj, scaler = model
        X_eval = scaler.transform(X_test)
        preds = model_obj.predict(X_eval)
        proba = model_obj.predict_proba(X_eval)
    else:
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, preds)

    try:
        ll = log_loss(y_test, proba)
        print(f"    {name:20s}: accuracy = {acc:.3f}  |  log_loss = {ll:.4f}")
    except Exception:
        print(f"    {name:20s}: accuracy = {acc:.3f}")

    return acc


# ---------------------------------------------------------------------------
# Train all model types for a single target
# ---------------------------------------------------------------------------

ALL_TRAINERS = [
    ("xgb",        train_xgboost,       False),
    ("lgbm",       train_lightgbm,      False),
    ("catboost",   train_catboost,      False),
    ("rf",         train_random_forest, False),
    ("et",         train_extra_trees,   False),
    ("nn",         train_neural_network, True),
]


def train_and_save(X, y, label: str, out_dir: Path) -> dict:
    """Train all model types, evaluate, save, and return accuracy dict."""
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    n_classes = len(np.unique(y))
    print(f"\n{'='*60}")
    print(f"  TARGET: {label}  ({n_classes} classes, {len(X_train):,} train / {len(X_test):,} test)")
    print(f"{'='*60}")

    accuracies = {}

    for model_tag, trainer_fn, is_nn in ALL_TRAINERS:
        try:
            model = trainer_fn(X_train, y_train, label)
            if model is None:
                continue

            acc = evaluate(model, X_test, y_test, model_tag, is_nn=is_nn)
            accuracies[f"{label}_{model_tag}"] = round(acc, 4)

            # Save model
            path = out_dir / f"{label}_{model_tag}.joblib"
            joblib.dump(model, path)
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"    Saved -> {path}  ({size_mb:.1f} MB)")

        except Exception as e:
            print(f"    {model_tag} FAILED: {e}")

    return accuracies


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not MATCHES_CSV.exists():
        print(f"ERROR: {MATCHES_CSV} not found.")
        print("Run  py scripts/fetch_history.py  first.")
        sys.exit(1)

    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        print("ERROR: scikit-learn not installed.  pip install scikit-learn xgboost lightgbm joblib")
        sys.exit(1)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X, y_result, y_over15, y_over25, y_btts = build_dataset(df)

    print(f"\nDataset summary:")
    print(f"  Samples   : {len(X):,}")
    print(f"  Home wins : {(y_result == 2).sum():,}  ({(y_result == 2).mean():.1%})")
    print(f"  Draws     : {(y_result == 1).sum():,}  ({(y_result == 1).mean():.1%})")
    print(f"  Away wins : {(y_result == 0).sum():,}  ({(y_result == 0).mean():.1%})")
    print(f"  Over 1.5  : {y_over15.sum():,}  ({y_over15.mean():.1%})")
    print(f"  Over 2.5  : {y_over25.sum():,}  ({y_over25.mean():.1%})")
    print(f"  BTTS      : {y_btts.sum():,}  ({y_btts.mean():.1%})")

    all_accuracies = {}

    accs = train_and_save(X, y_result, "match_result", MODELS_DIR)
    all_accuracies.update(accs)

    accs = train_and_save(X, y_over15, "over_1_5", MODELS_DIR)
    all_accuracies.update(accs)

    accs = train_and_save(X, y_over25, "over_2_5", MODELS_DIR)
    all_accuracies.update(accs)

    accs = train_and_save(X, y_btts, "btts", MODELS_DIR)
    all_accuracies.update(accs)

    # -----------------------------------------------------------------------
    # Train statistical models (ELO + Dixon-Coles)
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  STATISTICAL MODELS")
    print(f"{'='*60}")

    matches_for_stat = []
    for _, r in df.iterrows():
        try:
            matches_for_stat.append({
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "home_goals": int(r["home_score"]),
                "away_goals": int(r["away_score"]),
                "date": str(r["date"]),
            })
        except (ValueError, KeyError):
            continue

    try:
        from ml.elo_model import EloRatingSystem
        elo = EloRatingSystem()
        elo.train(matches_for_stat)
        elo_path = MODELS_DIR / "elo_ratings.json"
        elo.save(str(elo_path))
        print(f"  ELO ratings saved ({len(elo.ratings)} teams)")
    except Exception as e:
        print(f"  ELO training failed: {e}")

    try:
        from ml.dixon_coles_model import DixonColesModel
        dc = DixonColesModel()
        dc.train(matches_for_stat[-2000:])
        dc_path = MODELS_DIR / "dixon_coles.json"
        dc.save(str(dc_path))
        print(f"  Dixon-Coles saved ({len(dc.teams)} teams)")
    except Exception as e:
        print(f"  Dixon-Coles training failed: {e}")

    # -----------------------------------------------------------------------
    # Save model weights (based on test accuracy — used by ensemble voting)
    # -----------------------------------------------------------------------
    weights_path = MODELS_DIR / "model_weights.json"
    with open(weights_path, "w") as f:
        json.dump(all_accuracies, f, indent=2)
    print(f"\n  Model weights saved -> {weights_path}")

    # -----------------------------------------------------------------------
    # Save meta.json — tells the prediction service what to load
    # -----------------------------------------------------------------------
    all_model_names = sorted(all_accuracies.keys())

    meta = {
        "feature_columns": FEATURE_COLUMNS,
        "form_window":     FORM_WINDOW,
        "h2h_window":      H2H_WINDOW,
        "trained_at":      datetime.utcnow().isoformat(),
        "n_samples":       len(X),
        "models":          all_model_names,
        "model_types":     ["xgb", "lgbm", "catboost", "rf", "et", "nn"],
        "targets":         ["match_result", "over_1_5", "over_2_5", "btts"],
        "result_classes":  {0: "Away Win", 1: "Draw", 2: "Home Win"},
    }
    meta_path = MODELS_DIR / "meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    # -----------------------------------------------------------------------
    # Print final summary
    # -----------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("  FINAL ACCURACY SUMMARY")
    print(f"{'='*60}")

    targets = ["match_result", "over_1_5", "over_2_5", "btts"]
    model_types = ["xgb", "lgbm", "catboost", "rf", "et", "nn"]

    # Header
    header = f"  {'Target':20s}" + "".join(f"{mt:>10s}" for mt in model_types)
    print(header)
    print("  " + "-" * (20 + 10 * len(model_types)))

    for target in targets:
        row = f"  {target:20s}"
        for mt in model_types:
            key = f"{target}_{mt}"
            if key in all_accuracies:
                row += f"{all_accuracies[key]:>10.3f}"
            else:
                row += f"{'---':>10s}"
        print(row)

    # Best per target
    print()
    for target in targets:
        target_accs = {k: v for k, v in all_accuracies.items() if k.startswith(target)}
        if target_accs:
            best = max(target_accs, key=target_accs.get)
            print(f"  Best {target}: {best} ({target_accs[best]:.3f})")

    total_size = sum(
        f.stat().st_size for f in MODELS_DIR.glob("*.joblib")
    ) / (1024 * 1024)

    print(f"\n  Total models: {len(all_model_names)}")
    print(f"  Total size:   {total_size:.1f} MB")
    print(f"  Saved to:     {MODELS_DIR}/")
    print(f"\n  Restart the server -- predictions will use ALL models automatically.")


if __name__ == "__main__":
    main()
