"""
CatBoost models for football prediction.
Handles categorical features (team names, leagues) natively without encoding.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from catboost import CatBoostClassifier, Pool
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    logger.warning("CatBoost not installed. Run: pip install catboost")


class CatBoostMatchResultModel:
    CAT_FEATURES = ["home_team", "away_team", "league", "home_form", "away_form"]

    def __init__(self, iterations=500, depth=6, learning_rate=0.05):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.model = None
        self.is_trained = False
        self.label_map = {0: "home_win", 1: "draw", 2: "away_win"}

    def _available_cat_features(self, df):
        return [c for c in self.CAT_FEATURES if c in df.columns]

    def train(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed")
        cat_cols = self._available_cat_features(X)
        pool = Pool(X, label=y, cat_features=cat_cols)
        self.model = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth,
            learning_rate=self.learning_rate, loss_function="MultiClass",
            eval_metric="Accuracy", random_seed=42, verbose=0, early_stopping_rounds=50,
        )
        self.model.fit(pool)
        self.is_trained = True
        logger.info("CatBoostMatchResultModel trained successfully")

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(Pool(X, cat_features=self._available_cat_features(X)))

    def predict(self, X):
        proba = self.predict_proba(X)
        row = proba[0]
        best_idx = int(np.argmax(row))
        return {
            "prediction": self.label_map[best_idx],
            "confidence": float(row[best_idx]),
            "probabilities": {"home_win": float(row[0]), "draw": float(row[1]), "away_win": float(row[2])},
            "model_name": "catboost_match_result",
            "model_type": "catboost",
        }

    def save(self, path):
        if self.model:
            self.model.save_model(path)

    def load(self, path):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed")
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self.is_trained = True


class CatBoostBTTSModel:
    CAT_FEATURES = ["home_team", "away_team", "league"]

    def __init__(self, iterations=300, depth=5, learning_rate=0.05):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.model = None
        self.is_trained = False

    def _available_cat_features(self, df):
        return [c for c in self.CAT_FEATURES if c in df.columns]

    def train(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed")
        cat_cols = self._available_cat_features(X)
        self.model = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=self.learning_rate,
            loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=0, early_stopping_rounds=40,
        )
        self.model.fit(Pool(X, label=y, cat_features=cat_cols))
        self.is_trained = True
        logger.info("CatBoostBTTSModel trained successfully")

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(Pool(X, cat_features=self._available_cat_features(X)))

    def predict(self, X):
        row = self.predict_proba(X)[0]
        btts = bool(row[1] >= 0.5)
        return {
            "prediction": "yes" if btts else "no",
            "confidence": float(row[1] if btts else row[0]),
            "probabilities": {"yes": float(row[1]), "no": float(row[0])},
            "model_name": "catboost_btts",
            "model_type": "catboost",
        }

    def save(self, path):
        if self.model:
            self.model.save_model(path)

    def load(self, path):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed")
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self.is_trained = True


class CatBoostOverUnderModel:
    CAT_FEATURES = ["home_team", "away_team", "league"]

    def __init__(self, threshold=2.5, iterations=300, depth=5):
        self.threshold = threshold
        self.iterations = iterations
        self.depth = depth
        self.model = None
        self.is_trained = False

    def _available_cat_features(self, df):
        return [c for c in self.CAT_FEATURES if c in df.columns]

    def train(self, X, y):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed")
        cat_cols = self._available_cat_features(X)
        self.model = CatBoostClassifier(
            iterations=self.iterations, depth=self.depth, learning_rate=0.05,
            loss_function="Logloss", eval_metric="AUC", random_seed=42, verbose=0, early_stopping_rounds=40,
        )
        self.model.fit(Pool(X, label=y, cat_features=cat_cols))
        self.is_trained = True
        logger.info(f"CatBoostOverUnderModel ({self.threshold}) trained successfully")

    def predict_proba(self, X):
        if not self.is_trained:
            raise RuntimeError("Model not trained")
        return self.model.predict_proba(Pool(X, cat_features=self._available_cat_features(X)))

    def predict(self, X):
        row = self.predict_proba(X)[0]
        over = bool(row[1] >= 0.5)
        return {
            "prediction": f"over_{self.threshold}" if over else f"under_{self.threshold}",
            "confidence": float(row[1] if over else row[0]),
            "probabilities": {f"over_{self.threshold}": float(row[1]), f"under_{self.threshold}": float(row[0])},
            "model_name": f"catboost_over_under_{self.threshold}",
            "model_type": "catboost",
        }

    def save(self, path):
        if self.model:
            self.model.save_model(path)

    def load(self, path):
        if not CATBOOST_AVAILABLE:
            raise RuntimeError("CatBoost not installed")
        self.model = CatBoostClassifier()
        self.model.load_model(path)
        self.is_trained = True
