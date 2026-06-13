"""
Improved Ensemble Model

This module contains the improved ensemble model for football match prediction.
It uses a combination of scikit-learn models to predict match results, over/under 2.5 goals,
and both teams to score.
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import joblib
from typing import Dict, List, Tuple, Any, Union

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ml.base_model import BaseModel
from utils.common import ensure_directory_exists
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class MatchResultModel(BaseModel):
    """
    Model for predicting match results (home win, draw, away win).
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__("match_result_ensemble")
        self.model = None
        self.feature_scaler = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Dictionary with training results
        """
        try:
            # Save feature names
            self.feature_names = X.columns.tolist()

            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Create ensemble model
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model
            self.save()

            return {
                "status": "success",
                "accuracy": accuracy,
                "message": f"Match result model trained successfully with accuracy: {accuracy:.4f}"
            }

        except Exception as e:
            logger.error(f"Error training match result model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training match result model: {str(e)}"
            }

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions.

        Args:
            features: Features DataFrame

        Returns:
            Dictionary with predictions
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()

            # Check if model is loaded
            if self.model is None:
                return {
                    "status": "error",
                    "message": "Match result model not loaded"
                }

            # Ensure features have the correct columns
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0.0

            # Select and order features
            X = features[self.feature_names]

            # Scale features
            X_scaled = self.feature_scaler.transform(X)

            # Make predictions
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)

            # Calibrate confidence scores
            calibrated_proba = self.calibrate_confidence(y_proba)

            # Map predictions to labels
            labels = ["H", "D", "A"]
            predictions = [labels[p] for p in y_pred]

            # Create confidence scores with calibrated probabilities
            confidence = [float(p.max()) * 100 for p in calibrated_proba]

            # Add uncertainty estimates
            if hasattr(self.confidence_calibrator, 'get_confidence_with_uncertainty'):
                _, uncertainty = self.confidence_calibrator.get_confidence_with_uncertainty(y_proba)
                uncertainty_percent = [float(u) * 100 for u in uncertainty]
            else:
                # Default uncertainty estimate
                uncertainty_percent = [min(100 - conf, 20) for conf in confidence]

            return {
                "status": "success",
                "predictions": predictions,
                "confidence": confidence,
                "uncertainty": uncertainty_percent,
                "probabilities": calibrated_proba.tolist()
            }

        except Exception as e:
            logger.error(f"Error predicting match results: {str(e)}")
            return {
                "status": "error",
                "message": f"Error predicting match results: {str(e)}"
            }

class OverUnderModel(BaseModel):
    """
    Model for predicting over/under 2.5 goals.
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__("over_under_ensemble")
        self.model = None
        self.feature_scaler = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Dictionary with training results
        """
        try:
            # Save feature names
            self.feature_names = X.columns.tolist()

            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Create ensemble model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model
            self._save_model()

            return {
                "status": "success",
                "accuracy": accuracy,
                "message": f"Over/under model trained successfully with accuracy: {accuracy:.4f}"
            }

        except Exception as e:
            logger.error(f"Error training over/under model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training over/under model: {str(e)}"
            }

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions.

        Args:
            features: Features DataFrame

        Returns:
            Dictionary with predictions
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()

            # Check if model is loaded
            if self.model is None:
                return {
                    "status": "error",
                    "message": "Over/under model not loaded"
                }

            # Ensure features have the correct columns
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0.0

            # Select and order features
            X = features[self.feature_names]

            # Scale features
            X_scaled = self.feature_scaler.transform(X)

            # Make predictions
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)

            # Calibrate confidence scores
            calibrated_proba = self.calibrate_confidence(y_proba)

            # Create confidence scores with calibrated probabilities
            confidence = [float(p.max()) * 100 for p in calibrated_proba]

            # Add uncertainty estimates
            if hasattr(self.confidence_calibrator, 'get_confidence_with_uncertainty'):
                _, uncertainty = self.confidence_calibrator.get_confidence_with_uncertainty(y_proba)
                uncertainty_percent = [float(u) * 100 for u in uncertainty]
            else:
                # Default uncertainty estimate
                uncertainty_percent = [min(100 - conf, 20) for conf in confidence]

            return {
                "status": "success",
                "predictions": y_pred.tolist(),
                "confidence": confidence,
                "uncertainty": uncertainty_percent,
                "probabilities": calibrated_proba.tolist()
            }

        except Exception as e:
            logger.error(f"Error predicting over/under: {str(e)}")
            return {
                "status": "error",
                "message": f"Error predicting over/under: {str(e)}"
            }

class BTTSModel(BaseModel):
    """
    Model for predicting both teams to score.
    """

    def __init__(self):
        """Initialize the model."""
        super().__init__("btts_ensemble")
        self.model = None
        self.feature_scaler = None
        self.feature_names = None

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X: Features DataFrame
            y: Target Series

        Returns:
            Dictionary with training results
        """
        try:
            # Save feature names
            self.feature_names = X.columns.tolist()

            # Scale features
            self.feature_scaler = StandardScaler()
            X_scaled = self.feature_scaler.fit_transform(X)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42
            )

            # Create ensemble model
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                n_jobs=-1
            )

            # Train model
            self.model.fit(X_train, y_train)

            # Evaluate model
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Save model
            self._save_model()

            return {
                "status": "success",
                "accuracy": accuracy,
                "message": f"BTTS model trained successfully with accuracy: {accuracy:.4f}"
            }

        except Exception as e:
            logger.error(f"Error training BTTS model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training BTTS model: {str(e)}"
            }

    def predict(self, features: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions.

        Args:
            features: Features DataFrame

        Returns:
            Dictionary with predictions
        """
        try:
            # Load model if not loaded
            if self.model is None:
                self._load_model()

            # Check if model is loaded
            if self.model is None:
                return {
                    "status": "error",
                    "message": "BTTS model not loaded"
                }

            # Ensure features have the correct columns
            for col in self.feature_names:
                if col not in features.columns:
                    features[col] = 0.0

            # Select and order features
            X = features[self.feature_names]

            # Scale features
            X_scaled = self.feature_scaler.transform(X)

            # Make predictions
            y_pred = self.model.predict(X_scaled)
            y_proba = self.model.predict_proba(X_scaled)

            # Calibrate confidence scores
            calibrated_proba = self.calibrate_confidence(y_proba)

            # Create confidence scores with calibrated probabilities
            confidence = [float(p.max()) * 100 for p in calibrated_proba]

            # Add uncertainty estimates
            if hasattr(self.confidence_calibrator, 'get_confidence_with_uncertainty'):
                _, uncertainty = self.confidence_calibrator.get_confidence_with_uncertainty(y_proba)
                uncertainty_percent = [float(u) * 100 for u in uncertainty]
            else:
                # Default uncertainty estimate
                uncertainty_percent = [min(100 - conf, 20) for conf in confidence]

            return {
                "status": "success",
                "predictions": y_pred.tolist(),
                "confidence": confidence,
                "uncertainty": uncertainty_percent,
                "probabilities": calibrated_proba.tolist()
            }

        except Exception as e:
            logger.error(f"Error predicting BTTS: {str(e)}")
            return {
                "status": "error",
                "message": f"Error predicting BTTS: {str(e)}"
            }

class ImprovedEnsembleModel:
    """
    Improved ensemble model for football match prediction.
    """

    def __init__(self):
        """Initialize the model."""
        self.match_result_model = MatchResultModel()
        self.over_under_model = OverUnderModel()
        self.btts_model = BTTSModel()

    def train(self, features_df: pd.DataFrame, results_df: pd.DataFrame, historical_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train all models.

        Args:
            features_df: Features DataFrame
            results_df: Results DataFrame
            historical_df: Historical data DataFrame

        Returns:
            Dictionary with training results
        """
        try:
            # Preprocess data
            X, y_match_result, y_over_under, y_btts = self._preprocess_data(
                pd.merge(features_df, results_df, on="match_id")
            )

            # Train match result model
            match_result_result = self.match_result_model.train(X, y_match_result)

            # Train over/under model
            over_under_result = self.over_under_model.train(X, y_over_under)

            # Train BTTS model
            btts_result = self.btts_model.train(X, y_btts)

            return {
                "status": "success",
                "match_result_accuracy": match_result_result.get("accuracy", 0),
                "over_under_accuracy": over_under_result.get("accuracy", 0),
                "btts_accuracy": btts_result.get("accuracy", 0),
                "message": "All models trained successfully"
            }

        except Exception as e:
            logger.error(f"Error training ensemble model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training ensemble model: {str(e)}"
            }

    def predict(self, fixture: Dict[str, Any], historical_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions for a fixture.

        Args:
            fixture: Fixture data
            historical_df: Historical data DataFrame

        Returns:
            Dictionary with predictions
        """
        try:
            # Extract fixture data
            home_team = fixture["teams"]["home"]["name"]
            away_team = fixture["teams"]["away"]["name"]

            # Create features
            features = pd.DataFrame({
                "home_team": [home_team],
                "away_team": [away_team],
                "home_form": [0.5],
                "away_form": [0.5],
                "home_attack": [0.5],
                "away_attack": [0.5],
                "home_defense": [0.5],
                "away_defense": [0.5],
                "h2h_home_wins": [0],
                "h2h_away_wins": [0],
                "h2h_draws": [0],
                "league_position_diff": [0]
            })

            # Make predictions
            match_result_pred = self.match_result_model.predict(features)
            over_under_pred = self.over_under_model.predict(features)
            btts_pred = self.btts_model.predict(features)

            # Check if all predictions were successful
            if (match_result_pred.get("status") != "success" or
                over_under_pred.get("status") != "success" or
                btts_pred.get("status") != "success"):
                return {
                    "status": "error",
                    "message": "Error making predictions"
                }

            # Create predictions
            predictions = []

            # Add match result prediction
            predictions.append({
                "prediction_type": "Match Result",
                "prediction": match_result_pred["predictions"][0],
                "confidence": match_result_pred["confidence"][0],
                "uncertainty": match_result_pred.get("uncertainty", [10])[0],
                "explanation": self._get_match_result_explanation(
                    match_result_pred["predictions"][0],
                    match_result_pred["confidence"][0],
                    home_team,
                    away_team
                )
            })

            # Add over/under prediction
            predictions.append({
                "prediction_type": "Over/Under 2.5",
                "prediction": "Over" if over_under_pred["predictions"][0] == 1 else "Under",
                "confidence": over_under_pred["confidence"][0],
                "uncertainty": over_under_pred.get("uncertainty", [10])[0],
                "explanation": self._get_over_under_explanation(
                    over_under_pred["predictions"][0],
                    over_under_pred["confidence"][0],
                    home_team,
                    away_team
                )
            })

            # Add BTTS prediction
            predictions.append({
                "prediction_type": "Both Teams To Score",
                "prediction": "Yes" if btts_pred["predictions"][0] == 1 else "No",
                "confidence": btts_pred["confidence"][0],
                "uncertainty": btts_pred.get("uncertainty", [10])[0],
                "explanation": self._get_btts_explanation(
                    btts_pred["predictions"][0],
                    btts_pred["confidence"][0],
                    home_team,
                    away_team
                )
            })

            return {
                "status": "success",
                "predictions": predictions
            }

        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            return {
                "status": "error",
                "message": f"Error making predictions: {str(e)}"
            }

    def _preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Preprocess data for training.

        Args:
            data: DataFrame with features and results

        Returns:
            Tuple of (X, y_match_result, y_over_under, y_btts)
        """
        # Create a copy to avoid modifying the original
        df = data.copy()

        # Convert categorical variables to numeric
        if "match_result" in df.columns:
            df["match_result_num"] = df["match_result"].map({"H": 0, "D": 1, "A": 2})

        # Select features
        feature_cols = [
            "home_form", "away_form", "home_attack", "away_attack",
            "home_defense", "away_defense", "h2h_home_wins", "h2h_away_wins",
            "h2h_draws", "league_position_diff"
        ]

        # Add any missing columns
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.5

        # Create feature matrix
        X = df[feature_cols]

        # Create target variables
        y_match_result = df["match_result_num"] if "match_result_num" in df.columns else pd.Series([0] * len(df))
        y_over_under = df["over_2_5"] if "over_2_5" in df.columns else pd.Series([0] * len(df))
        y_btts = df["btts"] if "btts" in df.columns else pd.Series([0] * len(df))

        return X, y_match_result, y_over_under, y_btts

    def _get_match_result_explanation(self, prediction: str, confidence: float, home_team: str, away_team: str) -> str:
        """
        Get explanation for match result prediction.

        Args:
            prediction: Prediction (H, D, A)
            confidence: Confidence score
            home_team: Home team name
            away_team: Away team name

        Returns:
            Explanation string
        """
        if prediction == "H":
            return f"{home_team} win predicted with {confidence:.1f}% confidence"
        elif prediction == "A":
            return f"{away_team} win predicted with {confidence:.1f}% confidence"
        else:
            return f"Draw predicted with {confidence:.1f}% confidence"

    def _get_over_under_explanation(self, prediction: int, confidence: float, home_team: str, away_team: str) -> str:
        """
        Get explanation for over/under prediction.

        Args:
            prediction: Prediction (1 for over, 0 for under)
            confidence: Confidence score
            home_team: Home team name
            away_team: Away team name

        Returns:
            Explanation string
        """
        if prediction == 1:
            return f"Over 2.5 goals predicted with {confidence:.1f}% confidence"
        else:
            return f"Under 2.5 goals predicted with {confidence:.1f}% confidence"

    def _get_btts_explanation(self, prediction: int, confidence: float, home_team: str, away_team: str) -> str:
        """
        Get explanation for BTTS prediction.

        Args:
            prediction: Prediction (1 for yes, 0 for no)
            confidence: Confidence score
            home_team: Home team name
            away_team: Away team name

        Returns:
            Explanation string
        """
        if prediction == 1:
            return f"Both teams to score predicted with {confidence:.1f}% confidence"
        else:
            return f"Both teams not to score predicted with {confidence:.1f}% confidence"

# Create singleton instance
improved_ensemble_model = ImprovedEnsembleModel()
