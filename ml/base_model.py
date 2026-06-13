"""
Base Model Module

This module provides a base class for all machine learning models.
It defines common functionality for model training, prediction, and evaluation.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split

from utils.common import setup_logging, ensure_directory_exists
from utils.config import settings

# Set up logging
logger = setup_logging(__name__)

# Import confidence calibrator
try:
    from ml.confidence_calibrator import IsotonicCalibrator as ConfidenceCalibrator
except ImportError:
    logger.warning("Advanced confidence calibrator not available, using fallback")
    try:
        from ml.confidence_calibration import ConfidenceCalibrator
    except ImportError:
        logger.error("No confidence calibrator available")

        # Define a minimal fallback calibrator
        class ConfidenceCalibrator:
            def __init__(self, model_name=None):
                self.is_trained = False
                self.model_name = model_name

            def train(self, y_true, y_proba):
                self.is_trained = True
                return {"status": "success", "message": "Fallback calibrator doesn't actually calibrate"}

            def calibrate(self, y_proba):
                return y_proba

            def save(self, model_dir):
                return True

            def load(self, model_dir):
                return True



class BaseModel:
    """
    Base class for all machine learning models.

    Features:
    - Model training and evaluation
    - Model saving and loading
    - Prediction with confidence scores
    - Model versioning
    - Confidence calibration
    """

    def __init__(self, model_name: str):
        """
        Initialize the base model.

        Args:
            model_name: Name of the model
        """
        self.model_name = model_name
        self.model_dir = settings.ml.MODEL_DIR
        ensure_directory_exists(self.model_dir)

        self.model = None
        self.feature_scaler = None
        self.feature_names = None
        self.confidence_calibrator = ConfidenceCalibrator(model_name)
        self.model_info = {
            "name": model_name,
            "version": "0.1.0",
            "created_at": None,
            "updated_at": None,
            "metrics": {},
            "parameters": {}
        }

    def _get_model_path(self) -> str:
        """
        Get the path to the model file.

        Returns:
            Path to the model file
        """
        return os.path.join(self.model_dir, f"{self.model_name}.joblib")

    def _get_info_path(self) -> str:
        """
        Get the path to the model info file.

        Returns:
            Path to the model info file
        """
        return os.path.join(self.model_dir, f"{self.model_name}_info.joblib")

    def save(self) -> bool:
        """
        Save the model and its metadata.

        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model is None:
                logger.warning(f"Cannot save {self.model_name}: model not trained")
                return False

            # Update model info
            self.model_info["updated_at"] = datetime.now().isoformat()

            # Save model
            model_path = self._get_model_path()
            joblib.dump(self.model, model_path)
            logger.info(f"Model saved to {model_path}")

            # Save feature scaler if available
            if self.feature_scaler is not None:
                scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.joblib")
                joblib.dump(self.feature_scaler, scaler_path)
                logger.info(f"Feature scaler saved to {scaler_path}")

            # Save feature names if available
            if self.feature_names is not None:
                names_path = os.path.join(self.model_dir, f"{self.model_name}_feature_names.joblib")
                joblib.dump(self.feature_names, names_path)
                logger.info(f"Feature names saved to {names_path}")

            # Save confidence calibrator if trained
            self.confidence_calibrator.save(self.model_dir)

            # Save model info
            info_path = self._get_info_path()
            joblib.dump(self.model_info, info_path)
            logger.info(f"Model info saved to {info_path}")

            return True

        except Exception as e:
            logger.error(f"Error saving model {self.model_name}: {str(e)}")
            return False

    def load(self) -> bool:
        """
        Load the model and its metadata.

        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = self._get_model_path()

            if not os.path.exists(model_path):
                logger.warning(f"Model file not found: {model_path}")
                return False

            # Load model
            self.model = joblib.load(model_path)
            logger.info(f"Model loaded from {model_path}")

            # Load feature scaler if available
            scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.joblib")
            if os.path.exists(scaler_path):
                self.feature_scaler = joblib.load(scaler_path)
                logger.info(f"Feature scaler loaded from {scaler_path}")

            # Load feature names if available
            names_path = os.path.join(self.model_dir, f"{self.model_name}_feature_names.joblib")
            if os.path.exists(names_path):
                self.feature_names = joblib.load(names_path)
                logger.info(f"Feature names loaded from {names_path}")

            # Load confidence calibrator if available
            self.confidence_calibrator.load(self.model_dir)

            # Load model info
            info_path = self._get_info_path()
            if os.path.exists(info_path):
                self.model_info = joblib.load(info_path)
                logger.info(f"Model info loaded from {info_path}")

            return True

        except Exception as e:
            logger.error(f"Error loading model {self.model_name}: {str(e)}")
            return False

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42) -> Dict[str, Any]:
        """
        Train the model.

        Args:
            X: Features
            y: Target variable
            test_size: Proportion of data to use for testing
            random_state: Random state for reproducibility

        Returns:
            Dictionary with training results
        """
        raise NotImplementedError("Subclasses must implement train method")

    def predict(self, X: Union[pd.DataFrame, np.ndarray, List]) -> Dict[str, Any]:
        """
        Make predictions.

        Args:
            X: Features

        Returns:
            Dictionary with predictions
        """
        raise NotImplementedError("Subclasses must implement predict method")

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Evaluate the model and train the confidence calibrator.

        Args:
            X: Features
            y: Target variable

        Returns:
            Dictionary with evaluation metrics
        """
        if self.model is None:
            logger.warning(f"Cannot evaluate {self.model_name}: model not trained")
            return {"status": "error", "message": "Model not trained"}

        try:
            # Scale features if scaler is available
            if self.feature_scaler is not None:
                X_scaled = self.feature_scaler.transform(X)
            else:
                X_scaled = X

            # Make predictions
            y_pred = self.model.predict(X_scaled)

            # Get probability predictions for calibration
            if hasattr(self.model, 'predict_proba'):
                y_proba = self.model.predict_proba(X_scaled)

                # Train the confidence calibrator
                calibrator_result = self.confidence_calibrator.train(y, y_proba)
                logger.info(f"Confidence calibrator trained: {calibrator_result.get('message', 'No message')}")

            # Calculate metrics
            accuracy = accuracy_score(y, y_pred)
            precision = precision_score(y, y_pred, average='weighted')
            recall = recall_score(y, y_pred, average='weighted')
            f1 = f1_score(y, y_pred, average='weighted')

            # Update model info
            self.model_info["metrics"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "evaluated_at": datetime.now().isoformat()
            }

            # Save updated model info
            info_path = self._get_info_path()
            joblib.dump(self.model_info, info_path)

            return {
                "status": "success",
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                },
                "classification_report": classification_report(y, y_pred)
            }

        except Exception as e:
            logger.error(f"Error evaluating model {self.model_name}: {str(e)}")
            return {"status": "error", "message": str(e)}

    def calibrate_confidence(self, raw_probabilities: np.ndarray) -> np.ndarray:
        """
        Calibrate raw confidence scores to more realistic values.

        Args:
            raw_probabilities: Raw probability scores from the model

        Returns:
            Calibrated confidence scores
        """
        # Use the confidence calibrator if it's trained
        if self.confidence_calibrator.is_trained:
            return self.confidence_calibrator.calibrate(raw_probabilities)

        # Otherwise, apply a simple sigmoid calibration
        if raw_probabilities.ndim > 1:
            # For multi-class, calibrate the max probability
            max_probs = np.max(raw_probabilities, axis=1)
            calibrated_max = self._apply_sigmoid_calibration(max_probs)

            # Scale other probabilities proportionally
            calibrated_probs = np.zeros_like(raw_probabilities)
            for i in range(len(raw_probabilities)):
                if max_probs[i] > 0:
                    # Scale factor to maintain the same ratios between classes
                    scale_factor = calibrated_max[i] / max_probs[i]
                    calibrated_probs[i] = raw_probabilities[i] * scale_factor
                    # Normalize to ensure sum to 1
                    calibrated_probs[i] = calibrated_probs[i] / np.sum(calibrated_probs[i])
                else:
                    calibrated_probs[i] = raw_probabilities[i]

            return calibrated_probs
        else:
            # For binary classification
            return self._apply_sigmoid_calibration(raw_probabilities)

    def _apply_sigmoid_calibration(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Apply sigmoid calibration to probabilities.

        Args:
            probabilities: Raw probability scores

        Returns:
            Calibrated probabilities
        """
        # Parameters for sigmoid calibration
        a, b = 1.2, -0.1  # Moderate slope, slight shift

        # Apply sigmoid: 1 / (1 + exp(-a*(x-0.5) - b))
        return 1.0 / (1.0 + np.exp(-a * (probabilities - 0.5) - b))

    def _calculate_odds(self, confidence: float) -> float:
        """
        Calculate odds based on confidence.

        Args:
            confidence: Confidence score (0-1)

        Returns:
            Odds value
        """
        # Simple formula: odds = 1 / probability
        # Add a small margin for the bookmaker
        margin = 0.1
        probability = confidence - margin

        # Ensure probability is between 0.1 and 0.9
        probability = max(0.1, min(0.9, probability))

        odds = 1 / probability

        # Round to 2 decimal places
        return round(odds, 2)
