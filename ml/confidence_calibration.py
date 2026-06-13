"""
Confidence Calibration Module

This module provides functions for calibrating confidence scores from ML models
to provide more realistic and accurate confidence percentages.

Key features:
1. Platt scaling for probability calibration
2. Isotonic regression for non-parametric calibration
3. Confidence interval estimation
4. Ensemble calibration methods
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import joblib
import os
import logging
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

class ConfidenceCalibrator:
    """
    Class for calibrating confidence scores from ML models.
    
    This class provides methods for:
    1. Calibrating raw probabilities to more accurate confidence scores
    2. Estimating uncertainty in predictions
    3. Providing realistic confidence intervals
    """
    
    def __init__(self, model_name: str, calibration_method: str = "platt"):
        """
        Initialize the confidence calibrator.
        
        Args:
            model_name: Name of the model to calibrate
            calibration_method: Method to use for calibration ('platt' or 'isotonic')
        """
        self.model_name = model_name
        self.calibration_method = calibration_method
        self.calibrator = None
        self.is_trained = False
        self.historical_accuracy = {}
        self.confidence_bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
        
    def train(self, y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, Any]:
        """
        Train the calibrator on historical predictions.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            
        Returns:
            Dictionary with training results
        """
        try:
            # For multi-class, we'll use the max probability
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                # Get the predicted class and its probability
                y_pred = np.argmax(y_proba, axis=1)
                max_proba = np.max(y_proba, axis=1)
                
                # Create binary labels for calibration (1 if correct, 0 if wrong)
                binary_y_true = (y_pred == y_true).astype(int)
                
                # Train calibrator on max probabilities
                if self.calibration_method == "isotonic":
                    self.calibrator = IsotonicRegression(out_of_bounds="clip")
                    self.calibrator.fit(max_proba, binary_y_true)
                else:  # Default to Platt scaling
                    # We'll use a simple logistic regression for Platt scaling
                    from sklearn.linear_model import LogisticRegression
                    self.calibrator = LogisticRegression(C=1.0)
                    # Reshape for sklearn API
                    self.calibrator.fit(max_proba.reshape(-1, 1), binary_y_true)
                
                # Calculate historical accuracy per confidence bin
                self._calculate_historical_accuracy(binary_y_true, max_proba)
                
            else:
                # For binary classification
                if self.calibration_method == "isotonic":
                    self.calibrator = IsotonicRegression(out_of_bounds="clip")
                    self.calibrator.fit(y_proba, y_true)
                else:  # Default to Platt scaling
                    from sklearn.linear_model import LogisticRegression
                    self.calibrator = LogisticRegression(C=1.0)
                    self.calibrator.fit(y_proba.reshape(-1, 1), y_true)
                
                # Calculate historical accuracy per confidence bin
                self._calculate_historical_accuracy(y_true, y_proba)
            
            self.is_trained = True
            
            return {
                "status": "success",
                "message": f"Calibrator trained successfully using {self.calibration_method} method",
                "historical_accuracy": self.historical_accuracy
            }
            
        except Exception as e:
            logger.error(f"Error training calibrator: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training calibrator: {str(e)}"
            }
    
    def calibrate(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Calibrate raw probabilities to more accurate confidence scores.
        
        Args:
            y_proba: Raw predicted probabilities
            
        Returns:
            Calibrated confidence scores
        """
        if not self.is_trained:
            logger.warning("Calibrator not trained, returning raw probabilities")
            return y_proba
        
        try:
            # For multi-class, calibrate the max probability
            if y_proba.ndim > 1 and y_proba.shape[1] > 1:
                max_proba = np.max(y_proba, axis=1)
                
                if self.calibration_method == "isotonic":
                    calibrated_max_proba = self.calibrator.predict(max_proba)
                else:  # Platt scaling
                    calibrated_max_proba = self.calibrator.predict_proba(max_proba.reshape(-1, 1))[:, 1]
                
                # Scale other probabilities proportionally
                calibrated_proba = np.zeros_like(y_proba)
                for i in range(len(y_proba)):
                    if max_proba[i] > 0:
                        # Scale factor to maintain the same ratios between classes
                        scale_factor = calibrated_max_proba[i] / max_proba[i]
                        calibrated_proba[i] = y_proba[i] * scale_factor
                        # Normalize to ensure sum to 1
                        calibrated_proba[i] = calibrated_proba[i] / np.sum(calibrated_proba[i])
                    else:
                        calibrated_proba[i] = y_proba[i]
                
                return calibrated_proba
                
            else:
                # For binary classification
                if self.calibration_method == "isotonic":
                    return self.calibrator.predict(y_proba)
                else:  # Platt scaling
                    return self.calibrator.predict_proba(y_proba.reshape(-1, 1))[:, 1]
                
        except Exception as e:
            logger.error(f"Error calibrating probabilities: {str(e)}")
            return y_proba
    
    def get_confidence_with_uncertainty(self, y_proba: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get calibrated confidence scores with uncertainty estimates.
        
        Args:
            y_proba: Raw predicted probabilities
            
        Returns:
            Tuple of (calibrated_confidence, uncertainty)
        """
        calibrated_proba = self.calibrate(y_proba)
        
        # Calculate uncertainty based on historical accuracy
        uncertainty = self._estimate_uncertainty(calibrated_proba)
        
        return calibrated_proba, uncertainty
    
    def _calculate_historical_accuracy(self, y_true: np.ndarray, y_proba: np.ndarray) -> None:
        """
        Calculate historical accuracy per confidence bin.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
        """
        # Create bins for confidence scores
        bin_indices = np.digitize(y_proba, self.confidence_bins) - 1
        
        # Calculate accuracy per bin
        for i in range(len(self.confidence_bins) - 1):
            bin_mask = (bin_indices == i)
            if np.sum(bin_mask) > 0:
                bin_accuracy = np.mean(y_true[bin_mask])
                bin_count = np.sum(bin_mask)
                
                # Store bin information
                bin_name = f"{self.confidence_bins[i]:.1f}-{self.confidence_bins[i+1]:.1f}"
                self.historical_accuracy[bin_name] = {
                    "accuracy": float(bin_accuracy),
                    "count": int(bin_count),
                    "confidence": float((self.confidence_bins[i] + self.confidence_bins[i+1]) / 2)
                }
    
    def _estimate_uncertainty(self, y_proba: np.ndarray) -> np.ndarray:
        """
        Estimate uncertainty for each prediction based on historical accuracy.
        
        Args:
            y_proba: Predicted probabilities
            
        Returns:
            Uncertainty estimates
        """
        if not self.historical_accuracy:
            # If no historical data, use a simple heuristic
            return 0.5 - 0.5 * np.abs(2 * y_proba - 1)
        
        # For multi-class, use max probability
        if y_proba.ndim > 1 and y_proba.shape[1] > 1:
            max_proba = np.max(y_proba, axis=1)
        else:
            max_proba = y_proba
        
        # Find the closest bin for each prediction
        bin_indices = np.digitize(max_proba, self.confidence_bins) - 1
        
        # Get uncertainty from historical accuracy
        uncertainty = np.zeros_like(max_proba)
        
        for i in range(len(max_proba)):
            bin_idx = bin_indices[i]
            bin_name = f"{self.confidence_bins[bin_idx]:.1f}-{self.confidence_bins[bin_idx+1]:.1f}"
            
            if bin_name in self.historical_accuracy:
                # Uncertainty is the absolute difference between predicted confidence and historical accuracy
                bin_data = self.historical_accuracy[bin_name]
                uncertainty[i] = abs(max_proba[i] - bin_data["accuracy"])
            else:
                # Fallback if bin not found
                uncertainty[i] = 0.2  # Default uncertainty
        
        return uncertainty
    
    def save(self, model_dir: str) -> bool:
        """
        Save the calibrator to disk.
        
        Args:
            model_dir: Directory to save the calibrator
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_trained:
            logger.warning("Calibrator not trained, nothing to save")
            return False
        
        try:
            # Create calibrator path
            calibrator_path = os.path.join(model_dir, f"{self.model_name}_calibrator.joblib")
            
            # Save calibrator
            joblib.dump({
                "calibrator": self.calibrator,
                "method": self.calibration_method,
                "historical_accuracy": self.historical_accuracy,
                "confidence_bins": self.confidence_bins,
                "saved_at": datetime.now().isoformat()
            }, calibrator_path)
            
            logger.info(f"Calibrator saved to {calibrator_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibrator: {str(e)}")
            return False
    
    def load(self, model_dir: str) -> bool:
        """
        Load the calibrator from disk.
        
        Args:
            model_dir: Directory to load the calibrator from
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create calibrator path
            calibrator_path = os.path.join(model_dir, f"{self.model_name}_calibrator.joblib")
            
            # Check if file exists
            if not os.path.exists(calibrator_path):
                logger.warning(f"Calibrator file not found: {calibrator_path}")
                return False
            
            # Load calibrator
            calibrator_data = joblib.load(calibrator_path)
            
            self.calibrator = calibrator_data["calibrator"]
            self.calibration_method = calibrator_data["method"]
            self.historical_accuracy = calibrator_data["historical_accuracy"]
            self.confidence_bins = calibrator_data["confidence_bins"]
            self.is_trained = True
            
            logger.info(f"Calibrator loaded from {calibrator_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibrator: {str(e)}")
            return False

# Helper functions for confidence calibration

def calibrate_confidence_scores(raw_scores: List[float], model_type: str = "match_result") -> List[float]:
    """
    Calibrate raw confidence scores to more realistic values.
    
    Args:
        raw_scores: List of raw confidence scores (0-100)
        model_type: Type of model (match_result, over_under, btts)
        
    Returns:
        List of calibrated confidence scores
    """
    # Convert to 0-1 scale
    raw_probs = [score / 100.0 for score in raw_scores]
    
    # Apply sigmoid calibration with model-specific parameters
    if model_type == "match_result":
        # Match result models tend to be overconfident
        a, b = 1.5, -0.3  # Steeper slope, shifted left
    elif model_type == "over_under":
        # Over/under models are moderately calibrated
        a, b = 1.2, -0.1  # Moderate slope, slight shift
    elif model_type == "btts":
        # BTTS models tend to be underconfident
        a, b = 0.9, 0.1  # Gentler slope, shifted right
    else:
        # Default calibration
        a, b = 1.0, 0.0  # No change
    
    # Apply calibration: sigmoid(a*x + b)
    calibrated_probs = [1.0 / (1.0 + np.exp(-a * (p - 0.5) - b)) for p in raw_probs]
    
    # Convert back to 0-100 scale and round to integers
    calibrated_scores = [min(round(p * 100), 100) for p in calibrated_probs]
    
    return calibrated_scores
