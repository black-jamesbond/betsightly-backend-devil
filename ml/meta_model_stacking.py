"""
Meta-Model Stacking System for BetSightly

This module implements intelligent model stacking using a meta-classifier
to optimally blend predictions from XGBoost, LightGBM, Neural Networks, and LSTM models.
"""

import logging
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

from utils.config import settings
from utils.error_handling import ModelError

# Set up logging
logger = logging.getLogger(__name__)

class MetaModelStacker:
    """
    Intelligent meta-model stacking system.
    
    Instead of hard fallback, this system:
    1. Collects predictions from all available models
    2. Uses a meta-classifier to intelligently blend them
    3. Provides calibrated confidence scores
    4. Handles missing model predictions gracefully
    """
    
    def __init__(self, meta_model_type: str = "logistic"):
        """
        Initialize the meta-model stacker.
        
        Args:
            meta_model_type: Type of meta-model ('logistic' or 'neural')
        """
        self.meta_model_type = meta_model_type
        self.meta_models = {}  # {prediction_type: meta_model}
        self.base_models = {}  # {prediction_type: {model_name: model}}
        self.feature_scalers = {}  # For meta-model features
        self.model_weights = {}  # Individual model performance weights
        self.calibrators = {}  # Confidence calibrators
        
    def register_base_model(self, prediction_type: str, model_name: str, model_path: str):
        """
        Register a base model for stacking.
        
        Args:
            prediction_type: Type of prediction (match_result, over_under, btts)
            model_name: Name of the model (xgboost, lightgbm, neural_network, lstm)
            model_path: Path to the trained model
        """
        try:
            if prediction_type not in self.base_models:
                self.base_models[prediction_type] = {}
            
            # Load model
            model = joblib.load(model_path)
            self.base_models[prediction_type][model_name] = model
            
            logger.info(f"Registered {model_name} for {prediction_type} stacking")
            
        except Exception as e:
            logger.error(f"Failed to register model {model_name}: {str(e)}")
    
    def train_meta_model(self, prediction_type: str, X_train: pd.DataFrame, 
                        y_train: pd.Series, validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Train meta-model for a specific prediction type.
        
        Args:
            prediction_type: Type of prediction
            X_train: Training features
            y_train: Training targets
            validation_split: Validation split for meta-model training
            
        Returns:
            Training results
        """
        try:
            if prediction_type not in self.base_models:
                raise ModelError(f"No base models registered for {prediction_type}")
            
            base_models = self.base_models[prediction_type]
            
            if len(base_models) < 2:
                logger.warning(f"Only {len(base_models)} base models for {prediction_type}. Meta-model needs at least 2.")
                return {"status": "insufficient_models", "count": len(base_models)}
            
            # Generate base model predictions for meta-training
            meta_features = self._generate_meta_features(prediction_type, X_train)
            
            if meta_features is None or len(meta_features) == 0:
                raise ModelError("Failed to generate meta-features")
            
            # Split data for meta-model training
            split_idx = int(len(meta_features) * (1 - validation_split))
            X_meta_train = meta_features[:split_idx]
            X_meta_val = meta_features[split_idx:]
            y_meta_train = y_train.iloc[:split_idx]
            y_meta_val = y_train.iloc[split_idx:]
            
            # Scale meta-features
            scaler = StandardScaler()
            X_meta_train_scaled = scaler.fit_transform(X_meta_train)
            X_meta_val_scaled = scaler.transform(X_meta_val)
            
            # Train meta-model
            if self.meta_model_type == "logistic":
                meta_model = LogisticRegression(
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                )
            else:  # neural
                meta_model = MLPClassifier(
                    hidden_layer_sizes=(50, 25),
                    random_state=42,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.1
                )
            
            # Train with cross-validation
            cv_scores = cross_val_score(
                meta_model, X_meta_train_scaled, y_meta_train,
                cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
                scoring='accuracy'
            )
            
            # Fit final model
            meta_model.fit(X_meta_train_scaled, y_meta_train)
            
            # Validate
            y_pred = meta_model.predict(X_meta_val_scaled)
            y_pred_proba = meta_model.predict_proba(X_meta_val_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_meta_val, y_pred)
            precision = precision_score(y_meta_val, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_meta_val, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_meta_val, y_pred, average='weighted', zero_division=0)
            
            # Train confidence calibrator
            calibrator = CalibratedClassifierCV(meta_model, method='isotonic', cv=3)
            calibrator.fit(X_meta_train_scaled, y_meta_train)
            
            # Store models and scalers
            self.meta_models[prediction_type] = meta_model
            self.feature_scalers[prediction_type] = scaler
            self.calibrators[prediction_type] = calibrator
            
            # Calculate individual model weights based on performance
            self._calculate_model_weights(prediction_type, X_meta_val, y_meta_val)
            
            # Save meta-model
            self._save_meta_model(prediction_type)
            
            results = {
                "status": "success",
                "prediction_type": prediction_type,
                "meta_model_type": self.meta_model_type,
                "base_models_count": len(base_models),
                "cv_accuracy_mean": float(np.mean(cv_scores)),
                "cv_accuracy_std": float(np.std(cv_scores)),
                "validation_accuracy": float(accuracy),
                "validation_precision": float(precision),
                "validation_recall": float(recall),
                "validation_f1": float(f1),
                "model_weights": self.model_weights.get(prediction_type, {})
            }
            
            logger.info(f"Meta-model trained for {prediction_type}: {accuracy:.4f} accuracy")
            return results
            
        except Exception as e:
            logger.error(f"Meta-model training failed for {prediction_type}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _generate_meta_features(self, prediction_type: str, X: pd.DataFrame) -> Optional[np.ndarray]:
        """Generate meta-features from base model predictions."""
        try:
            base_models = self.base_models[prediction_type]
            meta_features_list = []
            
            for model_name, model in base_models.items():
                try:
                    # Get predictions and probabilities
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X)
                        predictions = model.predict(X)
                        
                        # Add probabilities as features
                        if len(probabilities.shape) > 1:
                            for i in range(probabilities.shape[1]):
                                meta_features_list.append(probabilities[:, i])
                        else:
                            meta_features_list.append(probabilities)
                        
                        # Add prediction confidence (max probability)
                        if len(probabilities.shape) > 1:
                            confidence = np.max(probabilities, axis=1)
                        else:
                            confidence = np.abs(probabilities - 0.5) * 2
                        meta_features_list.append(confidence)
                        
                    else:
                        # Fallback for models without predict_proba
                        predictions = model.predict(X)
                        meta_features_list.append(predictions.astype(float))
                        
                except Exception as e:
                    logger.warning(f"Failed to get predictions from {model_name}: {str(e)}")
                    continue
            
            if not meta_features_list:
                return None
            
            # Combine all meta-features
            meta_features = np.column_stack(meta_features_list)
            return meta_features
            
        except Exception as e:
            logger.error(f"Meta-feature generation failed: {str(e)}")
            return None
    
    def _calculate_model_weights(self, prediction_type: str, X_val: np.ndarray, y_val: pd.Series):
        """Calculate individual model performance weights."""
        try:
            base_models = self.base_models[prediction_type]
            weights = {}
            
            for model_name, model in base_models.items():
                try:
                    # Get individual model predictions
                    y_pred = model.predict(X_val)
                    accuracy = accuracy_score(y_val, y_pred)
                    weights[model_name] = float(accuracy)
                    
                except Exception as e:
                    logger.warning(f"Failed to calculate weight for {model_name}: {str(e)}")
                    weights[model_name] = 0.0
            
            # Normalize weights
            total_weight = sum(weights.values())
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            
            self.model_weights[prediction_type] = weights
            
        except Exception as e:
            logger.error(f"Weight calculation failed: {str(e)}")
    
    def predict_with_stacking(self, prediction_type: str, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions using meta-model stacking.
        
        Args:
            prediction_type: Type of prediction
            X: Input features
            
        Returns:
            Stacked prediction results
        """
        try:
            if prediction_type not in self.meta_models:
                raise ModelError(f"No meta-model trained for {prediction_type}")
            
            # Generate meta-features
            meta_features = self._generate_meta_features(prediction_type, X)
            
            if meta_features is None:
                raise ModelError("Failed to generate meta-features for prediction")
            
            # Scale meta-features
            scaler = self.feature_scalers[prediction_type]
            meta_features_scaled = scaler.transform(meta_features)
            
            # Get meta-model prediction
            meta_model = self.meta_models[prediction_type]
            predictions = meta_model.predict(meta_features_scaled)
            probabilities = meta_model.predict_proba(meta_features_scaled)
            
            # Get calibrated confidence scores
            calibrator = self.calibrators[prediction_type]
            calibrated_probabilities = calibrator.predict_proba(meta_features_scaled)
            
            # Calculate confidence scores
            confidence_scores = np.max(calibrated_probabilities, axis=1) * 100
            
            # Get individual model contributions
            model_contributions = self._get_model_contributions(prediction_type, X)
            
            return {
                "status": "success",
                "predictions": predictions.tolist(),
                "probabilities": probabilities.tolist(),
                "calibrated_probabilities": calibrated_probabilities.tolist(),
                "confidence_scores": confidence_scores.tolist(),
                "model_contributions": model_contributions,
                "meta_model_type": self.meta_model_type,
                "base_models_used": list(self.base_models[prediction_type].keys())
            }
            
        except Exception as e:
            logger.error(f"Stacked prediction failed for {prediction_type}: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def _get_model_contributions(self, prediction_type: str, X: pd.DataFrame) -> Dict[str, Any]:
        """Get individual model contributions to the final prediction."""
        try:
            base_models = self.base_models[prediction_type]
            weights = self.model_weights.get(prediction_type, {})
            contributions = {}
            
            for model_name, model in base_models.items():
                try:
                    predictions = model.predict(X)
                    weight = weights.get(model_name, 0.0)
                    
                    contributions[model_name] = {
                        "predictions": predictions.tolist() if hasattr(predictions, 'tolist') else [predictions],
                        "weight": float(weight),
                        "contribution_percentage": float(weight * 100)
                    }
                    
                except Exception as e:
                    logger.warning(f"Failed to get contribution from {model_name}: {str(e)}")
                    contributions[model_name] = {
                        "predictions": [],
                        "weight": 0.0,
                        "contribution_percentage": 0.0
                    }
            
            return contributions
            
        except Exception as e:
            logger.error(f"Model contribution calculation failed: {str(e)}")
            return {}
    
    def _save_meta_model(self, prediction_type: str):
        """Save meta-model and associated components."""
        try:
            model_dir = Path(settings.ml.MODEL_DIR) / "meta_models"
            model_dir.mkdir(exist_ok=True)
            
            # Save meta-model
            meta_model_path = model_dir / f"{prediction_type}_meta_model.joblib"
            joblib.dump(self.meta_models[prediction_type], meta_model_path)
            
            # Save scaler
            scaler_path = model_dir / f"{prediction_type}_meta_scaler.joblib"
            joblib.dump(self.feature_scalers[prediction_type], scaler_path)
            
            # Save calibrator
            calibrator_path = model_dir / f"{prediction_type}_calibrator.joblib"
            joblib.dump(self.calibrators[prediction_type], calibrator_path)
            
            # Save weights
            weights_path = model_dir / f"{prediction_type}_weights.joblib"
            joblib.dump(self.model_weights.get(prediction_type, {}), weights_path)
            
            logger.info(f"Meta-model saved for {prediction_type}")
            
        except Exception as e:
            logger.error(f"Failed to save meta-model: {str(e)}")
    
    def load_meta_model(self, prediction_type: str) -> bool:
        """Load trained meta-model and components."""
        try:
            model_dir = Path(settings.ml.MODEL_DIR) / "meta_models"
            
            # Load meta-model
            meta_model_path = model_dir / f"{prediction_type}_meta_model.joblib"
            if meta_model_path.exists():
                self.meta_models[prediction_type] = joblib.load(meta_model_path)
            else:
                return False
            
            # Load scaler
            scaler_path = model_dir / f"{prediction_type}_meta_scaler.joblib"
            if scaler_path.exists():
                self.feature_scalers[prediction_type] = joblib.load(scaler_path)
            
            # Load calibrator
            calibrator_path = model_dir / f"{prediction_type}_calibrator.joblib"
            if calibrator_path.exists():
                self.calibrators[prediction_type] = joblib.load(calibrator_path)
            
            # Load weights
            weights_path = model_dir / f"{prediction_type}_weights.joblib"
            if weights_path.exists():
                if prediction_type not in self.model_weights:
                    self.model_weights[prediction_type] = {}
                self.model_weights[prediction_type] = joblib.load(weights_path)
            
            logger.info(f"Meta-model loaded for {prediction_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load meta-model for {prediction_type}: {str(e)}")
            return False


# Global meta-model stacker instance
meta_stacker = MetaModelStacker(meta_model_type="logistic")
