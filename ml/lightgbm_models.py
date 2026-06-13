"""
LightGBM Models for Football Prediction

This module contains LightGBM-based models for various football prediction tasks.
LightGBM is a gradient boosting framework that uses tree-based learning algorithms.
It is designed to be distributed and efficient with faster training speed and higher efficiency.
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Dict, List, Any, Tuple
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml.base_model import BaseModel
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class LightGBMBTTSModel(BaseModel):
    """
    LightGBM model for predicting both teams to score (BTTS).
    """
    
    def __init__(self):
        """Initialize the model."""
        super().__init__("lightgbm_btts")
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
            
            # Define parameter grid for optimization
            param_grid = {
                'num_leaves': [31, 50, 100],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            
            # Create base model
            base_model = lgb.LGBMClassifier(
                objective='binary',
                metric='binary_logloss',
                random_state=42,
                verbose=-1
            )
            
            # Use grid search to find best parameters
            grid_search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                cv=3,
                scoring='roc_auc',
                verbose=1,
                n_jobs=-1
            )
            
            # Train model with best parameters
            grid_search.fit(X_train, y_train)
            self.model = grid_search.best_estimator_
            
            # Evaluate model
            y_pred = self.model.predict(X_test)
            y_proba = self.model.predict_proba(X_test)[:, 1]
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_proba)
            
            # Save model
            self.save()
            
            # Update model info
            self.model_info["metrics"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "best_params": grid_search.best_params_
            }
            
            # Get feature importance
            feature_importance = self.model.feature_importances_
            feature_importance_dict = dict(zip(self.feature_names, feature_importance))
            
            return {
                "status": "success",
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "best_params": grid_search.best_params_,
                "feature_importance": feature_importance_dict,
                "message": f"LightGBM BTTS model trained successfully with accuracy: {accuracy:.4f}, AUC: {auc:.4f}"
            }
            
        except Exception as e:
            logger.error(f"Error training LightGBM BTTS model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training LightGBM BTTS model: {str(e)}"
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
                    "message": "LightGBM BTTS model not loaded"
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
            
            # Create confidence scores
            confidence = [float(p.max()) * 100 for p in y_proba]
            
            return {
                "status": "success",
                "predictions": y_pred.tolist(),
                "confidence": confidence,
                "probabilities": y_proba.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error predicting with LightGBM BTTS model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error predicting with LightGBM BTTS model: {str(e)}"
            }
