"""
XGBoost Model Module

This module provides an implementation of XGBoost models for football match prediction.
It uses the advanced feature engineering module for better prediction accuracy.
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from datetime import datetime

from ml.base_model import BaseModel
from ml.advanced_feature_engineering import AdvancedFootballFeatureEngineer
from utils.common import setup_logging

# Set up logging
logger = setup_logging(__name__)

class XGBoostMatchResultModel(BaseModel):
    """
    XGBoost model for predicting match results (home win, draw, away win).
    
    Features:
    - Uses advanced feature engineering
    - Hyperparameter optimization
    - Feature importance analysis
    - Calibrated confidence scores
    """
    
    def __init__(self):
        """Initialize the XGBoost match result model."""
        super().__init__("xgboost_match_result")
        self.feature_engineer = AdvancedFootballFeatureEngineer()
        self.label_map = {0: "H", 1: "D", 2: "A"}  # 0 = Home win, 1 = Draw, 2 = Away win
        self.reverse_label_map = {"H": 0, "D": 1, "A": 2}
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            X: Features
            y: Target variable (H, D, A)
            
        Returns:
            Dictionary with training results
        """
        try:
            # Convert labels to numeric if they are strings
            if y.dtype == object:
                y_numeric = y.map(self.reverse_label_map)
            else:
                y_numeric = y
            
            # Split data into training and validation sets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y_numeric, test_size=0.2, random_state=42, stratify=y_numeric
            )
            
            # Set feature names
            self.feature_names = X.columns.tolist()
            
            # Use advanced feature engineering
            self.feature_engineer.set_historical_data(X)
            
            # Create DMatrix for XGBoost
            dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=self.feature_names)
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=self.feature_names)
            
            # Set XGBoost parameters
            params = {
                'objective': 'multi:softprob',
                'num_class': 3,
                'eval_metric': ['mlogloss', 'merror'],
                'eta': 0.05,
                'max_depth': 6,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'gamma': 0,
                'alpha': 0.1,
                'lambda': 1.0,
                'tree_method': 'hist',  # For faster training
                'seed': 42
            }
            
            # Train the model with early stopping
            evals = [(dtrain, 'train'), (dval, 'val')]
            self.model = xgb.train(
                params,
                dtrain,
                num_boost_round=1000,
                evals=evals,
                early_stopping_rounds=50,
                verbose_eval=100
            )
            
            # Get feature importance
            importance = self.model.get_score(importance_type='gain')
            importance = {k: v for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True)}
            
            # Make predictions on validation set
            y_pred = self.model.predict(dval)
            y_pred_class = np.argmax(y_pred, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_val, y_pred_class)
            precision = precision_score(y_val, y_pred_class, average='weighted')
            recall = recall_score(y_val, y_pred_class, average='weighted')
            f1 = f1_score(y_val, y_pred_class, average='weighted')
            loss = log_loss(y_val, y_pred)
            
            # Update model info
            self.model_info["created_at"] = datetime.now().isoformat()
            self.model_info["updated_at"] = datetime.now().isoformat()
            self.model_info["metrics"] = {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "log_loss": loss
            }
            self.model_info["parameters"] = params
            self.model_info["feature_importance"] = importance
            
            # Train the confidence calibrator
            self.confidence_calibrator.train(y_val, y_pred)
            
            # Save the model
            self.save()
            
            return {
                "status": "success",
                "message": "Model trained successfully",
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "log_loss": loss
                },
                "feature_importance": importance
            }
            
        except Exception as e:
            logger.error(f"Error training XGBoost match result model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make predictions.
        
        Args:
            X: Features
            
        Returns:
            Dictionary with predictions
        """
        try:
            if self.model is None:
                logger.warning("Model not trained, loading from disk")
                if not self.load():
                    return {"status": "error", "message": "Model not trained and not found on disk"}
            
            # Use advanced feature engineering if historical data is available
            if hasattr(self.feature_engineer, 'historical_data') and self.feature_engineer.historical_data is not None:
                # Extract fixture data
                fixture_data = {
                    'fixture': {'id': X.get('fixture_id', ['unknown'])[0]},
                    'teams': {
                        'home': {'name': X.get('home_team', ['unknown'])[0]},
                        'away': {'name': X.get('away_team', ['unknown'])[0]}
                    },
                    'fixture': {'date': X.get('match_date', [datetime.now()])[0]}
                }
                
                # Engineer features
                X_engineered = self.feature_engineer.engineer_features(fixture_data)
                
                # Check if we have engineered features
                if not X_engineered.empty:
                    X = X_engineered
            
            # Create DMatrix for prediction
            dtest = xgb.DMatrix(X, feature_names=self.feature_names)
            
            # Make predictions
            y_proba = self.model.predict(dtest)
            
            # Calibrate confidence scores
            calibrated_proba = self.calibrate_confidence(y_proba)
            
            # Get predicted class
            y_pred_class = np.argmax(calibrated_proba, axis=1)
            
            # Map numeric predictions to labels
            predictions = [self.label_map[p] for p in y_pred_class]
            
            # Create confidence scores
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
            logger.error(f"Error making predictions with XGBoost match result model: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Optimize hyperparameters using grid search.
        
        Args:
            X: Features
            y: Target variable
            
        Returns:
            Dictionary with optimization results
        """
        try:
            # Convert labels to numeric if they are strings
            if y.dtype == object:
                y_numeric = y.map(self.reverse_label_map)
            else:
                y_numeric = y
            
            # Define parameter grid
            param_grid = {
                'max_depth': [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'min_child_weight': [1, 3, 5],
                'gamma': [0, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0],
                'colsample_bytree': [0.6, 0.8, 1.0]
            }
            
            # Create XGBoost classifier
            xgb_model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                eval_metric='mlogloss',
                use_label_encoder=False,
                tree_method='hist',
                random_state=42
            )
            
            # Create grid search
            grid_search = GridSearchCV(
                estimator=xgb_model,
                param_grid=param_grid,
                cv=5,
                scoring='accuracy',
                n_jobs=-1,
                verbose=2
            )
            
            # Fit grid search
            grid_search.fit(X, y_numeric)
            
            # Get best parameters
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            
            return {
                "status": "success",
                "message": "Hyperparameter optimization completed",
                "best_params": best_params,
                "best_score": best_score
            }
            
        except Exception as e:
            logger.error(f"Error optimizing hyperparameters: {str(e)}")
            return {"status": "error", "message": str(e)}
