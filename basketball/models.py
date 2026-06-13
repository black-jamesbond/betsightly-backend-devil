"""
Basketball ML Models

This module provides basketball-specific ML models following the same pattern
as the football models but adapted for basketball predictions.

Models:
- XGBoost for Win/Loss prediction
- LightGBM for Over/Under prediction
- Neural Network for advanced predictions
"""

import logging
import pandas as pd
import numpy as np
import joblib
import os
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from datetime import datetime

# ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Try to import neural network components
try:
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    NEURAL_NETWORK_AVAILABLE = True
except ImportError:
    NEURAL_NETWORK_AVAILABLE = False

from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class BasketballModelFactory:
    """
    Factory for creating and managing basketball prediction models.
    
    Follows the same pattern as the football model factory but adapted
    for basketball-specific predictions.
    """
    
    def __init__(self):
        """Initialize the basketball model factory."""
        self.models_dir = os.path.join(settings.ml.MODEL_DIR, "basketball")
        os.makedirs(self.models_dir, exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        
        logger.info("Basketball Model Factory initialized")
    
    def create_win_loss_model(self) -> 'BasketballWinLossModel':
        """Create a Win/Loss prediction model."""
        return BasketballWinLossModel(self.models_dir)
    
    def create_over_under_model(self) -> 'BasketballOverUnderModel':
        """Create an Over/Under prediction model."""
        return BasketballOverUnderModel(self.models_dir)
    
    def create_neural_network_model(self) -> Optional['BasketballNeuralNetworkModel']:
        """Create a Neural Network model if available."""
        if NEURAL_NETWORK_AVAILABLE:
            return BasketballNeuralNetworkModel(self.models_dir)
        return None
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        models = ['win_loss_xgboost', 'over_under_lightgbm']
        if NEURAL_NETWORK_AVAILABLE:
            models.append('neural_network')
        return models

    def _get_models_status(self) -> Dict[str, bool]:
        """Get status of available models."""
        status = {}

        # Check Win/Loss model
        win_loss_model = self.create_win_loss_model()
        status['win_loss_xgboost'] = win_loss_model.model is not None

        # Check Over/Under model
        over_under_model = self.create_over_under_model()
        status['over_under_lightgbm'] = over_under_model.model is not None

        # Check Neural Network model
        neural_model = self.create_neural_network_model()
        status['neural_network'] = neural_model is not None and neural_model.model is not None

        return status

class BasketballWinLossModel:
    """
    XGBoost model for basketball Win/Loss prediction.
    """
    
    def __init__(self, models_dir: str):
        """Initialize the Win/Loss model."""
        self.models_dir = models_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = os.path.join(models_dir, "win_loss_xgboost.joblib")
        self.scaler_path = os.path.join(models_dir, "win_loss_scaler.joblib")
        
        # Try to load existing model
        self.load_model()
    
    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """
        Train the Win/Loss model.
        
        Args:
            X: Feature matrix
            y: Target variable (1 for home win, 0 for away win)
            
        Returns:
            Training results dictionary
        """
        logger.info("Training basketball Win/Loss XGBoost model")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create and train XGBoost model
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='logloss'
        )
        
        # Train model
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='accuracy')
        
        # Save model
        self.save_model()
        
        results = {
            'model_type': 'basketball_win_loss_xgboost',
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Win/Loss model trained. Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make Win/Loss predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions dictionary
        """
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'prediction': 'HOME_WIN' if pred == 1 else 'AWAY_WIN',
                'home_win_probability': prob[1],
                'away_win_probability': prob[0],
                'confidence': max(prob)
            }
            results.append(result)
        
        return {'predictions': results}
    
    def save_model(self):
        """Save the trained model."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Win/Loss model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Win/Loss model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Win/Loss model: {str(e)}")

class BasketballOverUnderModel:
    """
    LightGBM model for basketball Over/Under prediction.
    """
    
    def __init__(self, models_dir: str):
        """Initialize the Over/Under model."""
        self.models_dir = models_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = os.path.join(models_dir, "over_under_lightgbm.joblib")
        self.scaler_path = os.path.join(models_dir, "over_under_scaler.joblib")
        
        # Try to load existing model
        self.load_model()
    
    def train(self, X: pd.DataFrame, y: pd.Series, total_threshold: float = 220.0) -> Dict[str, Any]:
        """
        Train the Over/Under model.
        
        Args:
            X: Feature matrix
            y: Target variable (total points)
            total_threshold: Over/Under threshold
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training basketball Over/Under LightGBM model (threshold: {total_threshold})")
        
        # Convert to binary classification (Over/Under)
        y_binary = (y > total_threshold).astype(int)
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Create and train LightGBM model
        self.model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X_scaled, y_binary, cv=5, scoring='accuracy')
        
        # Save model
        self.save_model()
        
        results = {
            'model_type': 'basketball_over_under_lightgbm',
            'threshold': total_threshold,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': dict(zip(self.feature_names, self.model.feature_importances_)),
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        logger.info(f"Over/Under model trained. Accuracy: {accuracy:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        return results
    
    def predict(self, X: pd.DataFrame, total_threshold: float = 220.0) -> Dict[str, Any]:
        """
        Make Over/Under predictions.
        
        Args:
            X: Feature matrix
            total_threshold: Over/Under threshold
            
        Returns:
            Predictions dictionary
        """
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'prediction': 'OVER' if pred == 1 else 'UNDER',
                'threshold': total_threshold,
                'over_probability': prob[1],
                'under_probability': prob[0],
                'confidence': max(prob)
            }
            results.append(result)
        
        return {'predictions': results}
    
    def save_model(self):
        """Save the trained model."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Over/Under model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Over/Under model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Over/Under model: {str(e)}")

class BasketballNeuralNetworkModel:
    """
    Neural Network model for basketball predictions.
    """
    
    def __init__(self, models_dir: str):
        """Initialize the Neural Network model."""
        self.models_dir = models_dir
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.model_path = os.path.join(models_dir, "neural_network.joblib")
        self.scaler_path = os.path.join(models_dir, "neural_network_scaler.joblib")
        
        # Try to load existing model
        self.load_model()
    
    def train(self, X: pd.DataFrame, y: pd.Series, task_type: str = 'classification') -> Dict[str, Any]:
        """
        Train the Neural Network model.
        
        Args:
            X: Feature matrix
            y: Target variable
            task_type: 'classification' or 'regression'
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training basketball Neural Network model ({task_type})")
        
        # Store feature names
        self.feature_names = list(X.columns)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        # Create model based on task type
        if task_type == 'classification':
            self.model = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
        else:
            self.model = MLPRegressor(
                hidden_layer_sizes=(100, 50),
                activation='relu',
                solver='adam',
                alpha=0.001,
                max_iter=500,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        if task_type == 'classification':
            y_pred = self.model.predict(X_test)
            score = accuracy_score(y_test, y_pred)
            metric_name = 'accuracy'
        else:
            y_pred = self.model.predict(X_test)
            score = mean_squared_error(y_test, y_pred, squared=False)  # RMSE
            metric_name = 'rmse'
        
        # Save model
        self.save_model()
        
        results = {
            'model_type': f'basketball_neural_network_{task_type}',
            metric_name: score,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'task_type': task_type
        }
        
        logger.info(f"Neural Network model trained. {metric_name}: {score:.3f}")
        return results
    
    def predict(self, X: pd.DataFrame) -> Dict[str, Any]:
        """
        Make Neural Network predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Predictions dictionary
        """
        if self.model is None:
            return {'error': 'Model not trained'}
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Get probabilities if classification
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': pred,
                    'probabilities': prob.tolist(),
                    'confidence': max(prob)
                }
                results.append(result)
        else:
            results = [{'prediction': pred} for pred in predictions]
        
        return {'predictions': results}
    
    def save_model(self):
        """Save the trained model."""
        if self.model is not None:
            joblib.dump(self.model, self.model_path)
            joblib.dump(self.scaler, self.scaler_path)
            logger.info(f"Neural Network model saved to {self.model_path}")
    
    def load_model(self):
        """Load a trained model."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            try:
                self.model = joblib.load(self.model_path)
                self.scaler = joblib.load(self.scaler_path)
                logger.info("Neural Network model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Neural Network model: {str(e)}")
