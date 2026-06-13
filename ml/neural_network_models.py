"""
Neural Network Models for Football Prediction

This module contains neural network-based models for various football prediction tasks.
Neural networks are particularly good at capturing complex patterns in data.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, List, Any, Tuple
import logging
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml.base_model import BaseModel
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class NeuralNetworkOverUnderModel(BaseModel):
    """
    Neural Network model for predicting over/under goals.
    """
    
    def __init__(self, threshold=2.5):
        """
        Initialize the model.
        
        Args:
            threshold: Goal threshold for over/under prediction (default: 2.5)
        """
        super().__init__(f"nn_over_under_{str(threshold).replace('.', '_')}")
        self.model = None
        self.feature_scaler = None
        self.feature_names = None
        self.threshold = threshold
        
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
            
            # Convert to numpy arrays
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            # Define model architecture
            input_dim = X_train.shape[1]
            
            self.model = Sequential([
                Dense(128, activation='relu', input_shape=(input_dim,)),
                BatchNormalization(),
                Dropout(0.3),
                Dense(64, activation='relu'),
                BatchNormalization(),
                Dropout(0.2),
                Dense(32, activation='relu'),
                BatchNormalization(),
                Dense(1, activation='sigmoid')
            ])
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Define callbacks
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            model_path = os.path.join(self.model_dir, f"{self.model_name}.h5")
            model_checkpoint = ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True
            )
            
            # Train model
            history = self.model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            # Evaluate model
            y_pred_proba = self.model.predict(X_test)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='binary')
            recall = recall_score(y_test, y_pred, average='binary')
            f1 = f1_score(y_test, y_pred, average='binary')
            auc = roc_auc_score(y_test, y_pred_proba)
            
            # Save feature scaler
            scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.joblib")
            import joblib
            joblib.dump(self.feature_scaler, scaler_path)
            
            # Update model info
            self.model_info["metrics"] = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "training_history": {
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']],
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']]
                }
            }
            
            # Save model info
            self._save_model_info()
            
            return {
                "status": "success",
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "auc": float(auc),
                "message": f"Neural Network Over/Under {self.threshold} model trained successfully with accuracy: {accuracy:.4f}, AUC: {auc:.4f}"
            }
            
        except Exception as e:
            logger.error(f"Error training Neural Network Over/Under model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training Neural Network Over/Under model: {str(e)}"
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
                    "message": f"Neural Network Over/Under {self.threshold} model not loaded"
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
            y_pred_proba = self.model.predict(X_scaled)
            y_pred = (y_pred_proba > 0.5).astype(int).flatten()
            
            # Create confidence scores
            confidence = [float(abs(p - 0.5) * 2 * 100) for p in y_pred_proba]
            
            # Create probabilities array (for binary classification)
            probabilities = []
            for p in y_pred_proba:
                prob = float(p[0]) if isinstance(p, np.ndarray) else float(p)
                probabilities.append([1 - prob, prob])
            
            return {
                "status": "success",
                "predictions": y_pred.tolist(),
                "confidence": confidence,
                "probabilities": probabilities
            }
            
        except Exception as e:
            logger.error(f"Error predicting with Neural Network Over/Under model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error predicting with Neural Network Over/Under model: {str(e)}"
            }
    
    def _load_model(self) -> bool:
        """
        Load the model from disk.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_path = os.path.join(self.model_dir, f"{self.model_name}.h5")
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                
                # Load feature scaler
                scaler_path = os.path.join(self.model_dir, f"{self.model_name}_scaler.joblib")
                if os.path.exists(scaler_path):
                    import joblib
                    self.feature_scaler = joblib.load(scaler_path)
                
                # Load model info
                self._load_model_info()
                
                return True
            else:
                logger.warning(f"Model file not found: {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return False
