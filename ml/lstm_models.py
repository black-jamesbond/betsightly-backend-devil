"""
LSTM Models for Football Prediction

This module contains LSTM-based models for time-series football prediction tasks.
LSTM (Long Short-Term Memory) networks are well-suited for sequential data like
team performance over time.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from typing import Dict, List, Any, Tuple
import logging
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ml.base_model import BaseModel
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class LSTMTeamFormModel(BaseModel):
    """
    LSTM model for predicting team performance based on form over time.
    This model can be used for various prediction tasks by changing the output layer.
    """
    
    def __init__(self, prediction_type="match_result", sequence_length=5):
        """
        Initialize the model.
        
        Args:
            prediction_type: Type of prediction to make (match_result, btts, over_under)
            sequence_length: Number of previous matches to consider
        """
        super().__init__(f"lstm_{prediction_type}")
        self.model = None
        self.feature_scaler = None
        self.feature_names = None
        self.prediction_type = prediction_type
        self.sequence_length = sequence_length
        
    def _prepare_sequences(self, df: pd.DataFrame, team_id_col: str, date_col: str, 
                          feature_cols: List[str], target_col: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare sequences for LSTM training.
        
        Args:
            df: DataFrame with match data
            team_id_col: Column name for team ID
            date_col: Column name for match date
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Tuple of (X_sequences, y)
        """
        # Sort by team and date
        df = df.sort_values([team_id_col, date_col])
        
        # Group by team
        team_groups = df.groupby(team_id_col)
        
        X_sequences = []
        y_values = []
        
        for team_id, group in team_groups:
            # Get features and target
            X_team = group[feature_cols].values
            y_team = group[target_col].values
            
            # Create sequences
            for i in range(len(X_team) - self.sequence_length):
                X_sequences.append(X_team[i:i+self.sequence_length])
                y_values.append(y_team[i+self.sequence_length])
                
        return np.array(X_sequences), np.array(y_values)
        
    def train(self, df: pd.DataFrame, team_id_col: str, date_col: str, 
             feature_cols: List[str], target_col: str) -> Dict[str, Any]:
        """
        Train the model.
        
        Args:
            df: DataFrame with match data
            team_id_col: Column name for team ID
            date_col: Column name for match date
            feature_cols: List of feature column names
            target_col: Target column name
            
        Returns:
            Dictionary with training results
        """
        try:
            # Save feature names
            self.feature_names = feature_cols
            
            # Prepare sequences
            X_sequences, y = self._prepare_sequences(df, team_id_col, date_col, feature_cols, target_col)
            
            if len(X_sequences) == 0:
                return {
                    "status": "error",
                    "message": "Not enough sequential data to train LSTM model"
                }
            
            # Scale features
            self.feature_scaler = MinMaxScaler()
            n_samples, n_timesteps, n_features = X_sequences.shape
            X_reshaped = X_sequences.reshape(n_samples * n_timesteps, n_features)
            X_scaled = self.feature_scaler.fit_transform(X_reshaped)
            X_sequences = X_scaled.reshape(n_samples, n_timesteps, n_features)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y, test_size=0.2, random_state=42
            )
            
            # Define model architecture based on prediction type
            if self.prediction_type == "match_result":
                # Multi-class classification (Home, Draw, Away)
                output_units = 3
                activation = 'softmax'
                loss = 'sparse_categorical_crossentropy'
            else:
                # Binary classification (BTTS, Over/Under)
                output_units = 1
                activation = 'sigmoid'
                loss = 'binary_crossentropy'
            
            # Create model
            self.model = Sequential([
                LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(feature_cols))),
                Dropout(0.3),
                LSTM(32),
                Dropout(0.2),
                Dense(16, activation='relu'),
                BatchNormalization(),
                Dense(output_units, activation=activation)
            ])
            
            # Compile model
            self.model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss=loss,
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
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                callbacks=[early_stopping, model_checkpoint],
                verbose=1
            )
            
            # Evaluate model
            if self.prediction_type == "match_result":
                # Multi-class evaluation
                y_pred_proba = self.model.predict(X_test)
                y_pred = np.argmax(y_pred_proba, axis=1)
                
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted')
                recall = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
            else:
                # Binary evaluation
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
            
            # Save sequence info
            sequence_info = {
                "sequence_length": self.sequence_length,
                "feature_cols": feature_cols,
                "team_id_col": team_id_col,
                "date_col": date_col,
                "target_col": target_col
            }
            
            sequence_path = os.path.join(self.model_dir, f"{self.model_name}_sequence_info.joblib")
            joblib.dump(sequence_info, sequence_path)
            
            # Update model info
            metrics = {
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "training_history": {
                    "loss": [float(x) for x in history.history['loss']],
                    "val_loss": [float(x) for x in history.history['val_loss']],
                    "accuracy": [float(x) for x in history.history['accuracy']],
                    "val_accuracy": [float(x) for x in history.history['val_accuracy']]
                }
            }
            
            if self.prediction_type != "match_result":
                metrics["auc"] = float(auc)
                
            self.model_info["metrics"] = metrics
            self.model_info["sequence_info"] = sequence_info
            
            # Save model info
            self._save_model_info()
            
            result = {
                "status": "success",
                "accuracy": float(accuracy),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "message": f"LSTM {self.prediction_type} model trained successfully with accuracy: {accuracy:.4f}"
            }
            
            if self.prediction_type != "match_result":
                result["auc"] = float(auc)
                
            return result
            
        except Exception as e:
            logger.error(f"Error training LSTM model: {str(e)}")
            return {
                "status": "error",
                "message": f"Error training LSTM model: {str(e)}"
            }
