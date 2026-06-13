"""
Model Explainability System for BetSightly

This module provides SHAP and LIME explanations for all ML models,
making predictions transparent and trustworthy for users.
"""

import logging
import numpy as np
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json

from utils.config import settings
from utils.error_handling import ModelError

# Set up logging
logger = logging.getLogger(__name__)

class ModelExplainer:
    """
    Comprehensive model explainer using SHAP and LIME.
    
    Provides transparent explanations for all model types:
    - SHAP for XGBoost/LightGBM (tree-based models)
    - LIME for Neural Networks and other black-box models
    - Feature importance analysis
    - Prediction confidence breakdown
    """
    
    def __init__(self):
        """Initialize the model explainer."""
        self.shap_explainers = {}
        self.lime_explainers = {}
        self.feature_names = {}
        self.model_types = {}
        self.background_data = {}
        
    def initialize_explainers(self, model_path: str, model_type: str, 
                            feature_names: List[str], background_data: pd.DataFrame = None):
        """
        Initialize explainers for a specific model.
        
        Args:
            model_path: Path to the trained model
            model_type: Type of model (xgboost, lightgbm, neural_network, etc.)
            feature_names: List of feature names
            background_data: Background dataset for SHAP (optional)
        """
        try:
            model_name = Path(model_path).stem
            
            # Load model
            model = joblib.load(model_path)
            
            # Store metadata
            self.feature_names[model_name] = feature_names
            self.model_types[model_name] = model_type
            
            # Initialize appropriate explainer based on model type
            if model_type in ['xgboost', 'lightgbm', 'catboost']:
                self._initialize_shap_explainer(model_name, model, background_data)
            else:
                self._initialize_lime_explainer(model_name, model, background_data)
                
            logger.info(f"Initialized explainer for {model_name} ({model_type})")
            
        except Exception as e:
            logger.error(f"Failed to initialize explainer for {model_path}: {str(e)}")
            raise ModelError(f"Explainer initialization failed: {str(e)}")
    
    def _initialize_shap_explainer(self, model_name: str, model, background_data: pd.DataFrame = None):
        """Initialize SHAP explainer for tree-based models."""
        try:
            # For tree-based models, use TreeExplainer
            if hasattr(model, 'named_steps') and 'model' in model.named_steps:
                # Pipeline model - extract the actual model
                actual_model = model.named_steps['model']
            else:
                actual_model = model
            
            # Create SHAP explainer
            self.shap_explainers[model_name] = shap.TreeExplainer(actual_model)
            
            # Store background data for reference
            if background_data is not None:
                # Use a sample for efficiency
                sample_size = min(100, len(background_data))
                self.background_data[model_name] = background_data.sample(n=sample_size, random_state=42)
            
            logger.info(f"SHAP TreeExplainer initialized for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize SHAP explainer: {str(e)}")
            # Fallback to KernelExplainer
            self._initialize_lime_explainer(model_name, model, background_data)
    
    def _initialize_lime_explainer(self, model_name: str, model, background_data: pd.DataFrame = None):
        """Initialize LIME explainer for black-box models."""
        try:
            if background_data is None:
                logger.warning(f"No background data provided for LIME explainer: {model_name}")
                return
            
            # Prepare background data
            background_sample = background_data.sample(n=min(1000, len(background_data)), random_state=42)
            
            # Create LIME explainer
            self.lime_explainers[model_name] = lime.lime_tabular.LimeTabularExplainer(
                background_sample.values,
                feature_names=self.feature_names[model_name],
                class_names=['No', 'Yes'] if model_name != 'match_result' else ['Home', 'Draw', 'Away'],
                mode='classification',
                discretize_continuous=True
            )
            
            # Store model for predictions
            self.background_data[model_name] = model
            
            logger.info(f"LIME explainer initialized for {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LIME explainer: {str(e)}")
    
    def explain_prediction(self, model_name: str, features: pd.DataFrame, 
                         prediction: Any = None, top_features: int = 10) -> Dict[str, Any]:
        """
        Generate explanation for a prediction.
        
        Args:
            model_name: Name of the model
            features: Input features for prediction
            prediction: Model prediction (optional)
            top_features: Number of top features to include in explanation
            
        Returns:
            Dictionary containing explanation data
        """
        try:
            if model_name not in self.feature_names:
                raise ModelError(f"Model {model_name} not initialized for explanation")
            
            # Ensure features are in correct order
            feature_names = self.feature_names[model_name]
            features_ordered = features[feature_names]
            
            # Generate explanation based on available explainer
            if model_name in self.shap_explainers:
                return self._explain_with_shap(model_name, features_ordered, top_features)
            elif model_name in self.lime_explainers:
                return self._explain_with_lime(model_name, features_ordered, top_features)
            else:
                return self._basic_feature_importance(model_name, features_ordered, top_features)
                
        except Exception as e:
            logger.error(f"Failed to explain prediction for {model_name}: {str(e)}")
            return {
                "status": "error",
                "message": f"Explanation generation failed: {str(e)}",
                "explanation": {}
            }
    
    def _explain_with_shap(self, model_name: str, features: pd.DataFrame, top_features: int) -> Dict[str, Any]:
        """Generate SHAP explanation."""
        try:
            explainer = self.shap_explainers[model_name]
            
            # Get SHAP values
            shap_values = explainer.shap_values(features.values)
            
            # Handle multi-class vs binary classification
            if isinstance(shap_values, list):
                # Multi-class (e.g., match result: Home, Draw, Away)
                explanations = {}
                class_names = ['Home', 'Draw', 'Away'] if model_name == 'match_result' else ['Class_0', 'Class_1', 'Class_2']
                
                for i, class_name in enumerate(class_names):
                    class_shap = shap_values[i][0] if len(shap_values[i].shape) > 1 else shap_values[i]
                    explanations[class_name] = self._format_shap_explanation(
                        class_shap, features.columns, top_features
                    )
            else:
                # Binary classification
                shap_vals = shap_values[0] if len(shap_values.shape) > 1 else shap_values
                explanations = self._format_shap_explanation(shap_vals, features.columns, top_features)
            
            return {
                "status": "success",
                "method": "SHAP",
                "model_type": self.model_types[model_name],
                "explanation": explanations,
                "feature_count": len(features.columns)
            }
            
        except Exception as e:
            logger.error(f"SHAP explanation failed: {str(e)}")
            raise
    
    def _explain_with_lime(self, model_name: str, features: pd.DataFrame, top_features: int) -> Dict[str, Any]:
        """Generate LIME explanation."""
        try:
            explainer = self.lime_explainers[model_name]
            model = self.background_data[model_name]
            
            # Get LIME explanation
            explanation = explainer.explain_instance(
                features.values[0],
                model.predict_proba,
                num_features=top_features
            )
            
            # Format explanation
            lime_explanation = {}
            for feature_idx, importance in explanation.as_list():
                feature_name = self.feature_names[model_name][feature_idx] if isinstance(feature_idx, int) else feature_idx
                lime_explanation[feature_name] = {
                    "importance": float(importance),
                    "value": float(features.iloc[0, feature_idx]) if isinstance(feature_idx, int) else 0.0
                }
            
            return {
                "status": "success",
                "method": "LIME",
                "model_type": self.model_types[model_name],
                "explanation": lime_explanation,
                "feature_count": len(features.columns)
            }
            
        except Exception as e:
            logger.error(f"LIME explanation failed: {str(e)}")
            raise
    
    def _format_shap_explanation(self, shap_values: np.ndarray, feature_names: List[str], 
                                top_features: int) -> Dict[str, Any]:
        """Format SHAP values into readable explanation."""
        # Create feature importance dictionary
        feature_importance = {}
        
        for i, (feature, importance) in enumerate(zip(feature_names, shap_values)):
            feature_importance[feature] = {
                "importance": float(importance),
                "abs_importance": float(abs(importance)),
                "rank": 0  # Will be set below
            }
        
        # Sort by absolute importance and assign ranks
        sorted_features = sorted(
            feature_importance.items(),
            key=lambda x: x[1]["abs_importance"],
            reverse=True
        )
        
        # Assign ranks and limit to top features
        explanation = {}
        total_abs_importance = sum(abs(imp["importance"]) for _, imp in sorted_features[:top_features])
        
        for rank, (feature, importance_data) in enumerate(sorted_features[:top_features], 1):
            # Calculate percentage contribution
            percentage = (importance_data["abs_importance"] / total_abs_importance * 100) if total_abs_importance > 0 else 0
            
            explanation[feature] = {
                "importance": importance_data["importance"],
                "percentage": round(percentage, 1),
                "rank": rank,
                "direction": "positive" if importance_data["importance"] > 0 else "negative"
            }
        
        return explanation
    
    def _basic_feature_importance(self, model_name: str, features: pd.DataFrame, 
                                top_features: int) -> Dict[str, Any]:
        """Fallback basic feature importance when explainers are not available."""
        return {
            "status": "limited",
            "method": "Basic",
            "model_type": self.model_types.get(model_name, "unknown"),
            "explanation": {
                "message": "Detailed explanation not available for this model type",
                "feature_count": len(features.columns)
            },
            "feature_count": len(features.columns)
        }
    
    def generate_human_readable_explanation(self, explanation_data: Dict[str, Any], 
                                          prediction_result: str) -> str:
        """
        Generate human-readable explanation text.
        
        Args:
            explanation_data: Explanation data from explain_prediction
            prediction_result: The actual prediction result
            
        Returns:
            Human-readable explanation string
        """
        try:
            if explanation_data["status"] != "success":
                return "Explanation not available for this prediction."
            
            explanation = explanation_data["explanation"]
            method = explanation_data["method"]
            
            # Handle multi-class explanations
            if isinstance(explanation, dict) and any(key in explanation for key in ['Home', 'Draw', 'Away']):
                # Multi-class explanation (match result)
                if prediction_result in explanation:
                    relevant_explanation = explanation[prediction_result]
                else:
                    relevant_explanation = explanation
            else:
                relevant_explanation = explanation
            
            # Generate readable text
            if isinstance(relevant_explanation, dict) and any(isinstance(v, dict) for v in relevant_explanation.values()):
                # Feature-based explanation
                top_features = sorted(
                    relevant_explanation.items(),
                    key=lambda x: x[1].get("percentage", 0) if isinstance(x[1], dict) else 0,
                    reverse=True
                )[:5]  # Top 5 features
                
                explanation_parts = []
                for feature, data in top_features:
                    if isinstance(data, dict) and "percentage" in data:
                        direction = "supports" if data.get("direction") == "positive" else "opposes"
                        explanation_parts.append(f"{data['percentage']:.1f}% {feature} ({direction})")
                
                if explanation_parts:
                    return f"Prediction '{prediction_result}' based on: {', '.join(explanation_parts)}"
            
            return f"Prediction '{prediction_result}' generated using {method} analysis of key performance indicators."
            
        except Exception as e:
            logger.error(f"Failed to generate human-readable explanation: {str(e)}")
            return f"Prediction: {prediction_result} (explanation processing error)"


# Global explainer instance
model_explainer = ModelExplainer()
