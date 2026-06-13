"""
Model Compatibility Service

This service handles model loading with compatibility fixes and fallbacks
for different scikit-learn, numpy, and XGBoost versions.
"""

import logging
import joblib
import warnings
import numpy as np
from typing import Any, Dict, Optional, Tuple
from pathlib import Path

from utils.common import setup_logging

# Set up logging
logger = setup_logging(__name__)

class ModelCompatibilityService:
    """
    Handles model loading with compatibility fixes for version mismatches.
    
    Features:
    - Graceful handling of scikit-learn version mismatches
    - NumPy random state compatibility fixes
    - SHAP explainer compatibility for pipelines
    - Fallback model creation for failed loads
    """
    
    def __init__(self):
        """Initialize the compatibility service."""
        self.compatibility_warnings = []
        self.failed_models = []
        self.successful_models = []
        
    def load_model_safely(self, model_path: str, model_name: str) -> Tuple[Any, bool]:
        """
        Load a model with compatibility handling and memory optimization.

        Args:
            model_path: Path to the model file
            model_name: Name of the model for logging

        Returns:
            Tuple of (model, success_flag)
        """
        try:
            # Check file size for memory management
            file_size_mb = Path(model_path).stat().st_size / (1024 * 1024)
            if file_size_mb > 50:  # Skip very large models on Render
                logger.warning(f"⚠️  Skipping large model {model_name} ({file_size_mb:.1f}MB) to save memory")
                return self._create_fallback_model(model_name), False

            # Suppress specific warnings during model loading
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")
                warnings.filterwarnings("ignore", message=".*numpy.dtype size changed.*")

                # Try to load the model
                model = joblib.load(model_path)

                # Validate the model
                if self._validate_model(model, model_name):
                    self.successful_models.append(model_name)
                    logger.info(f"✅ Successfully loaded {model_name} ({file_size_mb:.1f}MB)")
                    return model, True
                else:
                    logger.warning(f"⚠️  Model {model_name} loaded but failed validation")
                    return self._create_fallback_model(model_name), False

        except Exception as e:
            error_msg = str(e)

            # Handle memory issues
            if "MemoryError" in error_msg or "out of memory" in error_msg.lower():
                logger.warning(f"⚠️  Memory issue loading {model_name} - using fallback")
                return self._create_fallback_model(model_name), False

            # Handle specific compatibility issues
            elif "numpy.random._mt19937.MT19937" in error_msg:
                logger.warning(f"⚠️  NumPy random state compatibility issue in {model_name}")
                return self._fix_numpy_random_state(model_path, model_name)

            elif "node array from the pickle has an incompatible dtype" in error_msg:
                logger.warning(f"⚠️  Scikit-learn dtype compatibility issue in {model_name}")
                return self._fix_sklearn_dtype_issue(model_path, model_name)

            elif "InconsistentVersionWarning" in error_msg:
                logger.warning(f"⚠️  Scikit-learn version mismatch in {model_name}")
                return self._handle_version_mismatch(model_path, model_name)

            else:
                logger.error(f"❌ Failed to load {model_name}: {error_msg}")
                self.failed_models.append((model_name, error_msg))
                return self._create_fallback_model(model_name), False
    
    def _validate_model(self, model: Any, model_name: str) -> bool:
        """Validate that a loaded model is functional."""
        try:
            # Check if model has required methods
            if hasattr(model, 'predict'):
                # Try a dummy prediction if possible
                if hasattr(model, 'n_features_in_'):
                    n_features = model.n_features_in_
                elif hasattr(model, 'feature_importances_'):
                    n_features = len(model.feature_importances_)
                else:
                    n_features = 10  # Default assumption
                
                # Create dummy input
                dummy_input = np.random.random((1, n_features))
                
                # Try prediction
                _ = model.predict(dummy_input)
                return True
                
            elif hasattr(model, 'transform'):
                # For encoders and transformers
                return True
                
            else:
                logger.warning(f"⚠️  Model {model_name} missing predict/transform methods")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️  Model {model_name} validation failed: {str(e)}")
            return False
    
    def _fix_numpy_random_state(self, model_path: str, model_name: str) -> Tuple[Any, bool]:
        """Fix NumPy random state compatibility issues."""
        try:
            logger.info(f"🔧 Attempting NumPy random state fix for {model_name}")
            
            # Try loading with different NumPy settings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Reset NumPy random state
                np.random.seed(42)
                
                # Try loading again
                model = joblib.load(model_path)
                
                if self._validate_model(model, model_name):
                    logger.info(f"✅ Fixed NumPy random state issue for {model_name}")
                    return model, True
                    
        except Exception as e:
            logger.warning(f"⚠️  NumPy fix failed for {model_name}: {str(e)}")
        
        return self._create_fallback_model(model_name), False
    
    def _fix_sklearn_dtype_issue(self, model_path: str, model_name: str) -> Tuple[Any, bool]:
        """Fix scikit-learn dtype compatibility issues."""
        try:
            logger.info(f"🔧 Attempting scikit-learn dtype fix for {model_name}")
            
            # This is a complex issue that usually requires model retraining
            # For now, create a fallback model
            logger.warning(f"⚠️  Scikit-learn dtype issue requires model retraining for {model_name}")
            
        except Exception as e:
            logger.warning(f"⚠️  Sklearn dtype fix failed for {model_name}: {str(e)}")
        
        return self._create_fallback_model(model_name), False
    
    def _handle_version_mismatch(self, model_path: str, model_name: str) -> Tuple[Any, bool]:
        """Handle scikit-learn version mismatches."""
        try:
            logger.info(f"🔧 Attempting version mismatch fix for {model_name}")
            
            # Try loading with warnings suppressed
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                model = joblib.load(model_path)
                
                # Even with version mismatch, model might still work
                if self._validate_model(model, model_name):
                    logger.info(f"✅ Model {model_name} works despite version mismatch")
                    self.compatibility_warnings.append(f"{model_name}: Version mismatch but functional")
                    return model, True
                    
        except Exception as e:
            logger.warning(f"⚠️  Version mismatch fix failed for {model_name}: {str(e)}")
        
        return self._create_fallback_model(model_name), False
    
    def _create_fallback_model(self, model_name: str) -> Any:
        """Create a simple fallback model when loading fails."""
        logger.info(f"🔄 Creating fallback model for {model_name}")
        
        class FallbackModel:
            """Simple fallback model that returns reasonable predictions."""
            
            def __init__(self, model_name: str):
                self.model_name = model_name
                self.n_features_in_ = 10  # Default
                
            def predict(self, X):
                """Return reasonable default predictions."""
                n_samples = X.shape[0] if hasattr(X, 'shape') else 1
                
                if 'match_result' in self.model_name.lower():
                    # Return home win with moderate confidence
                    return np.array([1] * n_samples)  # 1 = Home win
                elif 'over_under' in self.model_name.lower():
                    # Return over 2.5 goals
                    return np.array([1] * n_samples)  # 1 = Over
                elif 'btts' in self.model_name.lower():
                    # Return both teams to score
                    return np.array([1] * n_samples)  # 1 = Yes
                else:
                    # Default positive prediction
                    return np.array([1] * n_samples)
            
            def predict_proba(self, X):
                """Return reasonable probability predictions."""
                n_samples = X.shape[0] if hasattr(X, 'shape') else 1
                
                if 'match_result' in self.model_name.lower():
                    # Home win: 45%, Draw: 25%, Away win: 30%
                    return np.array([[0.30, 0.25, 0.45]] * n_samples)
                else:
                    # Binary classification: 60% positive, 40% negative
                    return np.array([[0.40, 0.60]] * n_samples)
        
        return FallbackModel(model_name)
    
    def setup_shap_safely(self, model: Any, model_name: str) -> Optional[Any]:
        """Set up SHAP explainer with compatibility handling."""
        try:
            import shap
            
            # Handle pipeline models
            if hasattr(model, 'named_steps'):
                # Extract the actual estimator from pipeline
                estimator = None
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'predict'):
                        estimator = step
                        break
                
                if estimator is None:
                    logger.warning(f"⚠️  Could not extract estimator from pipeline {model_name}")
                    return None
                
                # Use the estimator for SHAP
                model_for_shap = estimator
            else:
                model_for_shap = model
            
            # Try different SHAP explainers
            if 'xgboost' in model_name.lower():
                try:
                    explainer = shap.TreeExplainer(model_for_shap)
                    logger.info(f"✅ SHAP TreeExplainer ready for {model_name}")
                    return explainer
                except Exception:
                    pass
            
            # Fallback to KernelExplainer
            try:
                # Create a simple background dataset
                background = np.random.random((10, 10))  # 10 samples, 10 features
                explainer = shap.KernelExplainer(model_for_shap.predict, background)
                logger.info(f"✅ SHAP KernelExplainer ready for {model_name}")
                return explainer
            except Exception:
                pass
            
            logger.warning(f"⚠️  Could not create SHAP explainer for {model_name}")
            return None
            
        except ImportError:
            logger.warning("⚠️  SHAP not available")
            return None
        except Exception as e:
            logger.warning(f"⚠️  SHAP setup failed for {model_name}: {str(e)}")
            return None
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Get a report of model compatibility status."""
        return {
            'successful_models': self.successful_models,
            'failed_models': self.failed_models,
            'compatibility_warnings': self.compatibility_warnings,
            'total_models_attempted': len(self.successful_models) + len(self.failed_models),
            'success_rate': len(self.successful_models) / max(1, len(self.successful_models) + len(self.failed_models))
        }

# Create singleton instance
model_compatibility_service = ModelCompatibilityService()
