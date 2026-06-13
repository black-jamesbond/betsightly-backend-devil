"""
Model Factory for Football Prediction

This module provides a factory for creating and using different types of ML models
for football prediction tasks.
"""

import logging
from typing import Dict, Any, List, Optional, Union, Type

# Set up logging
logger = logging.getLogger(__name__)

# Import base model
from ml.base_model import BaseModel

# Import model implementations
from ml.ensemble_model_improved import MatchResultModel, OverUnderModel, BTTSModel

# Import advanced models
try:
    from ml.xgboost_model import XGBoostMatchResultModel
except ImportError:
    logger.warning("XGBoost model not available")
    XGBoostMatchResultModel = None

try:
    from ml.lightgbm_models import LightGBMBTTSModel
except ImportError:
    logger.warning("LightGBM model not available")
    LightGBMBTTSModel = None

try:
    from ml.neural_network_models import NeuralNetworkOverUnderModel
except ImportError:
    logger.warning("Neural Network model not available")
    NeuralNetworkOverUnderModel = None

try:
    from ml.lstm_models import LSTMTeamFormModel
except ImportError:
    logger.warning("LSTM model not available")
    LSTMTeamFormModel = None


try:
    from ml.catboost_model import CatBoostMatchResultModel, CatBoostBTTSModel, CatBoostOverUnderModel
    CATBOOST_AVAILABLE = True
except ImportError:
    logger.warning("CatBoost model not available")
    CatBoostMatchResultModel = CatBoostBTTSModel = CatBoostOverUnderModel = None
    CATBOOST_AVAILABLE = False

try:
    from ml.dixon_coles_model import DixonColesModel
    DIXON_COLES_AVAILABLE = True
except ImportError:
    logger.warning("Dixon-Coles model not available")
    DixonColesModel = None
    DIXON_COLES_AVAILABLE = False

try:
    from ml.elo_model import EloRatingSystem
    ELO_AVAILABLE = True
except ImportError:
    logger.warning("ELO model not available")
    EloRatingSystem = None
    ELO_AVAILABLE = False

try:
    from ml.bayesian_averaging import BayesianModelAverager
    BMA_AVAILABLE = True
except ImportError:
    logger.warning("Bayesian Model Averaging not available")
    BayesianModelAverager = None
    BMA_AVAILABLE = False


class ModelFactory:
    """
    Factory for creating and managing football prediction models.
    """

    def __init__(self):
        """Initialize the model factory."""
        self.models = {}
        self.register_default_models()

    def register_default_models(self):
        """Register default models."""
        # Register traditional models
        self.register_model("match_result", MatchResultModel)
        self.register_model("over_under", OverUnderModel)
        self.register_model("btts", BTTSModel)

        # Register XGBoost models if available
        if XGBoostMatchResultModel is not None:
            self.register_model("xgboost_match_result", XGBoostMatchResultModel)
            logger.info("Registered XGBoost match result model")

        # Register LightGBM models if available
        if LightGBMBTTSModel is not None:
            self.register_model("lightgbm_btts", LightGBMBTTSModel)
            logger.info("Registered LightGBM BTTS model")

        # Register Neural Network models if available
        if NeuralNetworkOverUnderModel is not None:
            self.register_model("nn_over_under_2_5", lambda: NeuralNetworkOverUnderModel(threshold=2.5))
            self.register_model("nn_over_under_1_5", lambda: NeuralNetworkOverUnderModel(threshold=1.5))
            self.register_model("nn_over_under_3_5", lambda: NeuralNetworkOverUnderModel(threshold=3.5))
            logger.info("Registered Neural Network over/under models")

        # Register LSTM models if available
        if LSTMTeamFormModel is not None:
            self.register_model("lstm_match_result", lambda: LSTMTeamFormModel(prediction_type="match_result"))
            self.register_model("lstm_btts", lambda: LSTMTeamFormModel(prediction_type="btts"))
            self.register_model("lstm_over_under", lambda: LSTMTeamFormModel(prediction_type="over_under"))
            logger.info("Registered LSTM team form models")

        # Register CatBoost models if available
        if CatBoostMatchResultModel is not None:
            self.register_model("catboost_match_result", CatBoostMatchResultModel)
            self.register_model("catboost_btts", CatBoostBTTSModel)
            self.register_model("catboost_over_under_2_5", lambda: CatBoostOverUnderModel(threshold=2.5))
            logger.info("Registered CatBoost models")

        # Register Dixon-Coles model if available
        if DixonColesModel is not None:
            self.register_model("dixon_coles", DixonColesModel)
            logger.info("Registered Dixon-Coles model")

        # Register ELO model if available
        if EloRatingSystem is not None:
            self.register_model("elo_rating", EloRatingSystem)
            logger.info("Registered ELO rating model")

        # Bayesian Model Averager (accessible via factory)
        if BayesianModelAverager is not None:
            self.register_model("bayesian_model_average", BayesianModelAverager)
            logger.info("Registered Bayesian Model Averaging")


    def register_model(self, model_type: str, model_class: Union[Type[BaseModel], callable]):
        """
        Register a model type with its implementation class.

        Args:
            model_type: Type identifier for the model
            model_class: Model class or factory function
        """
        self.models[model_type] = model_class

    def create_model(self, model_type: str) -> Optional[BaseModel]:
        """
        Create a model instance of the specified type.

        Args:
            model_type: Type of model to create

        Returns:
            Model instance or None if type not found
        """
        if model_type not in self.models:
            logger.error(f"Unknown model type: {model_type}")
            return None

        model_class = self.models[model_type]

        # Handle both class and factory function
        if callable(model_class) and not isinstance(model_class, type):
            return model_class()
        else:
            return model_class()

    def get_available_models(self) -> List[str]:
        """
        Get list of available model types.

        Returns:
            List of model type identifiers
        """
        return list(self.models.keys())

    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """
        Get information about a model type.

        Args:
            model_type: Type of model

        Returns:
            Dictionary with model information
        """
        if model_type not in self.models:
            return {"error": f"Unknown model type: {model_type}"}

        model = self.create_model(model_type)

        # Load model info
        model._load_model_info()

        return {
            "name": model.model_name,
            "info": model.model_info
        }

    def train_model(self, model_type: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Train a model of the specified type.

        Args:
            model_type: Type of model to train
            *args, **kwargs: Arguments to pass to the model's train method

        Returns:
            Dictionary with training results
        """
        model = self.create_model(model_type)

        if model is None:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        return model.train(*args, **kwargs)

    def predict(self, model_type: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Make predictions using a model of the specified type.

        Args:
            model_type: Type of model to use
            *args, **kwargs: Arguments to pass to the model's predict method

        Returns:
            Dictionary with predictions
        """
        model = self.create_model(model_type)

        if model is None:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        return model.predict(*args, **kwargs)

    def evaluate_model(self, model_type: str, *args, **kwargs) -> Dict[str, Any]:
        """
        Evaluate a model of the specified type.

        Args:
            model_type: Type of model to evaluate
            *args, **kwargs: Arguments to pass to the model's evaluate method

        Returns:
            Dictionary with evaluation results
        """
        model = self.create_model(model_type)

        if model is None:
            return {"status": "error", "message": f"Unknown model type: {model_type}"}

        return model.evaluate(*args, **kwargs)

# Create singleton instance
model_factory = ModelFactory()
