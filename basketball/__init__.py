"""
Basketball Prediction Module for BetSightly

This module provides basketball prediction capabilities integrated with the existing
football prediction infrastructure.

Features:
- NBA data fetching using free APIs
- XGBoost and LightGBM models for Win/Loss and Over/Under predictions
- Feature engineering for basketball-specific metrics
- Integration with existing BetSightly architecture
"""

__version__ = "1.0.0"
__author__ = "BetSightly Team"

# Import main components
from .data_fetcher import NBADataFetcher
from .feature_engineering import BasketballFeatureEngineer
from .models import BasketballModelFactory
from .prediction_service import BasketballPredictionService

__all__ = [
    "NBADataFetcher",
    "BasketballFeatureEngineer", 
    "BasketballModelFactory",
    "BasketballPredictionService"
]
