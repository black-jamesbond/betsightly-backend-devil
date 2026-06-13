"""
Streamlined Configuration Module

Centralized configuration for the production-ready ML prediction pipeline.
Focuses on essential settings for data sources, API integration, and ML models.
"""

import os
from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

# Set up logging
logger = logging.getLogger(__name__)

class DataSourceSettings(BaseSettings):
    """Data source configuration for training and live predictions."""

    # GitHub dataset for training (primary data source)
    GITHUB_DATASET_URL: str = Field(
        "https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv",
        description="GitHub football dataset URL for model training"
    )

    # Local data directories
    DATA_DIR: str = Field("data", description="Local data directory")
    MODELS_DIR: str = Field("models", description="ML models directory")
    CACHE_DIR: str = Field("cache", description="Cache directory")

    model_config = SettingsConfigDict(env_prefix="DATA_", case_sensitive=True)

class FootballDataSettings(BaseSettings):
    """Football-Data.org API configuration for live fixtures."""

    API_KEY: str = Field(default="", description="Football-Data.org API key")
    BASE_URL: str = Field("https://api.football-data.org/v4")
    DAILY_LIMIT: int = Field(100)
    DEFAULT_COMPETITIONS: str = Field("PL,PD,SA,BL1,FL1")

    model_config = SettingsConfigDict(env_prefix="FOOTBALL_DATA_", case_sensitive=True)

class APIFootballSettings(BaseSettings):
    """API Football configuration settings."""

    API_KEY: str = Field("")
    API_HOST: str = Field("api-football-v1.p.rapidapi.com")
    BASE_URL: str = Field("https://api-football-v1.p.rapidapi.com/v3")
    DAILY_LIMIT: int = Field(100)

    model_config = SettingsConfigDict(env_prefix="API_FOOTBALL_", case_sensitive=True)

class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    URL: str = Field("sqlite:///./football.db")
    ECHO: bool = Field(False)
    POOL_SIZE: int = Field(5)
    MAX_OVERFLOW: int = Field(10)

    model_config = SettingsConfigDict(env_prefix="DATABASE_", case_sensitive=True)

class MLSettings(BaseSettings):
    """Advanced ML configuration for production pipeline."""

    # Directories
    MODEL_DIR: str = Field("models")
    DATA_DIR: str = Field("data")
    CACHE_DIR: str = Field("cache")

    # Model priorities (advanced models first)
    PREFERRED_MODELS: str = Field(
        "xgboost,lightgbm,neural_network,lstm,ensemble",
        description="Comma-separated list of models in priority order"
    )

    # Training configuration
    TRAIN_TEST_SPLIT: float = Field(0.2, description="Test set ratio")
    CROSS_VALIDATION_FOLDS: int = Field(5, description="CV folds for model evaluation")
    HYPERPARAMETER_OPTIMIZATION: bool = Field(True, description="Enable hyperparameter tuning")

    # Feature engineering
    FEATURE_SELECTION: bool = Field(True, description="Enable automatic feature selection")
    MIN_FEATURE_IMPORTANCE: float = Field(0.01, description="Minimum feature importance threshold")

    # Prediction filtering
    MIN_CONFIDENCE_THRESHOLD: float = Field(0.65, description="Minimum confidence for predictions")
    MAX_PREDICTIONS_PER_CATEGORY: int = Field(10, description="Maximum predictions per category")

    # Caching
    FEATURE_CACHE_EXPIRY: int = Field(24, description="Feature cache expiry in hours")
    PREDICTION_CACHE_TTL: int = Field(1800, description="Prediction cache TTL in seconds")

    model_config = SettingsConfigDict(env_prefix="ML_", case_sensitive=True)

class BasketballSettings(BaseSettings):
    """Basketball prediction configuration settings."""

    # NBA API settings (free)
    NBA_API_ENABLED: bool = Field(True, description="Enable NBA API data fetching")
    NBA_API_BASE_URL: str = Field("https://stats.nba.com/stats", description="NBA Stats API base URL")

    # Data sources
    KAGGLE_DATASET_URL: str = Field(
        "https://www.kaggle.com/datasets/nathanlauga/nba-games",
        description="Kaggle NBA dataset URL for training"
    )

    # Prediction settings
    MIN_CONFIDENCE_THRESHOLD: float = Field(0.60, description="Minimum confidence for basketball predictions")
    MAX_PREDICTIONS_PER_CATEGORY: int = Field(8, description="Maximum basketball predictions per category")

    # Feature engineering
    TEAM_FORM_GAMES: int = Field(5, description="Number of recent games for form calculation")
    SEASON_START_MONTH: int = Field(10, description="NBA season start month (October)")

    # Model settings
    PREFERRED_MODELS: str = Field(
        "xgboost,lightgbm,neural_network",
        description="Preferred basketball models in priority order"
    )

    model_config = SettingsConfigDict(env_prefix="BASKETBALL_", case_sensitive=True)

class TelegramSettings(BaseSettings):
    """Telegram bot configuration settings."""

    BOT_TOKEN: str = Field("")
    CHAT_ID: str = Field("")
    WEBHOOK_URL: str = Field("")
    WEBHOOK_SECRET: str = Field("")

    model_config = SettingsConfigDict(env_prefix="TELEGRAM_", case_sensitive=True)

class PunterSettings(BaseSettings):
    """Punter configuration settings."""

    CACHE_DIR: str = Field("punter_cache")
    MIN_PREDICTIONS: int = Field(10)
    DEFAULT_CONFIDENCE: float = Field(0.5)

    model_config = SettingsConfigDict(env_prefix="PUNTER_", case_sensitive=True)

class OddsCategories(BaseSettings):
    """Odds categories configuration."""

    TWO_ODDS_MIN: float = Field(1.5)
    TWO_ODDS_MAX: float = Field(2.5)
    TWO_ODDS_MIN_CONFIDENCE: float = Field(70.0)
    TWO_ODDS_LIMIT: int = Field(5)
    TWO_ODDS_TARGET: float = Field(2.0)

    FIVE_ODDS_MIN: float = Field(2.5)
    FIVE_ODDS_MAX: float = Field(5.0)
    FIVE_ODDS_MIN_CONFIDENCE: float = Field(70.0)
    FIVE_ODDS_LIMIT: int = Field(3)
    FIVE_ODDS_TARGET: float = Field(5.0)

    TEN_ODDS_MIN: float = Field(5.0)
    TEN_ODDS_MAX: float = Field(10.0)
    TEN_ODDS_MIN_CONFIDENCE: float = Field(70.0)
    TEN_ODDS_LIMIT: int = Field(2)
    TEN_ODDS_TARGET: float = Field(10.0)

    ROLLOVER_MIN: float = Field(1.2)
    ROLLOVER_MAX: float = Field(2.0)
    ROLLOVER_MIN_CONFIDENCE: float = Field(70.0)
    ROLLOVER_TARGET: float = Field(3.0)
    ROLLOVER_DAYS: int = Field(10)

    model_config = SettingsConfigDict(env_prefix="ODDS_", case_sensitive=True)

class Settings(BaseSettings):
    """Main application settings."""

    # Application settings
    APP_NAME: str = Field("BetSightly")
    APP_VERSION: str = Field("1.0.0")
    DEBUG: bool = Field(False)
    ENVIRONMENT: str = Field("development")

    # Security settings
    SECRET_KEY: str = Field("", description="Secret key for JWT tokens and encryption")
    API_KEY: str = Field("", description="Static API key clients must send in X-API-Key header (leave blank to disable auth in dev)")
    API_KEY_HEADER: str = Field("X-API-Key", description="Header name for API key authentication")
    ALLOWED_HOSTS: str = Field("localhost,127.0.0.1", description="Comma-separated allowed hosts")

    # Rate limiting
    RATE_LIMIT_REQUESTS: int = Field(100, description="Number of requests per minute per IP")
    RATE_LIMIT_WINDOW: int = Field(60, description="Rate limit window in seconds")

    # Path settings
    BASE_DIR: str = Field(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    # Component settings
    data_source: DataSourceSettings = DataSourceSettings()       # GitHub dataset for training
    football_data: FootballDataSettings = FootballDataSettings() # Live fixtures API
    api_football: APIFootballSettings = APIFootballSettings()    # Alternative live fixtures API
    basketball: BasketballSettings = BasketballSettings()        # Basketball prediction settings
    database: DatabaseSettings = DatabaseSettings()
    ml: MLSettings = MLSettings()
    telegram: TelegramSettings = TelegramSettings()
    punter: PunterSettings = PunterSettings()
    odds_categories: OddsCategories = OddsCategories()

    model_config = SettingsConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Set absolute paths based on BASE_DIR
        self.ml.MODEL_DIR = os.path.join(self.BASE_DIR, self.ml.MODEL_DIR)
        self.ml.DATA_DIR = os.path.join(self.BASE_DIR, self.ml.DATA_DIR)
        self.ml.CACHE_DIR = os.path.join(self.BASE_DIR, self.ml.CACHE_DIR)
        self.punter.CACHE_DIR = os.path.join(self.BASE_DIR, self.punter.CACHE_DIR)

        # Create directories if they don't exist
        os.makedirs(self.ml.MODEL_DIR, exist_ok=True)
        os.makedirs(self.ml.DATA_DIR, exist_ok=True)
        os.makedirs(self.ml.CACHE_DIR, exist_ok=True)
        os.makedirs(self.punter.CACHE_DIR, exist_ok=True)

        # Convert odds categories to dictionary format for backward compatibility
        self.ODDS_CATEGORIES = {
            "2_odds": {
                "min_odds": self.odds_categories.TWO_ODDS_MIN,
                "max_odds": self.odds_categories.TWO_ODDS_MAX,
                "min_confidence": self.odds_categories.TWO_ODDS_MIN_CONFIDENCE,
                "limit": self.odds_categories.TWO_ODDS_LIMIT,
                "target_combined_odds": self.odds_categories.TWO_ODDS_TARGET
            },
            "5_odds": {
                "min_odds": self.odds_categories.FIVE_ODDS_MIN,
                "max_odds": self.odds_categories.FIVE_ODDS_MAX,
                "min_confidence": self.odds_categories.FIVE_ODDS_MIN_CONFIDENCE,
                "limit": self.odds_categories.FIVE_ODDS_LIMIT,
                "target_combined_odds": self.odds_categories.FIVE_ODDS_TARGET
            },
            "10_odds": {
                "min_odds": self.odds_categories.TEN_ODDS_MIN,
                "max_odds": self.odds_categories.TEN_ODDS_MAX,
                "min_confidence": self.odds_categories.TEN_ODDS_MIN_CONFIDENCE,
                "limit": self.odds_categories.TEN_ODDS_LIMIT,
                "target_combined_odds": self.odds_categories.TEN_ODDS_TARGET
            },
            "rollover": {
                "min_odds": self.odds_categories.ROLLOVER_MIN,
                "max_odds": self.odds_categories.ROLLOVER_MAX,
                "min_confidence": self.odds_categories.ROLLOVER_MIN_CONFIDENCE,
                "target_combined_odds": self.odds_categories.ROLLOVER_TARGET,
                "days": self.odds_categories.ROLLOVER_DAYS
            }
        }

# Create a singleton instance
settings = Settings()

# For backward compatibility
FOOTBALL_DATA_KEY = settings.football_data.API_KEY
FOOTBALL_DATA_BASE_URL = settings.football_data.BASE_URL
FOOTBALL_DATA_DEFAULT_COMPETITIONS = settings.football_data.DEFAULT_COMPETITIONS

API_FOOTBALL_KEY = settings.api_football.API_KEY
API_FOOTBALL_HOST = settings.api_football.API_HOST
API_FOOTBALL_BASE_URL = settings.api_football.BASE_URL

MODEL_DIR = settings.ml.MODEL_DIR
DATA_DIR = settings.ml.DATA_DIR
CACHE_DIR = settings.ml.CACHE_DIR
