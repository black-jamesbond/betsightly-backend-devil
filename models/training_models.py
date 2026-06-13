"""
Training Pipeline Database Models

This module defines SQLAlchemy models for the training pipeline,
prediction caching, and performance monitoring.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from datetime import datetime
import json

from database import Base

class CachedPrediction(Base):
    """
    Cached daily predictions for fast API responses.
    Stores ML predictions generated in batch to avoid real-time inference.
    """
    __tablename__ = "cached_predictions_v2"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(String, nullable=False, index=True)  # YYYY-MM-DD
    category = Column(String, nullable=False, index=True)  # 2_odds, 5_odds, 10_odds, rollover
    
    # Match details
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String, nullable=False)
    fixture_id = Column(String)
    match_time = Column(String)
    
    # Prediction details
    prediction = Column(String, nullable=False)  # Main prediction (e.g., "Home Win", "Over 2.5")
    confidence = Column(Float, nullable=False)   # Confidence score (0-100)
    odds = Column(Float, nullable=False)         # Calculated odds
    
    # ML Model metadata
    models_used = Column(JSON)                   # List of models that contributed
    service_used = Column(String, default="advanced_prediction_service")
    
    # Advanced features
    model_predictions = Column(JSON)             # Individual model predictions
    explanations = Column(JSON)                  # SHAP/LIME explanations
    advanced_features = Column(JSON)             # Meta-stacking, ensemble info
    
    # Timestamps
    generated_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    is_stale = Column(Boolean, default=False)
    
    # Indexes for fast querying
    __table_args__ = (
        Index('idx_cached_pred_date_category', 'prediction_date', 'category'),
        Index('idx_cached_pred_generated', 'generated_at'),
        Index('idx_cached_pred_expires', 'expires_at'),
    )

class PredictionBatch(Base):
    """
    Tracks daily prediction generation batches.
    Monitors when predictions were generated and their status.
    """
    __tablename__ = "prediction_batches"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    batch_date = Column(String, nullable=False, unique=True, index=True)  # YYYY-MM-DD
    
    # Generation status
    status = Column(String, nullable=False)      # 'running', 'completed', 'failed', 'partial'
    
    # Statistics
    fixtures_fetched = Column(Integer, default=0)
    predictions_generated = Column(Integer, default=0)
    categories_populated = Column(JSON)          # List of categories with predictions
    
    # Performance metrics
    generation_time_seconds = Column(Float, default=0.0)
    api_source = Column(String)                  # 'football-data', 'api-football', 'hybrid'
    
    # ML Model info
    models_loaded = Column(Integer, default=0)
    service_used = Column(String)
    advanced_features_used = Column(JSON)
    
    # Error handling
    error_message = Column(Text)
    warnings = Column(JSON)                      # List of warnings during generation
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'batch_date': self.batch_date,
            'status': self.status,
            'fixtures_fetched': self.fixtures_fetched,
            'predictions_generated': self.predictions_generated,
            'categories_populated': self.categories_populated,
            'generation_time_seconds': self.generation_time_seconds,
            'models_loaded': self.models_loaded,
            'service_used': self.service_used,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None
        }

class ModelTrainingRun(Base):
    """
    Tracks model training runs for continuous learning.
    Monitors training performance and model improvements.
    """
    __tablename__ = "model_training_runs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Training details
    training_date = Column(String, nullable=False, index=True)  # YYYY-MM-DD
    training_type = Column(String, nullable=False)  # 'scheduled', 'manual', 'triggered'
    trigger_reason = Column(String)                 # 'accuracy_drop', 'weekly_schedule', 'manual_request'
    
    # Training data
    training_data_size = Column(Integer, default=0)
    new_matches_added = Column(Integer, default=0)
    data_source = Column(String)                    # 'github_dataset', 'live_results', 'hybrid'
    
    # Models trained
    models_trained = Column(JSON)                   # List of model names
    training_algorithms = Column(JSON)              # XGBoost, LightGBM, etc.
    
    # Performance metrics
    training_time_seconds = Column(Float, default=0.0)
    
    # Model performance (before/after)
    previous_accuracy = Column(JSON)                # Accuracy before training
    new_accuracy = Column(JSON)                     # Accuracy after training
    performance_improvement = Column(JSON)          # Improvement metrics
    
    # Deployment status
    deployment_status = Column(String, default='pending')  # 'pending', 'deployed', 'rejected'
    deployment_reason = Column(String)
    
    # Error handling
    status = Column(String, nullable=False)         # 'running', 'completed', 'failed'
    error_message = Column(Text)
    warnings = Column(JSON)
    
    # Timestamps
    started_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime)
    deployed_at = Column(DateTime)
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'id': self.id,
            'training_date': self.training_date,
            'training_type': self.training_type,
            'trigger_reason': self.trigger_reason,
            'training_data_size': self.training_data_size,
            'new_matches_added': self.new_matches_added,
            'models_trained': self.models_trained,
            'training_time_seconds': self.training_time_seconds,
            'previous_accuracy': self.previous_accuracy,
            'new_accuracy': self.new_accuracy,
            'performance_improvement': self.performance_improvement,
            'deployment_status': self.deployment_status,
            'status': self.status,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'deployed_at': self.deployed_at.isoformat() if self.deployed_at else None
        }

class PredictionAccuracy(Base):
    """
    Tracks prediction accuracy for continuous monitoring.
    Compares predictions with actual match results.
    """
    __tablename__ = "prediction_accuracy"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Match details
    match_date = Column(String, nullable=False, index=True)
    home_team = Column(String, nullable=False)
    away_team = Column(String, nullable=False)
    league = Column(String, nullable=False)
    
    # Prediction details
    prediction_type = Column(String, nullable=False)  # 'match_result', 'over_under', 'btts'
    predicted_outcome = Column(String, nullable=False)
    confidence = Column(Float, nullable=False)
    odds = Column(Float, nullable=False)
    
    # Actual results
    actual_outcome = Column(String)
    home_score = Column(Integer)
    away_score = Column(Integer)
    
    # Accuracy metrics
    is_correct = Column(Boolean)
    accuracy_score = Column(Float)  # Weighted by confidence
    
    # Model info
    model_used = Column(String, nullable=False)
    service_used = Column(String, nullable=False)
    
    # Timestamps
    predicted_at = Column(DateTime, nullable=False)
    result_recorded_at = Column(DateTime)
    
    __table_args__ = (
        Index('idx_accuracy_date_model', 'match_date', 'model_used'),
        Index('idx_accuracy_prediction_type', 'prediction_type'),
    )

class CacheStatus(Base):
    """
    Tracks cache status and health metrics.
    Monitors cache performance and freshness.
    """
    __tablename__ = "cache_status"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    
    # Cache details
    cache_date = Column(String, nullable=False, unique=True, index=True)  # YYYY-MM-DD
    
    # Status metrics
    total_predictions = Column(Integer, default=0)
    predictions_by_category = Column(JSON)       # Count per category
    cache_hit_rate = Column(Float, default=0.0)  # Percentage of requests served from cache
    
    # Performance metrics
    average_response_time_ms = Column(Float, default=0.0)
    cache_size_mb = Column(Float, default=0.0)
    
    # Freshness indicators
    last_refresh_at = Column(DateTime)
    next_refresh_at = Column(DateTime)
    is_stale = Column(Boolean, default=False)
    
    # Health status
    health_status = Column(String, default='healthy')  # 'healthy', 'degraded', 'unhealthy'
    health_issues = Column(JSON)                       # List of issues if any
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self):
        """Convert to dictionary for API responses."""
        return {
            'cache_date': self.cache_date,
            'total_predictions': self.total_predictions,
            'predictions_by_category': self.predictions_by_category,
            'cache_hit_rate': self.cache_hit_rate,
            'average_response_time_ms': self.average_response_time_ms,
            'cache_size_mb': self.cache_size_mb,
            'last_refresh_at': self.last_refresh_at.isoformat() if self.last_refresh_at else None,
            'next_refresh_at': self.next_refresh_at.isoformat() if self.next_refresh_at else None,
            'is_stale': self.is_stale,
            'health_status': self.health_status,
            'health_issues': self.health_issues,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }
