"""
Daily Prediction Caching Service

This service handles daily batch prediction generation and caching
to optimize API performance and reduce ML inference load.
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_

from database import SessionLocal
from models.training_models import CachedPrediction, PredictionBatch, CacheStatus
from services.advanced_prediction_service import advanced_prediction_service
from utils.common import setup_logging

# Set up logging
logger = setup_logging(__name__)

class DailyPredictionCache:
    """
    Manages daily prediction caching for optimal API performance.
    
    Features:
    - Daily batch prediction generation
    - Database caching with expiration
    - Cache status monitoring
    - Fallback to real-time predictions
    - Performance optimization
    """
    
    def __init__(self):
        """Initialize the daily prediction cache service."""
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
        self.max_predictions_per_category = 10  # Limit predictions per category
        
    def generate_daily_predictions(self, date_str: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate and cache predictions for a specific date.
        
        Args:
            date_str: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with generation results and statistics
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        start_time = datetime.now()
        
        # Create database session
        db = SessionLocal()
        
        try:
            # Check if batch already exists
            existing_batch = db.query(PredictionBatch).filter(
                PredictionBatch.batch_date == date_str
            ).first()
            
            if existing_batch and existing_batch.status == 'completed':
                logger.info(f"Predictions for {date_str} already exist and are complete")
                return self._get_batch_summary(existing_batch)
            
            # Create or update batch record
            if existing_batch:
                batch = existing_batch
                batch.status = 'running'
                batch.started_at = start_time
            else:
                batch = PredictionBatch(
                    batch_date=date_str,
                    status='running',
                    started_at=start_time
                )
                db.add(batch)
            
            db.commit()
            
            # Generate predictions using advanced ML service
            logger.info(f"🚀 Starting daily prediction generation for {date_str}")
            
            predictions_result = advanced_prediction_service.get_predictions_for_date(date_str)
            
            if predictions_result.get('status') != 'success':
                raise Exception(f"Failed to generate predictions: {predictions_result.get('message', 'Unknown error')}")
            
            # Extract predictions and metadata
            predictions = predictions_result.get('predictions', [])
            categories = predictions_result.get('categories', {})
            metadata = predictions_result.get('metadata', {})
            
            # Cache predictions in database
            cached_count = self._cache_predictions(db, date_str, categories, metadata)
            
            # Update batch record
            generation_time = (datetime.now() - start_time).total_seconds()
            
            batch.status = 'completed'
            batch.fixtures_fetched = metadata.get('total_fixtures', 0)
            batch.predictions_generated = cached_count
            batch.categories_populated = list(categories.keys())
            batch.generation_time_seconds = generation_time
            batch.api_source = metadata.get('data_source', 'unknown')
            batch.models_loaded = len(metadata.get('models_used', []))
            batch.service_used = metadata.get('service', 'unknown')
            batch.advanced_features_used = metadata.get('advanced_features', {})
            batch.completed_at = datetime.now()
            
            db.commit()
            
            # Update cache status
            self._update_cache_status(db, date_str, cached_count, categories)
            
            logger.info(f"✅ Daily prediction generation completed for {date_str}")
            logger.info(f"📊 Generated {cached_count} predictions in {generation_time:.2f} seconds")
            
            return {
                'status': 'success',
                'date': date_str,
                'predictions_cached': cached_count,
                'categories_populated': list(categories.keys()),
                'generation_time_seconds': generation_time,
                'batch_id': batch.id,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Error generating daily predictions: {str(e)}")
            
            # Update batch with error
            if 'batch' in locals():
                batch.status = 'failed'
                batch.error_message = str(e)
                batch.completed_at = datetime.now()
                db.commit()
            
            return {
                'status': 'error',
                'date': date_str,
                'error': str(e),
                'predictions_cached': 0
            }
            
        finally:
            db.close()
    
    def _cache_predictions(self, db: Session, date_str: str, categories: Dict[str, List], 
                          metadata: Dict[str, Any]) -> int:
        """Cache predictions in the database."""
        cached_count = 0
        expiry_time = datetime.now() + timedelta(hours=self.cache_expiry_hours)
        
        # Clear existing cached predictions for this date
        db.query(CachedPrediction).filter(
            CachedPrediction.prediction_date == date_str
        ).delete()
        
        # Cache predictions by category
        for category, predictions in categories.items():
            for prediction in predictions[:self.max_predictions_per_category]:
                try:
                    cached_pred = CachedPrediction(
                        prediction_date=date_str,
                        category=category,
                        home_team=prediction.get('home_team', ''),
                        away_team=prediction.get('away_team', ''),
                        league=prediction.get('league', ''),
                        fixture_id=prediction.get('fixture_id', ''),
                        match_time=prediction.get('match_time', ''),
                        prediction=prediction.get('prediction', ''),
                        confidence=prediction.get('confidence', 0.0),
                        odds=prediction.get('odds', 0.0),
                        models_used=prediction.get('model_predictions', {}).keys() if prediction.get('model_predictions') else [],
                        service_used=metadata.get('service', 'advanced_prediction_service'),
                        model_predictions=prediction.get('model_predictions', {}),
                        explanations=prediction.get('explanations', {}),
                        advanced_features=prediction.get('advanced_features', {}),
                        expires_at=expiry_time
                    )
                    
                    db.add(cached_pred)
                    cached_count += 1
                    
                except Exception as e:
                    logger.error(f"Error caching prediction: {str(e)}")
                    continue
        
        db.commit()
        return cached_count
    
    def get_cached_predictions(self, date_str: Optional[str] = None, 
                             category: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieve cached predictions from database.
        
        Args:
            date_str: Date in YYYY-MM-DD format (default: today)
            category: Specific category to retrieve (optional)
            
        Returns:
            Dictionary with cached predictions or fallback to real-time
        """
        if not date_str:
            date_str = datetime.now().strftime("%Y-%m-%d")
        
        db = SessionLocal()
        
        try:
            # Build query
            query = db.query(CachedPrediction).filter(
                and_(
                    CachedPrediction.prediction_date == date_str,
                    CachedPrediction.expires_at > datetime.now(),
                    CachedPrediction.is_stale == False
                )
            )
            
            if category:
                query = query.filter(CachedPrediction.category == category)
            
            cached_predictions = query.all()
            
            if not cached_predictions:
                logger.warning(f"No cached predictions found for {date_str}, falling back to real-time")
                return self._fallback_to_realtime(date_str, category)
            
            # Organize predictions by category
            categories = {}
            for pred in cached_predictions:
                if pred.category not in categories:
                    categories[pred.category] = []
                
                categories[pred.category].append({
                    'home_team': pred.home_team,
                    'away_team': pred.away_team,
                    'league': pred.league,
                    'prediction': pred.prediction,
                    'confidence': pred.confidence,
                    'odds': pred.odds,
                    'model_predictions': pred.model_predictions,
                    'explanations': pred.explanations,
                    'advanced_features': pred.advanced_features,
                    'cached_at': pred.generated_at.isoformat()
                })
            
            # Get batch info
            batch = db.query(PredictionBatch).filter(
                PredictionBatch.batch_date == date_str
            ).first()
            
            return {
                'status': 'success',
                'date': date_str,
                'source': 'cache',
                'categories': categories,
                'metadata': {
                    'service': 'daily_prediction_cache',
                    'cache_hit': True,
                    'total_predictions': len(cached_predictions),
                    'batch_info': batch.to_dict() if batch else None,
                    'cached_at': cached_predictions[0].generated_at.isoformat() if cached_predictions else None
                }
            }
            
        except Exception as e:
            logger.error(f"Error retrieving cached predictions: {str(e)}")
            return self._fallback_to_realtime(date_str, category)
            
        finally:
            db.close()
    
    def _fallback_to_realtime(self, date_str: str, category: Optional[str] = None) -> Dict[str, Any]:
        """Fallback to real-time prediction generation."""
        logger.info(f"🔄 Falling back to real-time predictions for {date_str}")
        
        try:
            # Use advanced prediction service for real-time generation
            result = advanced_prediction_service.get_predictions_for_date(date_str)
            
            if result.get('status') == 'success':
                # Filter by category if specified
                if category and 'categories' in result:
                    filtered_categories = {category: result['categories'].get(category, [])}
                    result['categories'] = filtered_categories
                
                # Update metadata to indicate real-time source
                if 'metadata' in result:
                    result['metadata']['cache_hit'] = False
                    result['metadata']['source'] = 'real_time_fallback'
                
                return result
            else:
                return {
                    'status': 'error',
                    'date': date_str,
                    'message': 'Both cache and real-time prediction failed',
                    'categories': {'2_odds': [], '5_odds': [], '10_odds': [], 'over_1_5': [], 'rollover': []}
                }
                
        except Exception as e:
            logger.error(f"Real-time fallback failed: {str(e)}")
            return {
                'status': 'error',
                'date': date_str,
                'error': str(e),
                'categories': {'2_odds': [], '5_odds': [], '10_odds': [], 'over_1_5': [], 'rollover': []}
            }
    
    def _update_cache_status(self, db: Session, date_str: str, cached_count: int, 
                           categories: Dict[str, List]) -> None:
        """Update cache status metrics."""
        try:
            # Count predictions by category
            predictions_by_category = {cat: len(preds) for cat, preds in categories.items()}
            
            # Create or update cache status
            cache_status = db.query(CacheStatus).filter(
                CacheStatus.cache_date == date_str
            ).first()
            
            if cache_status:
                cache_status.total_predictions = cached_count
                cache_status.predictions_by_category = predictions_by_category
                cache_status.last_refresh_at = datetime.now()
                cache_status.next_refresh_at = datetime.now() + timedelta(hours=24)
                cache_status.is_stale = False
                cache_status.health_status = 'healthy'
                cache_status.updated_at = datetime.now()
            else:
                cache_status = CacheStatus(
                    cache_date=date_str,
                    total_predictions=cached_count,
                    predictions_by_category=predictions_by_category,
                    last_refresh_at=datetime.now(),
                    next_refresh_at=datetime.now() + timedelta(hours=24),
                    is_stale=False,
                    health_status='healthy'
                )
                db.add(cache_status)
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating cache status: {str(e)}")
    
    def _get_batch_summary(self, batch: PredictionBatch) -> Dict[str, Any]:
        """Get summary of existing batch."""
        return {
            'status': 'success',
            'date': batch.batch_date,
            'predictions_cached': batch.predictions_generated,
            'categories_populated': batch.categories_populated,
            'generation_time_seconds': batch.generation_time_seconds,
            'batch_id': batch.id,
            'message': 'Using existing cached predictions'
        }

# Create singleton instance
daily_prediction_cache = DailyPredictionCache()
