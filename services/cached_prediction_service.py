"""
Cached Prediction Service

High-performance prediction service with intelligent caching and background refresh.
Reduces API response time from 2-3 seconds to <500ms by serving cached predictions.

Features:
- Database-backed prediction caching
- Automatic cache expiration (45 minutes)
- Background refresh every 30 minutes
- Graceful degradation when generation fails
- Comprehensive logging and monitoring
"""

import json
import logging
import sqlite3
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from threading import Thread, Lock
import schedule

from services.quick_prediction_service import QuickPredictionService
from database import DB_FILE
from utils.config import settings

# Set up logging
logger = logging.getLogger(__name__)

class CachedPredictionService:
    """
    High-performance cached prediction service.
    
    Provides sub-500ms API responses by serving cached predictions
    while maintaining data freshness through background updates.
    """
    
    def __init__(self):
        """Initialize the cached prediction service."""
        self.prediction_service = QuickPredictionService()
        self.cache_lock = Lock()
        self.background_refresh_enabled = True
        
        # Cache configuration
        self.CACHE_DURATION_MINUTES = 45  # Cache expires after 45 minutes
        self.REFRESH_INTERVAL_MINUTES = 30  # Background refresh every 30 minutes
        self.STALE_THRESHOLD_MINUTES = 60  # Mark as stale after 1 hour
        
        # Initialize background scheduler
        self._setup_background_refresh()
        
        logger.info("Cached Prediction Service initialized")
    
    def get_predictions_for_date(self, date: str = None) -> Dict[str, Any]:
        """
        Get predictions with intelligent caching.
        
        Args:
            date: Date in YYYY-MM-DD format (default: today)
            
        Returns:
            Dictionary with predictions and categories (fast response)
        """
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")
        
        start_time = time.time()
        
        try:
            # Try to get from cache first
            cached_result = self._get_cached_predictions(date)
            
            if cached_result and not self._is_cache_expired(cached_result):
                # Serve from cache (fast path)
                response_time = (time.time() - start_time) * 1000
                logger.info(f"⚡ Served cached predictions in {response_time:.0f}ms")
                return self._format_cached_response(cached_result, date)
            
            # Cache miss or expired - generate fresh predictions
            logger.info(f"🔄 Cache miss/expired for {date}, generating fresh predictions")
            fresh_result = self._generate_and_cache_predictions(date)
            
            response_time = (time.time() - start_time) * 1000
            logger.info(f"🆕 Generated fresh predictions in {response_time:.0f}ms")
            
            return fresh_result
            
        except Exception as e:
            # Fallback to stale cache if available
            logger.error(f"❌ Error getting predictions: {str(e)}")
            stale_result = self._get_stale_cache_fallback(date)
            
            if stale_result:
                logger.warning(f"⚠️  Serving stale cache as fallback")
                return self._format_cached_response(stale_result, date, is_stale=True)
            
            # Last resort - return empty result
            return self._empty_result(date, error=str(e))
    
    def _get_cached_predictions(self, date: str) -> Optional[Dict[str, Any]]:
        """Get cached predictions from database."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT prediction_data, generated_at, expires_at, fixture_count
                FROM cached_predictions 
                WHERE prediction_date = ? AND category = 'all'
                ORDER BY generated_at DESC 
                LIMIT 1
            """, (date,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'data': json.loads(result[0]),
                    'generated_at': result[1],
                    'expires_at': result[2],
                    'fixture_count': result[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting cached predictions: {str(e)}")
            return None
    
    def _is_cache_expired(self, cached_result: Dict[str, Any]) -> bool:
        """Check if cached result is expired."""
        try:
            expires_at = datetime.fromisoformat(cached_result['expires_at'])
            return datetime.now() > expires_at
        except Exception:
            return True
    
    def _generate_and_cache_predictions(self, date: str) -> Dict[str, Any]:
        """Generate fresh predictions and cache them."""
        generation_start = time.time()
        
        try:
            # Generate predictions using existing service
            result = self.prediction_service.get_predictions_for_date(date)
            
            if result.get('status') == 'success':
                # Cache the successful result
                self._cache_predictions(date, result)
                
                # Log generation metrics
                generation_time = time.time() - generation_start
                self._log_generation_metrics(date, 'success', result, generation_time)
                
                return result
            else:
                # Log failed generation
                self._log_generation_metrics(date, 'error', result, 
                                           time.time() - generation_start)
                return result
                
        except Exception as e:
            # Log error
            self._log_generation_metrics(date, 'error', {'error': str(e)}, 
                                       time.time() - generation_start)
            raise
    
    def _cache_predictions(self, date: str, result: Dict[str, Any]) -> None:
        """Store predictions in cache."""
        try:
            with self.cache_lock:
                conn = sqlite3.connect(DB_FILE)
                cursor = conn.cursor()
                
                # Calculate expiration time
                expires_at = datetime.now() + timedelta(minutes=self.CACHE_DURATION_MINUTES)
                
                # Count total fixtures
                fixture_count = sum(
                    len(predictions) for predictions in result.get('categories', {}).values()
                    if isinstance(predictions, list)
                )
                
                # Store in cache
                cursor.execute("""
                    INSERT OR REPLACE INTO cached_predictions 
                    (prediction_date, category, prediction_data, fixture_count, 
                     generated_at, expires_at, is_stale)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    date, 'all', json.dumps(result), fixture_count,
                    datetime.now().isoformat(), expires_at.isoformat(), 0
                ))
                
                conn.commit()
                conn.close()
                
                logger.info(f"💾 Cached predictions for {date} (expires: {expires_at.strftime('%H:%M')})")
                
        except Exception as e:
            logger.error(f"❌ Error caching predictions: {str(e)}")
    
    def _format_cached_response(self, cached_result: Dict[str, Any], date: str, 
                               is_stale: bool = False) -> Dict[str, Any]:
        """Format cached result for API response."""
        response = cached_result['data'].copy()
        
        # Add cache metadata
        response['cache_info'] = {
            'served_from_cache': True,
            'generated_at': cached_result['generated_at'],
            'is_stale': is_stale,
            'fixture_count': cached_result.get('fixture_count', 0)
        }
        
        return response
    
    def _get_stale_cache_fallback(self, date: str) -> Optional[Dict[str, Any]]:
        """Get stale cache as fallback when fresh generation fails."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            # Get any cache within stale threshold
            stale_threshold = datetime.now() - timedelta(minutes=self.STALE_THRESHOLD_MINUTES)
            
            cursor.execute("""
                SELECT prediction_data, generated_at, expires_at, fixture_count
                FROM cached_predictions 
                WHERE prediction_date = ? AND category = 'all'
                AND datetime(generated_at) > ?
                ORDER BY generated_at DESC 
                LIMIT 1
            """, (date, stale_threshold.isoformat()))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'data': json.loads(result[0]),
                    'generated_at': result[1],
                    'expires_at': result[2],
                    'fixture_count': result[3]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting stale cache: {str(e)}")
            return None
    
    def _empty_result(self, date: str, error: str = None) -> Dict[str, Any]:
        """Return empty result when all else fails."""
        return {
            "status": "error" if error else "success",
            "date": date,
            "predictions": [],
            "categories": {
                "2_odds": [],
                "5_odds": [],
                "10_odds": [],
                "rollover": []
            },
            "message": error or "No predictions available",
            "cache_info": {
                "served_from_cache": False,
                "generated_at": datetime.now().isoformat(),
                "is_stale": False,
                "fixture_count": 0
            }
        }
    
    def _log_generation_metrics(self, date: str, status: str, result: Dict[str, Any], 
                               generation_time: float) -> None:
        """Log prediction generation metrics."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            
            fixtures_fetched = 0
            predictions_generated = 0
            error_message = None
            api_source = "unknown"
            
            if status == 'success':
                categories = result.get('categories', {})
                predictions_generated = sum(
                    len(preds) for preds in categories.values() 
                    if isinstance(preds, list)
                )
                fixtures_fetched = predictions_generated
                api_source = "football-data"  # Default assumption
            else:
                error_message = result.get('error', result.get('message', 'Unknown error'))
            
            cursor.execute("""
                INSERT INTO prediction_generation_log 
                (prediction_date, status, fixtures_fetched, predictions_generated,
                 generation_time_seconds, error_message, api_source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                date, status, fixtures_fetched, predictions_generated,
                generation_time, error_message, api_source
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"❌ Error logging metrics: {str(e)}")

    def _setup_background_refresh(self) -> None:
        """Setup background refresh scheduler."""
        try:
            # Schedule background refresh every 30 minutes
            schedule.every(self.REFRESH_INTERVAL_MINUTES).minutes.do(self._background_refresh_job)

            # Start scheduler thread
            scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()

            logger.info(f"🔄 Background refresh scheduled every {self.REFRESH_INTERVAL_MINUTES} minutes")

        except Exception as e:
            logger.error(f"❌ Error setting up background refresh: {str(e)}")

    def _run_scheduler(self) -> None:
        """Run the background scheduler."""
        while self.background_refresh_enabled:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"❌ Scheduler error: {str(e)}")
                time.sleep(60)

    def _background_refresh_job(self) -> None:
        """Background job to refresh predictions."""
        try:
            today = datetime.now().strftime("%Y-%m-%d")
            tomorrow = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

            logger.info("🔄 Starting background prediction refresh")

            # Refresh today's predictions
            self._refresh_predictions_if_needed(today)

            # Pre-generate tomorrow's predictions
            self._refresh_predictions_if_needed(tomorrow)

            logger.info("✅ Background refresh completed")

        except Exception as e:
            logger.error(f"❌ Background refresh error: {str(e)}")

    def _refresh_predictions_if_needed(self, date: str) -> None:
        """Refresh predictions if cache is expired or missing."""
        try:
            cached_result = self._get_cached_predictions(date)

            if not cached_result or self._is_cache_expired(cached_result):
                logger.info(f"🔄 Refreshing predictions for {date}")
                self._generate_and_cache_predictions(date)
            else:
                logger.debug(f"✅ Cache still valid for {date}")

        except Exception as e:
            logger.error(f"❌ Error refreshing predictions for {date}: {str(e)}")

    def force_refresh(self, date: str = None) -> Dict[str, Any]:
        """Force refresh predictions (useful for manual cache invalidation)."""
        if not date:
            date = datetime.now().strftime("%Y-%m-%d")

        logger.info(f"🔄 Force refreshing predictions for {date}")
        return self._generate_and_cache_predictions(date)

    def get_cache_status(self) -> Dict[str, Any]:
        """Get cache status and statistics."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            # Get cache statistics
            cursor.execute("""
                SELECT
                    prediction_date,
                    fixture_count,
                    generated_at,
                    expires_at,
                    is_stale,
                    CASE
                        WHEN datetime(expires_at) > datetime('now') THEN 'valid'
                        WHEN datetime(generated_at) > datetime('now', '-60 minutes') THEN 'stale'
                        ELSE 'expired'
                    END as status
                FROM cached_predictions
                WHERE category = 'all'
                ORDER BY prediction_date DESC
                LIMIT 7
            """)

            cache_entries = []
            for row in cursor.fetchall():
                cache_entries.append({
                    'date': row[0],
                    'fixture_count': row[1],
                    'generated_at': row[2],
                    'expires_at': row[3],
                    'is_stale': bool(row[4]),
                    'status': row[5]
                })

            # Get generation statistics
            cursor.execute("""
                SELECT
                    COUNT(*) as total_generations,
                    SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) as successful,
                    AVG(generation_time_seconds) as avg_time,
                    MAX(created_at) as last_generation
                FROM prediction_generation_log
                WHERE created_at > datetime('now', '-24 hours')
            """)

            stats_row = cursor.fetchone()
            generation_stats = {
                'total_generations_24h': stats_row[0] or 0,
                'successful_generations_24h': stats_row[1] or 0,
                'avg_generation_time': round(stats_row[2] or 0, 2),
                'last_generation': stats_row[3]
            }

            conn.close()

            return {
                'cache_entries': cache_entries,
                'generation_stats': generation_stats,
                'background_refresh_enabled': self.background_refresh_enabled,
                'cache_duration_minutes': self.CACHE_DURATION_MINUTES,
                'refresh_interval_minutes': self.REFRESH_INTERVAL_MINUTES
            }

        except Exception as e:
            logger.error(f"❌ Error getting cache status: {str(e)}")
            return {'error': str(e)}

    def cleanup_old_cache(self, days_to_keep: int = 7) -> None:
        """Clean up old cache entries."""
        try:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()

            cutoff_date = datetime.now() - timedelta(days=days_to_keep)

            cursor.execute("""
                DELETE FROM cached_predictions
                WHERE datetime(generated_at) < ?
            """, (cutoff_date.isoformat(),))

            deleted_count = cursor.rowcount

            cursor.execute("""
                DELETE FROM prediction_generation_log
                WHERE datetime(created_at) < ?
            """, (cutoff_date.isoformat(),))

            conn.commit()
            conn.close()

            logger.info(f"🧹 Cleaned up {deleted_count} old cache entries")

        except Exception as e:
            logger.error(f"❌ Error cleaning up cache: {str(e)}")


# Global instance for use across the application
cached_prediction_service = CachedPredictionService()
