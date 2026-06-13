"""
Database Optimization Utilities

This module provides utilities for optimizing database queries and preventing N+1 problems.
"""

import logging
import time
import hashlib
import json
from typing import List, Dict, Any, Optional, Callable
from functools import wraps
from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import and_, or_, desc, asc, event, Engine, text
from datetime import datetime, timedelta

# Set up logging
logger = logging.getLogger(__name__)


class OptimizedQueryBuilder:
    """Builder for creating optimized database queries."""
    
    def __init__(self, session: Session):
        self.session = session
        
    def get_predictions_with_fixtures(
        self,
        date: Optional[datetime] = None,
        category: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Any]:
        """
        Get predictions with their associated fixtures in a single query.
        
        Args:
            date: Filter by date
            category: Filter by category
            limit: Limit number of results
            
        Returns:
            List of predictions with fixtures loaded
        """
        from prediction import Prediction
        from fixture import Fixture
        
        query = self.session.query(Prediction).options(
            joinedload(Prediction.fixture)
        )
        
        if date:
            start_of_day = datetime.combine(date.date(), datetime.min.time())
            end_of_day = datetime.combine(date.date(), datetime.max.time())
            query = query.join(Fixture).filter(
                and_(
                    Fixture.date >= start_of_day,
                    Fixture.date <= end_of_day
                )
            )
        
        if category:
            query = query.filter(Prediction.category == category)
            
        if limit:
            query = query.limit(limit)
            
        return query.all()
    
    def get_betting_codes_with_relations(
        self,
        skip: int = 0,
        limit: int = 100,
        punter_id: Optional[int] = None,
        bookmaker_id: Optional[int] = None,
        featured: Optional[bool] = None
    ) -> List[Any]:
        """
        Get betting codes with punter and bookmaker data in a single query.
        
        Args:
            skip: Number of records to skip
            limit: Maximum number of records to return
            punter_id: Filter by punter ID
            bookmaker_id: Filter by bookmaker ID
            featured: Filter by featured status
            
        Returns:
            List of betting codes with relations loaded
        """
        from betting_code import BettingCode
        from punter import Punter
        from bookmaker import Bookmaker
        
        query = self.session.query(BettingCode).options(
            joinedload(BettingCode.punter),
            joinedload(BettingCode.bookmaker)
        )
        
        if punter_id:
            query = query.filter(BettingCode.punter_id == punter_id)
            
        if bookmaker_id:
            query = query.filter(BettingCode.bookmaker_id == bookmaker_id)
            
        if featured is not None:
            query = query.filter(BettingCode.featured == featured)
            
        return query.offset(skip).limit(limit).all()
    
    def get_prediction_combinations_optimized(
        self,
        category: str,
        date: Optional[datetime] = None,
        limit: int = 10
    ) -> List[Any]:
        """
        Get prediction combinations with all related data in optimized queries.
        
        Args:
            category: Category to filter by
            date: Date to filter by
            limit: Maximum number of combinations to return
            
        Returns:
            List of prediction combinations with relations loaded
        """
        from prediction_combination import PredictionCombination
        
        query = self.session.query(PredictionCombination).options(
            selectinload(PredictionCombination.items)
        ).filter(PredictionCombination.category == category)
        
        if date:
            start_of_day = datetime.combine(date.date(), datetime.min.time())
            end_of_day = datetime.combine(date.date(), datetime.max.time())
            query = query.filter(
                and_(
                    PredictionCombination.created_at >= start_of_day,
                    PredictionCombination.created_at <= end_of_day
                )
            )
        
        return query.order_by(
            desc(PredictionCombination.combined_confidence)
        ).limit(limit).all()


class AdvancedDatabaseCache:
    """Advanced in-memory cache with LRU eviction, statistics, and query optimization."""

    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        self.cache = {}
        self.access_times = {}
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'sets': 0,
            'total_queries': 0
        }

    def _generate_key(self, query: str, params: Dict = None) -> str:
        """Generate a cache key from query and parameters."""
        key_data = {'query': query, 'params': params or {}}
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with LRU tracking."""
        self.stats['total_queries'] += 1

        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl_seconds:
                self.access_times[key] = time.time()
                self.stats['hits'] += 1
                return value
            else:
                # Remove expired entry
                del self.cache[key]
                if key in self.access_times:
                    del self.access_times[key]

        self.stats['misses'] += 1
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with LRU eviction."""
        # Check if we need to evict items
        if len(self.cache) >= self.max_size and key not in self.cache:
            self._evict_lru()

        self.cache[key] = (value, time.time())
        self.access_times[key] = time.time()
        self.stats['sets'] += 1

    def _evict_lru(self) -> None:
        """Evict the least recently used item."""
        if not self.access_times:
            return

        lru_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[lru_key]
        del self.access_times[lru_key]
        self.stats['evictions'] += 1

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = (self.stats['hits'] / self.stats['total_queries'] * 100) if self.stats['total_queries'] > 0 else 0

        return {
            **self.stats,
            'size': len(self.cache),
            'hit_rate': round(hit_rate, 2),
            'memory_usage_mb': self._estimate_memory_usage()
        }

    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        import sys
        total_size = 0
        for key, (value, _) in self.cache.items():
            total_size += sys.getsizeof(key) + sys.getsizeof(value)
        return round(total_size / (1024 * 1024), 2)

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()
        self.access_times.clear()

    def remove(self, key: str) -> None:
        """Remove specific key from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]


class DatabaseCache:
    """Simple in-memory cache for frequently accessed database queries (backward compatibility)."""

    def __init__(self, ttl_seconds: int = 300):  # 5 minutes default TTL
        self.cache = {}
        self.ttl_seconds = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.ttl_seconds:
                return value
            else:
                # Remove expired entry
                del self.cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set value in cache with current timestamp."""
        self.cache[key] = (value, datetime.now().timestamp())

    def clear(self) -> None:
        """Clear all cached values."""
        self.cache.clear()

    def remove(self, key: str) -> None:
        """Remove specific key from cache."""
        if key in self.cache:
            del self.cache[key]


# Global cache instances
db_cache = DatabaseCache()
advanced_cache = AdvancedDatabaseCache(ttl_seconds=300, max_size=1000)


def query_performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor database query performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time

            if execution_time > 1.0:  # Log slow queries (>1 second)
                logger.warning(f"Slow query detected: {func.__name__} took {execution_time:.2f}s")
            else:
                logger.debug(f"Query {func.__name__} executed in {execution_time:.3f}s")

            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query {func.__name__} failed after {execution_time:.3f}s: {str(e)}")
            raise
    return wrapper


def cached_query(cache_key_prefix: str = None, ttl_seconds: int = 300):
    """Decorator to automatically cache query results."""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_prefix = cache_key_prefix or func.__name__
            key_data = {
                'function': func.__name__,
                'args': str(args),
                'kwargs': str(sorted(kwargs.items()))
            }
            cache_key = advanced_cache._generate_key(str(key_data))

            # Try to get from cache
            cached_result = advanced_cache.get(cache_key)
            if cached_result is not None:
                return cached_result

            # Execute query and cache result
            result = func(*args, **kwargs)
            advanced_cache.set(cache_key, result)

            return result
        return wrapper
    return decorator


def get_cached_or_query(
    cache_key: str,
    query_func,
    *args,
    **kwargs
) -> Any:
    """
    Get data from cache or execute query and cache the result.
    
    Args:
        cache_key: Unique key for caching
        query_func: Function to execute if cache miss
        *args: Arguments for query function
        **kwargs: Keyword arguments for query function
        
    Returns:
        Query result (from cache or fresh query)
    """
    # Try to get from cache first
    cached_result = db_cache.get(cache_key)
    if cached_result is not None:
        logger.debug(f"Cache hit for key: {cache_key}")
        return cached_result
    
    # Cache miss - execute query
    logger.debug(f"Cache miss for key: {cache_key}")
    result = query_func(*args, **kwargs)
    
    # Cache the result
    db_cache.set(cache_key, result)
    
    return result


def create_database_indexes():
    """
    Create database indexes for better query performance.
    This should be run during application startup or migration.
    """
    from database import engine
    
    indexes = [
        # Predictions table indexes
        "CREATE INDEX IF NOT EXISTS idx_predictions_fixture_id ON predictions (fixture_id);",
        "CREATE INDEX IF NOT EXISTS idx_predictions_category ON predictions (category);",
        "CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON predictions (created_at);",
        
        # Fixtures table indexes
        "CREATE INDEX IF NOT EXISTS idx_fixtures_date ON fixtures (date);",
        "CREATE INDEX IF NOT EXISTS idx_fixtures_league_id ON fixtures (league_id);",
        "CREATE INDEX IF NOT EXISTS idx_fixtures_status ON fixtures (status);",
        
        # Betting codes table indexes
        "CREATE INDEX IF NOT EXISTS idx_betting_codes_punter_id ON betting_codes (punter_id);",
        "CREATE INDEX IF NOT EXISTS idx_betting_codes_bookmaker_id ON betting_codes (bookmaker_id);",
        "CREATE INDEX IF NOT EXISTS idx_betting_codes_featured ON betting_codes (featured);",
        "CREATE INDEX IF NOT EXISTS idx_betting_codes_created_at ON betting_codes (created_at);",
        
        # Prediction combinations table indexes
        "CREATE INDEX IF NOT EXISTS idx_prediction_combinations_category ON prediction_combinations (category);",
        "CREATE INDEX IF NOT EXISTS idx_prediction_combinations_confidence ON prediction_combinations (combined_confidence);",
        "CREATE INDEX IF NOT EXISTS idx_prediction_combinations_created_at ON prediction_combinations (created_at);",
        
        # Composite indexes for common query patterns
        "CREATE INDEX IF NOT EXISTS idx_predictions_fixture_category ON predictions (fixture_id, category);",
        "CREATE INDEX IF NOT EXISTS idx_fixtures_date_league ON fixtures (date, league_id);",
        "CREATE INDEX IF NOT EXISTS idx_betting_codes_punter_featured ON betting_codes (punter_id, featured);",
    ]
    
    try:
        with engine.connect() as conn:
            for index_sql in indexes:
                try:
                    conn.execute(text(index_sql))
                    logger.info(f"Created index: {index_sql}")
                except Exception as e:
                    logger.warning(f"Failed to create index: {index_sql}, Error: {str(e)}")

            conn.commit()
            logger.info("Database indexes created successfully")
            
    except Exception as e:
        logger.error(f"Error creating database indexes: {str(e)}")


def optimize_query_performance():
    """
    Apply SQLite-specific optimizations for better performance.
    """
    from database import engine
    
    optimizations = [
        "PRAGMA journal_mode = WAL;",  # Write-Ahead Logging for better concurrency
        "PRAGMA synchronous = NORMAL;",  # Balance between safety and performance
        "PRAGMA cache_size = 10000;",  # Increase cache size
        "PRAGMA temp_store = MEMORY;",  # Store temporary tables in memory
        "PRAGMA mmap_size = 268435456;",  # 256MB memory-mapped I/O
    ]
    
    try:
        with engine.connect() as conn:
            for pragma in optimizations:
                conn.execute(text(pragma))
                logger.info(f"Applied optimization: {pragma}")

            logger.info("Database optimizations applied successfully")
            
    except Exception as e:
        logger.error(f"Error applying database optimizations: {str(e)}")
