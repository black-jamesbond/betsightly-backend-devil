"""
Redis-backed cache with in-memory fallback.

Usage
-----
from utils.redis_cache import cache

# Set a value (TTL in seconds)
cache.set("key", {"data": 123}, ttl=300)

# Get a value (returns None on miss)
val = cache.get("key")

# Delete
cache.delete("key")

# Decorator for functions
@cache.cached(ttl=600, key_fn=lambda date: f"predictions:{date}")
def get_predictions(date): ...

Configuration
-------------
Set REDIS_URL in the environment to enable Redis (e.g. redis://localhost:6379).
Without it the module falls back to an in-process dict cache that resets on restart
and is not shared across workers.
"""

from __future__ import annotations

import json
import logging
import os
import time
from functools import wraps
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------


class _InMemoryBackend:
    """Thread-unsafe in-process dict cache. Fine for single-worker dev."""

    def __init__(self):
        self._store: dict[str, tuple[Any, float]] = {}  # key → (value, expires_at)

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, expires_at = entry
        if expires_at and time.time() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        expires_at = time.time() + ttl if ttl else 0
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        self._store.pop(key, None)

    def flush(self) -> None:
        self._store.clear()

    @property
    def backend_name(self) -> str:
        return "in-memory"


class _RedisBackend:
    """Redis-backed cache using the redis-py client."""

    def __init__(self, url: str):
        import redis  # imported lazily so redis is optional
        self._client = redis.from_url(url, decode_responses=True)
        self._client.ping()  # fail fast if connection is broken
        logger.info(f"Redis cache connected: {url}")

    def get(self, key: str) -> Optional[Any]:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            return raw

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        serialized = json.dumps(value, default=str)
        if ttl:
            self._client.setex(key, ttl, serialized)
        else:
            self._client.set(key, serialized)

    def delete(self, key: str) -> None:
        self._client.delete(key)

    def flush(self) -> None:
        self._client.flushdb()

    @property
    def backend_name(self) -> str:
        return "redis"


# ---------------------------------------------------------------------------
# Unified cache interface
# ---------------------------------------------------------------------------


class Cache:
    """Single cache interface backed by Redis or in-memory."""

    def __init__(self):
        self._backend = self._connect()

    def _connect(self):
        redis_url = os.getenv("REDIS_URL", "")
        if redis_url:
            try:
                return _RedisBackend(redis_url)
            except Exception as e:
                logger.warning(f"Redis unavailable ({e}), falling back to in-memory cache")
        return _InMemoryBackend()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def get(self, key: str) -> Optional[Any]:
        return self._backend.get(key)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        self._backend.set(key, value, ttl)

    def delete(self, key: str) -> None:
        self._backend.delete(key)

    def flush(self) -> None:
        self._backend.flush()

    @property
    def backend(self) -> str:
        return self._backend.backend_name

    # ------------------------------------------------------------------
    # Decorator
    # ------------------------------------------------------------------

    def cached(
        self,
        ttl: int = 300,
        key_fn: Optional[Callable[..., str]] = None,
    ):
        """
        Decorator that caches the return value of a function.

        Parameters
        ----------
        ttl     Cache TTL in seconds.
        key_fn  Callable that receives the same args/kwargs as the decorated
                function and returns a cache key string. If omitted, the
                function name is used (i.e. one entry for all call combos).
        """

        def decorator(fn: Callable) -> Callable:
            @wraps(fn)
            def wrapper(*args, **kwargs):
                key = key_fn(*args, **kwargs) if key_fn else fn.__name__
                cached_val = self.get(key)
                if cached_val is not None:
                    return cached_val
                result = fn(*args, **kwargs)
                if result is not None:
                    self.set(key, result, ttl)
                return result

            return wrapper

        return decorator


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

cache = Cache()
