# coding=utf-8
"""
Query Cache Module with Performance Optimizations.

This module provides a thread-safe LRU cache for ClickHouse query results with:
- Thread-safe operations using RLocks
- TTL (Time-To-Live) support for cache entries
- Maximum result size limit to prevent memory bloat
- Fast MD5-based key generation
- Cache statistics tracking (hits, misses, hit rate)

Example:
    >>> from ClickSQL.utils import get_query_cache
    >>> cache = get_query_cache()
    >>> cache.set("SELECT 1", {"result": "data"})
    >>> result = cache.get("SELECT 1")
    >>> print(cache.stats())
"""
import hashlib
import threading
import time
from collections import OrderedDict
from functools import wraps
from typing import Any, Optional, Tuple

MAX_RESULT_SIZE = 10 * 1024 * 1024  # 10MB max cached result size


class QueryCache:
    """Thread-safe LRU cache for query results with optimizations."""

    def __init__(self, maxsize: int = 128, ttl: int = 300, max_result_size: int = MAX_RESULT_SIZE):
        self._cache = OrderedDict()
        self._maxsize = maxsize
        self._ttl = ttl
        self._max_result_size = max_result_size
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, sql: str, params: dict = None) -> str:
        """Fast key generation using MD5."""
        key_data = f"{sql}:{str(params or {})}".encode('utf-8')
        return hashlib.md5(key_data, usedforsecurity=False).hexdigest()

    def get(self, sql: str, params: dict = None) -> Optional[Any]:
        """Get cached result with TTL check."""
        key = self._make_key(sql, params)
        with self._lock:
            if key in self._cache:
                value, timestamp = self._cache[key]
                if time.time() - timestamp < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return value
                else:
                    del self._cache[key]
            self._misses += 1
        return None

    def set(self, sql: str, value: Any, params: dict = None) -> bool:
        """Set cache value with size limit check."""
        if value is None:
            return False
        
        try:
            import sys
            result_size = sys.getsizeof(value)
            if result_size > self._max_result_size:
                return False
        except (TypeError, ImportError):
            pass
        
        key = self._make_key(sql, params)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (value, time.time())
            if len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return True

    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def __contains__(self, sql: str) -> bool:
        """Check if SQL is in cache (without TTL check)."""
        return self._make_key(sql) in self._cache

    def __len__(self) -> int:
        return len(self._cache)
    
    def stats(self) -> dict:
        """Get cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0
        return {
            'size': len(self._cache),
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'maxsize': self._maxsize,
            'ttl': self._ttl
        }


_global_cache = QueryCache()


def get_query_cache() -> QueryCache:
    """Get the global query cache instance."""
    return _global_cache


def clear_query_cache():
    """Clear the global query cache."""
    _global_cache.clear()


def query_cache(ttl: int = 300, maxsize: int = 128, max_result_size: int = MAX_RESULT_SIZE):
    """Decorator for caching function results."""
    def decorator(func):
        cache = QueryCache(maxsize=maxsize, ttl=ttl, max_result_size=max_result_size)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if args:
                sql = args[0] if isinstance(args[0], str) else None
            else:
                sql = kwargs.get('sql')

            if sql:
                cached_result = cache.get(sql)
                if cached_result is not None:
                    return cached_result

            result = func(*args, **kwargs)

            if sql:
                cache.set(sql, result)

            return result

        wrapper.cache = cache
        wrapper.cache_clear = cache.clear
        wrapper.cache_stats = cache.stats
        return wrapper
    return decorator


if __name__ == '__main__':
    cache = QueryCache(maxsize=3, ttl=60)
    cache.set("SELECT 1", {"result": "test"})
    print(cache.get("SELECT 1"))
    print(cache.stats())
