"""
Caching - Request and result caching for performance

Implements LRU caching for frequently accessed operations.
"""

import hashlib
import json
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def _make_cache_key(args: tuple, kwargs: dict) -> str:
    """Create a cache key from function arguments.

    Args:
        args: Positional arguments
        kwargs: Keyword arguments

    Returns:
        Cache key string
    """
    # Convert args and kwargs to a JSON-serializable format
    key_data = {
        "args": [str(arg) for arg in args],
        "kwargs": {k: str(v) for k, v in kwargs.items()},
    }
    key_str = json.dumps(key_data, sort_keys=True)
    # Use hash to keep key length reasonable
    return hashlib.md5(key_str.encode()).hexdigest()


class QueryCache:
    """LRU cache for retrieval queries.

    Caches search results to avoid repeated vector store lookups
    for identical queries within a time window.
    """

    def __init__(self, maxsize: int = 128):
        """Initialize query cache.

        Args:
            maxsize: Maximum number of cached queries
        """
        self.maxsize = maxsize
        self._cache: Dict[str, Any] = {}
        self._access_order: List[str] = []

    def get(self, query: str, top_k: int) -> Optional[List[Dict[str, Any]]]:
        """Get cached results for a query.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Cached results or None if not found
        """
        key = f"{query}:{top_k}"
        if key in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(key)
            self._access_order.append(key)
            logger.debug(f"Query cache hit for: {query[:50]}...")
            return self._cache[key]
        return None

    def set(self, query: str, top_k: int, results: List[Dict[str, Any]]) -> None:
        """Cache results for a query.

        Args:
            query: Search query
            top_k: Number of results
            results: Results to cache
        """
        key = f"{query}:{top_k}"

        # Remove oldest entry if cache is full
        if len(self._cache) >= self.maxsize and key not in self._cache:
            oldest_key = self._access_order.pop(0)
            del self._cache[oldest_key]
            logger.debug(f"Evicted oldest cache entry")

        self._cache[key] = results
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
        logger.debug(f"Cached results for query: {query[:50]}...")

    def clear(self) -> None:
        """Clear all cached results."""
        self._cache.clear()
        self._access_order.clear()
        logger.info("Query cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "utilization": len(self._cache) / self.maxsize,
        }


# Global query cache instance
_query_cache = QueryCache(maxsize=128)


def get_query_cache() -> QueryCache:
    """Get the global query cache instance.

    Returns:
        QueryCache instance
    """
    return _query_cache


def clear_query_cache() -> None:
    """Clear the global query cache."""
    _query_cache.clear()


def get_cache_stats() -> Dict[str, Any]:
    """Get cache statistics.

    Returns:
        Dictionary with cache stats
    """
    return _query_cache.stats()


class CachedRetrieval:
    """Wrapper for cached retrieval operations."""

    def __init__(self, search_fn: Callable) -> None:
        """Initialize cached retrieval.

        Args:
            search_fn: The search function to wrap
        """
        self.search_fn = search_fn
        self.cache = _query_cache

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search with caching.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            Search results
        """
        # Check cache first
        cached_results = self.cache.get(query, top_k)
        if cached_results is not None:
            return cached_results

        # Perform search
        results = self.search_fn(query, top_k)

        # Cache results
        self.cache.set(query, top_k, results)

        return results
