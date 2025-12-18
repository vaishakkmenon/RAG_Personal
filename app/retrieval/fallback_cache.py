"""
Fallback cache for retrieval results.

Provides a fallback mechanism when ChromaDB is unavailable by caching
successful retrieval results and returning them for similar queries.

ARCHITECTURE:
=============
- In-memory LRU cache for quick access
- Fuzzy query matching for similar questions
- Automatic expiration of old entries
- Thread-safe operations

This ensures the application can continue serving cached responses
even when the primary vector database is unavailable.
"""

import logging
import threading
import time
from collections import OrderedDict
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


class RetrievalFallbackCache:
    """
    LRU cache for retrieval results with fuzzy query matching.

    When ChromaDB is unavailable, this cache can provide results
    for similar queries based on past successful retrievals.
    """

    def __init__(
        self,
        max_size: int = 100,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.8,
    ):
        """
        Initialize the fallback cache.

        Args:
            max_size: Maximum number of cached queries
            ttl_seconds: Time-to-live for cache entries (default: 1 hour)
            similarity_threshold: Minimum similarity score for fuzzy matching (0-1)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold

        # LRU cache: query -> (results, timestamp)
        self._cache: OrderedDict[str, tuple[List[Dict[str, Any]], float]] = (
            OrderedDict()
        )
        self._lock = threading.RLock()

        # Statistics
        self._hits = 0
        self._misses = 0
        self._fallback_uses = 0

        logger.info(
            f"Initialized retrieval fallback cache: "
            f"max_size={max_size}, ttl={ttl_seconds}s, "
            f"similarity_threshold={similarity_threshold}"
        )

    def cache_results(self, query: str, results: List[Dict[str, Any]]) -> None:
        """
        Cache retrieval results for a query.

        Args:
            query: The search query
            results: List of retrieval results (chunks)
        """
        if not results:
            return  # Don't cache empty results

        normalized_query = self._normalize_query(query)
        current_time = time.time()

        with self._lock:
            # If cache is full, remove oldest entry
            if len(self._cache) >= self.max_size:
                self._cache.popitem(last=False)

            # Add or update entry (move to end for LRU)
            self._cache[normalized_query] = (results, current_time)
            self._cache.move_to_end(normalized_query)

        logger.debug(f"Cached {len(results)} results for query: {query[:50]}...")

    def get_cached_results(
        self, query: str, use_fuzzy_match: bool = True
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get cached results for a query.

        Args:
            query: The search query
            use_fuzzy_match: If True, use fuzzy matching for similar queries

        Returns:
            Cached results if found and not expired, None otherwise
        """
        normalized_query = self._normalize_query(query)
        current_time = time.time()

        with self._lock:
            # Try exact match first
            if normalized_query in self._cache:
                results, timestamp = self._cache[normalized_query]

                # Check if expired
                if current_time - timestamp > self.ttl_seconds:
                    del self._cache[normalized_query]
                    self._misses += 1
                    logger.debug(f"Cache entry expired for query: {query[:50]}...")
                    return None

                # Move to end (LRU)
                self._cache.move_to_end(normalized_query)
                self._hits += 1
                logger.debug(f"Cache hit (exact) for query: {query[:50]}...")
                return results

            # Try fuzzy match if enabled
            if use_fuzzy_match:
                best_match = self._find_similar_query(normalized_query, current_time)
                if best_match:
                    results, _ = self._cache[best_match]
                    self._cache.move_to_end(best_match)
                    self._hits += 1
                    self._fallback_uses += 1
                    logger.info(
                        f"Cache hit (fuzzy) for query: {query[:50]}... "
                        f"(matched: {best_match[:50]}...)"
                    )
                    return results

            self._misses += 1
            return None

    def get_fallback_results(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Get fallback results when ChromaDB is unavailable.

        This method always uses fuzzy matching and logs that
        fallback mode is being used.

        Args:
            query: The search query

        Returns:
            Cached results from similar query if available
        """
        results = self.get_cached_results(query, use_fuzzy_match=True)

        if results:
            logger.warning(
                f"[FALLBACK] Using cached results for query: {query[:50]}... "
                "(ChromaDB unavailable)"
            )
        else:
            logger.warning(
                f"[FALLBACK] No cached results available for query: {query[:50]}..."
            )

        return results

    def clear(self) -> None:
        """Clear all cached entries."""
        with self._lock:
            self._cache.clear()
            logger.info("Cleared retrieval fallback cache")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

            return {
                "cache_size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "fallback_uses": self._fallback_uses,
                "ttl_seconds": self.ttl_seconds,
            }

    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        return query.lower().strip()

    def _find_similar_query(self, query: str, current_time: float) -> Optional[str]:
        """
        Find the most similar query in the cache using fuzzy matching.

        Args:
            query: The normalized query to match
            current_time: Current timestamp for expiry checking

        Returns:
            Best matching cached query if similarity above threshold
        """
        best_match = None
        best_score = 0.0

        for cached_query, (_, timestamp) in self._cache.items():
            # Skip expired entries
            if current_time - timestamp > self.ttl_seconds:
                continue

            # Calculate similarity
            similarity = SequenceMatcher(None, query, cached_query).ratio()

            if similarity > best_score and similarity >= self.similarity_threshold:
                best_score = similarity
                best_match = cached_query

        return best_match

    def cleanup_expired(self) -> int:
        """
        Remove expired entries from cache.

        Returns:
            Number of entries removed
        """
        current_time = time.time()
        expired_keys = []

        with self._lock:
            for query, (_, timestamp) in self._cache.items():
                if current_time - timestamp > self.ttl_seconds:
                    expired_keys.append(query)

            for key in expired_keys:
                del self._cache[key]

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

        return len(expired_keys)


# Global fallback cache instance
_fallback_cache: Optional[RetrievalFallbackCache] = None


def get_fallback_cache() -> RetrievalFallbackCache:
    """
    Get the global fallback cache instance.

    Returns:
        RetrievalFallbackCache singleton
    """
    global _fallback_cache

    if _fallback_cache is None:
        _fallback_cache = RetrievalFallbackCache(
            max_size=100,  # Cache up to 100 queries
            ttl_seconds=3600,  # 1 hour TTL
            similarity_threshold=0.8,  # 80% similarity for fuzzy match
        )

    return _fallback_cache
