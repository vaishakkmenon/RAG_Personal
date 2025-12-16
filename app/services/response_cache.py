"""
Response Cache Service for RAG System

Caches complete ChatResponse objects in Redis for common queries.
Target: <100ms response time for cache hits.

Features:
- Query normalization for cache key generation
- Conversation-aware caching (includes session_id in key)
- TTL-based expiration
- Prometheus metrics for hit/miss rates
- Configurable cache bypass for non-cacheable queries
"""

import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import timedelta

logger = logging.getLogger(__name__)

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - response caching disabled")


class ResponseCache:
    """
    Redis-based cache for RAG responses.

    Optimizations:
    - Connection pooling for performance
    - Query normalization to maximize cache hits
    - Separate caching for conversational vs standalone queries
    - Metrics tracking for cache effectiveness
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 3600,  # 1 hour default
        enabled: bool = True,
        max_cache_size_mb: int = 100,
    ):
        """
        Initialize response cache.

        Args:
            redis_url: Redis connection URL
            ttl_seconds: Time-to-live for cached responses
            enabled: Whether caching is enabled
            max_cache_size_mb: Maximum cache size in MB (soft limit)
        """
        self.ttl_seconds = ttl_seconds
        self.enabled = enabled
        self.max_cache_size_mb = max_cache_size_mb
        self._client: Optional[redis.Redis] = None

        if not enabled:
            logger.info("Response caching is disabled")
            return

        if not REDIS_AVAILABLE:
            logger.error("Redis not available - response caching disabled")
            self.enabled = False
            return

        try:
            # Connection pool for better performance
            pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=10,
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
                retry_on_timeout=True
            )
            self._client = redis.Redis(connection_pool=pool)
            self._client.ping()
            logger.info(f"Response cache initialized: {redis_url} (TTL: {ttl_seconds}s)")
        except Exception as e:
            logger.error(f"Failed to connect to Redis for caching: {e}")
            self.enabled = False
            return

        # Initialize Prometheus metrics
        try:
            from prometheus_client import Counter, Histogram, Gauge

            self.cache_hits = Counter(
                'response_cache_hits_total',
                'Total cache hits',
                ['session_aware']  # true/false
            )

            self.cache_misses = Counter(
                'response_cache_misses_total',
                'Total cache misses',
                ['session_aware']
            )

            self.cache_errors = Counter(
                'response_cache_errors_total',
                'Total cache errors',
                ['operation']  # get/set
            )

            self.cache_latency = Histogram(
                'response_cache_operation_duration_seconds',
                'Cache operation latency',
                ['operation']  # get/set
            )

            self.cache_size = Gauge(
                'response_cache_size_bytes',
                'Approximate cache size in bytes'
            )

            self._metrics_enabled = True
            logger.info("Cache metrics enabled")

        except ImportError:
            self._metrics_enabled = False
            logger.warning("Prometheus not available - cache metrics disabled")

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for cache key generation.

        Normalization:
        - Lowercase
        - Strip leading/trailing whitespace
        - Collapse multiple spaces
        - Remove trailing punctuation (?, !, .)

        Args:
            query: Raw user query

        Returns:
            Normalized query string
        """
        normalized = query.lower().strip()

        # Collapse multiple spaces
        normalized = ' '.join(normalized.split())

        # Remove trailing question marks, exclamation points, periods
        normalized = normalized.rstrip('?!.')

        return normalized

    def _generate_cache_key(
        self,
        question: str,
        session_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Generate cache key from query and parameters.

        Cache key includes:
        - Normalized question
        - Session ID (if provided) for conversation-aware caching
        - Critical parameters that affect the response

        Args:
            question: User question
            session_id: Optional session ID for conversation context
            params: Optional dict of parameters affecting response

        Returns:
            Cache key string
        """
        # Normalize question
        normalized_q = self._normalize_query(question)

        # Build cache key components
        key_components = [normalized_q]

        # Add session ID if provided (for conversation-aware caching)
        if session_id:
            key_components.append(f"session:{session_id}")

        # Add parameters that affect the response
        # Only include params that significantly change the answer
        if params:
            relevant_params = {
                k: params[k]
                for k in ['top_k', 'max_distance', 'temperature', 'model', 'doc_type']
                if k in params and params[k] is not None
            }
            if relevant_params:
                # Sort for consistency
                param_str = json.dumps(relevant_params, sort_keys=True)
                key_components.append(f"params:{param_str}")

        # Create hash for compact key
        key_str = "|".join(key_components)
        key_hash = hashlib.md5(key_str.encode()).hexdigest()

        # Prefix with namespace
        cache_key = f"rag:response:{key_hash}"

        return cache_key

    def get(
        self,
        question: str,
        session_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get cached response if available.

        Args:
            question: User question
            session_id: Optional session ID
            params: Optional parameters

        Returns:
            Cached response dict or None if not found
        """
        if not self.enabled or not self._client:
            return None

        start_time = time.time()
        session_aware = session_id is not None

        try:
            cache_key = self._generate_cache_key(question, session_id, params)

            # Try to get from cache
            cached_data = self._client.get(cache_key)

            if cached_data:
                # Cache hit!
                response = json.loads(cached_data)

                if self._metrics_enabled:
                    self.cache_hits.labels(session_aware=str(session_aware).lower()).inc()
                    latency = time.time() - start_time
                    self.cache_latency.labels(operation='get').observe(latency)

                logger.info(
                    f"Cache HIT: {self._normalize_query(question)[:50]}... "
                    f"(latency: {(time.time() - start_time) * 1000:.1f}ms)"
                )

                return response
            else:
                # Cache miss
                if self._metrics_enabled:
                    self.cache_misses.labels(session_aware=str(session_aware).lower()).inc()
                    latency = time.time() - start_time
                    self.cache_latency.labels(operation='get').observe(latency)

                logger.info(f"Cache MISS: {self._normalize_query(question)[:50]}...")

                return None

        except Exception as e:
            logger.error(f"Cache get error: {e}")
            if self._metrics_enabled:
                self.cache_errors.labels(operation='get').inc()
            return None

    def set(
        self,
        question: str,
        response: Dict[str, Any],
        session_id: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
        ttl_override: Optional[int] = None
    ) -> bool:
        """
        Cache a response.

        Args:
            question: User question
            response: Response dict to cache
            session_id: Optional session ID
            params: Optional parameters
            ttl_override: Optional TTL override in seconds

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.enabled or not self._client:
            return False

        start_time = time.time()

        try:
            cache_key = self._generate_cache_key(question, session_id, params)

            # Serialize response
            cached_data = json.dumps(response)

            # Set with TTL
            ttl = ttl_override if ttl_override is not None else self.ttl_seconds
            self._client.setex(
                cache_key,
                ttl,
                cached_data
            )

            if self._metrics_enabled:
                latency = time.time() - start_time
                self.cache_latency.labels(operation='set').observe(latency)

            logger.info(
                f"Cached response: {self._normalize_query(question)[:50]}... "
                f"(TTL: {ttl}s, latency: {(time.time() - start_time) * 1000:.1f}ms)"
            )

            return True

        except Exception as e:
            logger.error(f"Cache set error: {e}")
            if self._metrics_enabled:
                self.cache_errors.labels(operation='set').inc()
            return False

    def invalidate_pattern(self, pattern: str) -> int:
        """
        Invalidate all cache keys matching a pattern.

        Args:
            pattern: Redis key pattern (e.g., "rag:response:*")

        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self._client:
            return 0

        try:
            keys = list(self._client.scan_iter(match=pattern))
            if keys:
                deleted = self._client.delete(*keys)
                logger.info(f"Invalidated {deleted} cache entries matching: {pattern}")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Cache invalidation error: {e}")
            return 0

    def clear_all(self) -> bool:
        """
        Clear all cached responses.

        Returns:
            True if successful
        """
        return self.invalidate_pattern("rag:response:*") >= 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dict with cache stats
        """
        if not self.enabled or not self._client:
            return {"enabled": False}

        try:
            info = self._client.info('stats')
            keys_count = self._client.dbsize()

            return {
                "enabled": True,
                "total_keys": keys_count,
                "total_commands_processed": info.get('total_commands_processed', 0),
                "keyspace_hits": info.get('keyspace_hits', 0),
                "keyspace_misses": info.get('keyspace_misses', 0),
                "hit_rate": (
                    info.get('keyspace_hits', 0) /
                    max(1, info.get('keyspace_hits', 0) + info.get('keyspace_misses', 0))
                ),
            }
        except Exception as e:
            logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": True, "error": str(e)}


# Singleton instance
_response_cache_instance: Optional[ResponseCache] = None


def get_response_cache(
    redis_url: Optional[str] = None,
    ttl_seconds: Optional[int] = None,
    enabled: Optional[bool] = None,
    max_cache_size_mb: Optional[int] = None,
) -> ResponseCache:
    """
    Get singleton response cache instance.

    Args:
        redis_url: Redis URL (only for first initialization)
        ttl_seconds: TTL in seconds (only for first initialization)
        enabled: Whether caching is enabled (only for first initialization)
        max_cache_size_mb: Max cache size in MB (only for first initialization)

    Returns:
        ResponseCache instance
    """
    global _response_cache_instance

    if _response_cache_instance is None:
        from app.settings import settings

        _response_cache_instance = ResponseCache(
            redis_url=redis_url or settings.session.redis_url,
            ttl_seconds=ttl_seconds or settings.response_cache.ttl_seconds,
            enabled=enabled if enabled is not None else settings.response_cache.enabled,
            max_cache_size_mb=max_cache_size_mb or settings.response_cache.max_cache_size_mb,
        )

    return _response_cache_instance


__all__ = ["ResponseCache", "get_response_cache"]
