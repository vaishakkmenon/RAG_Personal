"""
============================================================================
PRIMARY IMPLEMENTATION: Redis Session Store (Production-Ready)
============================================================================

This is the PRIMARY session storage backend designed for production use.

FEATURES:
---------
* Persistent Storage: Sessions survive server restarts
* Distributed: Shareable across multiple application instances
* Connection Pooling: Optimized for high-concurrency environments
* Prometheus Metrics: Production monitoring and observability
* IP Indexing: Fast O(K) lookups for sessions by IP address
* Automatic TTL: Redis-native expiration management
* Health Checks: Built-in connection monitoring

USAGE:
------
This backend is automatically selected when STORAGE_BACKEND=redis in settings.
If Redis connection fails, the system automatically falls back to in-memory store.

See: app/storage/factory.py for fallback logic
See: app/storage/memory.py for fallback implementation
"""

import json
import logging
import time
import uuid
from datetime import datetime
from typing import List, Optional

from app.settings import settings
from app.storage.base import SessionStore
from app.storage.models import Session
from app.storage.utils import mask_session_id

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

# Import session metrics from central metrics module
try:
    from app.metrics import (
        rag_sessions_active,
        rag_session_operations_total,
    )

    SESSION_METRICS_ENABLED = True
except ImportError:
    SESSION_METRICS_ENABLED = False
    logger.warning("Session metrics not available")


class RedisSessionStore(SessionStore):
    """Redis-backed session storage with connection pooling and production features."""

    def __init__(self, redis_url: str):
        """Initialize Redis store with connection pooling.

        Args:
            redis_url: Redis connection URL (e.g., redis://localhost:6379/0)
        """
        if not REDIS_AVAILABLE:
            logger.error("Redis not available. Falling back to in-memory store.")
            raise ImportError("Redis not installed")

        try:
            # Connection pool for better performance under load
            pool = redis.ConnectionPool.from_url(
                redis_url,
                max_connections=20,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
            )
            self._client = redis.Redis(connection_pool=pool)
            self._client.ping()
            logger.info(f"Connected to Redis with connection pool: {redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

        # Prometheus metrics
        try:
            from prometheus_client import Counter, Gauge, Histogram

            self._metrics_enabled = True

            # Try to create metrics, but handle duplicates gracefully
            try:
                self.redis_operations = Counter(
                    "redis_operations_total",
                    "Total Redis operations",
                    ["operation", "status"],  # get/set/delete, success/error
                )
            except ValueError:
                # Metric already exists, retrieve it from registry
                from prometheus_client import REGISTRY

                for collector in list(REGISTRY._collector_to_names.keys()):
                    if (
                        hasattr(collector, "_name")
                        and collector._name == "redis_operations_total"
                    ):
                        self.redis_operations = collector
                        break
                else:
                    # Fallback: disable this metric
                    self.redis_operations = None

            try:
                self.redis_latency = Histogram(
                    "redis_operation_duration_seconds",
                    "Redis operation latency",
                    ["operation"],
                )
            except ValueError:
                from prometheus_client import REGISTRY

                for collector in list(REGISTRY._collector_to_names.keys()):
                    if (
                        hasattr(collector, "_name")
                        and collector._name == "redis_operation_duration_seconds"
                    ):
                        self.redis_latency = collector
                        break
                else:
                    self.redis_latency = None

            try:
                self.redis_memory = Gauge(
                    "redis_memory_usage_bytes", "Redis memory usage"
                )
            except ValueError:
                from prometheus_client import REGISTRY

                for collector in list(REGISTRY._collector_to_names.keys()):
                    if (
                        hasattr(collector, "_name")
                        and collector._name == "redis_memory_usage_bytes"
                    ):
                        self.redis_memory = collector
                        break
                else:
                    self.redis_memory = None

            try:
                self.redis_keys = Gauge("redis_keys_total", "Total keys in Redis")
            except ValueError:
                from prometheus_client import REGISTRY

                for collector in list(REGISTRY._collector_to_names.keys()):
                    if (
                        hasattr(collector, "_name")
                        and collector._name == "redis_keys_total"
                    ):
                        self.redis_keys = collector
                        break
                else:
                    self.redis_keys = None
        except ImportError:
            logger.warning("Prometheus client not available, metrics disabled")
            self._metrics_enabled = False

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"session:{session_id}"

    def _ip_index_key(self, ip_address: str) -> str:
        """Generate Redis key for IP address index."""
        return f"ip_index:{ip_address}"

    def health_check(self) -> bool:
        """Check Redis connection health.

        Returns:
            True if Redis is responsive, False otherwise
        """
        try:
            self._client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis health check failed: {e}")
            return False

    def get_info(self) -> dict:
        """Get Redis server info.

        Returns:
            Dictionary with Redis stats (memory, clients, etc.)
        """
        try:
            info = self._client.info()
            return {
                "used_memory_mb": info.get("used_memory", 0) / (1024 * 1024),
                "connected_clients": info.get("connected_clients", 0),
                "total_keys": self._client.dbsize(),
                "uptime_seconds": info.get("uptime_in_seconds", 0),
            }
        except Exception as e:
            logger.error(f"Redis info error: {e}")
            return {}

    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session from Redis."""
        start_time = time.time()

        try:
            data = self._client.get(self._key(session_id))

            # Record metrics
            if self._metrics_enabled:
                if self.redis_operations:
                    self.redis_operations.labels(
                        operation="get", status="success"
                    ).inc()
                if self.redis_latency:
                    self.redis_latency.labels(operation="get").observe(
                        time.time() - start_time
                    )

            if data:
                # Track session retrieval
                if SESSION_METRICS_ENABLED:
                    rag_session_operations_total.labels(operation="retrieved").inc()
                return Session.from_dict(json.loads(data))
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")

            # Record error metric
            if self._metrics_enabled and self.redis_operations:
                self.redis_operations.labels(operation="get", status="error").inc()

            return None

    def create_session(
        self,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Create new session in Redis with IP indexing."""
        start_time = time.time()

        if session_id is None:
            session_id = str(uuid.uuid4())

        now = datetime.now()
        session = Session(
            session_id=session_id,
            created_at=now,
            last_accessed=now,
            ip_address=ip_address,
            user_agent=user_agent,
        )

        try:
            # Use pipeline for atomic operations
            pipe = self._client.pipeline()

            # Store session with TTL
            pipe.setex(
                self._key(session_id),
                settings.session.ttl_seconds,
                json.dumps(session.to_dict()),
            )

            # Add to IP index with same TTL
            if ip_address:
                pipe.sadd(self._ip_index_key(ip_address), session_id)
                pipe.expire(
                    self._ip_index_key(ip_address), settings.session.ttl_seconds
                )

            pipe.execute()
            logger.info(f"Created session {mask_session_id(session_id)} in Redis")

            # Record metrics
            if self._metrics_enabled:
                if self.redis_operations:
                    self.redis_operations.labels(
                        operation="create", status="success"
                    ).inc()
                if self.redis_latency:
                    self.redis_latency.labels(operation="create").observe(
                        time.time() - start_time
                    )

            # Track session creation and update active count
            if SESSION_METRICS_ENABLED:
                rag_session_operations_total.labels(operation="created").inc()
                # Update active sessions count
                try:
                    active_count = self._client.dbsize()  # Approximate count
                    rag_sessions_active.set(active_count)
                except Exception:
                    pass  # Don't fail on metrics
        except Exception as e:
            logger.error(f"Redis create error: {e}")

            if self._metrics_enabled and self.redis_operations:
                self.redis_operations.labels(operation="create", status="error").inc()

        return session

    def update_session(self, session: Session) -> None:
        """Update session in Redis with TTL refresh and IP index update."""
        start_time = time.time()

        try:
            # Get old session to check if IP changed
            old_data = self._client.get(self._key(session.session_id))
            old_ip = None
            if old_data:
                old_session = Session.from_dict(json.loads(old_data))
                old_ip = old_session.ip_address

            pipe = self._client.pipeline()

            # Update session
            pipe.setex(
                self._key(session.session_id),
                settings.session.ttl_seconds,
                json.dumps(session.to_dict()),
            )

            # Update IP index if IP changed
            if old_ip != session.ip_address:
                if old_ip:
                    pipe.srem(self._ip_index_key(old_ip), session.session_id)
                if session.ip_address:
                    pipe.sadd(
                        self._ip_index_key(session.ip_address), session.session_id
                    )
                    pipe.expire(
                        self._ip_index_key(session.ip_address),
                        settings.session.ttl_seconds,
                    )

            pipe.execute()

            # Record metrics
            if self._metrics_enabled:
                if self.redis_operations:
                    self.redis_operations.labels(
                        operation="update", status="success"
                    ).inc()
                if self.redis_latency:
                    self.redis_latency.labels(operation="update").observe(
                        time.time() - start_time
                    )
        except Exception as e:
            logger.error(f"Redis update error: {e}")

            if self._metrics_enabled and self.redis_operations:
                self.redis_operations.labels(operation="update", status="error").inc()

    def delete_session(self, session_id: str) -> None:
        """Delete session from Redis and IP index."""
        start_time = time.time()

        try:
            # Get session to find IP
            data = self._client.get(self._key(session_id))

            if data:
                session = Session.from_dict(json.loads(data))

                # Remove from both session store and IP index
                pipe = self._client.pipeline()
                pipe.delete(self._key(session_id))
                if session.ip_address:
                    pipe.srem(self._ip_index_key(session.ip_address), session_id)
                pipe.execute()

                logger.debug(
                    f"Deleted session {mask_session_id(session_id)} from Redis"
                )

                # Record metrics
                if self._metrics_enabled:
                    if self.redis_operations:
                        self.redis_operations.labels(
                            operation="delete", status="success"
                        ).inc()
                    if self.redis_latency:
                        self.redis_latency.labels(operation="delete").observe(
                            time.time() - start_time
                        )
        except Exception as e:
            logger.error(f"Redis delete error: {e}")

            if self._metrics_enabled and self.redis_operations:
                self.redis_operations.labels(operation="delete", status="error").inc()

    def get_session_count(self) -> int:
        """Get total number of active sessions in Redis.

        Uses SCAN to count only session keys (non-blocking, safe for production).
        """
        try:
            count = 0
            cursor = 0

            # Use SCAN to safely iterate through session keys
            while True:
                cursor, keys = self._client.scan(cursor, match="session:*", count=100)
                count += len(keys)

                if cursor == 0:
                    break

            # Update metrics
            if self._metrics_enabled and self.redis_keys:
                self.redis_keys.set(count)

            return count
        except Exception as e:
            logger.error(f"Redis count error: {e}")
            return 0

    def get_sessions_by_ip(self, ip_address: str) -> List[Session]:
        """Get all sessions for an IP address using indexed lookup.

        O(K) where K = sessions for this IP, not total sessions.
        """
        start_time = time.time()

        try:
            # Get session IDs from IP index (O(1))
            session_ids = self._client.smembers(self._ip_index_key(ip_address))

            # Fetch sessions in batch using pipeline
            if not session_ids:
                return []

            pipe = self._client.pipeline()
            for sid in session_ids:
                pipe.get(self._key(sid))

            results = pipe.execute()

            # Parse sessions
            sessions = []
            for data in results:
                if data:
                    sessions.append(Session.from_dict(json.loads(data)))

            # Record metrics
            if self._metrics_enabled:
                if self.redis_operations:
                    self.redis_operations.labels(
                        operation="get_by_ip", status="success"
                    ).inc()
                if self.redis_latency:
                    self.redis_latency.labels(operation="get_by_ip").observe(
                        time.time() - start_time
                    )

            return sessions
        except Exception as e:
            logger.error(f"Redis IP query error: {e}")

            if self._metrics_enabled and self.redis_operations:
                self.redis_operations.labels(
                    operation="get_by_ip", status="error"
                ).inc()

            return []

    def clear_cache(self) -> bool:
        """Clear all session data from Redis (useful for testing).

        WARNING: This removes ALL sessions AND response cache. Use only for testing!

        Returns:
            True if successful, False otherwise
        """
        try:
            # Count keys before clearing
            before_count = self.get_session_count()

            # Flush session keys, IP indexes, AND response cache (safer than FLUSHDB)
            deleted = 0

            # Clear session keys
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match="session:*", count=1000)
                if keys:
                    deleted += self._client.delete(*keys)

                if cursor == 0:
                    break

            # Clear IP indexes
            cursor = 0
            while True:
                cursor, keys = self._client.scan(cursor, match="ip_index:*", count=1000)
                if keys:
                    deleted += self._client.delete(*keys)

                if cursor == 0:
                    break

            # Clear response cache (THIS WAS MISSING!)
            cursor = 0
            response_keys_deleted = 0
            while True:
                cursor, keys = self._client.scan(
                    cursor, match="rag:response:*", count=1000
                )
                if keys:
                    response_keys_deleted += self._client.delete(*keys)

                if cursor == 0:
                    break

            logger.info(
                f"Cleared Redis cache: {before_count} sessions, "
                f"{response_keys_deleted} response cache entries, "
                f"{deleted} total keys deleted"
            )

            # Update metrics
            if SESSION_METRICS_ENABLED:
                rag_sessions_active.set(0)

            return True
        except Exception as e:
            logger.error(f"Failed to clear Redis cache: {e}")
            return False
