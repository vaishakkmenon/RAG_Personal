"""
Factory for session store creation.

ARCHITECTURE:
=============
This module implements a dual-backend session storage system:

1. PRIMARY: Redis (Production-Ready)
   - Persistent, distributed session storage
   - Connection pooling for high performance
   - Prometheus metrics integration
   - IP-based indexing for fast lookups
   - Automatic TTL management

2. FALLBACK: In-Memory (High-Availability)
   - Activated automatically if Redis fails
   - Zero external dependencies
   - Sharded architecture for concurrency
   - Automatic cleanup of expired sessions

The factory automatically handles fallback scenarios:
- Redis connection failures
- Missing Redis credentials
- Redis library not installed
"""

import logging
from typing import Optional

from app.settings import settings
from app.storage.base import SessionStore
from app.storage.primary import RedisSessionStore
from app.storage.fallback import InMemorySessionStore

logger = logging.getLogger(__name__)

_session_store: Optional[SessionStore] = None


# ============================================================================
# PRIMARY IMPLEMENTATION: REDIS (Production)
# ============================================================================


def create_session_store(
    backend: str = "redis", redis_url: Optional[str] = None
) -> SessionStore:
    """Factory function to create session store with automatic fallback.

    Attempts to create Redis store first (production), falls back to
    in-memory store if Redis is unavailable.

    Args:
        backend: 'redis' (production) or 'memory' (fallback/dev)
        redis_url: Redis connection URL (e.g., redis://localhost:6379/0)

    Returns:
        SessionStore implementation (Redis with fallback to in-memory)
    """
    if backend == "redis":
        # ===== ATTEMPT REDIS CONNECTION (PRIMARY) =====
        if not redis_url:
            logger.warning("Redis URL not provided, falling back to in-memory store")
            return _create_fallback_store()

        try:
            logger.info(f"Attempting to connect to Redis: {redis_url}")
            redis_store = RedisSessionStore(redis_url)
            logger.info(
                "[SUCCESS] Redis session store initialized successfully (PRIMARY)"
            )
            return redis_store

        except (ImportError, Exception) as e:
            logger.error(f"[FAILED] Failed to initialize Redis store: {e}")
            return _create_fallback_store()
    else:
        # Explicitly requested in-memory store
        logger.info("In-memory store explicitly requested via backend='memory'")
        return InMemorySessionStore()


# ============================================================================
# FALLBACK IMPLEMENTATION: IN-MEMORY (High Availability)
# ============================================================================


def _create_fallback_store() -> InMemorySessionStore:
    """Create fallback in-memory store when Redis is unavailable.

    This ensures the application continues to function even if Redis
    connection fails, with graceful degradation to in-memory storage.

    Returns:
        InMemorySessionStore instance
    """
    logger.warning("[FALLBACK ACTIVATED] Using in-memory session store")
    logger.warning("Sessions will not persist across server restarts")
    return InMemorySessionStore()


def get_session_store() -> SessionStore:
    """Get the global session store instance.

    Initializes on first call based on settings.

    Returns:
        SessionStore singleton
    """
    global _session_store

    if _session_store is None:
        _session_store = create_session_store(
            backend=settings.session.storage_backend,
            redis_url=settings.session.redis_url,
        )

    return _session_store
