"""
Query Router Factory - Global router instance management

Provides factory functions for creating and retrieving the global router instance.
Implements singleton pattern with pre-initialization for optimal performance.
"""

import logging
from typing import Any, Dict, Optional

from .router import QueryRouter

logger = logging.getLogger(__name__)

# Global router instance - initialized at module load for performance
_router: Optional[QueryRouter] = None
_router_initialized: bool = False


def _initialize_router() -> QueryRouter:
    """Initialize the global router instance once at module load.

    Returns:
        Initialized QueryRouter instance
    """
    global _router, _router_initialized
    if not _router_initialized:
        try:
            _router = QueryRouter()
            _router_initialized = True
            logger.info("Global QueryRouter initialized at module load")
        except Exception as e:
            logger.error(f"Failed to initialize QueryRouter: {e}", exc_info=True)
            raise
    return _router


def get_router() -> QueryRouter:
    """Get the global router instance.

    Returns:
        Global QueryRouter instance

    Note:
        Router is pre-initialized at module load.
    """
    global _router

    # Ensure router is initialized
    if _router is None:
        _initialize_router()

    return _router


def route_query(
    question: str,
) -> Dict[str, Any]:
    """Convenience function to route a query using the global router.

    Args:
        question: The user's question

    Returns:
        Dictionary of routing parameters
    """
    router = get_router()
    return router.route(question)


# Pre-initialize router at module load for optimal performance
try:
    _initialize_router()
except Exception as e:
    logger.warning(f"Router pre-initialization deferred: {e}")


__all__ = ["get_router", "route_query", "QueryRouter"]
