"""
Query Router Factory - Global router instance management

Provides factory functions for creating and retrieving the global router instance.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .router import QueryRouter

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ..certifications import CertificationRegistry

logger = logging.getLogger(__name__)

# Global router instance
_router: Optional[QueryRouter] = None


def get_router(cert_registry: Optional["CertificationRegistry"] = None) -> QueryRouter:
    """Get or create the global router instance.

    Args:
        cert_registry: Optional certification registry to initialize with

    Returns:
        Global QueryRouter instance
    """
    global _router
    if _router is None:
        _router = QueryRouter(cert_registry=cert_registry)
    elif cert_registry is not None and _router.cert_registry is None:
        # Update registry if it wasn't set during initialization
        _router.cert_registry = cert_registry
        _router.analyzer.cert_registry = cert_registry
        # Recompile certificate patterns with the new registry
        _router.analyzer._certificate_regexes = _router.analyzer._compile_certificate_patterns()
    return _router


def route_query(
    question: str,
    cert_registry: Optional["CertificationRegistry"] = None,
) -> Dict[str, Any]:
    """Convenience function to route a query using the global router.

    Args:
        question: The user's question
        cert_registry: Optional certification registry

    Returns:
        Dictionary of routing parameters
    """
    router = get_router(cert_registry=cert_registry)
    return router.route(question)


__all__ = ["get_router", "route_query", "QueryRouter"]
