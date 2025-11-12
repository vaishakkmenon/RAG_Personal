"""
Query Router Package

This package provides functionality for analyzing and routing user queries
to the appropriate handlers based on content analysis.
"""

from .patterns import PatternMatcher
from .route_helpers.query_analyzer import QueryAnalyzer
from .route_helpers.response_builder import ResponseBuilder
from .router import QueryRouter
from .factory import get_router, route_query

__all__ = [
    "PatternMatcher",
    "QueryAnalyzer",
    "ResponseBuilder",
    "QueryRouter",
    "get_router",
    "route_query",
]
