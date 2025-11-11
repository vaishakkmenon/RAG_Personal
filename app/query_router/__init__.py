"""
Query Router Package

This package provides functionality for analyzing and routing user queries
to the appropriate handlers based on content analysis.
"""

from .route_helpers.pattern_matcher import PatternMatcher
from .route_helpers.query_analyzer import QueryAnalyzer
from .route_helpers.response_builder import ResponseBuilder

# Re-export commonly used functions for backward compatibility
from .route_helpers.pattern_matcher import (
    detect_technologies,
    categorize_text,
    get_question_type
)

__all__ = [
    'PatternMatcher',
    'QueryAnalyzer',
    'ResponseBuilder',
    'detect_technologies',
    'categorize_text',
    'get_question_type'
]
