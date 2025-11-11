"""
Main Query Router for Personal RAG System

This module provides the main interface for the query routing system,
delegating to specialized components for pattern matching, analysis,
and response building.
"""

import logging
from typing import Any, Dict, Optional, TYPE_CHECKING

from .settings import query_router_settings
from .query_router import (
    PatternMatcher,
    QueryAnalyzer,
    ResponseBuilder
)

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .certifications import CertificationRegistry

logger = logging.getLogger(__name__)

# Global router instance
_router = None

class QueryRouter:
    """Main query router that coordinates analysis and routing."""
    
    def __init__(self, config=None, cert_registry=None):
        """Initialize the query router with configuration and optional registry.
        
        Args:
            config: Configuration settings (defaults to query_router_settings)
            cert_registry: Optional certification registry for certificate lookups
        """
        self.config = config or query_router_settings
        self.cert_registry = cert_registry
        
        # Initialize components
        self.pattern_matcher = PatternMatcher(self.config)
        self.analyzer = QueryAnalyzer(self.pattern_matcher, cert_registry)
        self.response_builder = ResponseBuilder(self.config)
    
    def route(self, question: str) -> Dict[str, Any]:
        """Route a user question to the appropriate handler.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary of routing parameters
        """
        try:
            # 1. Analyze the question
            analysis = self.analyzer.analyze(question)
            
            # 2. Build the response
            return self.response_builder.build_response(analysis)
            
        except Exception as e:
            logger.error(f"Error routing query: {e}", exc_info=True)
            # Return safe defaults on error
            return {
                'top_k': 5,
                'rerank': False,
                'null_threshold': 0.5,
                'max_distance': 0.6,
                'confidence': 0.5,
                'error': str(e)
            }

def get_router() -> QueryRouter:
    """Get or create the global router instance."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router

def route_query(
    question: str,
    cert_registry: Optional["CertificationRegistry"] = None,
) -> Dict[str, Any]:
    """
    Convenience function to route a query using the global router.
    
    Args:
        question: The user's question
        cert_registry: Optional certification registry
        
    Returns:
        Dictionary of routing parameters
    """
    router = get_router()
    if cert_registry is not None:
        router.cert_registry = cert_registry
    return router.route(question)
