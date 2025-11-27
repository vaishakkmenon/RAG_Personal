"""
Query Router - Main routing orchestrator

Coordinates query analysis and response building for routing decisions.
"""

import logging
from typing import Any, Dict, Optional

from .route_helpers.query_analyzer import QueryAnalyzer
from .route_helpers.response_builder import ResponseBuilder
from ..settings import query_router_settings


logger = logging.getLogger(__name__)


class QueryRouter:
    """Main query router that coordinates analysis and routing."""

    def __init__(self, config=None):
        """Initialize the query router with configuration.

        Args:
            config: Configuration settings (defaults to query_router_settings)
        """
        self.config = config or query_router_settings

        # Initialize components
        self.analyzer = QueryAnalyzer(self.config)
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
            # Return safe defaults on error (use settings values)
            from ..settings import settings
            return {
                "top_k": settings.retrieval.top_k,
                "rerank": settings.retrieval.rerank,
                "null_threshold": settings.retrieval.null_threshold,
                "max_distance": settings.retrieval.max_distance,
                "confidence": 0.5,
                "error": str(e),
            }


__all__ = ["QueryRouter"]
