"""Query analysis functionality for the query router."""

import re
from typing import Dict, Any, List, Optional
import logging
from ..patterns import detect_technologies, categorize_text, get_question_type

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries to extract metadata and determine routing."""

    def __init__(self, config):
        """Initialize with configuration and optional certification registry.

        Args:
            config: Configuration object with technology_terms, categories, and question_patterns
        """
        self.config = config

    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyze a question and extract relevant metadata.

        Args:
            question: The user's question

        Returns:
            Dictionary containing analysis results
        """
        question_lower = question.lower()

        # Basic analysis using standalone functions
        tech_patterns = getattr(self.config, "technology_terms", {})
        category_patterns = getattr(self.config, "categories", {})
        question_patterns = getattr(self.config, "question_patterns", {})

        analysis = {
            "question": question,
            "technologies": list(detect_technologies(question_lower, tech_patterns)),
            "categories": list(categorize_text(question_lower, category_patterns)),
            "question_type": get_question_type(question_lower, question_patterns),
            "is_ambiguous": False,
            "needs_clarification": False,
            "confidence": 1.0,
        }

        # Determine if the query is ambiguous
        self._check_ambiguity(analysis)

        return analysis

    def _check_ambiguity(self, analysis: Dict[str, Any]) -> None:
        """Check if the query is ambiguous and needs clarification."""
        # Check for broad questions
        question_type = analysis.get("question_type", "")
        if question_type in ["broad", "general"]:
            analysis.update(
                {
                    "is_ambiguous": True,
                    "confidence": min(analysis.get("confidence", 1.0), 0.7),
                }
            )
