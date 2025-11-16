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

        # Check for structured summary queries (broad but clear intent)
        analysis["is_structured_summary"] = self._is_structured_summary(question_lower)
        analysis["summary_domains"] = self._extract_summary_domains(question_lower)

        # Determine if the query is ambiguous
        self._check_ambiguity(analysis)

        return analysis

    def _is_structured_summary(self, question: str) -> bool:
        """Check if query is a structured summary request with clear scope.

        A structured summary has:
        1. Summary intent keywords (summarize, overview, etc.)
        2. Multiple specific domains mentioned (education, work, etc.)

        Args:
            question: Lowercased question text

        Returns:
            True if this is a structured summary request
        """
        # Check for summary intent keywords
        summary_keywords = ['summarize', 'summary', 'overview', 'give me',
                           'tell me about', 'background', 'profile']
        has_summary_intent = any(kw in question for kw in summary_keywords)

        # Count specific domains mentioned
        domain_count = sum(1 for domain in
            ['education', 'work', 'experience', 'certification', 'skill',
             'academic', 'professional', 'career', 'qualifications', 'degree']
            if domain in question)

        # Structured if has intent AND mentions 2+ domains
        return has_summary_intent and domain_count >= 2

    def _extract_summary_domains(self, question: str) -> List[str]:
        """Extract specific domains mentioned in summary queries.

        Args:
            question: Lowercased question text

        Returns:
            List of domain names found in the question
        """
        domains = []
        domain_map = {
            'education': ['education', 'academic', 'degree', 'gpa', 'university', 'school'],
            'work': ['work', 'job', 'employment', 'professional', 'career', 'experience'],
            'certifications': ['certification', 'certificate', 'certified', 'cert'],
            'skills': ['skill', 'technical', 'programming', 'technology', 'knowledge'],
        }

        for domain, keywords in domain_map.items():
            if any(kw in question for kw in keywords):
                domains.append(domain)

        return domains

    def _check_ambiguity(self, analysis: Dict[str, Any]) -> None:
        """Check if the query is ambiguous and needs clarification."""
        # Structured summaries are NOT ambiguous (clear intent + scope)
        if analysis.get("is_structured_summary"):
            return

        # Check for broad questions
        question_type = analysis.get("question_type", "")
        if question_type in ["broad", "general"]:
            analysis.update(
                {
                    "is_ambiguous": True,
                    "confidence": min(analysis.get("confidence", 1.0), 0.7),
                }
            )
