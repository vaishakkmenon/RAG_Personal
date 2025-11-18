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

        # Create domain configurations for multi-domain retrieval
        domain_configs = self._create_domain_configs(analysis["summary_domains"])
        analysis["domain_configs"] = domain_configs

        # Extract keywords for guided retrieval and prompting
        extracted_keywords = self._extract_keywords(question_lower)
        analysis["keywords"] = extracted_keywords
        # Flatten all keywords for easy access
        all_keywords = (
            extracted_keywords['domains'] +
            extracted_keywords['entities'] +
            extracted_keywords['intent']
        )
        analysis["all_keywords"] = all_keywords

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

        # Use the same domain extraction logic for consistency
        domains = self._extract_summary_domains(question)

        # Structured if has intent AND mentions 2+ domains
        return has_summary_intent and len(domains) >= 2

    def _extract_summary_domains(self, question: str) -> List[str]:
        """Extract specific domains mentioned in summary queries.

        Args:
            question: Lowercased question text

        Returns:
            List of domain names found in the question
        """
        domains = []

        # Special handling for "professional background" which implies both education + work
        if 'professional background' in question or 'career background' in question:
            return ['education', 'work']

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

    def _extract_keywords(self, question: str) -> Dict[str, List[str]]:
        """Extract keywords from query for guided retrieval and prompting.

        Extracts three types of keywords:
        1. Domains: High-level areas (education, work, certifications, skills)
        2. Entities: Specific items (technologies, certifications, etc.)
        3. Intent: Action keywords (summarize, list, compare)

        Args:
            question: Lowercased question text

        Returns:
            Dict with 'domains', 'entities', and 'intent' lists
        """
        keywords = {
            'domains': [],
            'entities': [],
            'intent': []
        }

        # Extract domains (reuse existing logic)
        keywords['domains'] = self._extract_summary_domains(question)

        # Extract entities using pattern matching
        tech_patterns = getattr(self.config, "technology_terms", {})
        if tech_patterns:
            # Check for technologies mentioned
            for tech_category, tech_list in tech_patterns.items():
                for tech in tech_list:
                    if tech.lower() in question:
                        keywords['entities'].append(tech)

        # Extract specific entities
        entity_patterns = {
            'gpa': ['gpa', 'grade point average'],
            'degree': ['degree', 'bachelor', 'master', 'bs', 'ms'],
            'honors': ['honors', 'summa cum laude', 'magna cum laude', 'cum laude'],
            'certification': ['cka', 'aws', 'certified', 'certification'],
        }

        for entity, patterns in entity_patterns.items():
            if any(pattern in question for pattern in patterns):
                if entity not in keywords['entities']:
                    keywords['entities'].append(entity)

        # Extract intent keywords
        intent_keywords = {
            'summarize': ['summarize', 'summary', 'overview'],
            'list': ['list', 'what are', 'which'],
            'compare': ['compare', 'difference', 'versus', 'vs'],
            'detail': ['detail', 'explain', 'describe'],
        }

        for intent, patterns in intent_keywords.items():
            if any(pattern in question for pattern in patterns):
                keywords['intent'].append(intent)

        return keywords

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

    def _create_domain_configs(self, domains: List[str]) -> List[Dict[str, Any]]:
        """Create metadata filter configurations for detected domains.

        Maps semantic domains (education, work, certifications, skills) to
        actual metadata filters that can be used for retrieval.

        Args:
            domains: List of detected domain names

        Returns:
            List of domain configurations for multi-domain search
        """
        domain_configs = []

        # Domain-to-filter mapping based on actual metadata structure
        domain_filter_map = {
            'education': {
                'name': 'education',
                'filters': [
                    {'doc_type': 'term'},  # Academic term records
                    {'doc_type': 'transcript_analysis'},  # Overall academic summary
                    {'doc_type': 'resume', 'section': 'Education'},  # Resume education section
                ]
            },
            'work': {
                'name': 'work_experience',
                'filters': [
                    {'doc_type': 'resume'},  # Will post-filter for Work Experience sections
                ],
                'section_prefix': 'Work Experience'  # Post-filter for work sections
            },
            'certifications': {
                'name': 'certifications',
                'filters': [
                    {'doc_type': 'certificate'},  # Detailed cert documents
                    {'doc_type': 'resume', 'section': 'Certifications'},  # Resume cert list
                ]
            },
            'skills': {
                'name': 'skills',
                'filters': [
                    {'doc_type': 'resume', 'section': 'Skills'},  # Technical skills
                    {'doc_type': 'resume'},  # Will post-filter for project sections
                ],
                'section_prefix': 'Personal Projects'  # Post-filter for projects
            }
        }

        for domain in domains:
            if domain in domain_filter_map:
                domain_configs.append(domain_filter_map[domain])
            else:
                logger.debug(f"Unknown domain '{domain}', skipping")

        return domain_configs
