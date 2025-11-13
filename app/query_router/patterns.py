"""
Pattern libraries for query disambiguation.

This module contains patterns and utilities for detecting and categorizing
elements in user queries to improve routing and response generation.
"""

import re
from typing import Dict, List, Optional, Pattern, Tuple, Set

class PatternMatcher:
    """Efficient pattern matching for query analysis."""
    
    def __init__(self, patterns: Dict[str, List[str]]):
        """Initialize with patterns dictionary."""
        self.patterns = patterns
        self.compiled_patterns = self._compile_patterns(patterns)
    
    def _compile_patterns(self, patterns: Dict[str, List[str]]) -> Dict[str, List[Pattern]]:
        """Compile regex patterns for faster matching."""
        return {
            key: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for key, patterns in patterns.items()
        }
    
    def find_matches(self, text: str) -> List[str]:
        """Find all matches in text for the stored patterns."""
        matches = []
        text_lower = text.lower()
        for category, regex_list in self.compiled_patterns.items():
            if any(regex.search(text_lower) for regex in regex_list):
                matches.append(category)
        return matches


def detect_technologies(text: str, tech_patterns: Dict[str, List[str]]) -> Set[str]:
    """Detect technology terms in the text."""
    matcher = PatternMatcher(tech_patterns)
    return set(matcher.find_matches(text))


def categorize_text(text: str, category_patterns: Dict[str, List[str]]) -> Set[str]:
    """Categorize the text based on content."""
    matcher = PatternMatcher(category_patterns)
    return set(matcher.find_matches(text))


def get_question_type(question: str, question_patterns: Dict[str, List[str]]) -> Optional[str]:
    """Determine the type of question being asked."""
    matcher = PatternMatcher(question_patterns)
    matches = matcher.find_matches(question)
    return matches[0] if matches else None


def extract_entities(text: str, patterns: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """Extract and group all matching entities from text."""
    result = {}
    for entity_type, terms in patterns.items():
        matches = []
        for term in terms:
            # Simple word boundary matching for exact terms
            if re.search(rf'\b{re.escape(term)}\b', text, re.IGNORECASE):
                matches.append(term)
        if matches:
            result[entity_type] = matches
    return result
