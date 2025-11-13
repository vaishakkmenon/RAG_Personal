"""Pattern matching functionality for the query router."""

import re
from typing import Dict, List, Optional, Pattern, Tuple

class PatternMatcher:
    """Handles pattern matching for technologies, categories, and question types."""
    
    def __init__(self, config):
        """Initialize with configuration."""
        self.config = config
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile all regex patterns for matching."""
        # Technology patterns
        self.tech_patterns = self._compile_pattern_list(
            getattr(self.config, 'technology_terms', [])
        )
        
        # Category patterns
        self.category_patterns = self._compile_pattern_list(
            getattr(self.config, 'categories', [])
        )
        
        # Question type patterns
        self.question_patterns = self._compile_pattern_list(
            getattr(self.config, 'question_patterns', [])
        )
    
    def _compile_pattern_list(self, patterns: List[Dict]) -> List[Tuple[Pattern, str]]:
        """Compile a list of pattern dictionaries into regex patterns."""
        compiled = []
        for item in patterns:
            try:
                pattern = re.compile(item['pattern'], re.IGNORECASE)
                compiled.append((pattern, item.get('type', '')))
            except (KeyError, re.error) as e:
                continue
        return compiled
    
    def find_matches(self, text: str, pattern_type: str = 'all') -> List[str]:
        """
        Find matches in text based on pattern type.
        
        Args:
            text: Text to search in
            pattern_type: Type of patterns to use ('tech', 'category', 'question', 'all')
            
        Returns:
            List of matched terms
        """
        text_lower = text.lower()
        matches = set()
        
        if pattern_type in ['tech', 'all']:
            for pattern, _ in self.tech_patterns:
                if pattern.search(text_lower):
                    matches.add(_.lower())
        
        if pattern_type in ['category', 'all']:
            for pattern, category in self.category_patterns:
                if pattern.search(text_lower):
                    matches.add(category.lower())
        
        if pattern_type in ['question', 'all']:
            for pattern, q_type in self.question_patterns:
                if pattern.search(text_lower):
                    matches.add(q_type.lower())
        
        return list(matches)
    
    def detect_technologies(self, text: str) -> List[str]:
        """Detect technology terms in text."""
        return self.find_matches(text, 'tech')
    
    def categorize_text(self, text: str) -> List[str]:
        """Categorize text based on category patterns."""
        return self.find_matches(text, 'category')
    
    def get_question_type(self, question: str) -> Optional[str]:
        """Determine the type of question being asked."""
        matches = self.find_matches(question, 'question')
        return matches[0] if matches else None
