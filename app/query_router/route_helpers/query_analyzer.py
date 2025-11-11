"""Query analysis functionality for the query router."""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

class QueryAnalyzer:
    """Analyzes queries to extract metadata and determine routing."""
    
    def __init__(self, pattern_matcher, cert_registry=None):
        """Initialize with a pattern matcher and optional certification registry."""
        self.pattern_matcher = pattern_matcher
        self.cert_registry = cert_registry
        self._certificate_regexes = self._compile_certificate_patterns()
    
    def _compile_certificate_patterns(self):
        """Compile regex patterns for certificate detection."""
        if not self.cert_registry:
            return []
            
        patterns = []
        for cert in self.cert_registry.certifications.values():
            # Add patterns for official name and aliases
            for name in [cert.official_name] + cert.aliases:
                try:
                    pattern = re.compile(r'\b' + re.escape(name.lower()) + r'\b', re.IGNORECASE)
                    patterns.append((pattern, cert.id, name))
                except re.error as e:
                    logger.warning(f"Failed to compile pattern for {name}: {e}")
        return patterns
    
    def analyze(self, question: str) -> Dict[str, Any]:
        """
        Analyze a question and extract relevant metadata.
        
        Args:
            question: The user's question
            
        Returns:
            Dictionary containing analysis results
        """
        question_lower = question.lower()
        
        # Basic analysis
        analysis = {
            'question': question,
            'technologies': self.pattern_matcher.detect_technologies(question_lower),
            'categories': self.pattern_matcher.categorize_text(question_lower),
            'question_type': self.pattern_matcher.get_question_type(question_lower),
            'certificates': self._find_certificate_matches(question_lower),
            'is_ambiguous': False,
            'needs_clarification': False,
            'confidence': 1.0
        }
        
        # Determine if the query is ambiguous
        self._check_ambiguity(analysis)
        
        return analysis
    
    def _find_certificate_matches(self, question: str) -> List[Dict[str, Any]]:
        """Find certificate matches in the question."""
        if not self.cert_registry:
            return []
            
        matches = []
        for pattern, cert_id, name in self._certificate_regexes:
            if pattern.search(question):
                cert = self.cert_registry.get(cert_id)
                if cert:
                    matches.append({
                        'id': cert_id,
                        'name': name,
                        'confidence': min(0.9, 0.7 + (0.2 * (len(name.split()) / 2)))
                    })
        return matches
    
    def _check_ambiguity(self, analysis: Dict[str, Any]) -> None:
        """Check if the query is ambiguous and needs clarification."""
        # Check for multiple certificate matches
        cert_matches = analysis.get('certificates', [])
        if len(cert_matches) > 1:
            analysis.update({
                'is_ambiguous': True,
                'needs_clarification': True,
                'confidence': 0.5,
                'clarification_options': [m['name'] for m in cert_matches]
            })
        
        # Check for broad questions
        question_type = analysis.get('question_type', '')
        if question_type in ['broad', 'general']:
            analysis.update({
                'is_ambiguous': True,
                'confidence': min(analysis.get('confidence', 1.0), 0.7)
            })
