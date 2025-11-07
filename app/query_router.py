"""
Query Router for Personal RAG System

Analyzes user questions and automatically selects optimal retrieval parameters:
- Document type filters (resume, transcript_analysis, term, certificate, course)
- Number of chunks to retrieve (top_k)
- Confidence thresholds
- Reranking settings
"""

import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple
import logging
from .settings import query_router_settings

if TYPE_CHECKING:  # pragma: no cover - import for type checking only
    from .certifications import CertificationRegistry

AmbiguityResult = Tuple[bool, float]

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Routes queries to optimal retrieval settings based on question content.
    
    Configuration is loaded from settings.query_router_settings.
    """
    
    def __init__(
        self,
        config=None,
        cert_registry: Optional["CertificationRegistry"] = None,
    ):
        """
        Initialize the query router with configuration.

        Args:
            config: Optional QueryRouterSettings instance. If None, uses the global settings.
        """
        self.config = config or query_router_settings
        self.cert_registry = cert_registry
        self._certificate_regexes: List[Tuple[re.Pattern, str, str]] = []
        self._compile_patterns()
        self._build_certificate_terms()

    def _compile_patterns(self):
        """Compile all regex patterns for better performance."""
        
        # Document type patterns
        self.doc_type_regexes = {
            doc_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for doc_type, patterns in self.config.doc_type_patterns.items()
        }
        
        # Question type patterns
        self.broad_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.broad_question_patterns
        ]
        
        self.specific_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.specific_question_patterns
        ]
        
        self.cumulative_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.config.cumulative_patterns
        ]

    def set_cert_registry(
        self, cert_registry: Optional["CertificationRegistry"]
    ) -> None:
        """Attach (or detach) the certification registry at runtime."""

        self.cert_registry = cert_registry
        self._build_certificate_terms()

    def _build_certificate_terms(self) -> None:
        """Precompute certification name patterns for fast registry matching."""

        self._certificate_regexes = []
        if not self.cert_registry:
            return

        seen_terms: Set[str] = set()
        for cert_id, certification in self.cert_registry.certifications.items():
            names = [certification.official_name, *getattr(certification, "aliases", [])]
            for raw_name in names:
                if not raw_name:
                    continue
                normalized = raw_name.strip()
                if not normalized:
                    continue
                key = normalized.lower()
                if key in seen_terms:
                    continue
                seen_terms.add(key)
                pattern = re.compile(rf"\\b{re.escape(normalized)}\\b", re.IGNORECASE)
                self._certificate_regexes.append((pattern, cert_id, normalized))

    def detect_ambiguity(self, question: str) -> AmbiguityResult:
        """Return (is_ambiguous, confidence score) for a question."""
        cleaned = (question or "").strip()
        if not cleaned:
            return False, 0.0

        words = cleaned.split()
        word_count = len(words)
        char_count = len(cleaned)

        confidence = 0.0

        if word_count == 1 and char_count <= self.config.single_word_char_limit:
            confidence = 1.0
        elif word_count <= self.config.short_query_word_limit:
            confidence = 0.8

        lowered = cleaned.lower()
        if any(keyword in lowered for keyword in self.config.ambiguous_keywords):
            confidence = min(1.0, confidence + 0.3)

        is_ambiguous = confidence >= self.config.ambiguity_threshold
        logger.debug(
            "Ambiguity detection",
            extra={
                "question": cleaned,
                "word_count": word_count,
                "char_count": char_count,
                "confidence": confidence,
                "threshold": self.config.ambiguity_threshold,
                "is_ambiguous": is_ambiguous,
            },
        )
        return is_ambiguous, confidence
    
    def is_cumulative_query(self, question: str) -> bool:
        """Check if query is asking for cumulative/overall statistics."""
        return any(pattern.search(question) for pattern in self.cumulative_regexes)
        
    def detect_doc_type(self, question: str) -> Optional[str]:
        """
        Detect the most relevant document type based on question content.
        
        Priority order:
        1. certificate (most specific)
        2. resume (very specific)
        3. transcript_analysis (for overall/cumulative stats)
        4. term (for specific term/semester queries)
        5. course (most general academic)
        
        Args:
            question: User's question
        
        Returns:
            Document type string or None if no clear match
        """
        # Score all doc types and filter zeros in one pass
        scores = {
            doc_type: score
            for doc_type, patterns in self.doc_type_regexes.items()
            if (score := sum(1 for pattern in patterns if pattern.search(question))) > 0
        }
        
        if not scores:
            return None
        
        # Special handling: if both transcript_analysis and term match,
        # prefer transcript_analysis for cumulative queries
        if "transcript_analysis" in scores and "term" in scores:
            if self.is_cumulative_query(question):
                logger.info("Cumulative query detected: preferring transcript_analysis over term")
                scores["transcript_analysis"] += self.config.cumulative_query_boost
        
        # Return doc type with highest score
        best_match = max(scores.items(), key=lambda x: x[1])
        
        # Only return if we have at least 1 match
        if best_match[1] >= 1:
            logger.info(f"Detected doc_type: {best_match[0]} (score: {best_match[1]})")
            return best_match[0]
        
        return None
    
    def is_broad_question(self, question: str) -> bool:
        """Check if question is broad/general."""
        return any(pattern.search(question) for pattern in self.broad_regexes)
    
    def is_specific_question(self, question: str) -> bool:
        """Check if question is specific/narrow."""
        return any(pattern.search(question) for pattern in self.specific_regexes)
    
    def extract_term_info(self, question: str) -> Tuple[Optional[str], bool]:
        """
        Extract term_id from questions like 'Fall 2022'.
        
        Returns:
            (term_id, has_course_code) tuple
        """
        question_lower = question.lower()
        
        # Pattern 1: "Fall 2022" or "Spring 2023"
        # MUST have season + space + 4-digit year together
        match = re.search(r'\b(fall|spring|summer|winter)\s+(\d{4})\b', question_lower)
        if match:
            season, year = match.groups()
            term_id = f"{year}-{season}"
            has_course = bool(re.search(r'\b[A-Z]{2,4}\s*\d{3,4}\b', question))
            return term_id, has_course
        
        # Pattern 2: "2022 Fall" (reversed)
        match = re.search(r'\b(\d{4})\s+(fall|spring|summer|winter)\b', question_lower)
        if match:
            year, season = match.groups()
            term_id = f"{year}-{season}"
            has_course = bool(re.search(r'\b[A-Z]{2,4}\s*\d{3,4}\b', question))
            return term_id, has_course
        
        # Pattern 3: Course code without term
        has_course = bool(re.search(r'\b[A-Z]{2,4}\s*\d{3,4}\b', question))
        if has_course:
            return None, True
        
        return None, False
    
    def detect_certification(self, question: str) -> Optional[Dict[str, str]]:
        """Detect certification intent using registry-backed patterns only."""

        cleaned = (question or "").strip()
        if not cleaned or not self._certificate_regexes:
            return None

        for pattern, cert_id, term in self._certificate_regexes:
            if pattern.search(cleaned):
                logger.info(
                    "Certification detected via registry",
                    extra={"term": term, "cert_id": cert_id},
                )
                return {
                    "doc_type": "certificate",
                    "cert_id": cert_id,
                    "match_term": term,
                    "source": "registry",
                }

        return None
    
    def route(self, question: str) -> Dict[str, Any]:
        """
        Analyze question and return optimal retrieval parameters.
        
        Args:
            question: User's question
        
        Returns:
            Dict with: doc_type, top_k, null_threshold, max_distance, rerank, etc.
        """
        question = question.strip()

        is_ambiguous, ambiguity_score = self.detect_ambiguity(question)

        cert_detection = self.detect_certification(question)
        if cert_detection:
            params = {
                "doc_type": "certificate",
                "term_id": None,
                "top_k": self.config.default_top_k,
                "null_threshold": self.config.default_null_threshold,
                "max_distance": self.config.default_max_distance,
                "rerank": False,
                "rerank_lex_weight": 0.5,
                "is_ambiguous": is_ambiguous,
                "ambiguity_score": ambiguity_score,
            }
            if cert_detection.get("cert_id"):
                params["cert_id"] = cert_detection["cert_id"]
            params["certification_match_source"] = cert_detection.get("source")

            logger.info("Query routing decision (certificate): %s", params)
            return params

        term_id, has_course_code = self.extract_term_info(question)
        
        # Detect document type
        doc_type = self.detect_doc_type(question)
        
        # Detect question breadth
        is_broad = self.is_broad_question(question)
        is_specific = self.is_specific_question(question)
        is_cumulative = self.is_cumulative_query(question)
        
        # Default parameters from config
        params = {
            "doc_type": doc_type,
            "term_id": term_id,
            "top_k": self.config.default_top_k,
            "null_threshold": self.config.default_null_threshold,
            "max_distance": self.config.default_max_distance,
            "rerank": False,
            "rerank_lex_weight": 0.5,
            "is_ambiguous": is_ambiguous,
            "ambiguity_score": ambiguity_score,
        }
        
        # Special handling for cumulative queries
        if is_cumulative:
            params["top_k"] = 10  # More chunks to ensure we get the summary doc
            params["null_threshold"] = 0.55  # Slightly more lenient
            params["max_distance"] = 0.55
            logger.info("Cumulative query detected: increasing top_k to ensure summary retrieval")
        
        # Special handling for "personal projects" queries
        elif "personal" in question.lower() and "project" in question.lower():
            params["top_k"] = 15  # Cast wider net, filter in Python
            params["rerank"] = True
            params["rerank_lex_weight"] = 0.6
            params["null_threshold"] = 0.65
            params["max_distance"] = 0.65
            params["post_filter_section_prefix"] = "Personal Projects"  # Filter after retrieval
            logger.info("Personal projects query detected: will filter by section prefix after retrieval")
            logger.info("Personal projects query detected: increasing top_k to 15, enabling rerank, and expanding query")
        
        # Broad questions need more chunks
        elif is_broad:
            params["top_k"] = 10
            params["rerank"] = True
            params["null_threshold"] = 0.55  # More lenient
            params["max_distance"] = 0.55
            logger.info("Broad question detected: increasing top_k and enabling rerank")
        
        # Specific questions can use fewer, stricter chunks
        elif is_specific:
            params["top_k"] = 5
            params["null_threshold"] = 0.45  # Stricter
            params["max_distance"] = 0.50
            logger.info("Specific question detected: using default tight parameters")
        
        # If doc_type detected, can be more confident with fewer chunks
        # BUT don't override special handling (cumulative, personal projects, broad)
        if doc_type:
            # Only reduce top_k if we haven't already set it higher for a special case
            is_personal_projects = "personal" in question.lower() and "project" in question.lower()
            if not is_broad and not is_cumulative and not is_personal_projects:
                params["top_k"] = 5
            logger.info(f"Doc type filter: {doc_type}")
        
        # No doc_type detected and broad: cast a wide net
        if not doc_type and is_broad:
            params["top_k"] = 15
            params["rerank"] = True
            logger.info("Broad question without doc_type: maximizing retrieval")
        
        # Log final decision
        logger.info(f"Query routing decision: {params}")
        
        return params

# Global router instance with default configuration
_router = QueryRouter()

# Default routing parameters (for backward compatibility)
DEFAULT_ROUTING_PARAMS = {
    "top_k": query_router_settings.default_top_k,
    "null_threshold": query_router_settings.default_null_threshold,
    "max_distance": query_router_settings.default_max_distance,
    "is_ambiguous": False,
    "ambiguity_score": 0.0,
    "rerank": False,
    "rerank_lex_weight": 0.5,
}

def route_query(
    question: str,
    cert_registry: Optional["CertificationRegistry"] = None,
) -> Dict[str, Any]:
    """
    Convenience function to route a query.
    
    Args:
        question: User's question

    Returns:
        Dict of retrieval parameters
    """
    _router.set_cert_registry(cert_registry)
    return _router.route(question)