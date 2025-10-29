"""
Query Router for Personal RAG System

Analyzes user questions and automatically selects optimal retrieval parameters:
- Document type filters (resume, transcript_analysis, term, certificate, course)
- Number of chunks to retrieve (top_k)
- Confidence thresholds
- Reranking settings
"""

import re
from typing import Dict, Optional, Any, Tuple, List, Set
import logging
from .settings import query_router_settings

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Routes queries to optimal retrieval settings based on question content.
    
    Configuration is loaded from settings.query_router_settings.
    """
    
    def __init__(self, config=None):
        """
        Initialize the query router with configuration.
        
        Args:
            config: Optional QueryRouterSettings instance. If None, uses the global settings.
        """
        self.config = config or query_router_settings
        self._compile_patterns()
    
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
    
    def route(self, question: str) -> Dict[str, Any]:
        """
        Analyze question and return optimal retrieval parameters.
        
        Args:
            question: User's question
        
        Returns:
            Dict with: doc_type, top_k, null_threshold, max_distance, rerank, etc.
        """
        question = question.strip()
        
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
    "rerank": False,
    "rerank_lex_weight": 0.5,
}

def route_query(question: str) -> Dict[str, Any]:
    """
    Convenience function to route a query.
    
    Args:
        question: User's question
    
    Returns:
        Dict of retrieval parameters
    """
    return _router.route(question)