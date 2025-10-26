"""
Query Router for Personal RAG System

Analyzes user questions and automatically selects optimal retrieval parameters:
- Document type filters (resume, transcript_analysis, term, certificate, course)
- Number of chunks to retrieve (top_k)
- Confidence thresholds
- Reranking settings
"""

import re
from typing import Dict, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)

# ------------------------------
# Configuration
# ------------------------------
CUMULATIVE_QUERY_BOOST = 2

class QueryRouter:
    """
    Routes queries to optimal retrieval settings based on question content.
    """
    
    # Keyword patterns for document types
    # ORDER MATTERS: More specific patterns should come first
    DOC_TYPE_PATTERNS = {
        "certificate": [
            r"\bcertificat(e|ion)s?\b",
            r"\bcertified\b",
            r"\bcka\b",
            r"\baws\b.*\b(ccp|practitioner)\b",
            r"\bprofessional\s+credential",
        ],
        "resume": [
            r"\b(work|job|employment|position)\s+(experience|history)\b",
            r"\bcompan(y|ies)\b.*\bworked\b",
            r"\bintern(ship|ed)?\b",  # Changed: Now matches intern, interned, internship
            r"\b(current|previous|recent)\s+(role|job|position)\b",
            r"\bskills?\b",
            r"\bpersonal\s+project",  # Added: Matches "personal project(s)" queries
            r"\bproject",
            r"\bcompan(y|ies)\b",  # Added: Match "companies" alone
            r"\bmaven\s+wave\b",    # Added: Company name
        ],
        # NEW: transcript_analysis for OVERALL/CUMULATIVE stats
        # This should match BEFORE "term" for overall queries
        "transcript_analysis": [
            r"\b(overall|cumulative|total|complete)\b.*\b(gpa|credit|grade|academic)\b",
            r"\b(undergraduate|graduate)\b.*\b(gpa|cumulative)\b",
            r"\bhow\s+many\s+(total\s+)?credits?\b",
            r"\btotal\s+credits?\s+(earned|completed)\b",
            r"\bacademic\s+(summary|overview)\b",
            r"\bwhat\s+degrees?\s+(did|have)\b",
            r"\bgraduation\s+date\b",
            r"\bwhen\s+did.*graduate\b",
            r"\bsumma\s+cum\s+laude\b",
        ],
        "term": [
            r"\bterm\s+gpa\b",
            r"\bsemester\s+gpa\b",
            r"\bacademic\s+(record|standing)\b",
            r"\bhonors?\b",
            r"\bgpa\b",
            r"\bgrade\s+point\s+average\b",
            r"\bcredit(s)?\b.*\b(earned|completed)\b",
            # Course-related patterns
            r"\bcourse(s)?\b",
            r"\bclass(es)?\b",
            r"\b(cs|ee|ma|ph|eh)\s*\d{3}",
            r"\b(spring|fall|summer|winter)\s+\d{4}\b",
            r"\bsemester",
            r"\bstudy|studied|took",
        ],
    }
    
    # Question type patterns
    BROAD_QUESTION_PATTERNS = [
        r"\bsummar(y|ize|ization)\b",
        r"\boverall\b",
        r"\beverything\b",
        r"\ball\b.*\b(about|my)\b",
        r"\bbackground\b",
        r"\bprofile\b",
        r"\bqualifications?\b",
    ]
    
    SPECIFIC_QUESTION_PATTERNS = [
        r"\bwhat\s+was\b",
        r"\bwhen\s+did\b",
        r"\bwhere\b",
        r"\bhow\s+many\b",
        r"\bwhich\b",
        r"\blist\b",
        r"\bwhat\s+is\b",
    ]
    
    # Patterns that indicate need for cumulative/summary information
    CUMULATIVE_PATTERNS = [
        r"\b(overall|cumulative|total|complete|entire)\b",
        r"\b(undergraduate|graduate)\b.*\bgpa\b",
        r"\bhow\s+many\s+total\b",
        r"\bacademic\s+(summary|overview|performance)\b",
    ]
    
    def __init__(self):
        """Initialize the query router."""
        # Compile patterns for efficiency
        self.doc_type_regexes = {
            doc_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for doc_type, patterns in self.DOC_TYPE_PATTERNS.items()
        }
        
        self.broad_regexes = [
            re.compile(pattern, re.IGNORECASE) 
            for pattern in self.BROAD_QUESTION_PATTERNS
        ]
        
        self.specific_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SPECIFIC_QUESTION_PATTERNS
        ]
        
        self.cumulative_regexes = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CUMULATIVE_PATTERNS
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
                scores["transcript_analysis"] += CUMULATIVE_QUERY_BOOST
        
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
        
        # Default parameters
        params = {
            "doc_type": doc_type,
            "term_id": term_id, 
            "top_k": 5,
            "null_threshold": 0.50,
            "max_distance": 0.50,
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

# Global router instance
_router = QueryRouter()

DEFAULT_ROUTING_PARAMS = {
    "top_k": 5,
    "null_threshold": 0.50,
    "max_distance": 0.50,
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