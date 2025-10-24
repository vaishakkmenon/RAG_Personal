"""
Query Router for Personal RAG System

Analyzes user questions and automatically selects optimal retrieval parameters:
- Document type filters (resume, transcript, certificate, course)
- Number of chunks to retrieve (top_k)
- Confidence thresholds
- Reranking settings
"""

import re
from typing import Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)


class QueryRouter:
    """
    Routes queries to optimal retrieval settings based on question content.
    """
    
    # Keyword patterns for document types
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
            r"\binternship",
            r"\b(current|previous|recent)\s+(role|job|position)\b",
            r"\bskills?\b",
            r"\bproject",
        ],
        "term": [
            r"\b(gpa|grade\s+point\s+average)\b",
            r"\bacademic\s+(record|performance|standing)\b",
            r"\bdegree",
            r"\bgraduat(e|ion|ed)",
            r"\b(undergraduate|graduate)\b",
            r"\bcredit(s)?\b.*\b(earned|completed)\b",
            r"\bhonors?\b",
        ],
        "course": [
            r"\bcourse(s)?\b",
            r"\bclass(es)?\b",
            r"\b(cs|ee)\s*\d{3}",  # Course codes like CS 660
            r"\b(spring|fall|summer|winter)\s+\d{4}\b",  # Terms
            r"\bsemester",
            r"\bterm\b",
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
    ]
    
    SPECIFIC_QUESTION_PATTERNS = [
        r"\bwhat\s+was\b",
        r"\bwhen\s+did\b",
        r"\bwhere\b",
        r"\bhow\s+many\b",
        r"\bwhich\b",
        r"\blist\b",
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
        
    def detect_doc_type(self, question: str) -> Optional[str]:
        """
        Detect the most relevant document type based on question content.
        
        Args:
            question: User's question
        
        Returns:
            Document type string or None if no clear match
        """
        scores = {}
        
        for doc_type, patterns in self.doc_type_regexes.items():
            score = sum(1 for pattern in patterns if pattern.search(question))
            if score > 0:
                scores[doc_type] = score
        
        if not scores:
            return None
        
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
    
    def route(self, question: str) -> Dict[str, Any]:
        """
        Analyze question and return optimal retrieval parameters.
        
        Args:
            question: User's question
        
        Returns:
            Dict with: doc_type, top_k, null_threshold, max_distance, rerank, etc.
        """
        question = question.strip()
        
        # Detect document type
        doc_type = self.detect_doc_type(question)
        
        # Detect question breadth
        is_broad = self.is_broad_question(question)
        is_specific = self.is_specific_question(question)
        
        # Default parameters
        params = {
            "doc_type": doc_type,
            "top_k": 5,
            "null_threshold": 0.60,
            "max_distance": 0.60,
            "rerank": False,
            "rerank_lex_weight": 0.5,
        }
        
        # Adjust based on question type and doc_type
        
        # Broad questions need more chunks
        if is_broad:
            params["top_k"] = 10
            params["rerank"] = True
            params["null_threshold"] = 0.65  # More lenient
            params["max_distance"] = 0.65
            logger.info("Broad question detected: increasing top_k and enabling rerank")
        
        # Specific questions can use fewer, stricter chunks
        elif is_specific:
            params["top_k"] = 5
            params["null_threshold"] = 0.55  # Stricter
            params["max_distance"] = 0.60
            logger.info("Specific question detected: using default tight parameters")
        
        # If doc_type detected, can be more confident with fewer chunks
        if doc_type:
            # When filtering by doc_type, we can use fewer chunks
            if not is_broad:
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


def route_query(question: str) -> Dict[str, Any]:
    """
    Convenience function to route a query.
    
    Args:
        question: User's question
    
    Returns:
        Dict of retrieval parameters
    """
    return _router.route(question)