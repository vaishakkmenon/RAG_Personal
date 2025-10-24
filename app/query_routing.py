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
        "transcript": [
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