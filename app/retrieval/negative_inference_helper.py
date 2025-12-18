"""
Negative Inference Helper for RAG System

Detects when a query might require negative inference by checking if
specific entities in the query don't exist in the knowledge base.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

from app.retrieval.store import search

logger = logging.getLogger(__name__)

# Distance threshold: entities with best match > this likely don't exist in KB
ENTITY_NOT_FOUND_THRESHOLD = float(os.getenv("NEGATIVE_INFERENCE_THRESHOLD", "0.37"))

# Adaptive threshold method: 'fixed', 'gap_based', 'context_aware'
THRESHOLD_METHOD = os.getenv("NEGATIVE_INFERENCE_METHOD", "gap_based")


def extract_potential_entities(question: str) -> List[str]:
    """Extract potential entity names from a question.

    Looks for proper nouns, acronyms, and specific technical terms that might
    be entities we need to check.

    Args:
        question: User's question

    Returns:
        List of potential entity strings to check
    """
    entities = []

    # Pattern 1: Capitalized words (potential company names, products, etc.)
    # e.g., "Microsoft", "Google", "Oracle"
    capitalized = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", question)
    entities.extend(capitalized)

    # Pattern 2: All-caps acronyms (potential certs, technologies)
    # e.g., "AWS", "PhD", "EdD", "CCNA"
    acronyms = re.findall(r"\b[A-Z]{2,}\b", question)
    entities.extend(acronyms)

    # Pattern 3: Technical terms from common patterns
    tech_patterns = [
        r"\b(blockchain|mobile apps?|iOS|Android|React Native)\b",
        r"\b(PhD|EdD|JD|MBA)\b",  # Degrees
    ]
    for pattern in tech_patterns:
        matches = re.findall(pattern, question, re.IGNORECASE)
        entities.extend(matches)

    # Remove duplicates and common words
    stopwords = {"I", "Do", "Did", "Have", "My", "The", "A", "An"}
    entities = [e for e in entities if e not in stopwords]

    return list(set(entities))


def check_entity_exists(
    entity: str,
    k: int = 5,
    threshold: float = ENTITY_NOT_FOUND_THRESHOLD,
    method: str = THRESHOLD_METHOD,
) -> Tuple[bool, float]:
    """Check if an entity likely exists in the knowledge base.

    Supports multiple detection methods:
    - 'fixed': Use fixed threshold (default 0.37)
    - 'gap_based': Analyze distance gaps between top results
    - 'context_aware': Use entity-type specific thresholds

    Args:
        entity: Entity name to check (e.g., "Microsoft", "PhD", "blockchain")
        k: Number of results to retrieve
        threshold: Distance threshold for 'fixed' method
        method: Detection method to use

    Returns:
        Tuple of (exists: bool, best_distance: float)
    """
    chunks = search(query=entity, k=k, max_distance=1.0)

    if not chunks:
        return False, 1.0

    best_distance = chunks[0]["distance"]

    # Use adaptive threshold methods if enabled
    if method == "gap_based":
        from app.retrieval.adaptive_threshold import calculate_gap_based_threshold

        exists, distance, reason = calculate_gap_based_threshold(entity, k=k)
        logger.debug(f"Entity '{entity}' gap-based check: {reason}")
        return exists, distance

    elif method == "context_aware":
        from app.retrieval.adaptive_threshold import calculate_context_aware_threshold

        exists, distance, reason = calculate_context_aware_threshold(entity)
        logger.debug(f"Entity '{entity}' context-aware check: {reason}")
        return exists, distance

    # Default: fixed threshold
    exists = best_distance <= threshold

    logger.debug(
        f"Entity '{entity}' check: best_distance={best_distance:.4f}, "
        f"exists={exists} (threshold={threshold})"
    )

    return exists, best_distance


def detect_negative_inference_opportunity(question: str) -> Optional[Dict[str, any]]:
    """Detect if a question might need negative inference.

    Analyzes the question to determine if:
    1. It's asking about a specific entity ("Do I have X?", "Did I work at Y?")
    2. The entity doesn't appear to exist in the knowledge base
    3. We should search for the category instead to find a complete list

    Args:
        question: User's question

    Returns:
        Dict with analysis results, or None if not a negative inference case:
        {
            'is_negative_inference_candidate': bool,
            'missing_entities': List[str],
            'suggested_category_search': str,
        }
    """
    # Check if question matches negative inference patterns
    negative_patterns = [
        r"do\s+i\s+have\s+(?:a|an|any)?\s*(.+)\?",
        r"did\s+i\s+(?:work\s+at|work\s+for|intern\s+at)\s+(.+)\?",
        r"have\s+i\s+(?:built|created|developed)\s+(?:any)?\s*(.+)\?",
    ]

    matches_pattern = False
    for pattern in negative_patterns:
        if re.search(pattern, question, re.IGNORECASE):
            matches_pattern = True
            break

    if not matches_pattern:
        return None

    # Extract potential entities
    entities = extract_potential_entities(question)

    if not entities:
        return None

    # Check which entities don't exist
    missing_entities = []
    for entity in entities:
        exists, distance = check_entity_exists(entity)
        if not exists:
            missing_entities.append({"entity": entity, "distance": distance})

    if not missing_entities:
        return None

    # Determine category search based on question pattern
    category_mapping = {
        r"work|company|employer|job": "work experience companies employment history",
        r"certification|certified|cert": "certifications credentials professional certifications",
        r"degree|phd|edd|mba|bachelor|master": "education degrees academic",
        r"project|built|developed|created": "personal projects built developed",
        r"skill|technology|framework": "technical skills technologies",
    }

    suggested_category = None
    for pattern, category in category_mapping.items():
        if re.search(pattern, question, re.IGNORECASE):
            suggested_category = category
            break

    if not suggested_category:
        suggested_category = "experience background summary"

    return {
        "is_negative_inference_candidate": True,
        "missing_entities": missing_entities,
        "suggested_category_search": suggested_category,
        "question": question,
    }


__all__ = [
    "detect_negative_inference_opportunity",
    "check_entity_exists",
    "extract_potential_entities",
]
