"""
Ranking and Scoring Rules

Handles domain-specific logic for boosting search results based on query intent.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


# --- Keyword Constants ---
ACADEMIC_KEYWORDS = [
    "credit",
    "credits",
    "gpa",
    "grade",
    "course",
    "courses",
    "semester",
    "term",
    "degree",
    "undergraduate",
    "graduate",
    "transcript",
    "academic",
    "university",
    "college",
]
CERT_KEYWORDS = ["certification", "certified", "certificate", "cka", "aws"]
WORK_KEYWORDS = [
    "work",
    "job",
    "experience",
    "company",
    "role",
    "position",
    "employment",
]
AGGREGATION_KEYWORDS = [
    "total",
    "overall",
    "how many",
    "all of",
    "summary",
    "statistics",
    "combined",
]
SUMMARY_KEYWORDS = [
    "summary",
    "statistics",
    "overall",
    "gpa",
    "total",
    "performance",
]


def detect_query_intent(query: str) -> dict:
    """Analyze query to determine user intent."""
    query_lower = query.lower()
    return {
        "is_academic": any(k in query_lower for k in ACADEMIC_KEYWORDS),
        "is_cert": any(k in query_lower for k in CERT_KEYWORDS),
        "is_work": any(k in query_lower for k in WORK_KEYWORDS),
        "needs_summary": any(k in query_lower for k in AGGREGATION_KEYWORDS),
    }


def apply_boosting_rules(query: str, chunks: List[dict]) -> List[dict]:
    """Boost chunks from specific documents based on query intent.

    Intelligently boosts document types based on query intent:
    - Academic/credit queries -> boost transcript_analysis docs
    - Certification queries -> boost certificate docs
    - Work/experience queries -> boost resume docs
    - Aggregation queries -> boost summary/statistics sections

    Args:
        query: The search query
        chunks: Retrieved chunks

    Returns:
        Chunks with adjusted distances (boosted chunks), re-sorted.
    """
    if not chunks:
        return []

    # Detect Intent
    intent = detect_query_intent(query)
    is_academic = intent["is_academic"]
    is_cert = intent["is_cert"]
    is_work = intent["is_work"]
    needs_summary = intent["needs_summary"]

    # --- Step 2: Apply Boosts ---

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        doc_type = metadata.get("doc_type", "")
        section_name = (metadata.get("section_name") or "").lower()
        original_distance = chunk.get("distance")

        if original_distance is None:
            continue

        # 2a. Doc Type Boosting
        if is_academic and doc_type == "transcript_analysis":
            chunk["distance"] = original_distance * 0.5
            logger.debug(
                f"Boosted transcript_analysis chunk (dist: {original_distance:.3f} -> {chunk['distance']:.3f})"
            )

        elif is_cert and doc_type == "certificate":
            chunk["distance"] = original_distance * 0.6
            logger.debug(
                f"Boosted certificate chunk (dist: {original_distance:.3f} -> {chunk['distance']:.3f})"
            )

        elif is_work and doc_type == "resume":
            chunk["distance"] = original_distance * 0.7
            logger.debug(
                f"Boosted resume chunk (dist: {original_distance:.3f} -> {chunk['distance']:.3f})"
            )

        # 2b. Section Summary Boosting (Multiplicative)
        if needs_summary and any(k in section_name for k in SUMMARY_KEYWORDS):
            # Apply on top of existing boost
            current_distance = chunk["distance"]
            chunk["distance"] = current_distance * 0.8
            logger.debug(
                f"Boosted summary section '{section_name}' (dist: {current_distance:.3f} -> {chunk['distance']:.3f})"
            )

    # --- Step 3: Re-sort ---

    # Handle potentially None distances (put them at end)
    chunks.sort(
        key=lambda x: x.get("distance") if x.get("distance") is not None else 999.0
    )

    return chunks


def diversify_sources(
    chunks: List[dict], top_k: int, max_per_source: int = 2
) -> List[dict]:
    """Ensure no single source dominates the results.

    Enforces that top-K results include at most N chunks per source file.
    This prevents documents like master_profile from consuming multiple slots.

    Args:
        chunks: List of retrieved chunks
        top_k: Target number of chunks to return
        max_per_source: Max chunks allowed per source file

    Returns:
        Filtered lists of chunks
    """
    source_counts = {}
    diversified = []

    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        source = metadata.get("source", "unknown")

        if source_counts.get(source, 0) < max_per_source:
            diversified.append(chunk)
            source_counts[source] = source_counts.get(source, 0) + 1

        if len(diversified) >= top_k:
            break

    return diversified
