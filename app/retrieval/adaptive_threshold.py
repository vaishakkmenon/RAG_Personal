"""
Adaptive Threshold Calculation for Entity Existence Detection

Instead of a fixed threshold, dynamically calculates thresholds based on:
1. Percentile-based approach (e.g., top 10% of all distances)
2. Statistical outlier detection
3. Context-aware thresholds based on entity type
"""

import logging
from typing import List, Optional, Tuple

from app.retrieval.store import search

logger = logging.getLogger(__name__)


def calculate_percentile_threshold(
    entity: str,
    sample_size: int = 10,
    percentile: float = 0.85
) -> Tuple[float, float]:
    """Calculate threshold based on distance distribution percentile.

    Instead of a fixed threshold, we look at where the entity's best match
    falls in the distribution of all possible matches.

    Args:
        entity: Entity to check
        sample_size: Number of results to sample for distribution
        percentile: Percentile cutoff (0.85 = top 15% considered "exists")

    Returns:
        Tuple of (best_distance, threshold_at_percentile)
    """
    chunks = search(query=entity, k=sample_size, max_distance=1.0)

    if not chunks:
        return 1.0, 0.85  # No results = doesn't exist

    distances = [c['distance'] for c in chunks]
    best_distance = distances[0]

    # Sort distances and find the percentile
    sorted_distances = sorted(distances)
    percentile_index = int(len(sorted_distances) * percentile)
    threshold = sorted_distances[percentile_index] if percentile_index < len(sorted_distances) else 1.0

    return best_distance, threshold


def calculate_gap_based_threshold(entity: str, k: int = 5) -> Tuple[bool, float, str]:
    """Detect entity existence based on distance gap between top results.

    If an entity truly exists, there should be:
    - Low best distance (< 0.4)
    - Small gap between 1st and 2nd result (similar chunks about same entity)

    If entity doesn't exist:
    - High best distance (> 0.4)
    - OR large gap between results (random unrelated matches)

    Args:
        entity: Entity to check
        k: Number of results to examine

    Returns:
        Tuple of (exists, best_distance, reason)
    """
    chunks = search(query=entity, k=k, max_distance=1.0)

    if not chunks:
        return False, 1.0, "No results found"

    distances = [c['distance'] for c in chunks]
    best_distance = distances[0]

    # Rule 1: Very low distance = definitely exists
    if best_distance < 0.30:
        return True, best_distance, f"Strong match (distance={best_distance:.3f})"

    # Rule 2: Very high distance = definitely doesn't exist
    if best_distance > 0.42:
        return False, best_distance, f"Weak match (distance={best_distance:.3f})"

    # Rule 3: Check gap between top results
    if len(distances) >= 2:
        gap = distances[1] - distances[0]

        # Small gap = multiple similar chunks about same entity = exists
        if gap < 0.08 and best_distance < 0.38:
            return True, best_distance, f"Consistent results (gap={gap:.3f})"

        # Large gap = scattered results = doesn't exist
        if gap > 0.15:
            return False, best_distance, f"Inconsistent results (gap={gap:.3f})"

    # Rule 4: Gray area - use conservative threshold
    if best_distance < 0.37:
        return True, best_distance, f"Moderate match (distance={best_distance:.3f})"
    else:
        return False, best_distance, f"Below threshold (distance={best_distance:.3f})"


def calculate_context_aware_threshold(
    entity: str,
    entity_type: Optional[str] = None
) -> Tuple[bool, float, str]:
    """Calculate threshold based on entity type context.

    Different types of entities have different similarity characteristics:
    - Proper nouns (companies, people) tend to have higher distances when missing
    - Common terms (technologies) might have lower distances even when missing
    - Acronyms (PhD, EdD) can be tricky

    Args:
        entity: Entity to check
        entity_type: Optional type hint ('company', 'technology', 'degree', etc.)

    Returns:
        Tuple of (exists, best_distance, reason)
    """
    chunks = search(query=entity, k=3, max_distance=1.0)

    if not chunks:
        return False, 1.0, "No results found"

    best_distance = chunks[0]['distance']

    # Infer entity type if not provided
    if entity_type is None:
        entity_lower = entity.lower()

        if entity.isupper() and len(entity) <= 5:
            entity_type = 'acronym'
        elif entity[0].isupper() and ' ' not in entity:
            entity_type = 'proper_noun'
        elif any(word in entity_lower for word in ['certification', 'degree', 'phd', 'mba']):
            entity_type = 'credential'
        else:
            entity_type = 'general'

    # Type-specific thresholds
    thresholds = {
        'acronym': 0.38,  # Acronyms like CKA, AWS - higher threshold
        'proper_noun': 0.40,  # Companies, people - higher threshold
        'credential': 0.35,  # Degrees, certs - lower threshold (PhD/EdD overlap)
        'technology': 0.36,  # Technologies - medium threshold
        'general': 0.37,  # Default
    }

    threshold = thresholds.get(entity_type, 0.37)
    exists = best_distance <= threshold

    return exists, best_distance, f"Type={entity_type}, threshold={threshold:.3f}"


def check_entity_exists_adaptive(
    entity: str,
    method: str = 'gap_based'
) -> Tuple[bool, float]:
    """Check if entity exists using adaptive threshold.

    Args:
        entity: Entity name to check
        method: Method to use ('gap_based', 'percentile', 'context_aware')

    Returns:
        Tuple of (exists: bool, best_distance: float)
    """
    if method == 'gap_based':
        exists, distance, reason = calculate_gap_based_threshold(entity)
        logger.debug(f"Gap-based check for '{entity}': {exists}, {reason}")
        return exists, distance

    elif method == 'percentile':
        best_distance, threshold = calculate_percentile_threshold(entity)
        exists = best_distance <= threshold
        logger.debug(
            f"Percentile check for '{entity}': distance={best_distance:.3f}, "
            f"threshold={threshold:.3f}, exists={exists}"
        )
        return exists, best_distance

    elif method == 'context_aware':
        exists, distance, reason = calculate_context_aware_threshold(entity)
        logger.debug(f"Context-aware check for '{entity}': {exists}, {reason}")
        return exists, distance

    else:
        raise ValueError(f"Unknown method: {method}")


__all__ = [
    'check_entity_exists_adaptive',
    'calculate_gap_based_threshold',
    'calculate_percentile_threshold',
    'calculate_context_aware_threshold',
]
