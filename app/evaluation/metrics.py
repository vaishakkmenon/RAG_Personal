"""
Retrieval evaluation metrics.

Implements standard IR metrics for evaluating retrieval quality.
"""

import math
from typing import List, Set


def _is_match(retrieved_id: str, expected_id: str) -> bool:
    """Check if a retrieved ID matches an expected ID.
    
    Supports both exact matching and prefix matching:
    - "doc@v#section:5" matches "doc@v#section" (prefix match)
    - "doc@v#section" matches "doc@v#section" (exact match)
    - Also handles the expected ID having index suffix
    
    Args:
        retrieved_id: The chunk ID from retrieval
        expected_id: The expected chunk ID from test case
        
    Returns:
        True if IDs match
    """
    # Exact match
    if retrieved_id == expected_id:
        return True
    
    # Prefix match: retrieved_id = "section:5" matches expected_id = "section"
    # Strip the chunk index suffix from retrieved_id
    if ':' in retrieved_id:
        prefix = retrieved_id.rsplit(':', 1)[0]  # "doc@v#section:5" -> "doc@v#section"
        if prefix == expected_id:
            return True
        # Also check if expected_id is a prefix of prefix
        if prefix.startswith(expected_id):
            return True
    
    # Check if expected_id has an index suffix and matches
    if ':' in expected_id:
        expected_prefix = expected_id.rsplit(':', 1)[0]
        if retrieved_id == expected_prefix:
            return True
        if ':' in retrieved_id:
            retrieved_prefix = retrieved_id.rsplit(':', 1)[0]
            if retrieved_prefix == expected_prefix:
                return True
    
    return False


def _count_matches(retrieved_ids: List[str], relevant_ids: List[str]) -> int:
    """Count how many retrieved IDs match any relevant ID.
    
    Uses prefix matching to handle chunk index suffixes.
    """
    count = 0
    for rid in retrieved_ids:
        for eid in relevant_ids:
            if _is_match(rid, eid):
                count += 1
                break  # Each retrieved ID can only match once
    return count


def _find_first_match_rank(retrieved_ids: List[str], relevant_ids: List[str]) -> int:
    """Find the rank (1-indexed) of the first matching ID.
    
    Returns 0 if no match found.
    """
    for rank, rid in enumerate(retrieved_ids, start=1):
        for eid in relevant_ids:
            if _is_match(rid, eid):
                return rank
    return 0


def _has_any_match(retrieved_ids: List[str], relevant_ids: List[str]) -> bool:
    """Check if any retrieved ID matches any relevant ID."""
    for rid in retrieved_ids:
        for eid in relevant_ids:
            if _is_match(rid, eid):
                return True
    return False


def calculate_recall_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = None) -> float:
    """Calculate Recall@K.

    Recall@K = (Number of relevant items in top K) / (Total relevant items)
    
    Uses prefix matching to handle chunk index suffixes.

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider (None = all retrieved)

    Returns:
        Recall score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 0.0

    # Consider only top K results
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Count matches using prefix matching
    num_relevant_retrieved = _count_matches(retrieved_ids, relevant_ids)
    num_relevant_total = len(relevant_ids)

    return num_relevant_retrieved / num_relevant_total if num_relevant_total > 0 else 0.0


def calculate_precision_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = None) -> float:
    """Calculate Precision@K.

    Precision@K = (Number of relevant items in top K) / K
    
    Uses prefix matching to handle chunk index suffixes.

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider (None = all retrieved)

    Returns:
        Precision score between 0.0 and 1.0
    """
    if not retrieved_ids:
        return 0.0

    # Consider only top K results
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Count matches using prefix matching
    num_relevant_retrieved = _count_matches(retrieved_ids, relevant_ids)
    num_retrieved = len(retrieved_ids)

    return num_relevant_retrieved / num_retrieved if num_retrieved > 0 else 0.0


def calculate_mrr(retrieved_ids: List[str], relevant_ids: List[str]) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    MRR = 1 / (rank of first relevant item)
    
    Uses prefix matching to handle chunk index suffixes.

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs (ground truth)

    Returns:
        MRR score between 0.0 and 1.0
    """
    rank = _find_first_match_rank(retrieved_ids, relevant_ids)
    return 1.0 / rank if rank > 0 else 0.0


def calculate_ndcg_at_k(retrieved_ids: List[str], relevant_ids: List[str], k: int = None) -> float:
    """Calculate Normalized Discounted Cumulative Gain (NDCG@K).

    NDCG accounts for the position of relevant documents.
    Higher-ranked relevant documents contribute more to the score.
    
    Uses prefix matching to handle chunk index suffixes.

    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider (None = all retrieved)

    Returns:
        NDCG score between 0.0 and 1.0
    """
    if not relevant_ids:
        return 0.0

    # Consider only top K results
    if k is not None:
        retrieved_ids = retrieved_ids[:k]

    # Calculate DCG (Discounted Cumulative Gain)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        # Check if this doc matches any relevant ID using prefix matching
        is_relevant = any(_is_match(doc_id, rel_id) for rel_id in relevant_ids)
        if is_relevant:
            # Binary relevance: 1 if relevant, 0 otherwise
            relevance = 1.0
            # Discount by log2(rank + 1)
            dcg += relevance / math.log2(rank + 1)

    # Calculate IDCG (Ideal DCG) - best possible ranking
    idcg = 0.0
    for rank in range(1, min(len(relevant_ids), len(retrieved_ids)) + 1):
        idcg += 1.0 / math.log2(rank + 1)

    # Normalize
    return dcg / idcg if idcg > 0 else 0.0


def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculate F1 score from precision and recall.

    F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        precision: Precision score
        recall: Recall score

    Returns:
        F1 score between 0.0 and 1.0
    """
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)
