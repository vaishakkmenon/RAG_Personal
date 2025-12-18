"""
Reranking service for Personal RAG system.

Provides hybrid reranking combining lexical overlap with semantic similarity.
"""

import logging
import re
import time
from typing import List, Set

logger = logging.getLogger(__name__)

# Import reranking metrics
try:
    from app.metrics import (
        rag_rerank_total,
        rag_rerank_latency_seconds,
        rag_rerank_score_distribution,
    )

    RERANK_METRICS_ENABLED = True
except ImportError:
    RERANK_METRICS_ENABLED = False
    logger.debug("Reranking metrics not available")

# Tiny stopword set for lexical overlap calculation
_STOPWORDS: Set[str] = {
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "for",
    "and",
    "or",
    "if",
    "is",
    "are",
    "was",
    "were",
    "by",
    "with",
    "from",
    "as",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
    "be",
    "been",
    "being",
    "which",
    "who",
    "whom",
    "what",
    "when",
    "where",
    "why",
    "how",
}

# Regex for tokenization
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _tokset(text: str) -> Set[str]:
    """Convert text to set of normalized tokens excluding stopwords.

    Args:
        text: Input text to tokenize

    Returns:
        Set of lowercase tokens without stopwords
    """
    return {
        word.lower()
        for word in _WORD_RE.findall(text or "")
        if word.lower() not in _STOPWORDS
    }


class RerankerService:
    """Service for reranking search results using hybrid approach."""

    @staticmethod
    def calculate_lexical_overlap(query: str, text: str) -> float:
        """Calculate lexical overlap between query and text.

        Args:
            query: The search query
            text: The text to compare against

        Returns:
            Overlap score between 0 and 1
        """
        query_tokens = _tokset(query)
        text_tokens = _tokset(text)

        if not query_tokens:
            return 0.0

        overlap = len(query_tokens & text_tokens) / len(query_tokens)
        return overlap

    @staticmethod
    def calculate_hybrid_score(
        query: str, chunk: dict, lex_weight: float = 0.5
    ) -> float:
        """Calculate hybrid score combining lexical and semantic similarity.

        Score = lex_weight * overlap + (1 - lex_weight) * (1 - distance)

        Args:
            query: The search query
            chunk: Chunk dict with 'text' and 'distance' keys
            lex_weight: Weight for lexical vs semantic (0-1)

        Returns:
            Combined score between 0 and 1
        """
        text = chunk.get("text", "")
        distance = chunk.get("distance")

        # Handle None distance (from BM25/hybrid chunks without semantic distance)
        if distance is None:
            distance = 0.5  # Neutral distance when unknown

        # Lexical overlap
        overlap = RerankerService.calculate_lexical_overlap(query, text)

        # Semantic similarity (inverse of distance)
        similarity = 1.0 - max(0.0, min(1.0, distance))

        # Combine scores
        score = lex_weight * overlap + (1.0 - lex_weight) * similarity
        return score

    @staticmethod
    def rerank(
        question: str, chunks: List[dict], lex_weight: float = 0.5
    ) -> List[dict]:
        """Rerank chunks using hybrid lexical + semantic scoring.

        Args:
            question: The user's question
            chunks: List of chunk dicts with 'text' and 'distance' keys
            lex_weight: Weight for lexical overlap (0-1)
                       0 = pure semantic, 1 = pure lexical, 0.5 = balanced

        Returns:
            Reranked list of chunks (sorted by hybrid score, descending)
        """
        if not chunks:
            return []

        # Track reranking latency
        start_time = time.time()

        # Clamp weight to valid range
        lex_weight = max(0.0, min(1.0, lex_weight))

        # Calculate scores and sort
        scored_chunks = [
            (chunk, RerankerService.calculate_hybrid_score(question, chunk, lex_weight))
            for chunk in chunks
        ]

        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        logger.debug(f"Reranked {len(chunks)} chunks with lex_weight={lex_weight:.2f}")

        # Track metrics
        if RERANK_METRICS_ENABLED:
            method = "bm25"  # This is the hybrid BM25-style reranking
            rag_rerank_total.labels(method=method).inc()
            rag_rerank_latency_seconds.labels(method=method).observe(
                time.time() - start_time
            )

            # Track score distribution
            for chunk, score in scored_chunks:
                rag_rerank_score_distribution.labels(method=method).observe(score)

        return [chunk for chunk, score in scored_chunks]


# Convenience function for backward compatibility
def rerank_chunks(
    question: str, chunks: List[dict], lex_weight: float = 0.5
) -> List[dict]:
    """Convenience function for reranking chunks.

    Args:
        question: The user's question
        chunks: List of chunk dicts
        lex_weight: Weight for lexical overlap (0-1)

    Returns:
        Reranked list of chunks
    """
    return RerankerService.rerank(question, chunks, lex_weight)


__all__ = ["RerankerService", "rerank_chunks"]
