"""
BM25 keyword search for hybrid retrieval.

Combines BM25 keyword matching with semantic vector search using
Reciprocal Rank Fusion (RRF) for optimal results.
"""

import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class BM25Index:
    """BM25 index for keyword-based document retrieval."""

    def __init__(self, index_path: Optional[str] = None):
        """Initialize BM25 index.

        Args:
            index_path: Path to save/load BM25 index (pickle file)
        """
        self.index_path = index_path or "data/chroma/bm25_index.pkl"
        self.bm25: Optional[BM25Okapi] = None
        self.documents: List[Dict[str, Any]] = []
        self.doc_ids: List[str] = []

    def build_index(self, documents: List[Dict[str, Any]]):
        """Build BM25 index from documents.

        Args:
            documents: List of document dicts with 'id' and 'text' keys
        """
        logger.info(f"Building BM25 index with {len(documents)} documents...")

        self.documents = documents
        self.doc_ids = [doc["id"] for doc in documents]

        # Tokenize documents (simple whitespace tokenization)
        tokenized_corpus = [self._tokenize(doc["text"]) for doc in documents]

        # Build BM25 index
        self.bm25 = BM25Okapi(tokenized_corpus)

        logger.info(f"BM25 index built successfully with {len(documents)} documents")

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25.

        Simple whitespace + lowercase tokenization.
        Can be improved with stemming, stopword removal, etc.

        Args:
            text: Input text

        Returns:
            List of tokens
        """
        # Simple tokenization: lowercase + split on whitespace
        return text.lower().split()

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        """Search documents using BM25.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of documents with BM25 scores
        """
        if self.bm25 is None:
            raise ValueError("BM25 index not built. Call build_index() first.")

        # Tokenize query
        tokenized_query = self._tokenize(query)

        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)

        # Get top K document indices
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]

        # Build results
        results = []
        for idx in top_k_indices:
            if scores[idx] > 0:  # Only include documents with non-zero scores
                doc = self.documents[idx].copy()
                doc["bm25_score"] = float(scores[idx])
                # Add distance field for compatibility (BM25 doesn't use distance)
                doc["distance"] = None
                results.append(doc)

        logger.info(f"BM25 search for '{query}': {len(results)} results")
        return results

    def save_index(self):
        """Save BM25 index to disk."""
        if self.bm25 is None:
            logger.warning("No BM25 index to save")
            return

        # Create directory if needed
        Path(self.index_path).parent.mkdir(parents=True, exist_ok=True)

        # Save index data
        index_data = {
            "bm25": self.bm25,
            "documents": self.documents,
            "doc_ids": self.doc_ids
        }

        with open(self.index_path, "wb") as f:
            pickle.dump(index_data, f)

        logger.info(f"BM25 index saved to {self.index_path}")

    def load_index(self) -> bool:
        """Load BM25 index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not Path(self.index_path).exists():
            logger.info(f"BM25 index file not found: {self.index_path}")
            return False

        try:
            with open(self.index_path, "rb") as f:
                index_data = pickle.load(f)

            self.bm25 = index_data["bm25"]
            self.documents = index_data["documents"]
            self.doc_ids = index_data["doc_ids"]

            logger.info(f"BM25 index loaded from {self.index_path} ({len(self.documents)} docs)")
            return True

        except Exception as e:
            logger.error(f"Error loading BM25 index: {e}")
            return False


def reciprocal_rank_fusion(
    results_list: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """Merge multiple ranked result lists using Reciprocal Rank Fusion (RRF).

    RRF formula: score(doc) = sum(1 / (k + rank(doc, list_i))) for all lists

    Args:
        results_list: List of result lists (e.g., [bm25_results, semantic_results])
        k: RRF constant (default 60, standard value from literature)

    Returns:
        Merged and reranked results
    """
    # Collect all unique document IDs
    doc_scores: Dict[str, float] = {}
    doc_data: Dict[str, Dict[str, Any]] = {}

    for results in results_list:
        for rank, doc in enumerate(results, start=1):
            doc_id = doc["id"]

            # RRF score contribution from this ranking
            rrf_score = 1.0 / (k + rank)

            # Accumulate score
            if doc_id in doc_scores:
                doc_scores[doc_id] += rrf_score
                # Merge document data: prefer non-None distance from semantic search
                if doc_id in doc_data:
                    existing_distance = doc_data[doc_id].get("distance")
                    new_distance = doc.get("distance")
                    # If existing distance is None but new one isn't, update
                    if existing_distance is None and new_distance is not None:
                        doc_data[doc_id] = doc
            else:
                doc_scores[doc_id] = rrf_score
                doc_data[doc_id] = doc

    # Sort by RRF score
    sorted_doc_ids = sorted(doc_scores.keys(), key=lambda x: doc_scores[x], reverse=True)

    # Build final result list
    merged_results = []
    for doc_id in sorted_doc_ids:
        doc = doc_data[doc_id].copy()
        doc["rrf_score"] = doc_scores[doc_id]
        # Ensure distance field exists for compatibility
        if "distance" not in doc:
            doc["distance"] = None
        merged_results.append(doc)

    logger.info(f"RRF fusion: merged {len(results_list)} result lists into {len(merged_results)} unique docs")
    return merged_results


def hybrid_search(
    query: str,
    bm25_index: BM25Index,
    semantic_search_fn,
    k: int = 5,
    bm25_k: int = 20,
    semantic_k: int = 20
) -> List[Dict[str, Any]]:
    """Perform hybrid search combining BM25 and semantic search.

    Args:
        query: Search query
        bm25_index: BM25 index instance
        semantic_search_fn: Function for semantic search (e.g., ChromaDB search)
        k: Number of final results to return
        bm25_k: Number of results to retrieve from BM25
        semantic_k: Number of results to retrieve from semantic search

    Returns:
        Merged results using RRF
    """
    # Retrieve from both systems
    bm25_results = bm25_index.search(query, k=bm25_k)
    semantic_results = semantic_search_fn(query, k=semantic_k)

    logger.info(f"Hybrid search: BM25={len(bm25_results)} docs, Semantic={len(semantic_results)} docs")

    # Merge using RRF
    merged_results = reciprocal_rank_fusion([bm25_results, semantic_results])

    # Return top K
    return merged_results[:k]
