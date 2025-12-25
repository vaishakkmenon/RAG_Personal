"""
Search Engine Strategy

Orchestrates the retrieval strategy, combining:
- Semantic Search (via VectorStore)
- Keyword Search (via BM25)
- Hybrid Fusion (RRF)
- Query Rewriting
- Cross-Encoder Reranking
- Fallback Caching
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.settings import settings
from app.retrieval.vector_store import get_vector_store
from app.retrieval.ranking import apply_boosting_rules

logger = logging.getLogger(__name__)

# Import BM25 for hybrid search
try:
    from app.retrieval.bm25_search import BM25Index, reciprocal_rank_fusion

    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 not available - hybrid search disabled")


class SearchEngine:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.bm25_index: Optional[BM25Index] = None

        # Initialize BM25
        if BM25_AVAILABLE:
            bm25_path = Path(settings.chroma_dir) / "bm25_index.pkl"
            if bm25_path.exists():
                self.bm25_index = BM25Index(index_path=str(bm25_path))
                if self.bm25_index.load_index():
                    logger.info(
                        f"BM25 hybrid search enabled ({len(self.bm25_index.documents)} docs)"
                    )
                else:
                    self.bm25_index = None
                    logger.warning("BM25 index file found but failed to load")
            else:
                logger.info("BM25 index not found - hybrid search disabled")

    def search(
        self,
        query: str,
        k: Optional[int] = None,
        max_distance: Optional[float] = None,
        metadata_filter: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True,
        use_query_rewriting: bool = True,
        use_cross_encoder: Optional[bool] = None,
    ) -> List[dict]:
        """Execute search pipeline.

        Pipeline:
        1. Query Rewriting (optional)
        2. Retrieval (Hybrid or Semantic)
        3. Boosting (Business Rules)
        4. Caching (Fallback)
        5. Reranking (Cross-Encoder)
        """
        original_query = (query or "").strip()
        if not original_query:
            return []

        # 1. Query Rewriting
        search_query = original_query
        rewrite_metadata = None
        if use_query_rewriting and settings.query_rewriter.enabled:
            try:
                from app.retrieval.query_rewriter import get_query_rewriter

                rewriter = get_query_rewriter()
                rewritten_query, rewrite_metadata = rewriter.rewrite_query(
                    original_query
                )
                if rewrite_metadata:
                    search_query = rewritten_query
                    logger.info(
                        f"Rewrote query: '{original_query}' -> '{search_query}'"
                    )
            except Exception as e:
                logger.warning(f"Query rewriting failed: {e}")

        # Defaults
        k = k or settings.top_k
        max_distance = settings.max_distance if max_distance is None else max_distance

        # Cross-Encoder Param Setup
        if use_cross_encoder is None:
            use_cross_encoder = settings.cross_encoder.enabled

        retrieval_k = settings.cross_encoder.retrieval_k if use_cross_encoder else k
        final_k = k

        # 2. Retrieval Execution
        try:
            if use_hybrid and self.bm25_index:
                results = self._hybrid_search(
                    search_query, retrieval_k, max_distance, metadata_filter
                )
            else:
                results = self.vector_store.search(
                    search_query, retrieval_k, max_distance, metadata_filter
                )

            # 3. Boosting
            results = apply_boosting_rules(search_query, results)

            # 4. Caching (Success)
            if results:
                self._cache_results(search_query, results)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            # 4. Caching (Fallback)
            results = self._get_fallback_results(search_query)
            if not results:
                raise e

        # 5. Reranking
        if use_cross_encoder and results:
            try:
                from app.services.cross_encoder_reranker import (
                    get_cross_encoder_reranker,
                )

                cross_encoder = get_cross_encoder_reranker()
                results = cross_encoder.rerank(
                    query=search_query,
                    chunks=results,
                    top_k=settings.cross_encoder.top_k,
                )
            except Exception as e:
                logger.error(f"Reranking failed: {e}")
                results = results[:final_k]

        return results[:final_k]

    def rerank(
        self, query: str, chunks: List[dict], top_k: Optional[int] = None
    ) -> List[dict]:
        """Rerank chunks using cross-encoder."""
        try:
            from app.services.cross_encoder_reranker import get_cross_encoder_reranker

            cross_encoder = get_cross_encoder_reranker()
            target_k = top_k or settings.cross_encoder.top_k

            ranked = cross_encoder.rerank(query=query, chunks=chunks, top_k=target_k)
            return ranked
        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return chunks

    def _hybrid_search(
        self,
        query: str,
        k: int,
        max_distance: float,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Perform hybrid search (BM25 + Semantic)."""
        if not self.bm25_index:
            return self.vector_store.search(query, k, max_distance, metadata_filter)

        # Retrieve 4x candidates for fusion
        bm25_k = k * 4
        semantic_k = k * 4

        bm25_results = self.bm25_index.search(query, k=bm25_k)
        semantic_results = self.vector_store.search(
            query, semantic_k, max_distance, metadata_filter
        )

        merged = reciprocal_rank_fusion(
            [bm25_results, semantic_results], k=settings.bm25.rrf_k, query=query
        )
        return merged[:k]

    def _cache_results(self, query: str, results: List[dict]):
        try:
            from app.retrieval.fallback_cache import get_fallback_cache

            get_fallback_cache().cache_results(query, results)
        except Exception as e:
            logger.warning(f"Failed to cache results: {e}")

    def _get_fallback_results(self, query: str) -> List[dict]:
        try:
            from app.retrieval.fallback_cache import get_fallback_cache

            results = get_fallback_cache().get_fallback_results(query)
            if results:
                logger.warning(f"Using {len(results)} cached fallback results")
            return results
        except Exception:
            return []


# Singleton
_search_engine = None


def get_search_engine() -> SearchEngine:
    global _search_engine
    if _search_engine is None:
        _search_engine = SearchEngine()
    return _search_engine
