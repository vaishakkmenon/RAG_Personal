"""
Cross-Encoder Neural Reranker for RAG System

Optimized for production use with <200ms latency target.
Uses sentence-transformers CrossEncoder for query-document scoring.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Singleton instance (lazy loaded)
_cross_encoder_instance: Optional["CrossEncoderReranker"] = None


class CrossEncoderReranker:
    """
    Cross-encoder reranker using sentence-transformers.

    Optimizations:
    - Lazy loading (model only loaded on first use)
    - Singleton pattern (one model instance per process)
    - Batch processing for efficiency
    - Configurable top_k to reduce inference time
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        cache_dir: Optional[str] = None,
        max_latency_ms: float = 400.0,
    ):
        """
        Initialize cross-encoder reranker.

        Args:
            model_name: HuggingFace model name
            cache_dir: Directory to cache model files
            max_latency_ms: Warning threshold for latency
        """
        self.model_name = model_name
        self.cache_dir = cache_dir or "/tmp/cross-encoder"
        self.max_latency_ms = max_latency_ms
        self.model = None  # Lazy loaded

        logger.info(f"CrossEncoderReranker initialized (model: {model_name})")
        logger.info("Model will be loaded on first use (lazy loading)")

    def _load_model(self):
        """Load the cross-encoder model (lazy loading)."""
        if self.model is not None:
            return  # Already loaded

        try:
            import os
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading cross-encoder model: {self.model_name}")
            start_time = time.time()

            # Create cache directory
            Path(self.cache_dir).mkdir(parents=True, exist_ok=True)

            # Set HuggingFace cache environment variables BEFORE loading
            # This ensures models are downloaded to writable location
            os.environ["HF_HOME"] = self.cache_dir
            os.environ["HUGGINGFACE_HUB_CACHE"] = self.cache_dir

            logger.info(f"Using cache directory: {self.cache_dir}")

            # Load model with caching
            self.model = CrossEncoder(
                self.model_name,
                max_length=512,
                device="cpu",  # CPU for now, can be configured for GPU
            )

            load_time = (time.time() - start_time) * 1000
            logger.info(f"Cross-encoder model loaded successfully ({load_time:.0f}ms)")

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load cross-encoder model: {e}")
            raise

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Rerank chunks using cross-encoder.

        Args:
            query: User query
            chunks: List of chunk dicts with 'text' key
            top_k: Number of top chunks to return

        Returns:
            Reranked list of top_k chunks with 'cross_encoder_score' added
        """
        if not chunks:
            return []

        # Lazy load model
        if self.model is None:
            self._load_model()

        start_time = time.time()

        try:
            # Prepare query-chunk pairs for cross-encoder
            pairs = [[query, chunk.get("text", "")] for chunk in chunks]

            # Score all pairs (batch processing)
            scores = self.model.predict(pairs, show_progress_bar=False)

            # Combine chunks with scores
            scored_chunks = [
                {**chunk, "cross_encoder_score": float(score)}
                for chunk, score in zip(chunks, scores)
            ]

            # Sort by cross-encoder score (descending)
            scored_chunks.sort(key=lambda x: x["cross_encoder_score"], reverse=True)

            # Return top_k
            result = scored_chunks[:top_k]

            # Log latency
            latency_ms = (time.time() - start_time) * 1000
            logger.info(
                f"Cross-encoder reranked {len(chunks)} → {top_k} chunks "
                f"in {latency_ms:.1f}ms"
            )

            # Warn if latency exceeds threshold
            if latency_ms > self.max_latency_ms:
                logger.warning(
                    f"Cross-encoder latency ({latency_ms:.1f}ms) exceeds "
                    f"threshold ({self.max_latency_ms}ms)"
                )

            return result

        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            logger.warning("Falling back to original ranking")
            return chunks[:top_k]


def get_cross_encoder_reranker(
    model_name: Optional[str] = None,
    cache_dir: Optional[str] = None,
    max_latency_ms: Optional[float] = None,
) -> CrossEncoderReranker:
    """
    Get singleton cross-encoder reranker instance.

    Args:
        model_name: HuggingFace model name (if creating new instance)
        cache_dir: Cache directory (if creating new instance)
        max_latency_ms: Latency threshold (if creating new instance)

    Returns:
        CrossEncoderReranker instance
    """
    global _cross_encoder_instance

    if _cross_encoder_instance is None:
        # Create new instance with provided or default parameters
        from app.settings import settings

        _cross_encoder_instance = CrossEncoderReranker(
            model_name=model_name or settings.cross_encoder.model,
            cache_dir=cache_dir or settings.cross_encoder.cache_dir,
            max_latency_ms=max_latency_ms or settings.cross_encoder.max_latency_ms,
        )

    return _cross_encoder_instance


__all__ = ["CrossEncoderReranker", "get_cross_encoder_reranker"]
