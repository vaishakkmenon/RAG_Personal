"""
Embedding Cache Service

Caches generated vector embeddings to avoid re-computation for identical text.
Critical for reducing latency and API costs (if using API-based embeddings).
"""

import hashlib
import json
import logging
from typing import List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logger.warning("Redis not available - embedding caching disabled")


class EmbeddingCache:
    """Redis-based cache for embeddings."""

    def __init__(
        self,
        redis_url: str,
        enabled: bool = True,
        ttl_seconds: int = 86400 * 7,  # 7 days default
    ):
        self.enabled = enabled and REDIS_AVAILABLE
        self.ttl_seconds = ttl_seconds
        self._client: Optional[redis.Redis] = None

        if self.enabled:
            try:
                # Use a dedicated pool logic or reuse connection URL
                # For simplicity here, new client
                self._client = redis.Redis.from_url(
                    redis_url,
                    decode_responses=False,  # Keep as bytes for efficiency if possible, or use JSON
                )
                self._client.ping()
                logger.info(f"Embedding cache initialized (TTL: {ttl_seconds}s)")
            except Exception as e:
                logger.error(f"Failed to connect to Redis for embedding cache: {e}")
                self.enabled = False

    def _get_key(self, text: str, model: str) -> str:
        """Generate cache key: rag:embed:<model_hash>:<text_hash>"""
        # Normalize text slightly
        text_norm = text.strip()

        text_hash = hashlib.sha256(text_norm.encode()).hexdigest()
        return f"rag:embed:{model}:{text_hash}"

    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        if not self.enabled or not self._client:
            return None

        try:
            key = self._get_key(text, model)
            cached_bytes = self._client.get(key)

            if cached_bytes:
                # Decoded from bytes -> JSON -> List
                return json.loads(cached_bytes)
        except Exception as e:
            logger.warning(f"Embedding cache get failed: {e}")

        return None

    def set_embedding(
        self, text: str, embedding: Union[List[float], np.ndarray], model: str
    ):
        if not self.enabled or not self._client:
            return

        try:
            key = self._get_key(text, model)

            # Convert numpy to list if needed
            if isinstance(embedding, np.ndarray):
                embedding = embedding.tolist()

            serialized = json.dumps(embedding)
            self._client.setex(key, self.ttl_seconds, serialized)
        except Exception as e:
            logger.warning(f"Embedding cache set failed: {e}")


# Global Entry Point
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    global _embedding_cache
    if _embedding_cache is None:
        from app.settings import settings

        # We can reuse session redis URL for now
        _embedding_cache = EmbeddingCache(
            redis_url=settings.session.redis_url,
            # We could add specific settings for this, but defaults serve well
        )
    return _embedding_cache
