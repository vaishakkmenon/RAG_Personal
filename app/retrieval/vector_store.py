"""
Vector Store Operations - ChromaDB integration

Handles low-level interaction with ChromaDB: connection, document storage, and semantic search.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.settings import settings

logger = logging.getLogger(__name__)

# Configuration
EMBED_MODEL: str = settings.embed_model
CHROMA_PATH: str = settings.chroma_dir
COLLECTION_NAME: str = settings.collection_name


class VectorStore:
    def __init__(self):
        self._embed = SentenceTransformerEmbeddingFunction(EMBED_MODEL)
        self._client = chromadb.PersistentClient(
            path=CHROMA_PATH,
            settings=ChromaSettings(allow_reset=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # Cosine distance for BGE models
            embedding_function=self._embed,
        )
        logger.info(
            f"VectorStore initialized (Collection: '{COLLECTION_NAME}', Model: '{EMBED_MODEL}')"
        )

        # BGE v1.5 models benefit from query instruction prefix
        self.BGE_V15_QUERY_PREFIX = (
            "Represent this sentence for searching relevant passages: "
        )

    def get_client(self):
        """Get the ChromaDB client instance."""
        return self._client

    def count(self) -> int:
        return self._collection.count()

    def _should_use_query_instruction(self) -> bool:
        """Check if we're using a BGE v1.5 model that needs query prefix."""
        model_name = EMBED_MODEL.lower()
        return "bge" in model_name and "v1.5" in model_name and "m3" not in model_name

    def _get_source(self, metadata: Any) -> str:
        """Safely extract source from metadata."""
        return (
            metadata.get("source", "unknown")
            if isinstance(metadata, dict)
            else "unknown"
        )

    def add_documents(self, docs: List[dict]) -> None:
        """Add chunks to the vector store.

        Args:
            docs: List of dicts with keys: id, text, metadata
        """
        if not docs:
            logger.warning("add_documents called with empty list")
            return

        self._collection.upsert(
            ids=[d["id"] for d in docs],
            documents=[d["text"] for d in docs],
            metadatas=[d.get("metadata", {}) for d in docs],
        )

        logger.info(f"Upserted {len(docs)} documents to collection")

    def search(
        self,
        query: str,
        k: int,
        max_distance: float,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[dict]:
        """Perform semantic-only search using ChromaDB.

        Args:
            query: Search query
            k: Number of results
            max_distance: Maximum distance threshold
            metadata_filter: Optional metadata filter

        Returns:
            List of results
        """

        # Add query instruction for BGE v1.5 models
        query_text = query
        if self._should_use_query_instruction():
            query_text = self.BGE_V15_QUERY_PREFIX + query
            logger.debug("Added BGE v1.5 query instruction prefix")

        # Build ChromaDB WHERE clause from metadata filter
        where_clause = None
        if metadata_filter:
            where_conditions = []
            for key, value in metadata_filter.items():
                where_conditions.append({key: {"$eq": value}})

            if len(where_conditions) == 1:
                where_clause = where_conditions[0]
            else:
                where_clause = {"$and": where_conditions}

            logger.debug(f"Using metadata filter: {where_clause}")

        # Query ChromaDB
        try:
            # Optimization: Use Embedding Cache
            # We defer this dependency import to avoid circular imports if cache depends on store
            try:
                from app.services.embedding_cache import get_embedding_cache

                embed_cache = get_embedding_cache()
                embedding = embed_cache.get_embedding(query_text, EMBED_MODEL)

                if not embedding:
                    # Cache miss - compute embedding
                    embeddings = self._embed([query_text])
                    if embeddings:
                        embedding = embeddings[0]
                        if hasattr(embedding, "tolist"):
                            embedding = embedding.tolist()
                        embed_cache.set_embedding(query_text, embedding, EMBED_MODEL)

                if embedding:
                    results = self._collection.query(
                        query_embeddings=[embedding],
                        n_results=k,
                        where=where_clause,
                        include=["documents", "metadatas", "distances"],
                    )
                else:
                    raise Exception("Failed to generate embedding")

            except Exception:
                # Fallback to internal embedding function if cache fails
                # logger.debug(f"Embedding cache skipped/failed: {e}")
                results = self._collection.query(
                    query_texts=[query_text],
                    n_results=k,
                    where=where_clause,
                    include=["documents", "metadatas", "distances"],
                )

        except Exception as e:
            logger.error(f"ChromaDB query failed: {e}")
            raise

        # Parse results
        ids = (results.get("ids") or [[]])[0]
        docs = (results.get("documents") or [[]])[0]
        metas = (results.get("metadatas") or [[]])[0]
        dists = (results.get("distances") or [[]])[0]

        # Filter by max_distance and build output
        output: List[dict] = []
        for chunk_id, text, metadata, distance in zip(ids, docs, metas, dists):
            try:
                if distance is not None and distance <= max_distance:
                    output.append(
                        {
                            "id": chunk_id,
                            "text": text,
                            "source": self._get_source(metadata),
                            "distance": distance,
                            "metadata": metadata,
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to process chunk {chunk_id}: {e}")
                continue

        # Sort results by distance (ascending)
        output.sort(
            key=lambda x: x["distance"] if x["distance"] is not None else float("inf")
        )

        return output

    def get_sample_chunks(self, n: int = 10) -> List[dict]:
        """Return up to n random example chunks for testing/debugging."""
        n = max(1, min(n, 100))
        count = self._collection.count()

        if count == 0:
            return []

        start = random.randint(0, max(0, count - n))

        try:
            results = self._collection.get(
                include=["documents", "metadatas"], limit=n, offset=start
            )
        except Exception as e:
            logger.error(f"Failed to get sample chunks: {e}")
            return []

        ids = results.get("ids") or []
        docs = results.get("documents") or []
        metas = results.get("metadatas") or []

        output: List[dict] = []
        for chunk_id, text, metadata in zip(ids, docs, metas):
            output.append(
                {
                    "id": chunk_id,
                    "text": text,
                    "source": self._get_source(metadata),
                    "distance": None,
                    "metadata": metadata,
                }
            )

        return output

    def get_all_documents(self) -> List[dict]:
        """Retrieve all documents from the collection (for re-indexing)."""
        try:
            result = self._collection.get(include=["documents", "metadatas"])

            output = []
            if result and result["ids"]:
                for i, doc_id in enumerate(result["ids"]):
                    text = result["documents"][i] if result["documents"] else ""
                    metadata = result["metadatas"][i] if result["metadatas"] else {}
                    output.append({"id": doc_id, "text": text, "metadata": metadata})

            return output
        except Exception as e:
            logger.error(f"Failed to get all documents: {e}")
            return []

    def heartbeat(self) -> int:
        """Check connection to ChromaDB."""
        return self._client.heartbeat()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database."""
        count = self._collection.count()

        if count == 0:
            return {
                "total_documents": 0,
                "collection_name": COLLECTION_NAME,
                "embed_model": EMBED_MODEL,
            }

        samples = self._collection.get(limit=min(100, count), include=["metadatas"])
        metas = samples.get("metadatas") or []

        sources = set()
        doc_types = set()
        for meta in metas:
            if isinstance(meta, dict):
                if "source" in meta:
                    sources.add(meta["source"])
                if "doc_type" in meta:
                    doc_types.add(meta["doc_type"])

        return {
            "total_documents": count,
            "unique_sources": len(sources),
            "doc_types": sorted(doc_types),
            "collection_name": COLLECTION_NAME,
            "embed_model": EMBED_MODEL,
            "sample_sources": sorted(list(sources))[:10],
        }

    def reset(self) -> None:
        """DANGER: Delete all documents from the collection."""
        try:
            count = self._collection.count()
            logger.warning(
                f"Resetting collection '{COLLECTION_NAME}' - will delete {count} documents"
            )
            self._client.delete_collection(COLLECTION_NAME)
        except Exception as e:
            logger.info(f"Collection doesn't exist or already deleted: {e}")

        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
            embedding_function=self._embed,
        )
        logger.info("Collection reset complete")


# Singleton instance
_vector_store = None


def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store
