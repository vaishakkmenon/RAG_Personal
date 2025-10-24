"""
Retrieval module for Personal RAG system.

Handles:
- ChromaDB vector store initialization
- BGE-small-en-v1.5 embeddings
- Semantic search with metadata filtering
- Document ingestion
"""

import logging
import random
from typing import List, Optional, Dict, Any

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from .settings import settings

logger = logging.getLogger(__name__)

# ------------------------------
# Configuration
# ------------------------------

EMBED_MODEL: str = settings.embed_model
CHROMA_PATH: str = settings.chroma_dir
COLLECTION_NAME: str = settings.collection_name
TOP_K_DEFAULT: int = settings.top_k
MAX_DISTANCE_DEFAULT: float = settings.max_distance

logger.info(f"Initializing retrieval with model: {EMBED_MODEL}")
logger.info(f"ChromaDB path: {CHROMA_PATH}")

# ------------------------------
# Embeddings & Chroma collection
# ------------------------------

_embed = SentenceTransformerEmbeddingFunction(EMBED_MODEL)
_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=ChromaSettings(allow_reset=False)
)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # Cosine distance for BGE models
    embedding_function=_embed,
)

logger.info(f"Collection '{COLLECTION_NAME}' ready with {_collection.count()} documents")

# ------------------------------
# Public API
# ------------------------------

def add_documents(docs: List[dict]) -> None:
    """
    Add chunks to the vector store.
    
    Args:
        docs: List of dicts, each with keys:
              - id: Unique chunk identifier
              - text: Chunk text content
              - metadata: Dict with source, doc_type, term_id, etc.
    
    Example:
        docs = [{
            "id": "resume.md:0",
            "text": "Experience: AI Engineer at...",
            "metadata": {
                "source": "resume.md",
                "doc_type": "resume",
                "skills": ["Python", "ML"]
            }
        }]
    """
    if not docs:
        logger.warning("add_documents called with empty list")
        return
    
    _collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d.get("metadata", {}) for d in docs],
    )
    
    logger.info(f"Added {len(docs)} documents to collection")


# BGE v1.5 models benefit from query instruction prefix
BGE_V15_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _should_use_query_instruction() -> bool:
    """Check if we're using a BGE v1.5 model that needs query prefix."""
    model_name = EMBED_MODEL.lower()
    return "bge" in model_name and "v1.5" in model_name and "m3" not in model_name


def search(
    query: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """
    Retrieve top-k chunks within a distance threshold, optionally filtered by metadata.
    
    Args:
        query: User's question or search text
        k: Number of results to return (default: TOP_K_DEFAULT)
        max_distance: Maximum cosine distance threshold (default: MAX_DISTANCE_DEFAULT)
                     Lower = more similar (0 = identical, 2 = opposite)
        metadata_filter: Optional dict for filtering, e.g.:
                        {"doc_type": "transcript", "level": "graduate"}
                        Multiple conditions are AND-ed together.
    
    Returns:
        List of dicts with keys: id, text, source, distance, metadata
        Sorted by distance (most similar first)
    
    Example:
        # Search only graduate transcripts
        chunks = search(
            "What was my GPA?",
            k=5,
            max_distance=0.60,
            metadata_filter={"doc_type": "transcript", "level": "graduate"}
        )
    """
    query = (query or "").strip()
    if not query:
        logger.info("Search skipped: empty query")
        return []
    
    k = k or TOP_K_DEFAULT
    max_dist = MAX_DISTANCE_DEFAULT if max_distance is None else max_distance
    
    # Add query instruction for BGE v1.5 models
    query_text = query
    if _should_use_query_instruction():
        query_text = BGE_V15_QUERY_PREFIX + query
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
        results = _collection.query(
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
            if distance is not None and float(distance) <= float(max_dist):
                source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
                output.append({
                    "id": chunk_id,
                    "text": text,
                    "source": source,
                    "distance": distance,
                    "metadata": metadata  # Include full metadata for inspection
                })
        except Exception as e:
            logger.warning(f"Failed to process chunk {chunk_id}: {e}")
            continue
    
    logger.info(
        f"Search '{query[:50]}...': {len(output)}/{len(ids)} results within max_distance {max_dist:.3f}"
    )
    
    return output


# Alias for compatibility with main.py
retrieve = search


def get_sample_chunks(n: int = 10) -> List[dict]:
    """
    Return up to n random example chunks for testing/debugging.
    
    Args:
        n: Number of samples to return
    
    Returns:
        List of sample chunks with id, text, source, metadata
    """
    n = max(1, min(n, 100))  # Cap at 100 for safety
    count = _collection.count()
    
    if count == 0:
        logger.warning("get_sample_chunks: collection is empty")
        return []
    
    # Random offset for variety
    start = random.randint(0, max(0, count - n))
    
    try:
        results = _collection.get(
            include=["documents", "metadatas"],
            limit=n,
            offset=start
        )
    except Exception as e:
        logger.error(f"Failed to get sample chunks: {e}")
        return []
    
    ids = results.get("ids") or []
    docs = results.get("documents") or []
    metas = results.get("metadatas") or []
    
    output: List[dict] = []
    for chunk_id, text, metadata in zip(ids, docs, metas):
        source = metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
        output.append({
            "id": chunk_id,
            "text": text,
            "source": source,
            "distance": None,  # No distance for random samples
            "metadata": metadata
        })
    
    logger.info(f"Returned {len(output)} sample chunks")
    return output


def get_collection_stats() -> Dict[str, Any]:
    """
    Get statistics about the vector database.
    
    Returns:
        Dict with count, sample sources, etc.
    """
    count = _collection.count()
    
    if count == 0:
        return {
            "total_documents": 0,
            "collection_name": COLLECTION_NAME,
            "embed_model": EMBED_MODEL
        }
    
    # Get a few samples to show document types
    samples = _collection.get(limit=min(100, count), include=["metadatas"])
    metas = samples.get("metadatas") or []
    
    # Collect unique sources and doc types
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
        "sample_sources": sorted(list(sources))[:10]  # First 10 sources
    }


def reset_collection() -> None:
    """
    DANGER: Delete all documents from the collection.
    Only use for testing or when you need a fresh start.
    """
    count = _collection.count()
    logger.warning(f"Resetting collection '{COLLECTION_NAME}' - will delete {count} documents")
    
    _client.delete_collection(COLLECTION_NAME)
    
    # Recreate collection
    global _collection
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embed,
    )
    
    logger.info("Collection reset complete")


# Export for external use
__all__ = [
    "add_documents",
    "search",
    "retrieve",
    "get_sample_chunks",
    "get_collection_stats",
    "reset_collection",
]