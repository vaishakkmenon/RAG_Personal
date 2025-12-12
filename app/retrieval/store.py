"""
Vector Store Operations - ChromaDB integration

Handles semantic search, document storage, and metadata filtering.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..settings import settings

logger = logging.getLogger(__name__)

# Import BM25 for hybrid search
try:
    from .bm25_search import BM25Index, reciprocal_rank_fusion
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    logger.warning("BM25 not available - hybrid search disabled")

# Configuration
EMBED_MODEL: str = settings.embed_model
CHROMA_PATH: str = settings.chroma_dir
COLLECTION_NAME: str = settings.collection_name
TOP_K_DEFAULT: int = settings.top_k
MAX_DISTANCE_DEFAULT: float = settings.max_distance

logger.info(f"Initializing retrieval with model: {EMBED_MODEL}")
logger.info(f"ChromaDB path: {CHROMA_PATH}")

# Embeddings & Chroma collection
_embed = SentenceTransformerEmbeddingFunction(EMBED_MODEL)
_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=ChromaSettings(allow_reset=False),
)
_collection = _client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},  # Cosine distance for BGE models
    embedding_function=_embed,
)

logger.info(
    f"Collection '{COLLECTION_NAME}' ready with {_collection.count()} documents"
)

# Initialize BM25 index for hybrid search
_bm25_index = None
if BM25_AVAILABLE:
    bm25_index_path = Path(CHROMA_PATH) / "bm25_index.pkl"
    if bm25_index_path.exists():
        _bm25_index = BM25Index(index_path=str(bm25_index_path))
        if _bm25_index.load_index():
            logger.info(f"BM25 hybrid search enabled ({_bm25_index.documents and len(_bm25_index.documents)} docs)")
        else:
            _bm25_index = None
            logger.warning("BM25 index file found but failed to load")
    else:
        logger.info(f"BM25 index not found at {bm25_index_path} - run build_bm25_index.py to enable hybrid search")
else:
    logger.info("BM25 library not installed - hybrid search disabled")


# Helpers
def _get_source(metadata: Any) -> str:
    """Safely extract source from metadata."""
    return (
        metadata.get("source", "unknown") if isinstance(metadata, dict) else "unknown"
    )


# BGE v1.5 models benefit from query instruction prefix
BGE_V15_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def _should_use_query_instruction() -> bool:
    """Check if we're using a BGE v1.5 model that needs query prefix."""
    model_name = EMBED_MODEL.lower()
    return "bge" in model_name and "v1.5" in model_name and "m3" not in model_name


# Public API
def add_documents(docs: List[dict]) -> None:
    """Add chunks to the vector store.

    Uses upsert to prevent duplicates - if a document with the same ID already exists,
    it will be updated instead of creating a duplicate.

    Args:
        docs: List of dicts with keys: id, text, metadata
    """
    if not docs:
        logger.warning("add_documents called with empty list")
        return

    _collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d.get("metadata", {}) for d in docs],
    )

    logger.info(f"Upserted {len(docs)} documents to collection")


def search(
    query: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
    use_hybrid: bool = True,
    use_query_rewriting: bool = True,
    use_cross_encoder: Optional[bool] = None,
) -> List[dict]:
    """Retrieve top-k chunks using hybrid search (BM25 + semantic) or semantic only.

    Args:
        query: User's question or search text
        k: Number of results to return
        max_distance: Maximum cosine distance threshold
        metadata_filter: Optional dict for filtering
        use_hybrid: Whether to use BM25 + semantic (default True)
        use_query_rewriting: Whether to apply pattern-based query rewriting (default True)
        use_cross_encoder: Whether to use cross-encoder reranking (default from settings)

    Returns:
        List of dicts with keys: id, text, source, distance, metadata
    """
    import time
    from ..settings import settings

    original_query = (query or "").strip()
    if not original_query:
        logger.info("Search skipped: empty query")
        return []

    # Apply query rewriting if enabled
    search_query = original_query
    rewrite_metadata = None
    if use_query_rewriting and settings.query_rewriter.enabled:
        try:
            from .query_rewriter import get_query_rewriter
            rewriter = get_query_rewriter()
            rewritten_query, rewrite_metadata = rewriter.rewrite_query(original_query)
            
            if rewrite_metadata:
                search_query = rewritten_query
                logger.info(
                    f"Query rewritten by '{rewrite_metadata.pattern_name}': "
                    f"'{original_query[:40]}...' → '{search_query[:60]}...' "
                    f"(latency: {rewrite_metadata.latency_ms:.2f}ms)"
                )
        except Exception as e:
            logger.warning(f"Query rewriting failed, using original: {e}")
            search_query = original_query

    query = search_query

    # Determine if cross-encoder should be used
    if use_cross_encoder is None:
        use_cross_encoder = settings.cross_encoder.enabled

    k = k or TOP_K_DEFAULT
    max_dist = MAX_DISTANCE_DEFAULT if max_distance is None else max_distance

    # Adjust retrieval k if using cross-encoder (need more candidates)
    if use_cross_encoder:
        retrieval_k = settings.cross_encoder.retrieval_k  # Default 15
        final_k = k
        logger.info(f"Cross-encoder enabled: retrieving {retrieval_k} candidates for reranking")
    else:
        retrieval_k = k
        final_k = k

    # Use hybrid search if BM25 is available and enabled
    if use_hybrid and _bm25_index is not None:
        results = _hybrid_search(query, retrieval_k, max_dist, metadata_filter)
    else:
        results = _semantic_search(query, retrieval_k, max_dist, metadata_filter)

    # Apply cross-encoder reranking if enabled
    if use_cross_encoder and len(results) > 0:
        try:
            from app.services.cross_encoder_reranker import get_cross_encoder_reranker
            cross_encoder = get_cross_encoder_reranker()
            results = cross_encoder.rerank(
                query=query,
                chunks=results,
                top_k=settings.cross_encoder.top_k
            )
            logger.info(f"Cross-encoder reranked to {len(results)} results")
        except Exception as e:
            logger.error(f"Cross-encoder reranking failed: {e}")
            logger.warning("Falling back to original ranking")
            results = results[:final_k]

    return results[:final_k]


def _semantic_search(
    query: str,
    k: int,
    max_dist: float,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Perform semantic-only search using ChromaDB.

    Args:
        query: Search query
        k: Number of results
        max_dist: Maximum distance threshold
        metadata_filter: Optional metadata filter

    Returns:
        List of results
    """

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
            if distance is not None and distance <= max_dist:
                output.append(
                    {
                        "id": chunk_id,
                        "text": text,
                        "source": _get_source(metadata),
                        "distance": distance,
                        "metadata": metadata,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to process chunk {chunk_id}: {e}")
            continue

    logger.info(
        f"Semantic search '{query[:50]}...': {len(output)}/{len(ids)} results within max_distance {max_dist:.3f}"
    )

    return output


def _hybrid_search(
    query: str,
    k: int,
    max_dist: float,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Perform hybrid search combining BM25 and semantic search.

    Args:
        query: Search query
        k: Number of final results
        max_dist: Maximum distance threshold for semantic results
        metadata_filter: Optional metadata filter

    Returns:
        Merged results using RRF
    """
    # Retrieve more candidates for better fusion
    bm25_k = k * 4  # Get 4x candidates from BM25
    semantic_k = k * 4  # Get 4x candidates from semantic

    # BM25 search
    bm25_results = _bm25_index.search(query, k=bm25_k)
    logger.info(f"BM25 retrieved {len(bm25_results)} candidates")

    # Semantic search
    semantic_results = _semantic_search(query, semantic_k, max_dist, metadata_filter)
    logger.info(f"Semantic retrieved {len(semantic_results)} candidates")

    # Merge using RRF with configurable k parameter
    merged = reciprocal_rank_fusion([bm25_results, semantic_results], k=settings.bm25.rrf_k)

    # Return top K
    final_results = merged[:k]

    logger.info(
        f"Hybrid search '{query[:50]}...': {len(final_results)} results (merged from BM25 + semantic)"
    )

    return final_results


def get_sample_chunks(n: int = 10) -> List[dict]:
    """Return up to n random example chunks for testing/debugging.

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
                "source": _get_source(metadata),
                "distance": None,
                "metadata": metadata,
            }
        )

    logger.info(f"Returned {len(output)} sample chunks")
    return output


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the vector database.

    Returns:
        Dict with count, sample sources, etc.
    """
    count = _collection.count()

    if count == 0:
        return {
            "total_documents": 0,
            "collection_name": COLLECTION_NAME,
            "embed_model": EMBED_MODEL,
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
        "sample_sources": sorted(list(sources))[:10],
    }


def reset_collection() -> None:
    """DANGER: Delete all documents from the collection.

    Only use for testing or when you need a fresh start.
    """
    global _collection

    try:
        count = _collection.count()
        logger.warning(
            f"Resetting collection '{COLLECTION_NAME}' - will delete {count} documents"
        )

        _client.delete_collection(COLLECTION_NAME)
    except Exception as e:
        # Collection might already be deleted or not exist
        logger.info(f"Collection doesn't exist or already deleted: {e}")

    # Always recreate collection to ensure it exists
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embed,
    )

    logger.info("Collection reset complete")
