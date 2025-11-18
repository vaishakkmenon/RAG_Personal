"""
Vector Store Operations - ChromaDB integration

Handles semantic search, document storage, and metadata filtering.
"""

import logging
import random
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from ..settings import settings

logger = logging.getLogger(__name__)

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

    Args:
        docs: List of dicts with keys: id, text, metadata
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


def search(
    query: str,
    k: Optional[int] = None,
    max_distance: Optional[float] = None,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Retrieve top-k chunks within a distance threshold.

    Args:
        query: User's question or search text
        k: Number of results to return
        max_distance: Maximum cosine distance threshold
        metadata_filter: Optional dict for filtering

    Returns:
        List of dicts with keys: id, text, source, distance, metadata
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
        f"Search '{query[:50]}...': {len(output)}/{len(ids)} results within max_distance {max_dist:.3f}"
    )

    return output


def multi_query_search(
    main_query: str,
    keywords: List[str],
    k: int,
    max_distance: float,
    metadata_filter: Optional[Dict[str, Any]] = None,
) -> List[dict]:
    """Perform multiple searches with keywords and merge results.

    This function improves retrieval by:
    1. Searching with the full question (semantic similarity)
    2. Searching with individual keywords (exact matches)
    3. Deduplicating and merging results
    4. Returning top k chunks by best distance

    Args:
        main_query: The original full question
        keywords: List of extracted keywords to search
        k: Number of total results to return
        max_distance: Maximum cosine distance threshold
        metadata_filter: Optional metadata filter

    Returns:
        List of deduplicated chunks sorted by distance
    """
    if not keywords or len(keywords) == 0:
        # Fall back to regular search if no keywords
        return search(main_query, k, max_distance, metadata_filter)

    # Dictionary to store chunks by ID for deduplication
    all_chunks = {}  # id -> {chunk, min_distance}

    # Perform main search with full query
    logger.info(f"Multi-query search: main query + {len(keywords)} keywords")
    main_results = search(main_query, k, max_distance, metadata_filter)
    for chunk in main_results:
        chunk_id = chunk["id"]
        all_chunks[chunk_id] = {
            "chunk": chunk,
            "min_distance": chunk["distance"],
            "sources": ["main"]
        }

    # Perform keyword searches (limit to 5 keywords to control latency)
    for keyword in keywords[:5]:
        kw_results = search(keyword, k // 2, max_distance, metadata_filter)
        for chunk in kw_results:
            chunk_id = chunk["id"]
            if chunk_id in all_chunks:
                # Chunk found in multiple searches - update min distance
                current_min = all_chunks[chunk_id]["min_distance"]
                new_dist = chunk["distance"]
                all_chunks[chunk_id]["min_distance"] = min(current_min, new_dist)
                all_chunks[chunk_id]["sources"].append(keyword)
            else:
                # New chunk from keyword search
                all_chunks[chunk_id] = {
                    "chunk": chunk,
                    "min_distance": chunk["distance"],
                    "sources": [keyword]
                }

    # Extract chunks and update distances to minimum found
    deduplicated_chunks = []
    for chunk_id, data in all_chunks.items():
        chunk = data["chunk"].copy()
        chunk["distance"] = data["min_distance"]  # Use best distance
        deduplicated_chunks.append(chunk)

    # Sort by distance (lower is better) and take top k
    sorted_chunks = sorted(deduplicated_chunks, key=lambda x: x["distance"])
    result = sorted_chunks[:k]

    logger.info(
        f"Multi-query retrieved {len(deduplicated_chunks)} unique chunks from {len(keywords) + 1} queries, returning top {len(result)}"
    )

    return result


def multi_domain_search(
    query: str,
    domain_configs: List[Dict[str, Any]],
    k: int,
    max_distance: float,
) -> List[dict]:
    """Search across multiple domains with balanced representation.

    This function ensures balanced coverage across different content domains
    (e.g., education, work experience, certifications) by searching each domain
    separately and merging results with equal representation.

    Args:
        query: User's full question (used for all searches)
        domain_configs: List of domain configurations, each with:
            - name: domain name (for logging)
            - filters: list of metadata filter dicts to try
            - section_prefix: optional string for post-filtering by section prefix
        k: Total number of chunks to return
        max_distance: Maximum cosine distance threshold

    Returns:
        List of chunks with balanced representation across domains

    Example domain_config:
        {
            'name': 'education',
            'filters': [
                {'doc_type': 'term'},
                {'doc_type': 'resume', 'section': 'Education'}
            ],
            'section_prefix': None
        }
    """
    if not domain_configs:
        logger.warning("multi_domain_search called with no domain configs, falling back to regular search")
        return search(query, k, max_distance)

    num_domains = len(domain_configs)
    per_domain_k = max(1, k // num_domains)

    logger.info(f"Multi-domain search across {num_domains} domains, {per_domain_k} chunks per domain")

    all_chunks = []

    for domain_config in domain_configs:
        domain_name = domain_config.get('name', 'unknown')
        filters = domain_config.get('filters', [])
        section_prefix = domain_config.get('section_prefix')

        domain_chunks = []

        # Try each filter for this domain
        for metadata_filter in filters:
            try:
                chunks = search(
                    query=query,
                    k=per_domain_k,
                    max_distance=max_distance,
                    metadata_filter=metadata_filter if metadata_filter else None
                )

                # Apply section prefix post-filter if specified
                if section_prefix and chunks:
                    original_count = len(chunks)
                    chunks = [
                        c for c in chunks
                        if c.get('metadata', {}).get('section', '').startswith(section_prefix)
                    ]
                    if original_count != len(chunks):
                        logger.debug(
                            f"Section prefix '{section_prefix}' filter: {original_count} → {len(chunks)}"
                        )

                domain_chunks.extend(chunks)

                # Stop if we have enough chunks for this domain
                if len(domain_chunks) >= per_domain_k:
                    break

            except Exception as e:
                logger.warning(f"Search failed for domain '{domain_name}' with filter {metadata_filter}: {e}")
                continue

        # Deduplicate within domain and take top per_domain_k
        seen_ids = set()
        unique_domain_chunks = []
        for chunk in domain_chunks:
            chunk_id = chunk['id']
            if chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                unique_domain_chunks.append(chunk)
                if len(unique_domain_chunks) >= per_domain_k:
                    break

        logger.info(
            f"Domain '{domain_name}': retrieved {len(unique_domain_chunks)} chunks"
        )

        all_chunks.extend(unique_domain_chunks)

    # Deduplicate across domains (in case of overlap)
    seen_ids = set()
    final_chunks = []
    for chunk in all_chunks:
        chunk_id = chunk['id']
        if chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            final_chunks.append(chunk)

    # Sort by distance and take top k
    final_chunks.sort(key=lambda x: x['distance'])
    result = final_chunks[:k]

    logger.info(
        f"Multi-domain search: {len(result)} total chunks from {num_domains} domains"
    )

    return result


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

    count = _collection.count()
    logger.warning(
        f"Resetting collection '{COLLECTION_NAME}' - will delete {count} documents"
    )

    _client.delete_collection(COLLECTION_NAME)

    # Recreate collection
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
        embedding_function=_embed,
    )

    logger.info("Collection reset complete")
