"""
Retrieval Package - Vector store and semantic search operations

Handles ChromaDB integration, embeddings, and document retrieval.
"""

from app.retrieval.vector_store import get_vector_store
from app.retrieval.search_engine import get_search_engine
from app.retrieval.query_rewriter import get_query_rewriter

# Facade for backward compatibility and simpler import API


def search(*args, **kwargs):
    """Execute search using the configured SearchEngine."""
    return get_search_engine().search(*args, **kwargs)


def add_documents(*args, **kwargs):
    """Add documents to the vector store."""
    return get_vector_store().add_documents(*args, **kwargs)


def get_sample_chunks(*args, **kwargs):
    """Get random sample chunks from the vector store."""
    return get_vector_store().get_sample_chunks(*args, **kwargs)


def get_collection_stats(*args, **kwargs):
    """Get statistics about the vector collection."""
    return get_vector_store().get_stats(*args, **kwargs)


def reset_collection(*args, **kwargs):
    """Reset the vector collection (delete all data)."""
    return get_vector_store().reset(*args, **kwargs)


def rewrite_query(query: str, conversation_history: list = None):
    """
    Rewrites the query using the configured strategies.

    Args:
        query: The user query to rewrite
        conversation_history: Optional history (currently unused by PatternMatcher but kept for interface compact)
    """
    rewriter = get_query_rewriter()
    # The PatternMatcher.rewrite_query takes (query), returns (rewritten, metadata)
    rewritten, metadata = rewriter.rewrite_query(query)
    return rewritten, metadata


__all__ = [
    "add_documents",
    "search",
    "get_sample_chunks",
    "get_collection_stats",
    "reset_collection",
    "rewrite_query",
]
