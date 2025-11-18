"""
Retrieval Package - Vector store and semantic search operations

Handles ChromaDB integration, embeddings, and document retrieval.
"""

from .store import (
    add_documents,
    search,
    multi_query_search,
    multi_domain_search,
    get_sample_chunks,
    get_collection_stats,
    reset_collection,
)

__all__ = [
    "add_documents",
    "search",
    "multi_query_search",
    "multi_domain_search",
    "get_sample_chunks",
    "get_collection_stats",
    "reset_collection",
]
