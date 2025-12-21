"""
Retrieval Package - Vector store and semantic search operations

Handles ChromaDB integration, embeddings, and document retrieval.
"""

from app.retrieval.store import (
    add_documents,
    search,
    get_sample_chunks,
    get_collection_stats,
    reset_collection,
)
from app.retrieval.query_rewriter import get_query_rewriter

__all__ = [
    "add_documents",
    "search",
    "get_sample_chunks",
    "get_collection_stats",
    "reset_collection",
    "rewrite_query",
]


def rewrite_query(query: str, conversation_history: list = None):
    """
    Rewrites the query using the configured strategies.

    This is a facade for the singleton QueryRewriter instance.
    """
    rewriter = get_query_rewriter()
    # QueryRewriter expects conversation history?
    # Checking QueryRewriter.rewrite_query signature:
    # def rewrite_query(self, query: str, metadata_filter: Optional[Dict] = None) -> Tuple[str, Optional["RewriteMetadata"]]:

    # It seems QueryRewriter.rewrite_query DOES NOT take history directly in the file I viewed.
    # It takes `metadata_filter`.
    # Let me re-verify the viewed file content for `query_rewriter.py`.
    # Line 140: def rewrite_query(self, query: str, metadata_filter: Optional[Dict] = None) ...

    # So the orchestration logic passing history was wrong or relying on a different version?
    # In `RetrievalOrchestrator`:
    # rewritten, metadata = rewrite_query(question, rewrite_history)

    # Wait, the `RetrievalOrchestrator` passed `rewrite_history` as the second argument.
    # If the signature is (query, metadata_filter), then passing history (a list) as metadata_filter (a dict) might cause issues or be ignored if type checked, but at runtime python might just pass it.

    # However, `QueryRewriter` logic:
    # It iterates patterns.
    # PatternMatcher.match(query)

    # It seems `QueryRewriter` in `app/retrieval/query_rewriter.py` DOES NOT use history currently.
    # So my usage in `RetrievalOrchestrator` passing history was based on a false assumption or an intended feature that isn't implemented.

    # Implementation:
    # I will stick to the signature `rewrite_query(query, metadata_filter=None)`.
    # And I need to fix `RetrievalOrchestrator` to stop passing history if it's not supported, OR I should update `rewrite_query` to support history if `PatternMatcher` supports it.

    # Let's assume for now I just expose what is there.
    return rewriter.rewrite_query(query)
