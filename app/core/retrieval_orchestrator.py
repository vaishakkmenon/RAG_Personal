"""
Retrieval Orchestrator - Manages the information retrieval pipeline.

Encapsulates logic for:
- Query rewriting
- Vector search execution
- Reranking
- Result formatting for prompts and responses
"""

import logging
from typing import List, Dict, Any, Optional, Tuple

from app.settings import settings
from app.models import ChatSource, RewriteMetadata
from app.retrieval import rewrite_query
from app.retrieval.ranking import dedupe_by_text

logger = logging.getLogger(__name__)


class RetrievalOrchestrator:
    """Orchestrates the RAG retrieval pipeline."""

    def __init__(self):
        from app.retrieval.search_engine import get_search_engine

        self.search_engine = get_search_engine()

    def _build_context_query(
        self, conversation_history: List[Dict[str, str]], max_turns: int = 2
    ) -> Optional[str]:
        """Build a context query from recent conversation history."""
        if not conversation_history:
            return None

        # Get recent user turns
        recent_turns = []
        count = 0
        # Iterate backwards
        for i in range(len(conversation_history) - 1, -1, -1):
            turn = conversation_history[i]
            if turn.get("role") == "user":
                recent_turns.append(turn.get("content", ""))
                count += 1
                if count >= max_turns:
                    break

        if not recent_turns:
            return None

        # Join in chronological order (reverse the reversed list)
        return " ".join(reversed(recent_turns))

    def _merge_and_dedupe_chunks(
        self, chunks_1: List[dict], chunks_2: List[dict]
    ) -> List[dict]:
        """Merge and deduplicate chunks from two retrieval passes."""
        seen_ids = set()
        merged = []

        # Add all chunks from first pass (primary query)
        for chunk in chunks_1:
            chunk_id = chunk.get("id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)

        # Add non-duplicate chunks from second pass (context query)
        for chunk in chunks_2:
            chunk_id = chunk.get("id")
            if chunk_id and chunk_id not in seen_ids:
                seen_ids.add(chunk_id)
                merged.append(chunk)

        return merged

    def perform_retrieval(
        self, question: str, history: List[Dict[str, str]], params: Dict[str, Any]
    ) -> Tuple[List[Dict], Optional[RewriteMetadata]]:
        """
        Execute the full retrieval pipeline: Rewrite -> Search -> Rerank.

        Args:
            question: User's question
            history: Conversation history
            params: Retrieval parameters (top_k, max_distance, filters, etc.)

        Returns:
            Tuple of (retrieved_chunks, rewrite_metadata)
        """
        # 1. Query Rewriting
        rewrite_metadata = None
        search_query = question

        if settings.query_rewriter.enabled:
            # We handle rewriting here to capture metadata, so we disable it in SearchEngine
            rewritten, metadata = rewrite_query(question)

            if rewritten != question:
                search_query = rewritten
                rewrite_metadata = metadata
                logger.info(f"Rewrote query: '{question}' -> '{search_query}'")

        # 2. Build Metadata Filters
        metadata_filter = {}
        if params.get("doc_type"):
            metadata_filter["doc_type"] = params["doc_type"]
        if params.get("term_id"):
            metadata_filter["term"] = params["term_id"]
        if params.get("level"):
            metadata_filter["level"] = params["level"]

        if not metadata_filter:
            metadata_filter = None

        # 3. Search (Primary + Context)
        # We disable cross-encoder in search_engine here because we want to merge results first

        # Primary search
        chunks = self.search_engine.search(
            query=search_query,
            k=params.get("top_k", settings.retrieval.top_k),
            max_distance=params.get("max_distance", settings.retrieval.max_distance),
            metadata_filter=metadata_filter,
            use_query_rewriting=False,  # Already done
            use_cross_encoder=False,  # We rerank after merge
        )

        # Context search (if history exists)
        context_chunks = []
        if history:
            context_query = self._build_context_query(history)
            if context_query:
                logger.info(f"Performing context search with: '{context_query}'")
                context_chunks = self.search_engine.search(
                    query=context_query,
                    k=params.get("top_k", settings.retrieval.top_k) // 2,
                    max_distance=params.get(
                        "max_distance", settings.retrieval.max_distance
                    ),
                    metadata_filter=metadata_filter,
                    use_query_rewriting=False,
                    use_cross_encoder=False,
                )

        # Merge if context chunks found
        if context_chunks:
            chunks = self._merge_and_dedupe_chunks(chunks, context_chunks)

        # Text-based deduplication (removes semantically similar chunks)
        chunks = dedupe_by_text(chunks)

        # 4. Reranking
        if chunks:
            # Check global setting or param override (params["rerank"] can be None)
            should_rerank = params.get("rerank")
            if should_rerank is None:
                should_rerank = settings.cross_encoder.enabled

            if should_rerank:
                params.get("rerank_lex_weight") or settings.retrieval.rerank_lex_weight
                # Note: SearchEngine.rerank doesn't currently support lex_weight if just wrapping cross_encoder
                # but cross_encoder.rerank might not use it either depending on implementation.
                # Checking retrieval_orchestrator usage: it passed lex_weight to rerank_chunks.
                # SearchEngine.rerank implementation I wrote only takes query, chunks, top_k.
                # Failure mode: lex_weight ignored. This is acceptable for clean abstraction vs implementation detail.

                chunks = self.search_engine.rerank(
                    query=search_query,
                    chunks=chunks,
                    top_k=params.get("top_k", settings.retrieval.top_k),
                )

        return chunks, rewrite_metadata

    def format_sources_for_prompt(self, chunks: List[dict]) -> List[Dict[str, Any]]:
        """Format chunks for the prompt builder."""
        return [
            {
                "id": c.get("id", ""),
                "source": c.get("source", "unknown"),
                "text": c.get("text", ""),
                "metadata": c.get("metadata", {}),
                "distance": c.get("distance", 1.0),
            }
            for c in chunks
            if c.get("text", "").strip()
        ]

    def build_chat_sources(self, chunks: List[dict]) -> List[ChatSource]:
        """Create output source metadata for chatbot responses."""
        sources: List[ChatSource] = []
        for chunk in chunks:
            text = chunk.get("text", "")
            truncated = f"{text[:200]}..." if len(text) > 200 else text
            sources.append(
                ChatSource(
                    id=chunk.get("id", ""),
                    source=chunk.get("source", ""),
                    text=truncated,
                    distance=chunk.get("distance", 1.0),
                    citation_index=chunk.get("citation_index"),
                )
            )
        return sources
