"""
Chat service for Personal RAG system.

Main business logic for handling chat requests, orchestrating retrieval,
prompt building, and LLM generation.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional

from fastapi import BackgroundTasks, HTTPException, status

from app.models import (
    ChatRequest,
    ChatResponse,
    ChatSource,
    AmbiguityMetadata,
    RewriteMetadata,
)
from app.monitoring import time_execution_info
from app.monitoring.pattern_analytics import get_pattern_analytics
from app.monitoring.pattern_suggester import get_pattern_suggester
from app.prompting import build_clarification_message, create_default_prompt_builder
from app.retrieval import search
from app.services.llm import generate_with_llm
from app.services.prompt_guard import get_prompt_guard
from app.services.reranker import rerank_chunks
from app.services.response_cache import get_response_cache
from app.settings import settings
from app.storage import get_session_store, Session
from app.storage.utils import mask_session_id

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from app.metrics import (
        rag_retrieval_chunks,
        rag_retrieval_distance,
        rag_grounding_total,
        rag_ambiguity_checks_total,
        rag_clarification_requests_total,
        rag_llm_token_usage_total,
        rag_llm_cost_total,
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for chat service")


# PII patterns to redact from LLM responses
PII_PATTERNS = [
    (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED SSN]"),  # SSN format XXX-XX-XXXX
    (r"\b\d{16}\b", "[REDACTED CARD]"),  # 16-digit credit card
    (
        r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",
        "[REDACTED CARD]",
    ),  # Credit card with spaces/dashes
]


def _filter_pii(text: str) -> str:
    """Redact potential PII from LLM responses.

    Args:
        text: The text to filter

    Returns:
        Text with PII patterns redacted
    """
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _is_chitchat(question: str) -> tuple[bool, str]:
    """Detect conversational/social interactions that don't need retrieval.

    Args:
        question: User's question

    Returns:
        Tuple of (is_chitchat, response_message)
    """
    q = question.strip().lower()

    # Greetings
    greetings = {
        "hello",
        "hi",
        "hey",
        "good morning",
        "good afternoon",
        "good evening",
        "howdy",
        "greetings",
    }
    if any(
        greeting == q or q.startswith(greeting + " ") or q.startswith(greeting + "!")
        for greeting in greetings
    ):
        return (
            True,
            "Hello! I can help answer questions about your background, experience, certifications, and education. What would you like to know?",
        )

    # Gratitude
    if any(
        word in q for word in ["thank you", "thanks", "thx", "appreciate", "grateful"]
    ):
        return True, "You're welcome! Is there anything else you'd like to know?"

    # Farewells
    farewells = {"bye", "goodbye", "see you", "later", "farewell"}
    if any(farewell in q for farewell in farewells):
        return True, "Goodbye! Feel free to come back if you have more questions."

    return False, ""


def _clean_answer(answer: str, question: str) -> str:
    """Clean up common answer formatting issues.

    Args:
        answer: The generated answer
        question: The original question

    Returns:
        Cleaned answer
    """
    if not answer:
        return answer

    # Fix pattern: "No, I have X but..." → "I have X"
    # This happens when LLM incorrectly starts with "No" for listing questions
    if answer.startswith("No, ") and " but " in answer:
        parts = answer.split(" but ", 1)
        if len(parts) == 2:
            # Check if the question is asking "Which" or "What" (listing questions)
            q_lower = question.lower().strip()
            if any(q_lower.startswith(word) for word in ("which", "what", "list")):
                cleaned = parts[1].strip()
                # Capitalize first letter
                if cleaned:
                    cleaned = cleaned[0].upper() + cleaned[1:]
                logger.info(
                    "Cleaned answer: removed 'No, ... but' prefix for listing question"
                )
                return cleaned

    return answer


def _is_truly_ambiguous(
    question: str, conversation_history: Optional[List[Dict[str, str]]] = None
) -> bool:
    """Detect truly vague questions using simple rules.

    Trust the system prompt (Rule 11) to handle most ambiguity.
    Only catch extremely minimal queries here.

    Args:
        question: User's question
        conversation_history: Optional conversation history for context-aware detection

    Returns:
        True if question is obviously too vague
    """
    q = question.strip()

    # Empty or just punctuation
    if not q or len(q) <= 2:
        return True

    # Remove punctuation and count words
    words = q.replace("?", "").replace(".", "").replace("!", "").strip().split()

    # With conversation context, allow very short follow-ups
    if conversation_history and len(conversation_history) > 0:
        if len(words) >= 1 and len(q) > 2:
            return False  # Trust context

    # Without context: only flag truly minimal queries
    if len(words) <= 1:
        return True

    # Let the system prompt (Rule 11) handle everything else
    return False


def _get_client_ip(request: ChatRequest) -> Optional[str]:
    """Extract client IP from request.

    In production, this would come from FastAPI's Request object.
    For now, we'll get it from the route layer (Phase 7).

    Args:
        request: Chat request

    Returns:
        IP address or None
    """
    # This will be properly implemented in Phase 7 when we have access to FastAPI Request
    # For now, return None (sessions will still work, just no IP-based limiting)
    return None


def _build_context_query(
    conversation_history: List[Dict[str, str]], max_turns: int = 2
) -> Optional[str]:
    """Build a context query from recent conversation history.

    Args:
        conversation_history: List of conversation turns
        max_turns: Maximum number of recent turns to include

    Returns:
        Context query string or None if no history
    """
    if not conversation_history:
        return None

    # Take last N turns (both user and assistant)
    recent_turns = conversation_history[
        -max_turns * 2 :
    ]  # Each turn is user + assistant

    # Extract only user questions for context (avoid hallucinated assistant responses)
    user_queries = [
        turn["content"]
        for turn in recent_turns
        if turn.get("role") == "user" and turn.get("content", "").strip()
    ]

    if not user_queries:
        return None

    # Combine recent user questions
    context_query = " ".join(user_queries)
    logger.info(f"Built context query from {len(user_queries)} previous question(s)")
    return context_query


def _merge_and_dedupe_chunks(chunks_1: List[dict], chunks_2: List[dict]) -> List[dict]:
    """Merge and deduplicate chunks from two retrieval passes.

    Args:
        chunks_1: Chunks from first retrieval pass
        chunks_2: Chunks from second retrieval pass

    Returns:
        Merged and deduplicated list of chunks
    """
    # Use chunk ID for deduplication
    seen_ids = set()
    merged = []

    # Process all chunks, maintaining order preference (chunks_1 first)
    for chunk in chunks_1 + chunks_2:
        chunk_id = chunk.get("id")
        if chunk_id and chunk_id not in seen_ids:
            seen_ids.add(chunk_id)
            merged.append(chunk)
        elif not chunk_id:
            # If no ID, include it (shouldn't happen but handle gracefully)
            merged.append(chunk)

    logger.info(
        f"Merged chunks: {len(chunks_1)} + {len(chunks_2)} = {len(merged)} (after deduplication)"
    )
    return merged


class ChatService:
    """Service for handling RAG chat requests."""

    def __init__(self, session_store=None):
        """Initialize chat service."""
        self.prompt_builder = create_default_prompt_builder()

        if session_store is None:
            session_store = get_session_store()

        self.session_store = session_store
        logger.info(f"ChatService initialized with {type(session_store).__name__}")

    def _format_sources(self, chunks: List[dict]) -> List[Dict[str, Any]]:
        """Format chunks for the prompt builder.

        Args:
            chunks: List of retrieved chunks with text, source, and metadata

        Returns:
            List of formatted sources
        """
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

    def _build_chat_sources(self, chunks: List[dict]) -> List[ChatSource]:
        """Create output source metadata for chatbot responses.

        Args:
            chunks: List of retrieved chunks with text, source, and metadata.
                    Each chunk should have a 'citation_index' from _format_context.

        Returns:
            List of ChatSource objects with citation indices
        """
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

    @time_execution_info
    def _normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent caching.

        - Lowercase
        - Strip whitespace
        - Remove trailing punctuation
        """
        normalized = query.lower().strip()
        return normalized.rstrip("?!.")

    def handle_chat(
        self,
        request: ChatRequest,
        grounded_only: Optional[bool] = None,
        null_threshold: Optional[float] = None,
        max_distance: Optional[float] = None,
        top_k: Optional[int] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        rerank: Optional[bool] = None,
        rerank_lex_weight: Optional[float] = None,
        doc_type: Optional[str] = None,
        term_id: Optional[str] = None,
        level: Optional[str] = None,
        model: Optional[str] = None,
        skip_route_cache: bool = False,
        api_key: Optional[str] = None,
        background_tasks: Optional[BackgroundTasks] = None,
        request_id: Optional[str] = None,
    ) -> ChatResponse:
        """
        Handle a chat request with RAG.

        Args:
            request: Chat request with question
            grounded_only: Whether to require grounded responses
            null_threshold: Distance threshold for grounding
            max_distance: Maximum distance for retrieval
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            rerank: Whether to rerank results
            rerank_lex_weight: Weight for lexical reranking
            doc_type: Document type filter
            term_id: Term ID filter
            level: Level filter
            model: LLM model override
            skip_route_cache: whether to skip the route cache
            api_key: Optional API key
            background_tasks: Optional background tasks
            request_id: Optional request ID

        Returns:
            ChatResponse with answer and metadata
        """
        time.time()

        # MEANINGFUL CHANGE: Normalize query to improve cache hit rates
        # "What is RAG?" and "what is rag" should be treated identical for caching & logic
        request.question = self._normalize_query(request.question)

        # Apply defaults from settings if not provided
        temperature = (
            temperature if temperature is not None else settings.llm.temperature
        )
        max_tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens
        # Don't pass model parameter - let LLM service use its configured default
        # Passing None lets the service choose based on provider
        model_name = model  # Only use if explicitly provided

        # Default parameters
        params = {
            "model": model_name,
            "top_k": top_k if top_k is not None else settings.retrieval.top_k,
            "max_distance": (
                max_distance
                if max_distance is not None
                else settings.retrieval.max_distance
            ),
            "null_threshold": (
                null_threshold
                if null_threshold is not None
                else settings.retrieval.null_threshold
            ),
            "rerank": rerank if rerank is not None else settings.retrieval.rerank,
            "rerank_lex_weight": (
                rerank_lex_weight
                if rerank_lex_weight is not None
                else settings.retrieval.rerank_lex_weight
            ),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "doc_type": doc_type,
            "term_id": term_id,
            "level": level,
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
        }

        try:
            session = self.session_store.get_or_create_session(
                session_id=request.session_id,
                ip_address=_get_client_ip(request),
                user_agent=None,  # Will be added in Phase 7
            )
            logger.info(
                f"Session: {mask_session_id(session.session_id)} "
                f"(created: {session.created_at.strftime('%Y-%m-%d %H:%M:%S')})"
            )
        except HTTPException as e:
            # Session limit exceeded or rate limit hit
            logger.error(f"Session creation failed: {e.detail}")
            raise
        except Exception as e:
            # Unexpected error - log but don't block the request
            logger.error(f"Session management error: {e}")
            # Create a temporary session for this request
            import uuid
            from datetime import datetime

            session = Session(
                session_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
            )

        if not self.session_store.check_rate_limit(session):
            logger.warning(
                f"Rate limit exceeded for session: {mask_session_id(session.session_id)}"
            )
            raise HTTPException(status_code=429, detail="Rate limit exceeded")

        conversation_history = session.get_truncated_history()
        logger.info(f"Conversation history: {len(conversation_history)} messages")

        logger.info(f"Question: {request.question}")

        # ===== PROMPT GUARD =====
        if settings.prompt_guard.enabled:
            guard = get_prompt_guard()
            # Check input with context
            history_dicts = [
                {"role": t.get("role"), "content": t.get("content")}
                for t in conversation_history
            ]

            guard_result = guard.check_input(
                user_input=request.question, conversation_history=history_dicts
            )

            if guard_result["blocked"]:
                logger.warning(
                    f"PromptGuard blocked request: {guard_result.get('label', 'BLOCKED')} "
                    f"(reason: {guard_result.get('reason', 'Unknown')})"
                )
                # Return refused response immediately
                return ChatResponse(
                    answer="I cannot answer this request as it triggers safety guardrails.",
                    sources=[],
                    grounded=False,
                    session_id=session.session_id,
                    rewrite_metadata=None,
                    ambiguity=AmbiguityMetadata(
                        is_ambiguous=False, score=0.0, clarification_requested=False
                    ),
                )

        # ===== RESPONSE CACHING =====
        # Try to get cached response before doing expensive retrieval/generation
        # Only check if not already checked at route level (skip_route_cache=False)
        # This handles conversation-aware caching with session_id
        response_cache = get_response_cache()
        cache_params = {
            "top_k": params["top_k"],
            "max_distance": params["max_distance"],
            "temperature": params["temperature"],
            "model": params["model"],
            "doc_type": params["doc_type"],
        }

        cached_response = None
        if not skip_route_cache:
            # Check cache with session context (for conversational queries)
            cached_response = response_cache.get(
                question=request.question,
                session_id=session.session_id
                if len(conversation_history) > 0
                else None,
                params=cache_params,
            )

        if cached_response:
            # Cache hit! Return cached response immediately
            # Add session_id to response
            cached_response["session_id"] = session.session_id

            # Convert to ChatResponse model
            return ChatResponse(**cached_response)
        # ===== END RESPONSE CACHING =====

        # STEP 1: Check for chitchat (greetings, thanks, etc.)
        is_chitchat, chitchat_response = _is_chitchat(request.question)
        if is_chitchat:
            logger.info(f"Chitchat detected: '{request.question}'")

            try:
                session.add_turn("user", request.question)
                session.add_turn("assistant", chitchat_response)
                self.session_store.update_session(session)
            except Exception as e:
                logger.error(f"Failed to update session for chitchat: {e}")

            return ChatResponse(
                answer=chitchat_response,
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=None,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=False,
                    score=0.0,
                    clarification_requested=False,
                ),
            )

        # STEP 2: Check for ambiguity BEFORE retrieval
        # Pass conversation history to make ambiguity detection context-aware
        is_ambiguous = _is_truly_ambiguous(
            request.question, conversation_history=conversation_history
        )

        # Track ambiguity detection
        if METRICS_ENABLED:
            rag_ambiguity_checks_total.labels(
                result="ambiguous" if is_ambiguous else "clear"
            ).inc()

        if is_ambiguous:
            logger.info(f"Pre-retrieval: Question is too vague: '{request.question}'")
            clarification = build_clarification_message(
                request.question, self.prompt_builder.config
            )

            # Track clarification request
            if METRICS_ENABLED:
                rag_clarification_requests_total.inc()

            return ChatResponse(
                answer=clarification,
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=None,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=True,
                    score=0.9,
                    clarification_requested=True,
                ),
            )

        # Build metadata filter
        metadata_filter = {
            k: v
            for k, v in {
                "doc_type": params.get("doc_type"),
                "term_id": params.get("term_id"),
                "level": level,
            }.items()
            if v is not None
        }

        # ===== QUERY REWRITING =====
        # Apply pattern-based query rewriting before retrieval
        from app.retrieval.query_rewriter import get_query_rewriter

        rewriter = get_query_rewriter()
        rewritten_query, rewrite_metadata = rewriter.rewrite_query(
            request.question, metadata_filter=metadata_filter
        )

        # Merge metadata filter enhancements from rewrite
        if rewrite_metadata and rewrite_metadata.metadata_filter_addition:
            metadata_filter.update(rewrite_metadata.metadata_filter_addition)

        # Use rewritten query for retrieval
        search_query = rewritten_query

        # Log rewrite if pattern matched
        if rewrite_metadata:
            logger.info(
                f"Query rewritten by '{rewrite_metadata.pattern_name}': "
                f"'{request.question[:50]}...' → '{rewritten_query[:50]}...' "
                f"(latency: {rewrite_metadata.latency_ms:.2f}ms, confidence: {rewrite_metadata.confidence:.2f})"
            )
        # ===== END QUERY REWRITING =====

        # Retrieve chunks with hybrid retrieval if conversation history exists
        # If reranking is enabled, retrieve more chunks initially for better reranking
        retrieval_k = params["top_k"]
        if params["rerank"]:
            retrieval_k = settings.retrieval.rerank_retrieval_k
            logger.info(
                f"Reranking enabled: retrieving {retrieval_k} chunks "
                f"(will rerank to top {params['top_k']})"
            )

        try:
            # Primary retrieval: current question
            # Note: use_query_rewriting=False because ChatService handles rewriting
            # for metadata tracking purposes
            chunks_primary = search(
                query=search_query,
                k=retrieval_k,
                max_distance=params["max_distance"],
                metadata_filter=metadata_filter if metadata_filter else None,
                use_query_rewriting=False,  # Already rewritten above
            )
            logger.info(f"Primary retrieval: {len(chunks_primary)} chunks")

            # Secondary retrieval: conversation context (if exists)
            chunks_context = []
            if conversation_history:
                context_query = _build_context_query(conversation_history, max_turns=1)
                if context_query:
                    # Retrieve fewer chunks for context (about 40% of primary)
                    context_k = max(2, retrieval_k // 2)
                    chunks_context = search(
                        query=context_query,
                        k=context_k,
                        max_distance=params["max_distance"],
                        metadata_filter=metadata_filter if metadata_filter else None,
                        use_query_rewriting=False,  # Context query doesn't need rewriting
                    )
                    logger.info(f"Context retrieval: {len(chunks_context)} chunks")

            # Merge and deduplicate
            if chunks_context:
                chunks = _merge_and_dedupe_chunks(chunks_primary, chunks_context)
            else:
                chunks = chunks_primary

        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to retrieve documents",
            )

        # DEBUG: Log retrieved chunk distances
        if chunks:
            distances = [c.get("distance", "N/A") for c in chunks[:5]]
            logger.info(
                f"Retrieved {len(chunks)} chunks (merged), top 5 distances: {distances}"
            )

        # Reranking (if enabled)
        if params["rerank"] and chunks:
            chunks = rerank_chunks(
                request.question, chunks, lex_weight=params["rerank_lex_weight"]
            )
            logger.info(f"Hybrid reranker: {len(chunks)} chunks")

            # Trim to top_k after reranking
            if len(chunks) > params["top_k"]:
                chunks = chunks[: params["top_k"]]
                logger.info(f"Trimmed to top {params['top_k']} chunks after reranking")

        if METRICS_ENABLED:
            rag_retrieval_chunks.observe(len(chunks))

            # Track retrieval distances
            for chunk in chunks:
                distance = chunk.get("distance")
                if distance is not None:
                    rag_retrieval_distance.observe(distance)

        # Grounding check - no chunks
        if not chunks:
            logger.warning("No chunks retrieved")

            # ===== ANALYTICS: Failed query (no chunks) =====
            if settings.query_rewriter.analytics_enabled:
                analytics = get_pattern_analytics()
                suggester = get_pattern_suggester()

                analytics.log_query(
                    query=request.question,
                    rewrite_metadata=rewrite_metadata,
                    retrieval_distance=None,
                    grounded=False,
                )

                suggester.capture_failed_query(
                    query=request.question,
                    distance=None,
                    pattern_matched=rewrite_metadata.pattern_name
                    if rewrite_metadata
                    else None,
                    grounded=False,
                )
            # ===== END ANALYTICS =====

            # Track ungrounded response
            if METRICS_ENABLED:
                rag_grounding_total.labels(grounded="false").inc()

            # Question is specific (passed ambiguity check) but no relevant docs found
            return ChatResponse(
                answer="I don't know. I couldn't find any relevant information in my documents.",
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=False,
                    score=0.0,
                    clarification_requested=False,
                ),
            )

        # Grounding check - distance threshold
        best_distance = chunks[0]["distance"]

        # Handle None distance (from BM25/hybrid search)
        if best_distance is None:
            logger.info(
                "Best chunk has no distance (BM25/hybrid search) - skipping distance check"
            )
        else:
            logger.info(
                f"Best chunk distance: {best_distance:.3f}, threshold: {params['null_threshold']}"
            )

        if best_distance is not None and best_distance > params["null_threshold"]:
            logger.info(
                f"Refusing: best distance {best_distance:.3f} > threshold {params['null_threshold']}"
            )

            # ===== ANALYTICS: Failed query (distance > threshold) =====
            if settings.query_rewriter.analytics_enabled:
                analytics = get_pattern_analytics()
                suggester = get_pattern_suggester()

                analytics.log_query(
                    query=request.question,
                    rewrite_metadata=rewrite_metadata,
                    retrieval_distance=best_distance,
                    grounded=False,
                )

                suggester.capture_failed_query(
                    query=request.question,
                    distance=best_distance,
                    pattern_matched=rewrite_metadata.pattern_name
                    if rewrite_metadata
                    else None,
                    grounded=False,
                )
            # ===== END ANALYTICS =====

            # Keep sources for transparency even when refusing
            return ChatResponse(
                answer="I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
                sources=self._build_chat_sources(chunks),  # Keep sources for debugging
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=params.get("is_ambiguous", False),
                    score=params.get("ambiguity_score", 0.0),
                    clarification_requested=False,
                ),
            )

        # Standard RAG flow
        return self._handle_standard_query(
            request.question,
            params,
            chunks,
            session,
            conversation_history,
            rewrite_metadata,
        )

    def _handle_standard_query(
        self,
        question: str,
        params: Dict[str, Any],
        chunks: List[dict],
        session: "Session",
        conversation_history: List[Dict[str, str]],
        rewrite_metadata: Optional[RewriteMetadata] = None,
    ) -> ChatResponse:
        """Handle standard RAG query with LLM generation.

        Args:
            question: User's question
            params: Query parameters
            chunks: Retrieved chunks
            session: Session object
            conversation_history: Conversation history
            rewrite_metadata: Query rewrite metadata

        Returns:
            ChatResponse
        """
        # ===== TOKEN LIMIT ENFORCEMENT =====
        # prevent silent failures from context overflow
        MODEL_CONTEXT_LIMIT = 7000  # Safe buffer for 8k models
        RESPONSE_BUFFER = params.get("max_tokens", 1000) or 1000
        SYSTEM_TOKENS = 1500  # Estimate for system prompt + robust guidelines

        def estimate_tokens(text: str) -> int:
            return len(text) // 3  # Conservative estimate (3 chars/token)

        # 1. Calculate History Tokens
        history_text = ""
        if conversation_history:
            history_text = "\n".join(
                [f"{t.get('role')}: {t.get('content')}" for t in conversation_history]
            )

        history_tokens = estimate_tokens(history_text)
        question_tokens = estimate_tokens(question)

        available_tokens = (
            MODEL_CONTEXT_LIMIT - RESPONSE_BUFFER - SYSTEM_TOKENS - question_tokens
        )

        # 2. Truncate History if too large (keep recent, discard old)
        if history_tokens > available_tokens * 0.25:  # Cap history at 25% of available
            max_history_tokens = int(available_tokens * 0.25)
            logger.info(
                f"Truncating conversation history (current: {history_tokens}, max: {max_history_tokens})"
            )

            # Keep popping pairs until fit or only 1 pair left
            while conversation_history and len(conversation_history) > 2:
                # Remove oldest pair if possible, or just oldest messages
                # Try to preserve user/assistant content structure if possible
                conversation_history.pop(0)
                if conversation_history:
                    conversation_history.pop(0)

                # Re-estimate
                temp_text = "\n".join(
                    [
                        f"{t.get('role')}: {t.get('content')}"
                        for t in conversation_history
                    ]
                )
                if estimate_tokens(temp_text) <= max_history_tokens:
                    break

            # Recalculate usage
            history_text = "\n".join(
                [f"{t.get('role')}: {t.get('content')}" for t in conversation_history]
            )
            history_tokens = estimate_tokens(history_text)

        # 3. Truncate Chunks (Core Context)
        available_for_context = available_tokens - history_tokens

        # Ensure we have at least SOME space for context
        available_for_context = max(available_for_context, 500)

        current_context_tokens = 0
        final_chunks = []

        for chunk in chunks:
            text = chunk.get("text", "")
            chunk_tokens = estimate_tokens(text) + 50  # +50 for metadata overhead

            if current_context_tokens + chunk_tokens > available_for_context:
                logger.warning(
                    f"Token limit reached, dropping remaining chunks (Used: {current_context_tokens}, Available: {available_for_context})"
                )
                break

            current_context_tokens += chunk_tokens
            final_chunks.append(chunk)

        chunks = final_chunks
        # ===== END TOKEN LIMIT ENFORCEMENT =====

        # Prepare context for prompt builder
        formatted_chunks = self._format_sources(chunks)

        # Build and validate prompt
        prompt_result = self.prompt_builder.build_prompt(
            question=question,
            context_chunks=formatted_chunks,
            keywords=params.get("all_keywords", []),
            conversation_history=conversation_history,
        )

        # Handle ambiguous questions or missing context
        if prompt_result.status == "ambiguous":
            params["is_ambiguous"] = True
            clarification = prompt_result.message or build_clarification_message(
                question, self.prompt_builder.config
            )
            return ChatResponse(
                answer=clarification,
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=True,
                    score=max(params.get("ambiguity_score", 0.0), 0.85),
                    clarification_requested=True,
                ),
            )

        if prompt_result.status == "no_context":
            logger.info("Prompt builder reported no context; issuing refusal")
            return ChatResponse(
                answer="I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=params.get("is_ambiguous", False),
                    score=params.get("ambiguity_score", 0.0),
                    clarification_requested=False,
                ),
            )

        # Generate response using the validated prompt
        try:
            response_text = generate_with_llm(
                prompt=prompt_result.prompt,
                temperature=params.get("temperature", 0.1),
                max_tokens=params.get("max_tokens", 1000),
                model=params.get("model"),
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return ChatResponse(
                answer="I'm temporarily unable to generate a complete answer. "
                "However, I found relevant information in the following sources.",
                sources=self._build_chat_sources(formatted_chunks),
                grounded=True,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=params.get("is_ambiguous", False),
                    score=params.get("ambiguity_score", 0.0),
                    clarification_requested=False,
                ),
            )

        # Use natural language response directly
        answer = (response_text or "").strip()

        # ===== METRICS: Token Usage & Cost =====
        if METRICS_ENABLED:
            model_used = params.get("model", "unknown")
            input_tokens = len(prompt_result.prompt) // 3
            output_tokens = len(answer) // 3

            rag_llm_token_usage_total.labels(type="input", model=model_used).inc(
                input_tokens
            )
            rag_llm_token_usage_total.labels(type="output", model=model_used).inc(
                output_tokens
            )

            # Estimate cost (very rough approximation)
            # Groq Llama 3 8B: ~$0.05 / 1M input, ~$0.08 / 1M output
            # Ollama: $0
            cost = 0.0
            if "groq" in settings.llm.provider:
                # avg $0.10 per million tokens (conservative estimate)
                cost = (input_tokens + output_tokens) / 1_000_000 * 0.10

            rag_llm_cost_total.labels(model=model_used).inc(cost)
        # ===== END METRICS =====

        # Validate answer has meaningful content
        MIN_ANSWER_LENGTH = 10
        if not answer or len(answer) < MIN_ANSWER_LENGTH:
            logger.warning(
                f"LLM returned empty or minimal response (len={len(answer)})"
            )
            return ChatResponse(
                answer="I was unable to generate a complete answer. "
                "Please try rephrasing your question.",
                sources=self._build_chat_sources(formatted_chunks),
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=False,
                    score=0.0,
                    clarification_requested=False,
                ),
            )

        # Clean answer to fix common formatting issues
        answer = _clean_answer(answer, question)

        # Filter potential PII from response
        answer = _filter_pii(answer)
        logger.info(f"Generated answer: {answer[:100]}...")

        # Check for refusal
        if self.prompt_builder.is_refusal(answer):
            logger.info("Answer is a refusal")
            return ChatResponse(
                answer=answer,
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=params.get("is_ambiguous", False),
                    score=params.get("ambiguity_score", 0.0),
                    clarification_requested=False,
                ),
            )

        # Check if answer is a clarification request (heuristic)
        clarification_indicators = [
            "i can help with:",
            "what would you like to know",
            "could you clarify",
            "please specify",
            "which aspect",
        ]
        is_clarification = any(
            indicator in answer.lower() for indicator in clarification_indicators
        )

        if is_clarification:
            logger.info("Answer appears to be a clarification request")
            return ChatResponse(
                answer=answer,
                sources=[],
                grounded=False,
                session_id=session.session_id,
                rewrite_metadata=rewrite_metadata,
                ambiguity=AmbiguityMetadata(
                    is_ambiguous=True,
                    score=0.8,
                    clarification_requested=True,
                ),
            )

        try:
            session.add_turn("user", question)
            session.add_turn("assistant", answer)
            self.session_store.update_session(session)
            logger.info(
                f"Updated session {mask_session_id(session.session_id)} "
                f"(total turns: {len(session.history)})"
            )
        except Exception as e:
            # Log error but don't fail the request
            logger.error(f"Failed to update session history: {e}")

        # ===== ANALYTICS: Successful grounded response =====
        if settings.query_rewriter.analytics_enabled:
            best_distance = chunks[0]["distance"] if chunks else None

            analytics = get_pattern_analytics()
            analytics.log_query(
                query=question,
                rewrite_metadata=rewrite_metadata,
                retrieval_distance=best_distance,
                grounded=True,
            )

            # Note: Don't capture in suggester for successful queries
        # ===== END ANALYTICS =====

        # Build successful grounded response
        response = ChatResponse(
            answer=answer,
            sources=self._build_chat_sources(formatted_chunks),
            grounded=True,
            session_id=session.session_id,
            rewrite_metadata=rewrite_metadata,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=params.get("is_ambiguous", False),
                score=params.get("ambiguity_score", 0.0),
                clarification_requested=False,
            ),
        )

        # Track grounding success
        if METRICS_ENABLED:
            rag_grounding_total.labels(grounded="true").inc()

        # ===== CACHE RESPONSE =====
        # Cache grounded responses for future requests
        # Don't cache conversational responses (they depend on history)
        response_cache = get_response_cache()
        cache_session_id = session.session_id if len(conversation_history) > 0 else None

        # Build cache params from params dict
        cache_params = {
            "top_k": params.get("top_k"),
            "max_distance": params.get("max_distance"),
            "temperature": params.get("temperature"),
            "model": params.get("model"),
            "doc_type": params.get("doc_type"),
        }

        response_cache.set(
            question=question,
            response=response.model_dump(),
            session_id=cache_session_id,
            params=cache_params,
        )
        # ===== END CACHE =====

        return response


__all__ = ["ChatService"]
