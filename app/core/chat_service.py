"""
Chat service for Personal RAG system.

Main business logic for handling chat requests, orchestrating retrieval,
prompt building, and LLM generation.
"""

import logging
from typing import Any, Dict, List, Optional

from fastapi import HTTPException, status

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
from app.services.llm import generate_with_ollama
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
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for chat service")


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


def _is_truly_ambiguous(
    question: str, conversation_history: Optional[List[Dict[str, str]]] = None
) -> bool:
    """Detect ONLY obviously vague questions (1-2 words).

    Catches the clear-cut cases where questions are too short to be meaningful.
    Lets the LLM handle gray areas via the system prompt.

    If conversation history exists, be more lenient with short questions since
    they may be valid follow-ups that rely on context.

    Args:
        question: User's question
        conversation_history: Optional conversation history for context-aware detection

    Returns:
        True if question is obviously too vague (1-2 words with filler)
    """
    q = question.strip()

    # Empty or just punctuation
    if not q or len(q) <= 2:
        return True

    # Remove punctuation and split into words
    words = q.replace("?", "").replace(".", "").replace("!", "").strip().split()

    # If we have conversation history, be more lenient with short questions
    # They might be valid context-dependent follow-ups
    if conversation_history and len(conversation_history) > 0:
        # Allow single-word questions if context exists (e.g., "Expiration?", "When?", "Why?")
        # Only reject truly empty or nonsensical queries
        if len(words) >= 1 and len(q) > 2:
            # Has at least one meaningful word and context - not ambiguous
            return False

    # Without conversation history, apply stricter rules
    # Single word queries (always vague without context)
    if len(words) <= 1:
        return True

    # Two words with filler words (very likely vague)
    if len(words) == 2:
        filler_words = {"my", "the", "a", "an", "your"}
        if any(word.lower() in filler_words for word in words):
            return True  # "My experience?", "The background?"

    # Everything else: let the LLM decide via system prompt
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


def _check_retrieval_quality(
    query: str, chunks: List[dict], min_term_overlap: int = 1
) -> dict:
    """Check if retrieved chunks contain important query terms.

    This pre-check runs BEFORE the LLM call to detect weak retrieval
    that might lead to incorrect answers.

    Args:
        query: The user's query
        chunks: Retrieved chunks
        min_term_overlap: Minimum number of query terms that should appear in chunks

    Returns:
        Dict with 'is_weak', 'reason', and 'found_terms'
    """
    if not chunks:
        return {"is_weak": True, "reason": "no_chunks", "found_terms": []}

    # Extract important terms from query (skip common words)
    stop_words = {
        "what",
        "which",
        "where",
        "when",
        "who",
        "how",
        "why",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "my",
        "i",
        "me",
        "you",
        "your",
        "any",
        "get",
    }

    query_words = (
        query.lower().replace("?", "").replace(".", "").replace(",", "").split()
    )
    important_terms = [w for w in query_words if w not in stop_words and len(w) > 2]

    if not important_terms:
        # No important terms to check - consider OK
        return {"is_weak": False, "reason": "no_terms_to_check", "found_terms": []}

    # Check if important terms appear in any chunk
    all_chunk_text = " ".join([c.get("text", "").lower() for c in chunks])
    found_terms = [term for term in important_terms if term in all_chunk_text]

    # Check for weak retrieval
    overlap_ratio = len(found_terms) / len(important_terms) if important_terms else 1.0

    if len(found_terms) < min_term_overlap or overlap_ratio < 0.3:
        return {
            "is_weak": True,
            "reason": "low_term_overlap",
            "found_terms": found_terms,
            "missing_terms": [t for t in important_terms if t not in found_terms],
            "overlap_ratio": overlap_ratio,
        }

    return {
        "is_weak": False,
        "reason": "ok",
        "found_terms": found_terms,
        "overlap_ratio": overlap_ratio,
    }


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
                "source": c.get("source", "unknown"),
                "text": c.get("text", ""),
                "metadata": c.get("metadata", {}),
            }
            for c in chunks
            if c.get("text", "").strip()
        ]

    def _build_chat_sources(self, chunks: List[dict]) -> List[ChatSource]:
        """Create output source metadata for chatbot responses.

        Args:
            chunks: List of retrieved chunks with text, source, and metadata

        Returns:
            List of ChatSource objects
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
                )
            )
        return sources

    @time_execution_info
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
    ) -> ChatResponse:
        """Handle a chat request with RAG.

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

        Returns:
            ChatResponse with answer and metadata
        """
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

        # Check for negative inference opportunity
        # If the question asks about a specific entity that doesn't exist in the KB,
        # search for the category instead to find complete lists
        from app.retrieval.negative_inference_helper import (
            detect_negative_inference_opportunity,
        )

        neg_inf_result = detect_negative_inference_opportunity(search_query)

        if neg_inf_result and neg_inf_result["is_negative_inference_candidate"]:
            # Use category search instead of entity search
            search_query = neg_inf_result["suggested_category_search"]
            logger.info(
                f"Negative inference detected: '{rewritten_query}' → "
                f"searching for category: '{search_query}'"
            )
            missing = [e["entity"] for e in neg_inf_result["missing_entities"]]
            logger.info(f"Missing entities: {missing}")

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

        # ===== RETRIEVAL QUALITY PRE-CHECK =====
        # Check if retrieved chunks actually contain relevant query terms
        # This helps detect weak retrieval before sending to LLM
        quality_check = _check_retrieval_quality(request.question, chunks)
        if quality_check["is_weak"]:
            logger.warning(
                f"Weak retrieval detected: {quality_check['reason']} - "
                f"missing terms: {quality_check.get('missing_terms', [])}"
            )
            # Log for debugging but continue - the LLM may still be able to answer
            # In production, could add retry logic here
        else:
            logger.info(
                f"Retrieval quality OK: found terms {quality_check['found_terms']} "
                f"(overlap: {quality_check.get('overlap_ratio', 1.0):.0%})"
            )
        # ===== END QUALITY PRE-CHECK =====

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
            neg_inf_result,
            rewrite_metadata,
        )

    def _handle_standard_query(
        self,
        question: str,
        params: Dict[str, Any],
        chunks: List[dict],
        session: "Session",
        conversation_history: List[Dict[str, str]],
        neg_inf_result: Optional[Dict[str, Any]] = None,
        rewrite_metadata: Optional[RewriteMetadata] = None,
    ) -> ChatResponse:
        """Handle standard RAG query with LLM generation.

        Args:
            question: User's question
            params: Query parameters
            chunks: Retrieved chunks
            session: Session object
            conversation_history: Conversation history
            neg_inf_result: Optional negative inference detection result

        Returns:
            ChatResponse
        """
        # Prepare context for prompt builder
        formatted_chunks = self._format_sources(chunks)

        # Prepare negative inference hint if detected
        negative_inference_hint = None
        if neg_inf_result and neg_inf_result.get("is_negative_inference_candidate"):
            # Map category search query to human-readable category name
            category_map = {
                "work experience": "employers/companies",
                "certifications": "professional certifications",
                "personal projects": "personal projects",
                "education": "degrees",
            }

            category_search = neg_inf_result.get("suggested_category_search", "")
            category_name = "items"
            for key, value in category_map.items():
                if key in category_search.lower():
                    category_name = value
                    break

            negative_inference_hint = {
                "missing_entities": [
                    e["entity"] for e in neg_inf_result["missing_entities"]
                ],
                "category": category_name,
            }

        # Build and validate prompt
        prompt_result = self.prompt_builder.build_prompt(
            question=question,
            context_chunks=formatted_chunks,
            keywords=params.get("all_keywords", []),
            negative_inference_hint=negative_inference_hint,
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
        response_text = generate_with_ollama(
            prompt=prompt_result.prompt,
            temperature=params.get("temperature", 0.1),
            max_tokens=params.get("max_tokens", 1000),
            model=params.get("model"),
        )

        # Use natural language response directly
        answer = response_text.strip()
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
            sources=self._build_chat_sources(chunks),
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
