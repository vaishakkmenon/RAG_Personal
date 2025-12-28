"""
Chat service for Personal RAG system.

Main business logic for handling chat requests, orchestrating retrieval,
prompt building, and LLM generation.
"""

import logging
import re
import time
from typing import Any, Dict, List, Optional, AsyncIterator, Tuple
import json
from dataclasses import dataclass

from fastapi import HTTPException, status

from app.models import (
    ChatRequest,
    ChatResponse,
    AmbiguityMetadata,
    RewriteMetadata,
)
from app.core.session_manager import SessionManager
from app.core.query_validator import QueryValidator
from app.core.retrieval_orchestrator import RetrievalOrchestrator

from app.monitoring.pattern_analytics import get_pattern_analytics
from app.monitoring.pattern_suggester import get_pattern_suggester
from app.prompting import build_clarification_message, create_default_prompt_builder
from app.services.llm import generate_with_llm
from app.services.response_cache import get_response_cache
from app.settings import settings
from app.storage import Session
from app.core.parsing import ChunkType

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from app.metrics import (
        rag_grounding_total,
        rag_ambiguity_checks_total,
        rag_clarification_requests_total,
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for chat service")


@dataclass
class ChatOptions:
    """Configuration options for chat request processing."""

    grounded_only: Optional[bool] = None
    null_threshold: Optional[float] = None
    max_distance: Optional[float] = None
    top_k: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    rerank: Optional[bool] = None
    rerank_lex_weight: Optional[float] = None
    doc_type: Optional[str] = None
    term_id: Optional[str] = None
    level: Optional[str] = None
    model: Optional[str] = None
    show_thinking: Optional[bool] = (
        None  # Enable thinking process display (Qwen models)
    )
    skip_route_cache: bool = False
    api_key: Optional[str] = None


@dataclass
class ChatContext:
    """Holds shared context for a chat request (sync or async)."""

    session: "Session"
    conversation_history: List[Dict[str, str]]
    chunks: List[Dict]
    formatted_chunks: List[Dict]
    query: str
    params: Dict[str, Any]
    rewrite_metadata: Optional[RewriteMetadata] = None
    # If set, we should return early with this specific response type
    early_return_type: Optional[str] = (
        None  # 'error', 'chitchat', 'ambiguous', 'rate_limit', 'blocked', 'empty', 'cache_hit'
    )
    early_return_payload: Optional[Any] = None


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
    """Redact potential PII from LLM responses."""
    for pattern, replacement in PII_PATTERNS:
        text = re.sub(pattern, replacement, text)
    return text


def _clean_answer(answer: str, question: str) -> str:
    """Clean up common answer formatting issues."""
    if not answer:
        return answer

    # === TEMPORARILY DISABLED FOR RAW OUTPUT TESTING ===
    # To re-enable, uncomment the code below and remove "return answer"
    return answer

    # Fix pattern: "No, I have X but..." -> "I have X"
    # if answer.startswith("No, ") and " but " in answer:
    #     parts = answer.split(" but ", 1)
    #     if len(parts) == 2:
    #         q_lower = question.lower().strip()
    #         if any(q_lower.startswith(word) for word in ("which", "what", "list")):
    #             cleaned = parts[1].strip()
    #             cleaned = cleaned[0].upper() + cleaned[1:]
    #             return cleaned

    # # Apply comprehensive output sanitization:
    # # - Strips trailing "References:", "Sources:", citation lists
    # # - Checks for prompt leakage
    # # - Handles internal terminology warnings
    # answer, had_issues = sanitize_response(answer, strict=True)

    # # If sanitize_response returned a fallback due to prompt leakage, use it
    # if had_issues and "encountered an issue" in answer:
    #     logger.warning("Prompt leakage detected in response, returning safe fallback")
    #     return (
    #         "I can help answer questions about Vaishak's professional background. "
    #         "Could you please rephrase your question?"
    #     )

    # # Additional cleanup: Strip inline citation references at very end of answer
    # # e.g., "...graduated in 2024 [1][2]" -> "...graduated in 2024"
    # answer = re.sub(
    #     r"\s*(?:\[\d+\](?:\s*,?\s*)?)+\s*$",
    #     "",
    #     answer,
    # ).strip()

    # return answer


class ChatService:
    """Service for handling RAG chat requests."""

    def __init__(self, session_store=None):
        """Initialize chat service."""
        self.session_manager = SessionManager(session_store)
        self.validator = QueryValidator()
        self.retrieval = RetrievalOrchestrator()
        self.prompt_builder = create_default_prompt_builder()

        logger.info("ChatService initialized with decomposed modules")

    def _prepare_chat_context(
        self,
        request: ChatRequest,
        options: ChatOptions,
    ) -> ChatContext:
        """Prepare context for chat (retrieval, session, validation).

        Args:
            request: Chat request with question
            options: Configuration options for processing

        Returns:
            ChatContext with prepared data or early return indicator
        """
        # 1. Validation & Normalization
        question = self.validator.normalize_query(request.question)

        # 2. Session Management
        session = self.session_manager.get_or_create_session(request)

        # 3. Rate Limit
        if not self.session_manager.check_rate_limit(session):
            return ChatContext(
                session=session,
                conversation_history=[],
                chunks=[],
                formatted_chunks=[],
                query=question,
                params={},
                early_return_type="rate_limit",
            )

        # 4. Conversation History
        history = session.get_truncated_history()

        # Consolidation of params from options
        params = {
            "top_k": options.top_k
            if options.top_k is not None
            else settings.retrieval.top_k,
            "max_distance": options.max_distance
            if options.max_distance is not None
            else settings.retrieval.max_distance,
            "temperature": options.temperature
            if options.temperature is not None
            else settings.llm.temperature,
            "max_tokens": options.max_tokens
            if options.max_tokens is not None
            else settings.llm.max_tokens,
            "model": options.model,
            "show_thinking": options.show_thinking or False,
            "doc_type": options.doc_type,
            "rerank": options.rerank,
            "rerank_lex_weight": options.rerank_lex_weight,
            "term_id": options.term_id,
            "level": options.level,
            "null_threshold": options.null_threshold
            if options.null_threshold is not None
            else settings.retrieval.null_threshold,
            "is_ambiguous": False,
            "ambiguity_score": 0.0,
            "all_keywords": [],
        }

        # 5. Safety Check (Prompt Guard)
        guard_result = self.validator.check_safety(question, history)
        if guard_result["blocked"]:
            return ChatContext(
                session=session,
                conversation_history=history,
                chunks=[],
                formatted_chunks=[],
                query=question,
                params=params,
                early_return_type="blocked",
            )

        # 6. Route Cache
        if not options.skip_route_cache:
            response_cache = get_response_cache()
            cache_params = {
                "top_k": params["top_k"],
                "max_distance": params["max_distance"],
                "temperature": params["temperature"],
                "model": params["model"],
                "doc_type": params["doc_type"],
            }

            cached_response = response_cache.get(
                question=request.question,  # Use original for cache key
                session_id=session.session_id if len(history) > 0 else None,
                params=cache_params,
            )

            if cached_response:
                cached_response["session_id"] = session.session_id
                return ChatContext(
                    session=session,
                    conversation_history=history,
                    chunks=[],
                    formatted_chunks=[],
                    query=question,
                    params=params,
                    early_return_type="cache_hit",
                    early_return_payload=cached_response,
                )

        # 7. Chitchat Detection
        is_chitchat, chitchat_response = self.validator.detect_chitchat(question)
        if is_chitchat:
            return ChatContext(
                session=session,
                conversation_history=history,
                chunks=[],
                formatted_chunks=[],
                query=question,
                params=params,
                early_return_type="chitchat",
                early_return_payload=chitchat_response,
            )

        # 8. Ambiguity Detection
        if self.validator.detect_ambiguity(question, history):
            clarification = build_clarification_message(
                question, self.prompt_builder.config
            )
            return ChatContext(
                session=session,
                conversation_history=history,
                chunks=[],
                formatted_chunks=[],
                query=question,
                params=params,
                early_return_type="ambiguous",
                early_return_payload=clarification,
            )

        # 9. Retrieval execution
        try:
            chunks, rewrite_metadata = self.retrieval.perform_retrieval(
                question, history, params
            )
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to retrieve documents: {str(e)}",
            )

        # 10. Grounding Check
        # 10. Grounding Check
        if not chunks:
            return ChatContext(
                session=session,
                conversation_history=history,
                chunks=[],
                formatted_chunks=[],
                query=question,
                params=params,
                early_return_type="empty",
                rewrite_metadata=rewrite_metadata,
            )

        # 11. Token Management (Truncation)
        history, chunks = self._enforce_token_limits(
            question=question,
            history=history,
            chunks=chunks,
            max_tokens=params["max_tokens"],
        )

        # Format chunks
        formatted_chunks = self.retrieval.format_sources_for_prompt(chunks)

        return ChatContext(
            session=session,
            conversation_history=history,
            chunks=chunks,
            formatted_chunks=formatted_chunks,
            query=question,
            params=params,
            rewrite_metadata=rewrite_metadata,
        )

    def handle_chat(
        self,
        request: ChatRequest,
        options: Optional[ChatOptions] = None,
    ) -> ChatResponse:
        """Handle a chat request with RAG.

        Args:
            request: Chat request with question
            options: Optional configuration options (uses defaults if None)

        Returns:
            ChatResponse with answer and metadata
        """
        time.time()  # Ensure import is used

        # Use default options if not provided
        options = options or ChatOptions()

        # Step 1: Prepare Context
        ctx = self._prepare_chat_context(request=request, options=options)

        # Step 2: Handle Early Returns
        if ctx.early_return_type:
            return self._handle_early_return_sync(ctx, request)

        # Step 3: Standard RAG Flow
        if METRICS_ENABLED:
            rag_ambiguity_checks_total.labels(result="clear").inc()

        # Unpack context
        session = ctx.session
        history = ctx.conversation_history
        rewrite_metadata = ctx.rewrite_metadata
        formatted_chunks = ctx.formatted_chunks
        chunks = ctx.chunks
        params = ctx.params
        question = ctx.query

        # Build prompt (model-aware: selects Qwen vs Llama prompt template)
        prompt_result = self.prompt_builder.build_prompt(
            question=question,
            context_chunks=formatted_chunks,
            conversation_history=history,
            model=params.get("model"),
        )

        if prompt_result.status == "ambiguous":
            params["is_ambiguous"] = True
            clarification = prompt_result.message or build_clarification_message(
                question, self.prompt_builder.config
            )
            return self._create_response(
                clarification,
                [],
                False,
                session,
                rewrite_metadata,
                ambiguity=True,
                clarification=True,
                params=params,
            )

        if prompt_result.status == "no_context":
            return self._create_response(
                "I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
                [],
                False,
                session,
                rewrite_metadata,
                params=params,
            )

        # Generate
        try:
            response_text = generate_with_llm(
                prompt=prompt_result.prompt,
                temperature=params.get("temperature", 0.1),
                max_tokens=params.get("max_tokens", 1000),
                model=params.get("model"),
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return self._create_response(
                "I'm temporarily unable to generate a complete answer. However, I found relevant information in the following sources.",
                formatted_chunks,
                True,
                session,
                rewrite_metadata,
                params=params,
            )

        answer = (response_text or "").strip()

        # Validations
        if not answer or len(answer) < 10:
            return self._create_response(
                "I was unable to generate a complete answer. Please try rephrasing your question.",
                formatted_chunks,
                False,
                session,
                rewrite_metadata,
                params=params,
            )

        answer = _clean_answer(answer, question)
        answer = _filter_pii(answer)

        if self.prompt_builder.is_refusal(answer):
            return self._create_response(
                answer, [], False, session, rewrite_metadata, params=params
            )

        # Check clarification in answer
        if any(
            x in answer.lower()
            for x in [
                "i can help with:",
                "what would you like to know",
                "could you clarify",
            ]
        ):
            return self._create_response(
                answer,
                [],
                False,
                session,
                rewrite_metadata,
                ambiguity=True,
                clarification=True,
                params=params,
            )

        # Success - update session
        try:
            session.add_turn("user", question)
            session.add_turn("assistant", answer)
            self.session_manager.update_session(session)
        except Exception:
            pass

        # Log Metrics/Analytics
        self._log_success_analytics(question, rewrite_metadata, chunks)

        # Cache
        self._cache_response(
            question, session, params, answer, formatted_chunks, rewrite_metadata
        )

        return self._create_response(
            answer, formatted_chunks, True, session, rewrite_metadata, params=params
        )

    async def handle_chat_stream(
        self,
        request: ChatRequest,
        options: Optional[ChatOptions] = None,
    ) -> AsyncIterator[str]:
        """Handle streaming chat request.

        Args:
            request: Chat request with question
            options: Optional configuration options (uses defaults if None, skip_route_cache forced to True)

        Yields:
            Server-Sent Events as strings
        """
        # Use default options if not provided, force skip_route_cache for streaming
        options = options or ChatOptions()
        options.skip_route_cache = True  # Always skip cache for streaming

        # 1. Prepare
        ctx = self._prepare_chat_context(request, options)

        # 2. Early Returns (Yield SSE)
        if ctx.early_return_type:
            async for event in self._handle_early_return_stream(ctx, request):
                yield event
            return

        # 3. Standard Flow
        if METRICS_ENABLED:
            rag_ambiguity_checks_total.labels(result="clear").inc()

        # Unpack
        question = ctx.query
        params = ctx.params
        chunks = ctx.chunks
        session = ctx.session
        history = ctx.conversation_history
        rewrite_metadata = ctx.rewrite_metadata
        formatted_chunks = ctx.formatted_chunks

        # Distance check for streaming (refusal)
        # Handle case where distance might be None or missing
        top_distance = chunks[0].get("distance") if chunks else 0.0
        if top_distance is None:
            top_distance = 0.0

        if top_distance > params["null_threshold"]:
            msg = "I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently."
            yield self._sse_metadata([], False, session.session_id)
            yield self._sse_token(msg)
            return

        # Build Prompt (model-aware: selects Qwen vs Llama prompt template)
        prompt_result = self.prompt_builder.build_prompt(
            question=question,
            context_chunks=formatted_chunks,
            conversation_history=history,
            model=params.get("model"),
        )

        if prompt_result.status in ["ambiguous", "no_context"]:
            msg = prompt_result.message or "I don't know."
            yield self._sse_metadata(
                [],
                False,
                session.session_id,
                is_ambiguous=(prompt_result.status == "ambiguous"),
            )
            yield self._sse_token(msg)
            return

        # Stream Generation
        # Metadata first
        sources = self.retrieval.build_chat_sources(formatted_chunks)
        metadata = {
            "sources": [s.model_dump() for s in sources],
            "grounded": True,
            "session_id": session.session_id,
            "rewrite_metadata": rewrite_metadata.model_dump()
            if rewrite_metadata
            else None,
            "is_ambiguous": False,
        }
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        full_answer = []
        full_thinking = []
        show_thinking = params.get("show_thinking", False)

        try:
            if show_thinking:
                # Use thinking-aware streaming for models that support it
                from app.services.llm import async_generate_stream_with_thinking

                async for chunk in async_generate_stream_with_thinking(
                    prompt=prompt_result.prompt,
                    temperature=params.get("temperature", 0.1),
                    max_tokens=params.get("max_tokens", 1000),
                    model=params.get("model"),
                ):
                    if chunk.content:
                        if chunk.type == ChunkType.THINKING:
                            # Stream thinking process for frontend display
                            full_thinking.append(chunk.content)
                            yield self._sse_thinking(chunk.content)
                        else:
                            # Stream answer tokens
                            full_answer.append(chunk.content)
                            yield self._sse_token(chunk.content)
            else:
                # Standard streaming without thinking separation
                from app.services.llm import async_generate_stream_with_llm

                async for token in async_generate_stream_with_llm(
                    prompt=prompt_result.prompt,
                    temperature=params.get("temperature", 0.1),
                    max_tokens=params.get("max_tokens", 1000),
                    model=params.get("model"),
                ):
                    if token:
                        full_answer.append(token)
                        yield self._sse_token(token)

        except Exception as e:
            logger.error(f"Stream generation failed: {e}")
            yield f"event: error\ndata: {json.dumps({'detail': 'Generation failed'})}\n\n"
            return

        # Send thinking_done event if there was thinking
        if full_thinking:
            yield "event: thinking_done\ndata: \n\n"

        yield "event: done\ndata: \n\n"

        # Post-process
        answer = "".join(full_answer).strip()
        try:
            session.add_turn("user", question)
            session.add_turn("assistant", answer)
            self.session_manager.update_session(session)
        except Exception:
            pass

        if METRICS_ENABLED:
            rag_grounding_total.labels(grounded="true").inc()
            self._log_success_analytics(question, rewrite_metadata, chunks)

    # --- Helpers for ChatService (Private) ---

    def _create_response(
        self,
        answer,
        chunks,
        grounded,
        session,
        rewrite_meta,
        ambiguity=False,
        clarification=False,
        params=None,
        thinking=None,
    ):
        return ChatResponse(
            answer=answer,
            sources=self.retrieval.build_chat_sources(chunks),
            grounded=grounded,
            session_id=session.session_id,
            rewrite_metadata=rewrite_meta,
            ambiguity=AmbiguityMetadata(
                is_ambiguous=ambiguity,
                score=params.get("ambiguity_score", 0.0) if params else 0.0,
                clarification_requested=clarification,
            ),
            thinking=thinking,
        )

    def _create_blocked_response(self, session):
        return ChatResponse(
            answer="I cannot answer this request as it triggers safety guardrails.",
            sources=[],
            grounded=False,
            session_id=session.session_id,
            rewrite_metadata=None,
            ambiguity=AmbiguityMetadata(False, 0.0, False),
            thinking=None,
        )

    def _handle_early_return_sync(self, ctx, request):
        if ctx.early_return_type == "rate_limit":
            raise HTTPException(status_code=429, detail="Rate limit exceeded")
        elif ctx.early_return_type == "blocked":
            return self._create_blocked_response(ctx.session)
        elif ctx.early_return_type == "cache_hit":
            return ChatResponse(**ctx.early_return_payload)
        elif ctx.early_return_type == "chitchat":
            try:
                ctx.session.add_turn("user", request.question)
                ctx.session.add_turn("assistant", ctx.early_return_payload)
                self.session_manager.update_session(ctx.session)
            except Exception:
                pass
            return self._create_response(
                ctx.early_return_payload, [], False, ctx.session, None
            )
        elif ctx.early_return_type == "ambiguous":
            if METRICS_ENABLED:
                rag_ambiguity_checks_total.labels(result="ambiguous").inc()
                rag_clarification_requests_total.inc()
            return self._create_response(
                ctx.early_return_payload,
                [],
                False,
                ctx.session,
                None,
                ambiguity=True,
                clarification=True,
            )
        elif ctx.early_return_type == "empty":
            self._log_failed_analytics(request.question, ctx.rewrite_metadata)
            return self._create_response(
                "I don't know. I couldn't find sufficiently relevant information in my documents to answer this question confidently.",
                [],
                False,
                ctx.session,
                ctx.rewrite_metadata,
                params=ctx.params,
            )
        return None

    async def _handle_early_return_stream(self, ctx, request):
        if ctx.early_return_type == "rate_limit":
            yield f"event: error\ndata: {json.dumps({'detail': 'Rate limit exceeded'})}\n\n"
        elif ctx.early_return_type == "blocked":
            yield f"event: error\ndata: {json.dumps({'detail': 'Request blocked by safety guardrails'})}\n\n"
        elif ctx.early_return_type == "chitchat":
            meta = {
                "sources": [],
                "grounded": False,
                "session_id": ctx.session.session_id,
                "is_ambiguous": False,
            }
            yield f"event: metadata\ndata: {json.dumps(meta)}\n\n"
            yield self._sse_token(ctx.early_return_payload)
            try:
                ctx.session.add_turn("user", request.question)
                ctx.session.add_turn("assistant", ctx.early_return_payload)
                self.session_manager.update_session(ctx.session)
            except Exception:
                pass
        elif ctx.early_return_type == "ambiguous":
            if METRICS_ENABLED:
                rag_ambiguity_checks_total.labels(result="ambiguous").inc()
                rag_clarification_requests_total.inc()
            meta = {
                "sources": [],
                "grounded": False,
                "session_id": ctx.session.session_id,
                "is_ambiguous": True,
            }
            yield f"event: metadata\ndata: {json.dumps(meta)}\n\n"
            yield self._sse_token(ctx.early_return_payload)
        elif ctx.early_return_type == "empty":
            self._log_failed_analytics(request.question, ctx.rewrite_metadata)
            meta = {
                "sources": [],
                "grounded": False,
                "session_id": ctx.session.session_id,
                "is_ambiguous": False,
            }
            yield f"event: metadata\ndata: {json.dumps(meta)}\n\n"
            yield self._sse_token(
                "I don't know. I couldn't find any relevant information in my documents."
            )

    def _sse_token(self, text):
        return f"event: token\ndata: {json.dumps(text)}\n\n"

    def _sse_thinking(self, text):
        """Emit SSE event for model thinking process.

        Frontend can display this in a collapsible "Thinking..." section.
        """
        return f"event: thinking\ndata: {json.dumps(text)}\n\n"

    def _sse_metadata(self, sources, grounded, session_id, is_ambiguous=False):
        meta = {
            "sources": sources,
            "grounded": grounded,
            "session_id": session_id,
            "is_ambiguous": is_ambiguous,
        }
        return f"event: metadata\ndata: {json.dumps(meta)}\n\n"

    def _log_success_analytics(self, question, rewrite_metadata, chunks):
        if settings.query_rewriter.analytics_enabled:
            analytics = get_pattern_analytics()
            analytics.log_query(
                query=question,
                rewrite_metadata=rewrite_metadata,
                retrieval_distance=chunks[0]["distance"] if chunks else None,
                grounded=True,
            )

    def _log_failed_analytics(self, question, rewrite_metadata):
        if settings.query_rewriter.analytics_enabled:
            get_pattern_analytics().log_query(
                query=question,
                rewrite_metadata=rewrite_metadata,
                retrieval_distance=None,
                grounded=False,
            )
            get_pattern_suggester().capture_failed_query(
                query=question,
                distance=None,
                pattern_matched=rewrite_metadata.pattern_name
                if rewrite_metadata
                else None,
                grounded=False,
            )

    def _cache_response(
        self, question, session, params, answer, chunks, rewrite_metadata
    ):
        response_cache = get_response_cache()
        cache_session_id = session.session_id if len(session.history) > 0 else None
        cache_params = {
            "top_k": params.get("top_k"),
            "max_distance": params.get("max_distance"),
            "temperature": params.get("temperature"),
            "model": params.get("model"),
            "doc_type": params.get("doc_type"),
        }

        # Reconstruct response object for caching
        resp_obj = self._create_response(
            answer, chunks, True, session, rewrite_metadata, params=params
        )

        response_cache.set(
            question=question,
            response=resp_obj.model_dump(),
            session_id=cache_session_id,
            params=cache_params,
        )

    def _enforce_token_limits(
        self,
        question: str,
        history: List[Dict[str, str]],
        chunks: List[dict],
        max_tokens: int,
    ) -> Tuple[List[Dict[str, str]], List[dict]]:
        """Enforce strict token limits to prevent context window overflow."""

        def estimate_tokens(text: str) -> int:
            return len(text) // 3  # Conservative estimate

        # 1. Calculate Limits
        MAX_CONTEXT_WINDOW = 8192  # Safe default for Llama3-8b/70b (actually 128k but we enable 8k for speed)
        SAFETY_BUFFER = 500  # For template structure

        # Reserve output tokens
        available_context = MAX_CONTEXT_WINDOW - max_tokens - SAFETY_BUFFER

        # 2. Check Question Cost
        question_tokens = estimate_tokens(question)
        remaining = available_context - question_tokens

        if remaining < 0:
            # Extreme case: Question is too long
            # Just empty everything else
            return [], []

        # 3. Allocations
        # History gets up to 20% of remaining space
        history_budget = int(remaining * 0.20)
        chunks_budget = remaining - history_budget

        # 4. Truncate History
        # Keep most recent turns first
        truncated_history = []
        current_history_tokens = 0

        # Reverse to keep recent turns
        for turn in reversed(history):
            content = turn.get("content", "")
            tokens = estimate_tokens(content) + 10  # +10 for role markers

            if current_history_tokens + tokens <= history_budget:
                truncated_history.insert(0, turn)
                current_history_tokens += tokens
            else:
                break

        # 5. Truncate Chunks (Reclaim unused history budget)
        unused_history_budget = history_budget - current_history_tokens
        final_chunks_budget = chunks_budget + unused_history_budget

        truncated_chunks = []
        current_chunks_tokens = 0

        for chunk in chunks:
            text = chunk.get("text", "")
            metadata = str(chunk.get("metadata", ""))
            # Estimate: Text + Metadata + [Citation] overhead
            tokens = estimate_tokens(text) + estimate_tokens(metadata) + 20

            if current_chunks_tokens + tokens <= final_chunks_budget:
                truncated_chunks.append(chunk)
                current_chunks_tokens += tokens
            else:
                # Stop adding chunks once we hit the limit
                break

        return truncated_history, truncated_chunks
