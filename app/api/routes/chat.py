"""
Chat endpoint for Personal RAG system.
"""

import time
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, Request

from app.models import ChatRequest, ChatResponse, ChatSource
from app.api.dependencies import check_api_key, get_chat_service
from app.core import ChatService
from app.retrieval import search
from app.services.llm import generate_with_ollama
from app.services.prompt_guard import get_prompt_guard
from app.prompting import create_default_prompt_builder
from app.settings import settings
import logging
from app.exceptions import LLMException, RetrievalException, RAGException

router = APIRouter()
logger = logging.getLogger(__name__)

# Import end-to-end metrics
try:
    from app.metrics import (
        rag_request_total,
        rag_request_latency_seconds,
        rag_errors_total,
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("End-to-end metrics not available")


@router.post(
    "/chat/simple",
    response_model=ChatResponse,
    summary="Simple RAG chat without advanced features",
    description="""
    Simplified chat endpoint for testing and debugging. No query routing, filtering, or reranking.

    **Use Cases:**
    - Testing RAG functionality
    - Debugging retrieval issues
    - Simple question answering without advanced features

    **What's Different from /chat:**
    - No query routing or domain detection
    - No metadata filtering
    - No hybrid reranking
    - No session support (no conversation history)
    - Faster but less accurate

    **Example Request:**
    ```json
    {
      "question": "What is Vaishak's GPA?"
    }
    ```

    **Example Response:**
    ```json
    {
      "answer": "Based on the transcript, Vaishak's GPA is 3.85.",
      "sources": [...],
      "grounded": true,
      "session_id": "generated-uuid"
    }
    ```
    """,
    responses={
        200: {
            "description": "Successful response",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "Vaishak's GPA is 3.85 based on his undergraduate transcript.",
                        "sources": [],
                        "grounded": True,
                        "session_id": "temp-uuid-123",
                    }
                }
            },
        },
        400: {"description": "Bad request or prompt injection detected"},
        401: {"description": "Invalid or missing API key"},
        500: {"description": "Internal server error"},
        503: {"description": "LLM service unavailable"},
    },
)
def simple_chat(
    request: ChatRequest,
    # Optional parameters (with sensible defaults)
    top_k: int = 10,
    max_distance: float = 0.6,
    temperature: float = 0.1,
    max_tokens: int = 500,
    # Dependencies
    api_key: str = Depends(check_api_key),
):
    """Bare-bones RAG chat for testing. See description for details."""
    # Track request start time
    start_time = time.time()
    endpoint = "/chat/simple"

    try:
        logger.info(f"Simple chat - Question: {request.question}")

        # Step 0: Check for prompt injection
        # Note: simple_chat doesn't support sessions, so no conversation history
        guard = get_prompt_guard()
        guard_result = guard.check_input(
            request.question,
            conversation_history=None,  # Simple endpoint has no session support
        )
        if guard_result["blocked"]:
            logger.warning(f"Prompt injection blocked: {guard_result['label']}")
            raise HTTPException(
                status_code=400,
                detail="Your request could not be processed. Please rephrase your question.",
            )

        # Step 1: Simple search - no filters, no routing
        try:
            chunks = search(
                query=request.question,
                k=top_k,
                max_distance=max_distance,
                metadata_filter=None,  # No filtering!
            )
        except Exception as e:
            logger.error(f"Retrieval failed in simple_chat: {e}")
            raise RetrievalException("Failed to search knowledge base")

        logger.info(f"Retrieved {len(chunks)} chunks")

        # Step 2: Build prompt
        prompt_builder = create_default_prompt_builder()

        # Format sources for prompt
        formatted_sources = [
            {
                "source": chunk.get("source", "unknown"),
                "text": chunk.get("text", ""),
                "metadata": chunk.get("metadata", {}),
            }
            for chunk in chunks
        ]

        prompt_result = prompt_builder.build_prompt(
            question=request.question,
            context_chunks=formatted_sources,  # Fixed: parameter is context_chunks not sources
            conversation_history=[],  # Simple endpoint doesn't support conversation history
        )

        # Step 3: Generate answer (use prompt from PromptResult)
        try:
            answer = generate_with_ollama(
                prompt=prompt_result.prompt,  # Fixed: use prompt_result.prompt not just prompt
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            logger.error(f"LLM generation failed in simple_chat: {e}")
            raise LLMException("Failed to generate response")

        # Step 4: Format response
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

        grounded = len(chunks) > 0

        # Generate session_id if not provided (simple endpoint doesn't use sessions but needs to return one)
        import uuid

        session_id = request.session_id or str(uuid.uuid4())

        response = ChatResponse(
            answer=answer,
            sources=sources,
            grounded=grounded,
            session_id=session_id,
        )

        # Track success metrics
        if METRICS_ENABLED:
            rag_request_total.labels(endpoint=endpoint, status="success").inc()
            rag_request_latency_seconds.labels(endpoint=endpoint).observe(
                time.time() - start_time
            )

        return response

    except HTTPException as e:
        # Track HTTP errors
        if METRICS_ENABLED:
            rag_request_total.labels(endpoint=endpoint, status="error").inc()
            rag_errors_total.labels(
                component="api", error_type=f"http_{e.status_code}"
            ).inc()
        raise

    except RAGException:
        # Re-raise custom exceptions to be handled by app-level handler
        # We don't track these here because the handler/middleware will track them?
        # Actually proper metrics tracking for these custom exceptions might be good.
        # For now, just re-raise.
        raise

    except Exception as e:
        # Track unexpected errors
        logger.error(f"Unexpected error in {endpoint}: {str(e)}")
        if METRICS_ENABLED:
            rag_request_total.labels(endpoint=endpoint, status="error").inc()
            rag_errors_total.labels(component="api", error_type="unexpected").inc()
        raise


@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Ask a question about Vaishak's background",
    description="""
    Ask a question and receive an AI-generated answer based on Vaishak's documents (resume, transcripts, certifications).

    **Features:**
    - Retrieval Augmented Generation (RAG) for accurate, grounded responses
    - Semantic search across all documents
    - Optional hybrid reranking for better relevance
    - Session-based conversation history
    - Response caching for fast repeated queries
    - Prompt injection protection

    **Rate Limits:**
    - 50 queries per hour per session (adjustable in production)
    - 5 sessions per IP address
    - 1000 max total sessions

    **Session Management:**
    - Provide `session_id` to maintain conversation context
    - Sessions expire after 6 hours
    - Previous messages are used for follow-up questions

    **Performance:**
    - Mean latency: ~2.1s (cache miss), ~1.4s (cache hit)
    - 100% success rate with Groq Developer tier
    - Response caching enabled by default

    **Example Request:**
    ```json
    {
      "question": "What AI courses has Vaishak taken?",
      "session_id": "optional-session-id"
    }
    ```

    **Example Response:**
    ```json
    {
      "answer": "Vaishak has taken several AI/ML courses including...",
      "sources": [
        {
          "id": "chunk_123",
          "source": "transcript_fall_2024.md",
          "text": "CS 498: Applied Machine Learning...",
          "distance": 0.23
        }
      ],
      "grounded": true,
      "session_id": "abc-123-def"
    }
    ```
    """,
    responses={
        200: {
            "description": "Successful response with answer and sources",
            "content": {
                "application/json": {
                    "example": {
                        "answer": "Vaishak has taken several AI courses including CS 498 Applied Machine Learning, CS 444 Deep Learning, and CS 410 Text Information Systems.",
                        "sources": [
                            {
                                "id": "chunk_42",
                                "source": "transcript_fall_2024.md",
                                "text": "CS 498: Applied Machine Learning (Grade: A)",
                                "distance": 0.23,
                            }
                        ],
                        "grounded": True,
                        "session_id": "session-abc-123",
                    }
                }
            },
        },
        400: {
            "description": "Bad request - invalid input or prompt injection detected",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Your request could not be processed. Please rephrase your question."
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - invalid or missing API key",
            "content": {
                "application/json": {
                    "example": {"detail": "Invalid or missing API key"}
                }
            },
        },
        403: {
            "description": "Forbidden - origin not allowed",
            "content": {
                "application/json": {"example": {"detail": "Origin not allowed"}}
            },
        },
        429: {
            "description": "Too many requests - rate limit exceeded",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Rate limit exceeded. Please wait before making more requests."
                    }
                }
            },
        },
        500: {
            "description": "Internal server error",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "An internal error occurred. Please try again later."
                    }
                }
            },
        },
        503: {
            "description": "Service unavailable - LLM service error",
            "content": {
                "application/json": {
                    "example": {"detail": "Failed to generate response"}
                }
            },
        },
    },
)
def chat(
    request: ChatRequest,
    http_request: Request,
    # Optional query parameters
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
    # Dependencies
    api_key: str = Depends(check_api_key),
    chat_service: ChatService = Depends(get_chat_service),
):
    """This docstring is shown in the endpoint list. Full docs above."""
    # Track request start time
    start_time = time.time()
    endpoint = "/chat"

    try:
        # OPTIMIZATION: Check response cache FIRST before expensive operations
        # This allows cached responses to skip prompt guard checks entirely
        # Note: We don't have session context yet, so we check without session_id
        # The chat_service will do a more specific cache check with session_id
        from app.services.response_cache import get_response_cache

        response_cache = get_response_cache()

        # Build cache params from query parameters (use defaults from settings if not provided)
        cache_params = {
            "top_k": top_k if top_k is not None else settings.retrieval.top_k,
            "max_distance": max_distance
            if max_distance is not None
            else settings.retrieval.max_distance,
            "temperature": temperature
            if temperature is not None
            else settings.llm.temperature,
            "model": model
            if model is not None
            else (
                settings.llm.groq_model
                if settings.llm.provider == "groq"
                else settings.llm.ollama_model
            ),
            "doc_type": doc_type,
        }

        # Try cache without session_id first (for non-conversational queries)
        cached_response = response_cache.get(
            question=request.question, session_id=None, params=cache_params
        )

        if cached_response:
            logger.info(
                "Cache hit at route level - skipping prompt guard and RAG pipeline"
            )
            # Add session_id from request if provided
            if request.session_id:
                cached_response["session_id"] = request.session_id

            # Track success for cached response
            if METRICS_ENABLED:
                rag_request_total.labels(endpoint=endpoint, status="success").inc()
                rag_request_latency_seconds.labels(endpoint=endpoint).observe(
                    time.time() - start_time
                )

            return ChatResponse(**cached_response)

        # Cache miss - proceed with prompt guard check
        # Get session and conversation history for context-aware checking
        from app.storage import get_session_store

        session_store = get_session_store()
        client_ip = http_request.client.host if http_request.client else "unknown"
        session = session_store.get_or_create_session(request.session_id, client_ip)
        conversation_history = session.get_truncated_history() if session else []

        guard = get_prompt_guard()
        guard_result = guard.check_input(
            request.question, conversation_history=conversation_history
        )
        if guard_result["blocked"]:
            logger.warning(f"Prompt injection blocked: {guard_result['label']}")
            raise HTTPException(
                status_code=400,
                detail="Your request could not be processed. Please rephrase your question.",
            )

        # Pass skip_route_cache=True to avoid double-checking cache in handle_chat
        response = chat_service.handle_chat(
            request=request,
            grounded_only=grounded_only,
            null_threshold=null_threshold,
            max_distance=max_distance,
            top_k=top_k,
            temperature=temperature,
            max_tokens=max_tokens,
            rerank=rerank,
            rerank_lex_weight=rerank_lex_weight,
            doc_type=doc_type,
            term_id=term_id,
            level=level,
            model=model,
            skip_route_cache=True,
        )

        # Track success metrics
        if METRICS_ENABLED:
            rag_request_total.labels(endpoint=endpoint, status="success").inc()
            rag_request_latency_seconds.labels(endpoint=endpoint).observe(
                time.time() - start_time
            )

        return response

    except HTTPException as e:
        # Track HTTP errors
        if METRICS_ENABLED:
            rag_request_total.labels(endpoint=endpoint, status="error").inc()
            rag_errors_total.labels(
                component="api", error_type=f"http_{e.status_code}"
            ).inc()
        raise

    except RAGException:
        raise

    except Exception as e:
        # Track unexpected errors
        logger.error(f"Unexpected error in {endpoint}: {str(e)}")
        if METRICS_ENABLED:
            rag_request_total.labels(endpoint=endpoint, status="error").inc()
            rag_errors_total.labels(component="api", error_type="unexpected").inc()
        raise
