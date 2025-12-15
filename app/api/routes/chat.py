"""
Chat endpoint for Personal RAG system.
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException

from ...models import ChatRequest, ChatResponse, ChatSource
from ..dependencies import check_api_key, get_chat_service
from ...core import ChatService
from ...retrieval import search
from ...services.llm import generate_with_ollama
from ...services.prompt_guard import get_prompt_guard
from ...prompting import create_default_prompt_builder
from ...settings import settings
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/chat/simple", response_model=ChatResponse)
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
    """Simple RAG chat endpoint without routing or filtering.

    This endpoint provides a bare-bones RAG experience:
    - Direct semantic search without query routing
    - No metadata filtering or doc_type restrictions
    - No reranking or multi-domain logic
    - Simple prompt building and generation

    Perfect for testing, debugging, or when you want straightforward retrieval.

    Args:
        request: Chat request with question
        top_k: Number of chunks to retrieve (default: 10)
        max_distance: Maximum distance for retrieval (default: 0.6)
        temperature: LLM temperature (default: 0.1)
        max_tokens: Maximum tokens to generate (default: 500)
        api_key: API key (from dependency)

    Returns:
        ChatResponse with answer and sources
    """
    logger.info(f"Simple chat - Question: {request.question}")

    # Step 0: Check for prompt injection
    # Note: simple_chat doesn't support sessions, so no conversation history
    guard = get_prompt_guard()
    guard_result = guard.check_input(
        request.question,
        conversation_history=None  # Simple endpoint has no session support
    )
    if guard_result["blocked"]:
        logger.warning(f"Prompt injection blocked: {guard_result['label']}")
        raise HTTPException(
            status_code=400,
            detail="Your request could not be processed. Please rephrase your question."
        )

    # Step 1: Simple search - no filters, no routing
    chunks = search(
        query=request.question,
        k=top_k,
        max_distance=max_distance,
        metadata_filter=None  # No filtering!
    )

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

    prompt = prompt_builder.build_prompt(
        question=request.question,
        sources=formatted_sources,
        conversation_history=[]  # Simple endpoint doesn't support conversation history
    )

    # Step 3: Generate answer
    answer = generate_with_ollama(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

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

    return ChatResponse(
        answer=answer,
        sources=sources,
        grounded=grounded,
        metadata={
            "num_chunks": len(chunks),
            "top_k": top_k,
            "max_distance": max_distance,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "routing_enabled": False,
        }
    )


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
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
    """Answer a question using RAG with filtering and reranking.

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
        api_key: API key (from dependency)
        chat_service: Chat service (from dependency)

    Returns:
        ChatResponse with answer and metadata
    """
    # OPTIMIZATION: Check response cache FIRST before expensive operations
    # This allows cached responses to skip prompt guard checks entirely
    # Note: We don't have session context yet, so we check without session_id
    # The chat_service will do a more specific cache check with session_id
    from ...services.response_cache import get_response_cache

    response_cache = get_response_cache()

    # Build cache params from query parameters (use defaults from settings if not provided)
    cache_params = {
        "top_k": top_k if top_k is not None else settings.retrieval.top_k,
        "max_distance": max_distance if max_distance is not None else settings.retrieval.max_distance,
        "temperature": temperature if temperature is not None else settings.llm.temperature,
        "model": model if model is not None else (settings.llm.groq_model if settings.llm.provider == "groq" else settings.llm.ollama_model),
        "doc_type": doc_type,
    }

    # Try cache without session_id first (for non-conversational queries)
    cached_response = response_cache.get(
        question=request.question,
        session_id=None,
        params=cache_params
    )

    if cached_response:
        logger.info("Cache hit at route level - skipping prompt guard and RAG pipeline")
        # Add session_id from request if provided
        if request.session_id:
            cached_response["session_id"] = request.session_id
        return ChatResponse(**cached_response)

    # Cache miss - proceed with prompt guard check
    # Get session and conversation history for context-aware checking
    from ...storage import get_session_store

    session_store = get_session_store()
    session = session_store.get_or_create_session(request.session_id, request.client_ip)
    conversation_history = session.get_truncated_history() if session else []

    guard = get_prompt_guard()
    guard_result = guard.check_input(
        request.question,
        conversation_history=conversation_history
    )
    if guard_result["blocked"]:
        logger.warning(f"Prompt injection blocked: {guard_result['label']}")
        raise HTTPException(
            status_code=400,
            detail="Your request could not be processed. Please rephrase your question."
        )

    # Pass skip_route_cache=True to avoid double-checking cache in handle_chat
    return chat_service.handle_chat(
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
