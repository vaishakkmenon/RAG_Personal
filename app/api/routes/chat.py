"""
Chat endpoint for Personal RAG system.
"""

from typing import Optional, List, Dict, Any

from fastapi import APIRouter, Depends

from ...models import ChatRequest, ChatResponse, ChatSource
from ..dependencies import check_api_key, get_chat_service
from ...core import ChatService
from ...retrieval import search
from ...services.llm import generate_with_ollama
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
    use_router: Optional[bool] = None,  # None = use settings default
    # Dependencies
    api_key: str = Depends(check_api_key),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Answer a question using RAG with full routing and filtering.

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
        use_router: Whether to use query router
        api_key: API key (from dependency)
        chat_service: Chat service (from dependency)

    Returns:
        ChatResponse with answer and metadata
    """
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
        use_router=use_router,
    )
