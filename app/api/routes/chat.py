"""
Chat endpoint for Personal RAG system.
"""

from typing import Optional

from fastapi import APIRouter, Depends

from ...models import ChatRequest, ChatResponse
from ..dependencies import check_api_key, get_chat_service
from ...core import ChatService

router = APIRouter()


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
    use_router: bool = True,
    # Dependencies
    api_key: str = Depends(check_api_key),
    chat_service: ChatService = Depends(get_chat_service),
):
    """Answer a question using RAG.

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
