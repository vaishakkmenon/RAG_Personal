"""
Debug endpoints for Personal RAG system.
"""

from typing import Optional

from fastapi import APIRouter

from ...retrieval import search, get_sample_chunks
from ...settings import settings

router = APIRouter()


@router.get("/search")
async def debug_search(
    q: str, k: Optional[int] = None, max_distance: Optional[float] = None
):
    """Debug endpoint for testing search functionality.

    Args:
        q: Search query
        k: Number of results to return (defaults to settings.retrieval.top_k)
        max_distance: Maximum cosine distance (defaults to settings.retrieval.max_distance)

    Returns:
        Search results
    """
    return search(
        q,
        k=k if k is not None else settings.retrieval.top_k,
        max_distance=(
            max_distance
            if max_distance is not None
            else settings.retrieval.max_distance
        ),
    )


@router.get("/samples")
async def debug_samples(n: int = 4):
    """Debug endpoint to retrieve sample chunks from the vector store.

    Args:
        n: Number of samples to return (capped at 20)

    Returns:
        Sample chunks
    """
    return get_sample_chunks(min(n, 20))  # Cap at 20 samples for safety
