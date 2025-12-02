"""
Health check endpoint for Personal RAG system.
"""

import socket

from fastapi import APIRouter

from ...settings import settings

router = APIRouter()


@router.get("/health")
async def health():
    """Health check endpoint.

    Returns basic system information including provider, model, host, and hostname.
    """
    # Get model based on provider
    if settings.llm.provider == "ollama":
        model = settings.llm.ollama_model
    elif settings.llm.provider == "groq":
        model = settings.llm.groq_model
    else:
        model = "unknown"

    return {
        "status": "ok",
        "provider": settings.llm.provider,
        "model": model,
        "socket": socket.gethostname(),
    }
