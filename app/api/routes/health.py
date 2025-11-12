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

    Returns basic system information including model, host, and hostname.
    """
    return {
        "status": "ok",
        "model": settings.ollama_model,
        "ollama_host": settings.ollama_host,
        "socket": socket.gethostname(),
    }
