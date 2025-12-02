"""
API layer for Personal RAG system.

Contains FastAPI routes and HTTP-related functionality.
"""

from fastapi import APIRouter
from .routes import health, ingest, chat, debug, admin

def create_api_router() -> APIRouter:
    """Create and configure the main API router."""
    router = APIRouter()

    # Register all route modules
    router.include_router(health.router, tags=["health"])
    router.include_router(ingest.router, tags=["ingest"])
    router.include_router(chat.router, tags=["chat"])
    router.include_router(debug.router, prefix="/debug", tags=["debug"])
    router.include_router(admin.router, prefix="/admin", tags=["admin"])

    return router

__all__ = ["create_api_router"]
