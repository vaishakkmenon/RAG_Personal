"""
API layer for Personal RAG system.

Contains FastAPI routes and HTTP-related functionality.
"""

from fastapi import APIRouter
from app.api.routes import chat, ingest, health, admin, debug, feedback


def create_api_router() -> APIRouter:
    """Create the main API router with all sub-routers included."""
    api_router = APIRouter()

    api_router.include_router(health.router, tags=["Health"])
    api_router.include_router(chat.router, tags=["Chat"])
    api_router.include_router(ingest.router, tags=["Ingest"])
    api_router.include_router(feedback.router, tags=["Feedback"])
    api_router.include_router(admin.router, prefix="/admin", tags=["Admin"])
    api_router.include_router(debug.router, prefix="/debug", tags=["Debug"])

    return api_router


__all__ = ["create_api_router"]
