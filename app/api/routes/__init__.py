"""
API routes for Personal RAG system.
"""

from app.api.routes import health, ingest, chat, debug, admin

__all__ = ["health", "ingest", "chat", "debug", "admin"]
