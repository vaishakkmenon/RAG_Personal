"""
API routes for Personal RAG system.
"""

from . import health, ingest, chat, debug, admin

__all__ = ["health", "ingest", "chat", "debug", "admin"]
