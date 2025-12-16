"""
Ingest Package - Document ingestion and chunking

Handles file discovery, metadata extraction, chunking, and batch insertion to vector store.
"""

from app.ingest.processor import ingest_paths

__all__ = ["ingest_paths"]
