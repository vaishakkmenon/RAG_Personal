"""
Chunker Module

Implements different strategies for splitting documents into retrieval-ready chunks.
"""

import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from app.ingest.loader import IngestDocument
from app.ingest.chunking import chunk_by_headers, chunk_by_terms
from app.settings import ingest_settings

logger = logging.getLogger(__name__)


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk(self, document: IngestDocument) -> List[Dict[str, Any]]:
        """Split document into chunks."""
        pass


class HeaderChunkingStrategy(ChunkingStrategy):
    """
    Standard markdown header-based chunking.
    Used for resumes, certificates, and general docs.
    """

    def __init__(
        self, chunk_size: int = None, overlap: int = None, split_level: int = 2
    ):
        self.chunk_size = chunk_size or ingest_settings.chunk_size
        self.overlap = overlap or ingest_settings.chunk_overlap
        self.split_level = split_level

    def chunk(self, document: IngestDocument) -> List[Dict[str, Any]]:
        return chunk_by_headers(
            text=document.text_content,
            base_metadata=document.metadata,
            source_path=document.source_path,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
            split_level=self.split_level,
        )


class TermChunkingStrategy(ChunkingStrategy):
    """
    Term-based chunking for academic transcripts.
    Preserves semantic boundaries of academic terms.
    """

    def chunk(self, document: IngestDocument) -> List[Dict[str, Any]]:
        return chunk_by_terms(
            text=document.text_content,
            base_metadata=document.metadata,
            source_path=document.source_path,
        )


class Chunker:
    """
    Factory and orchestrator for chunking documents.
    Routes to appropriate strategy based on document type.
    """

    def __init__(self):
        self.strategies = {
            "default": HeaderChunkingStrategy(),
            "transcript_analysis": TermChunkingStrategy(),
            # specialized strategies could be added here
        }

    def get_strategy(self, doc_type: str) -> ChunkingStrategy:
        """Select strategy based on document type."""
        if doc_type == "transcript_analysis":
            return self.strategies["transcript_analysis"]
        return self.strategies["default"]

    def process(self, document: IngestDocument) -> List[Dict[str, Any]]:
        """
        Chunk the given document using the appropriate strategy.

        Args:
            document: The loaded document to chunk

        Returns:
            List of chunk dictionaries ready for embedding
        """
        strategy = self.get_strategy(document.doc_type)
        logger.debug(
            f"Chunking {document.source_path} using {strategy.__class__.__name__}"
        )

        try:
            chunks = strategy.chunk(document)
            logger.debug(f"Generated {len(chunks)} chunks for {document.doc_id}")
            return chunks
        except Exception as e:
            logger.error(f"Failed to chunk {document.source_path}: {e}", exc_info=True)
            return []
