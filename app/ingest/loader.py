"""
Document Loader

Responsible for discovering, reading, and extracting metadata from source files.
Encapsulates file system interactions and raw data preparation.
"""

import os
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

from app.settings import settings, ingest_settings
from app.ingest.discovery import find_files
from app.ingest.chunking import extract_doc_id
from app.ingest.metadata import (
    extract_frontmatter,
    read_text,
    generate_version_identifier,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestDocument:
    """Represents a raw document loaded from disk."""

    source_path: str
    text_content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_id: str = ""
    doc_type: str = "unknown"
    content_hash: str = ""


class Loader:
    """
    Handles discovery and loading of documents from the filesystem.
    """

    def __init__(self, base_dirs: Optional[List[str]] = None):
        """
        Initialize Loader.

        Args:
            base_dirs: List of directories to search. Defaults to settings.docs_dir.
        """
        self.base_dirs = base_dirs or [settings.docs_dir]
        self.max_file_size = ingest_settings.max_file_size

    def discover(self) -> List[str]:
        """Find all valid files in base directories."""
        return find_files(self.base_dirs)

    def load_file(self, file_path: str) -> Optional[IngestDocument]:
        """
        Load a single file, extracting text and metadata.

        Args:
            file_path: Absolute path to the file

        Returns:
            IngestDocument if successful, None if skipped/failed
        """
        try:
            # Size check
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"Skipping {file_path}: too large ({file_size} bytes)")
                return None

            # Read content
            raw_text = read_text(file_path)

            # Extract frontmatter
            metadata, body = extract_frontmatter(raw_text)

            # Basic metadata
            metadata["source"] = file_path
            metadata["filename"] = os.path.basename(file_path)

            # Extract identifiers
            doc_id, inferred_doc_type = extract_doc_id(file_path)

            # Use explicit doc_type from frontmatter if available, otherwise use inferred
            doc_type = metadata.get("doc_type") or inferred_doc_type

            # Calculate hash
            content_hash = hashlib.sha256(body.encode("utf-8")).hexdigest()

            # Update metadata
            metadata.update(
                {
                    "doc_id": doc_id,
                    "doc_type": doc_type,
                    "content_hash": content_hash,
                    "ingestion_timestamp": datetime.now().isoformat(),
                }
            )

            # Versioning
            # Note: This calls ChromaDB to check existing versions
            version = generate_version_identifier(metadata, doc_id, content_hash)
            metadata["version_identifier"] = version

            return IngestDocument(
                source_path=file_path,
                text_content=body,
                metadata=metadata,
                doc_id=doc_id,
                doc_type=doc_type,
                content_hash=content_hash,
            )

        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}", exc_info=True)
            return None

    def load_all(self) -> List[IngestDocument]:
        """Discover and load all documents."""
        files = self.discover()
        documents = []
        for fp in files:
            doc = self.load_file(fp)
            if doc:
                documents.append(doc)
        return documents
