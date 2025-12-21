"""
Ingestion Processor - Main orchestration

Handles file processing, chunking, deduplication, and batch insertion.
Delegates to the modular IngestionPipeline.
"""

import logging
from typing import List, Optional

from app.ingest.pipeline import IngestionPipeline

logger = logging.getLogger(__name__)


def ingest_paths(paths: Optional[List[str]] = None, batch_size: int = None) -> int:
    """Ingest text files from the given paths.

    Args:
        paths: List of file or directory paths to ingest.
                Defaults to [settings.docs_dir] if None.
        batch_size: Number of documents to process in a batch.
                    If None, uses the value from configuration.

    Returns:
        Total number of chunks added
    """
    pipeline = IngestionPipeline(batch_size=batch_size)
    stats = pipeline.run(paths=paths)
    return stats.chunks_added
