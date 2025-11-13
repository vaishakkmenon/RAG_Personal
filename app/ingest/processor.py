"""
Ingestion Processor - Main orchestration

Handles file processing, chunking, deduplication, and batch insertion.
"""

import hashlib
import logging
import os
from typing import Dict, List, Optional

from ..retrieval import add_documents
from ..settings import ingest_settings
from .chunking import chunk_text_with_section_metadata
from .discovery import find_files
from .metadata import extract_frontmatter, read_text

logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = ingest_settings.batch_size
MAX_FILE_SIZE = ingest_settings.max_file_size

# Optional: Import Prometheus metrics if available
try:
    from ..metrics import rag_ingested_chunks_total, rag_ingest_skipped_files_total

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.info("Prometheus metrics not available (metrics.py not found)")


def _record_metric(metric_func, value=1):
    """Record metric if metrics are enabled."""
    if METRICS_ENABLED:
        try:
            metric_func(value)
        except TypeError:
            metric_func()


def _process_file(fp: str) -> Optional[List[Dict]]:
    """Process a single file, return chunks or None if failed."""
    try:
        # Check size
        file_size = os.path.getsize(fp)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping {fp}: too large ({file_size} bytes)")
            _record_metric(
                lambda: rag_ingest_skipped_files_total.labels(reason="too_large").inc()
            )
            return None

        # Read file
        text = read_text(fp)
        metadata, body = extract_frontmatter(text)
        metadata["source"] = fp

        return chunk_text_with_section_metadata(
            body, ingest_settings.chunk_size, ingest_settings.chunk_overlap, metadata
        )

    except Exception as e:
        logger.warning(f"Failed to process {fp}: {e}")
        _record_metric(
            lambda: rag_ingest_skipped_files_total.labels(reason="error").inc()
        )
        return None


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
    if batch_size is None:
        batch_size = BATCH_SIZE

    from ..settings import settings

    base_paths = paths or [settings.docs_dir]
    files = find_files(base_paths)

    docs_batch: List[Dict] = []
    added_total = 0
    duplicates_total = 0
    seen_hashes: set[str] = set()

    logger.info(f"Starting ingestion of {len(files)} files")

    for fp in files:
        chunk_dicts = _process_file(fp)
        if not chunk_dicts:
            continue

        # Per-file counters
        file_added = 0
        file_dupes = 0

        # Add chunks with metadata
        for idx, chunk_dict in enumerate(chunk_dicts):
            # Normalize and hash for deduplication
            normalized = chunk_dict["text"].strip()
            if not normalized:
                continue

            chunk_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()

            if chunk_hash in seen_hashes:
                file_dupes += 1
                duplicates_total += 1
                logger.debug(
                    f"Skipped duplicate chunk {fp}:{idx} (hash={chunk_hash[:8]})"
                )
                continue

            seen_hashes.add(chunk_hash)

            # Create chunk record with section-enriched metadata
            docs_batch.append(
                {
                    "id": f"{fp}:{idx}",
                    "text": normalized,
                    "metadata": chunk_dict["metadata"],
                }
            )

            file_added += 1

            # Process files in batches
            if len(docs_batch) >= batch_size:
                add_documents(docs_batch)
                _record_metric(lambda: rag_ingested_chunks_total.inc(len(docs_batch)))
                added_total += len(docs_batch)
                logger.info(f"Added batch of {len(docs_batch)} chunks to ChromaDB")
                docs_batch.clear()

        logger.info(
            f"{os.path.basename(fp)}: added {file_added} chunks, skipped {file_dupes} duplicates"
        )

    # Flush any remaining chunks
    if docs_batch:
        add_documents(docs_batch)
        _record_metric(lambda: rag_ingested_chunks_total.inc(len(docs_batch)))
        added_total += len(docs_batch)
        logger.info(f"Added final batch of {len(docs_batch)} chunks to ChromaDB")

    logger.info(
        f"✅ Ingestion complete: added {added_total} chunks from {len(files)} files; "
        f"skipped {duplicates_total} duplicate chunks"
    )

    return added_total
