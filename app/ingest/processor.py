"""
Ingestion Processor - Main orchestration

Handles file processing, chunking, deduplication, and batch insertion.
"""

import hashlib
import logging
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from app.retrieval import add_documents
from app.settings import ingest_settings
from app.ingest.chunking import (
    chunk_by_headers,
    extract_doc_id,
)
from app.ingest.discovery import find_files
from app.ingest.metadata import (
    extract_frontmatter,
    read_text,
    generate_version_identifier,
)

logger = logging.getLogger(__name__)

# Configuration
BATCH_SIZE = ingest_settings.batch_size
MAX_FILE_SIZE = ingest_settings.max_file_size

# Optional: Import Prometheus metrics if available
try:
    from app.metrics import rag_ingested_chunks_total, rag_ingest_skipped_files_total

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.info("Prometheus metrics not available (metrics.py not found)")


def _generate_chunk_id(
    doc_id: str, version: str, section_slug: str, chunk_idx: int
) -> str:
    """Generate a chunk ID with format: {doc_id}@{version}#{section_slug}:{chunk_idx}"""
    return f"{doc_id}@{version}#{section_slug}:{chunk_idx}"


def _record_metric(metric_func, value=1):
    """Record metric if metrics are enabled."""
    if METRICS_ENABLED:
        try:
            metric_func(value)
        except TypeError:
            metric_func()


def _process_file(fp: str) -> List[Dict]:
    """Process a single file, return chunks or empty list if failed."""
    try:
        # Existing size check
        file_size = os.path.getsize(fp)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping {fp}: too large ({file_size} bytes)")
            _record_metric(
                lambda: rag_ingest_skipped_files_total.labels(reason="too_large").inc()
            )
            return []

        # Read file and extract metadata
        text = read_text(fp)
        metadata, body = extract_frontmatter(text)

        # Add source file info to metadata
        metadata["source"] = fp
        metadata["filename"] = os.path.basename(fp)

        # Extract doc_id and doc_type
        doc_id, doc_type = extract_doc_id(fp)
        metadata.update(
            {
                "doc_id": doc_id,
                "doc_type": doc_type,
                "ingestion_timestamp": datetime.now().isoformat(),
            }
        )

        # Generate version identifier with collision detection
        version = generate_version_identifier(metadata, doc_id)
        metadata["version_identifier"] = version

        # Determine split level based on doc_type
        # Use level 2 (##) for transcript files to ensure Academic Summary sections are chunked
        # This includes critical sections like "Degrees Earned" and "Overall Academic Performance"
        # as well as individual terms like "Fall Term 2022"
        split_level = 2 if doc_type == "transcript_analysis" else 2

        # NEW: Single-stage header-based chunking (replaces two-stage process)
        all_chunks = chunk_by_headers(
            text=body,
            base_metadata=metadata,
            source_path=fp,
            chunk_size=ingest_settings.chunk_size,
            overlap=ingest_settings.chunk_overlap,
            split_level=split_level,
        )

        return all_chunks

    except Exception as e:
        logger.error(f"Failed to process {fp}: {str(e)}", exc_info=True)
        _record_metric(
            lambda: rag_ingest_skipped_files_total.labels(reason="error").inc()
        )
        return []


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

    from app.settings import settings

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
        for chunk_dict in chunk_dicts:
            # Get normalized text and generate hash for deduplication
            normalized = chunk_dict["text"].strip()
            if not normalized:
                continue

            chunk_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            if chunk_hash in seen_hashes:
                file_dupes += 1
                duplicates_total += 1
                logger.debug(f"Skipped duplicate chunk (hash={chunk_hash[:8]})")
                continue

            seen_hashes.add(chunk_hash)

            # Use chunk ID from chunk_by_headers() (already includes version, section, and index)
            chunk_id = chunk_dict["id"]
            metadata = chunk_dict["metadata"]

            # Create chunk record with enriched metadata
            docs_batch.append(
                {"id": chunk_id, "text": normalized, "metadata": metadata}
            )

            file_added += 1

            # Process in batches
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
        f"[COMPLETION] Ingestion complete: added {added_total} chunks from {len(files)} files; "
        f"skipped {duplicates_total} duplicate chunks"
    )

    return added_total
