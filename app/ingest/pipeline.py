"""
Ingestion Pipeline

Orchestrates the Flow: Loader -> Chunker -> Vector Store
"""

import logging
import hashlib
from typing import List, Optional, Set
from dataclasses import dataclass

from app.ingest.loader import Loader
from app.ingest.chunker import Chunker
from app.settings import ingest_settings

logger = logging.getLogger(__name__)

# Metrics (Optional import)
try:
    from app.metrics import rag_ingested_chunks_total, rag_ingest_skipped_files_total

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False


@dataclass
class IngestionStats:
    files_processed: int = 0
    files_skipped: int = 0
    chunks_generated: int = 0
    chunks_added: int = 0
    chunks_duplicate: int = 0


class IngestionPipeline:
    """
    Orchestrates the document ingestion process.
    """

    def __init__(self, batch_size: int = None):
        self.batch_size = batch_size or ingest_settings.batch_size
        self.loader = Loader()
        self.chunker = Chunker()
        self.stats = IngestionStats()
        self.seen_hashes: Set[str] = set()

    def run(self, paths: Optional[List[str]] = None) -> IngestionStats:
        """
        Run the ingestion pipeline on the given paths.

        Args:
            paths: List of directories/files to ingest. Defaults to configured docs_dir.

        Returns:
            IngestionStats object with results
        """
        # Override loader paths if provided
        if paths:
            self.loader.base_dirs = paths

        files = self.loader.discover()
        logger.info(f"Starting ingestion pipeline for {len(files)} files")

        docs_batch: List[dict] = []

        for file_path in files:
            self._process_file_path(file_path, docs_batch)

        # Flush remaining
        if docs_batch:
            self._flush_batch(docs_batch)

        self._log_completion()
        return self.stats

    def _process_file_path(self, file_path: str, batch: List[dict]):
        """Process a single file path."""
        doc = self.loader.load_file(file_path)

        if not doc:
            self.stats.files_skipped += 1
            if METRICS_ENABLED:
                rag_ingest_skipped_files_total.labels(reason="load_error").inc()
            return

        self.stats.files_processed += 1
        chunks = self.chunker.process(doc)
        self.stats.chunks_generated += len(chunks)

        for chunk in chunks:
            if self._is_duplicate(chunk):
                self.stats.chunks_duplicate += 1
                continue

            batch.append(chunk)

            if len(batch) >= self.batch_size:
                self._flush_batch(batch)

    def _is_duplicate(self, chunk: dict) -> bool:
        """Check if chunk is a duplicate based on content hash."""
        text = chunk.get("text", "").strip()
        if not text:
            return True

        chunk_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

        if chunk_hash in self.seen_hashes:
            return True

        self.seen_hashes.add(chunk_hash)
        return False

    def _flush_batch(self, batch: List[dict]):
        """Write batch to Vector Store."""
        if not batch:
            return

        try:
            from app.retrieval.vector_store import get_vector_store

            get_vector_store().add_documents(batch)

            count = len(batch)
            self.stats.chunks_added += count
            if METRICS_ENABLED:
                rag_ingested_chunks_total.inc(count)

            logger.info(f"Flushed batch of {count} chunks")
            batch.clear()
        except Exception as e:
            logger.error(f"Failed to flush batch: {e}")
            # We don't verify individual failures in batch here, simplified for now

    def _log_completion(self):
        logger.info(
            f"Ingestion complete. "
            f"Files: {self.stats.files_processed} processed, {self.stats.files_skipped} skipped. "
            f"Chunks: {self.stats.chunks_added} added, {self.stats.chunks_duplicate} duplicates."
        )
