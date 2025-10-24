"""
Document ingestion module for Personal RAG system.

Handles:
- Reading markdown/text files
- Extracting YAML front-matter metadata
- Chunking with paragraph/sentence awareness
- Deduplication via SHA-256
- Batch insertion to ChromaDB
"""

import os
import re
import hashlib
import logging
from typing import List, Dict, Optional, Tuple
from datetime import date, datetime

import yaml
from fastapi import HTTPException

from .retrieval import add_documents
from .settings import settings

logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXT = {".txt", ".md"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB max per file
BATCH_SIZE = 500

# Optional: Import Prometheus metrics if available
try:
    from .metrics import rag_ingested_chunks_total, rag_ingest_skipped_files_total
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.info("Prometheus metrics not available (metrics.py not found)")


def extract_frontmatter(text: str) -> Tuple[Dict, str]:
    """
    Extract YAML front-matter from Markdown and normalize for ChromaDB.
    
    ChromaDB metadata requirements:
    - Values must be: str, int, float, or bool (NOT None!)
    - No date objects, no nested dicts, no None values
    
    Args:
        text: Full file content
    
    Returns:
        Tuple of (metadata_dict, body_text)
        If no front-matter found, returns ({}, original_text)
    """
    if not text.startswith('---'):
        return {}, text
    
    parts = text.split('---', 2)
    if len(parts) < 3:
        logger.warning("Malformed YAML front-matter (missing closing ---)")
        return {}, text
    
    try:
        metadata = yaml.safe_load(parts[1])
        if metadata is None:
            metadata = {}
        
        # Normalize metadata for ChromaDB compatibility
        normalized_metadata = {}
        for key, value in metadata.items():
            # Skip None values entirely (ChromaDB doesn't like them)
            if value is None:
                continue
            elif isinstance(value, bool):
                normalized_metadata[key] = value
            elif isinstance(value, (int, float)):
                normalized_metadata[key] = value
            elif isinstance(value, str):
                normalized_metadata[key] = value
            elif isinstance(value, (date, datetime)):
                # Convert dates to ISO format strings
                normalized_metadata[key] = value.isoformat()
            elif isinstance(value, list):
                # Convert list to comma-separated string
                if value:  # Only if list is not empty
                    normalized_metadata[key] = ", ".join(str(item) for item in value)
            elif isinstance(value, dict):
                # Skip nested dicts (not supported by ChromaDB)
                logger.warning(f"Skipping nested dict metadata for key '{key}'")
                continue
            else:
                # Convert anything else to string (but skip if it's falsy)
                str_value = str(value)
                if str_value:
                    normalized_metadata[key] = str_value
        
        body = parts[2].strip()
        
        logger.debug(f"Extracted metadata keys: {list(normalized_metadata.keys())}")
        return normalized_metadata, body
    
    except yaml.YAMLError as e:
        logger.warning(f"Failed to parse YAML front-matter: {e}")
        return {}, text


def read_text(path: str) -> str:
    """
    Read and return the full text content of a file using UTF-8 encoding.
    
    Args:
        path: File path to read
    
    Returns:
        File contents as a single string
    
    Raises:
        Exception: If file cannot be read
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def find_files(base_paths: List[str]) -> List[str]:
    """
    Recursively find all allowed text files (.md, .txt) in the given paths.
    Accepts both files and directories.
    
    Args:
        base_paths: List of file or directory paths to search
    
    Returns:
        List of discovered file paths
    
    Raises:
        HTTPException: If no files are found to ingest
    """
    files = []
    base_docs_dir = os.path.abspath(settings.docs_dir)
    
    logger.info(f"Searching for files in: {base_paths}")
    logger.info(f"Base docs directory: {base_docs_dir}")

    for base in base_paths:
        abs_base = os.path.abspath(base)

        # Security: Only allow files within docs_dir
        if not abs_base.startswith(base_docs_dir):
            logger.warning(f"Skipping {base}: outside docs_dir {base_docs_dir}")
            if METRICS_ENABLED:
                rag_ingest_skipped_files_total.labels(reason="outside_docs_dir").inc()
            continue

        if os.path.isfile(abs_base):
            ext = os.path.splitext(abs_base)[1].lower()
            if ext in ALLOWED_EXT:
                files.append(abs_base)
                logger.debug(f"Found file: {abs_base}")
            else:
                logger.warning(f"Skipping {abs_base}: invalid extension {ext}")
                if METRICS_ENABLED:
                    rag_ingest_skipped_files_total.labels(reason="invalid_ext").inc()
        
        elif os.path.isdir(abs_base):
            for root, _, filenames in os.walk(abs_base):
                for name in filenames:
                    fp = os.path.join(root, name)
                    ext = os.path.splitext(name)[1].lower()
                    abs_fp = os.path.abspath(fp)
                    
                    if abs_fp.startswith(base_docs_dir) and ext in ALLOWED_EXT:
                        files.append(abs_fp)
                        logger.debug(f"Found file: {abs_fp}")
                    else:
                        logger.debug(f"Skipping {fp}: invalid or outside docs_dir")
        else:
            logger.warning(f"Path does not exist: {abs_base}")

    if not files:
        raise HTTPException(
            status_code=400,
            detail=f"No valid .txt or .md files found in {base_paths}"
        )
    
    logger.info(f"Found {len(files)} files to process")
    return files


def chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Split text into overlapping chunks, preferring natural boundaries (paragraphs/sentences).
    
    Strategy:
    1. Split by paragraphs (\\n\\n)
    2. If paragraph > chunk_size, split by sentences
    3. Pack into chunks with overlap
    
    Args:
        text: Raw text to chunk
        chunk_size: Target maximum size of each chunk (in characters)
        overlap: Number of characters to overlap between consecutive chunks
    
    Returns:
        List of non-empty text chunks
    """
    if not text.strip():
        return []

    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph exceeds chunk_size
        if len(current_chunk) + len(para) + 2 > chunk_size:  # +2 for \n\n
            # Save current chunk if not empty
            if current_chunk:
                chunks.append(current_chunk.strip())
            
            # If paragraph itself is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                
                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 > chunk_size:
                        if temp_chunk:
                            chunks.append(temp_chunk.strip())
                        temp_chunk = sent + " "
                    else:
                        temp_chunk += sent + " "
                
                # Save last sentence chunk with overlap for continuity
                if temp_chunk.strip():
                    current_chunk = temp_chunk[-overlap:] if len(temp_chunk) > overlap else temp_chunk
                else:
                    current_chunk = ""
            else:
                # Start new chunk with this paragraph
                current_chunk = para + "\n\n"
        else:
            # Add paragraph to current chunk
            current_chunk += para + "\n\n"
    
    # Don't forget last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks


def ingest_paths(paths: Optional[List[str]] = None) -> int:
    """
    Ingest text files from the given paths, chunking them and adding to the retrieval database.
    
    Process:
    1. Find all .md/.txt files in paths
    2. Extract YAML front-matter from each file
    3. Chunk the body text
    4. Deduplicate chunks via SHA-256
    5. Batch insert to ChromaDB with full metadata
    
    Args:
        paths: List of file or directory paths to ingest.
                Defaults to [settings.docs_dir] if None.
    
    Returns:
        Total number of text chunks successfully ingested
    
    Raises:
        HTTPException: If no valid files found
    """
    base_paths = paths or [settings.docs_dir]
    files = find_files(base_paths)

    docs_batch: List[Dict] = []
    added_total = 0
    duplicates_total = 0
    seen_hashes: set[str] = set()

    logger.info(f"Starting ingestion of {len(files)} files")

    for fp in files:
        # Check file size
        try:
            file_size = os.path.getsize(fp)
        except Exception as e:
            logger.warning(f"Could not stat file {fp}: {e}")
            continue

        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping {fp}: file too large ({file_size} bytes)")
            if METRICS_ENABLED:
                rag_ingest_skipped_files_total.labels(reason="too_large").inc()
            continue

        # Read file
        try:
            text = read_text(fp)
        except Exception as e:
            logger.warning(f"Could not read {fp}: {e}")
            if METRICS_ENABLED:
                rag_ingest_skipped_files_total.labels(reason="read_error").inc()
            continue

        # Extract YAML front-matter
        metadata, body = extract_frontmatter(text)
        
        # Ensure source is always in metadata
        metadata["source"] = fp

        # Chunk the body (not the YAML header)
        chunks = chunk_text(body, settings.chunk_size, settings.chunk_overlap)
        
        if not chunks:
            logger.info(f"{fp}: produced no valid chunks (empty or whitespace)")
            continue

        # Per-file counters
        file_added = 0
        file_dupes = 0

        # Add chunks with metadata
        for idx, text_chunk in enumerate(chunks):
            # Normalize and hash for deduplication
            normalized = text_chunk.strip()
            if not normalized:
                continue
            
            chunk_hash = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
            
            if chunk_hash in seen_hashes:
                file_dupes += 1
                duplicates_total += 1
                logger.debug(f"Skipped duplicate chunk {fp}:{idx} (hash={chunk_hash[:8]})")
                continue

            seen_hashes.add(chunk_hash)
            
            # Create chunk record with full metadata
            docs_batch.append({
                "id": f"{fp}:{idx}",
                "text": normalized,
                "metadata": metadata.copy()  # Each chunk gets full file metadata
            })
            
            file_added += 1

            # Flush batch to ChromaDB
            if len(docs_batch) >= BATCH_SIZE:
                add_documents(docs_batch)
                if METRICS_ENABLED:
                    rag_ingested_chunks_total.inc(len(docs_batch))
                added_total += len(docs_batch)
                logger.info(f"Added batch of {len(docs_batch)} chunks to ChromaDB")
                docs_batch.clear()

        logger.info(f"{os.path.basename(fp)}: added {file_added} chunks, skipped {file_dupes} duplicates")

    # Flush any remaining chunks
    if docs_batch:
        add_documents(docs_batch)
        if METRICS_ENABLED:
            rag_ingested_chunks_total.inc(len(docs_batch))
        added_total += len(docs_batch)
        logger.info(f"Added final batch of {len(docs_batch)} chunks to ChromaDB")

    logger.info(
        f"✅ Ingestion complete: added {added_total} chunks from {len(files)} files; "
        f"skipped {duplicates_total} duplicate chunks"
    )
    
    return added_total