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
from .settings import settings, ingest_settings

logger = logging.getLogger(__name__)

# Configuration - using settings from ingest_settings
ALLOWED_EXT = ingest_settings.allowed_extensions
MAX_FILE_SIZE = ingest_settings.max_file_size
BATCH_SIZE = ingest_settings.batch_size

# Optional: Import Prometheus metrics if available
try:
    from .metrics import rag_ingested_chunks_total, rag_ingest_skipped_files_total
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.info("Prometheus metrics not available (metrics.py not found)")
    
# ------------------------------
# Helper functions
# ------------------------------
def _record_metric(metric_func, value=1):
    """Record metric if metrics are enabled."""
    if METRICS_ENABLED:
        metric_func(value)
        
def _process_file(fp: str) -> Optional[List[Dict]]:
    """Process a single file, return chunks or None if failed."""
    try:
        # Check size
        file_size = os.path.getsize(fp)
        if file_size > MAX_FILE_SIZE:
            logger.warning(f"Skipping {fp}: too large ({file_size} bytes)")
            _record_metric(lambda: rag_ingest_skipped_files_total.labels(reason="too_large").inc())
            return None
        
        # Read file
        text = read_text(fp)
        metadata, body = extract_frontmatter(text)
        metadata["source"] = fp
        
        return chunk_text_with_section_metadata(body, ingest_settings.chunk_size, ingest_settings.chunk_overlap, metadata)
        
    except Exception as e:
        logger.warning(f"Failed to process {fp}: {e}")
        _record_metric(lambda: rag_ingest_skipped_files_total.labels(reason="error").inc())
        return None

# ------------------------------
# Core functionality
# ------------------------------
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
            _record_metric(lambda: rag_ingest_skipped_files_total.labels(reason="outside_docs_dir").inc())
            continue

        if os.path.isfile(abs_base):
            ext = os.path.splitext(abs_base)[1].lower()
            if ext in ALLOWED_EXT:
                files.append(abs_base)
                logger.debug(f"Found file: {abs_base}")
            else:
                logger.warning(f"Skipping {abs_base}: invalid extension {ext}")
                _record_metric(lambda: rag_ingest_skipped_files_total.labels(reason="invalid_ext").inc())
        
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


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """
    Split text into overlapping chunks, preferring natural boundaries (paragraphs/sentences).
    
    If chunk_size or overlap are not provided, they will be taken from the configuration.
    """
    if chunk_size is None:
        chunk_size = ingest_settings.chunk_size
    if overlap is None:
        overlap = ingest_settings.chunk_overlap
    
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

def chunk_text_with_section_metadata(
    text: str, 
    chunk_size: int = None, 
    overlap: int = None,
    base_metadata: Optional[Dict] = None
):
    """
    Split text into chunks while extracting section information from markdown headers.
    
    If chunk_size or overlap are not provided, they will be taken from the configuration.
    """
    if chunk_size is None:
        chunk_size = ingest_settings.chunk_size
    if overlap is None:
        overlap = ingest_settings.chunk_overlap
    if base_metadata is None:
        base_metadata = {}
    
    if not text.strip():
        return []
    
    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    
    chunks = []
    current_chunk = ""
    current_section = None  # Track current section
    section_stack = []  # Stack for nested headers
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Check if this paragraph is a markdown header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', para)
        
        if header_match:
            # Save current chunk BEFORE changing section
            if current_chunk.strip():
                chunk_metadata = base_metadata.copy()
                if current_section:
                    chunk_metadata["section"] = current_section
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
                current_chunk = ""  # Start fresh
            
            # NOW update section for the new header
            level = len(header_match.group(1))  # Number of #
            title = header_match.group(2).strip()
            
            # Update section stack based on header level
            if level == 1:
                section_stack = [title]
            elif level == 2:
                section_stack = section_stack[:1] + [title]
            elif level == 3:
                section_stack = section_stack[:2] + [title]
            else:
                section_stack = section_stack[:level-1] + [title]
            
            current_section = " > ".join(section_stack)
            
            # Add header to new chunk with correct section
            current_chunk = para + "\n\n"
            continue
        
        # Add paragraph to current chunk
        if len(current_chunk) + len(para) + 2 > chunk_size:
            # Save current chunk if not empty
            if current_chunk.strip():
                chunk_metadata = base_metadata.copy()
                if current_section:
                    chunk_metadata["section"] = current_section
                
                chunks.append({
                    "text": current_chunk.strip(),
                    "metadata": chunk_metadata
                })
            
            # If paragraph itself is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                temp_chunk = ""
                
                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 > chunk_size:
                        if temp_chunk.strip():
                            chunk_metadata = base_metadata.copy()
                            if current_section:
                                chunk_metadata["section"] = current_section
                            
                            chunks.append({
                                "text": temp_chunk.strip(),
                                "metadata": chunk_metadata
                            })
                        temp_chunk = sent + " "
                    else:
                        temp_chunk += sent + " "
                
                # Overlap for continuity
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
        chunk_metadata = base_metadata.copy()
        if current_section:
            chunk_metadata["section"] = current_section
        
        chunks.append({
            "text": current_chunk.strip(),
            "metadata": chunk_metadata
        })
    
    return chunks


def ingest_paths(paths: Optional[List[str]] = None, batch_size: int = None) -> int:
    """
    Ingest text files from the given paths, chunking them and adding to the retrieval database.
    
    Args:
        paths: List of file or directory paths to ingest.
              Defaults to [settings.docs_dir] if None.
        batch_size: Number of documents to process in a batch.
                   If None, uses the value from configuration.
    """
    if batch_size is None:
        batch_size = ingest_settings.batch_size
    
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
                logger.debug(f"Skipped duplicate chunk {fp}:{idx} (hash={chunk_hash[:8]})")
                continue

            seen_hashes.add(chunk_hash)
            
            # Create chunk record with section-enriched metadata
            docs_batch.append({
                "id": f"{fp}:{idx}",
                "text": normalized,
                "metadata": chunk_dict["metadata"]  # Now includes section info!
            })
            
            file_added += 1

            # Process files in batches
            if len(docs_batch) >= batch_size:
                add_documents(docs_batch)
                _record_metric(lambda: rag_ingested_chunks_total.inc(len(docs_batch)))
                added_total += len(docs_batch)
                logger.info(f"Added batch of {len(docs_batch)} chunks to ChromaDB")
                docs_batch.clear()

        logger.info(f"{os.path.basename(fp)}: added {file_added} chunks, skipped {file_dupes} duplicates")

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