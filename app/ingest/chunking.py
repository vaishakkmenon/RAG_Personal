"""
Chunking - Text splitting with section awareness

Handles splitting text into overlapping chunks while preserving section information.
"""

import logging
import re
from typing import Dict, List, Optional

from ..settings import ingest_settings

logger = logging.getLogger(__name__)


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    """Split text into overlapping chunks, preferring natural boundaries.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (uses config if None)
        overlap: Overlap between chunks (uses config if None)

    Returns:
        List of text chunks
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
                sentences = re.split(r"(?<=[.!?])\s+", para)
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
                    current_chunk = (
                        temp_chunk[-overlap:]
                        if len(temp_chunk) > overlap
                        else temp_chunk
                    )
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
    base_metadata: Optional[Dict] = None,
) -> List[Dict]:
    """Split text into chunks while extracting section information from markdown headers.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (uses config if None)
        overlap: Overlap between chunks (uses config if None)
        base_metadata: Base metadata to include in all chunks

    Returns:
        List of dicts with 'text' and 'metadata' keys
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
        header_match = re.match(r"^(#{1,6})\s+(.+)$", para)

        if header_match:
            # Save current chunk BEFORE changing section
            if current_chunk.strip():
                chunk_metadata = base_metadata.copy()
                if current_section:
                    chunk_metadata["section"] = current_section

                chunks.append({"text": current_chunk.strip(), "metadata": chunk_metadata})
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
                section_stack = section_stack[: level - 1] + [title]

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

                chunks.append({"text": current_chunk.strip(), "metadata": chunk_metadata})

            # If paragraph itself is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                temp_chunk = ""

                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 > chunk_size:
                        if temp_chunk.strip():
                            chunk_metadata = base_metadata.copy()
                            if current_section:
                                chunk_metadata["section"] = current_section

                            chunks.append(
                                {"text": temp_chunk.strip(), "metadata": chunk_metadata}
                            )
                        temp_chunk = sent + " "
                    else:
                        temp_chunk += sent + " "

                # Overlap for continuity
                if temp_chunk.strip():
                    current_chunk = (
                        temp_chunk[-overlap:]
                        if len(temp_chunk) > overlap
                        else temp_chunk
                    )
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

        chunks.append({"text": current_chunk.strip(), "metadata": chunk_metadata})

    # Filter out tiny chunks (header-only chunks with no content)
    # Keep chunks that are at least 50 characters (prevents "# Header" chunks)
    MIN_CHUNK_SIZE = 50
    filtered_chunks = []

    for chunk in chunks:
        text = chunk["text"]

        # If chunk is too small and is just a header, try to merge with next chunk
        if len(text) < MIN_CHUNK_SIZE:
            # Check if it's just a markdown header
            if re.match(r"^#{1,6}\s+.+$", text.strip()):
                # Skip this header-only chunk - it will be included in the next chunk
                # via the section metadata already
                logger.debug(f"Filtering out header-only chunk: {text[:50]}")
                continue

        filtered_chunks.append(chunk)

    return filtered_chunks
