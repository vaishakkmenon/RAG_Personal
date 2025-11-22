"""
Chunking - Text splitting with section awareness

Handles splitting text into overlapping chunks while preserving section information.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional

from ..settings import ingest_settings
from .metadata import generate_version_identifier

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
    section_metadata: Optional[Dict] = None,
) -> List[Dict]:
    """Split text into chunks while extracting section information from markdown headers.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (uses config if None)
        overlap: Overlap between chunks (uses config if None)
        base_metadata: Base metadata to include in all chunks
        section_metadata: Section-specific metadata to include in chunks

    Returns:
        List of dicts with 'text' and 'metadata' keys
    """
    if chunk_size is None:
        chunk_size = ingest_settings.chunk_size
    if overlap is None:
        overlap = ingest_settings.chunk_overlap
    if base_metadata is None:
        base_metadata = {}
    if section_metadata is None:
        section_metadata = {}

    # Create a deep copy of base_metadata to avoid modifying the original
    metadata = base_metadata.copy()

    # Update with section metadata, preserving any existing values
    for key, value in section_metadata.items():
        if key not in metadata or not metadata[key]:
            metadata[key] = value

    if not text.strip():
        return []

    # Split into paragraphs
    paragraphs = re.split(r"\n\s*\n", text)

    chunks = []
    current_chunk = ""
    current_section = metadata.get("section")  # Get section from metadata if available
    section_stack = metadata.get("section_stack", [])  # Get section stack if available

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Check if this paragraph is a markdown header
        header_match = re.match(r"^(#{1,6})\s+(.+)$", para)

        if header_match:
            # Save current chunk BEFORE changing section
            if current_chunk.strip():
                chunk_metadata = metadata.copy()
                if current_section:
                    chunk_metadata["section"] = current_section
                if section_stack:
                    chunk_metadata["section_stack"] = section_stack

                chunks.append(
                    {"text": current_chunk.strip(), "metadata": chunk_metadata}
                )
                current_chunk = ""  # Start fresh

            # Update section for the new header
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
                chunk_metadata = metadata.copy()
                if current_section:
                    chunk_metadata["section"] = current_section
                if section_stack:
                    chunk_metadata["section_stack"] = section_stack

                chunks.append(
                    {"text": current_chunk.strip(), "metadata": chunk_metadata}
                )

            # If paragraph itself is too large, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                temp_chunk = ""

                for sent in sentences:
                    if len(temp_chunk) + len(sent) + 1 > chunk_size:
                        if temp_chunk.strip():
                            chunk_metadata = metadata.copy()
                            if current_section:
                                chunk_metadata["section"] = current_section
                            if section_stack:
                                chunk_metadata["section_stack"] = section_stack

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
        chunk_metadata = metadata.copy()
        if current_section:
            chunk_metadata["section"] = current_section
        if section_stack:
            chunk_metadata["section_stack"] = section_stack

        chunks.append({"text": current_chunk.strip(), "metadata": chunk_metadata})

    # Filter out tiny chunks (header-only chunks with no content)
    MIN_CHUNK_SIZE = 50
    filtered_chunks = []

    for chunk in chunks:
        text = chunk["text"]
        chunk_metadata = chunk["metadata"]

        # If chunk is too small and is just a markdown header
        if len(text) < MIN_CHUNK_SIZE:
            if re.match(r"^#{1,6}\s+.+$", text.strip()):
                logger.debug(f"Filtering out header-only chunk: {text[:50]}")
                continue

            # For small non-header chunks, try to merge with previous chunk
            if filtered_chunks:
                last_chunk = filtered_chunks[-1]
                last_chunk["text"] += " " + text
                # Update the last chunk's metadata if needed
                if (
                    "section" not in last_chunk["metadata"]
                    and "section" in chunk_metadata
                ):
                    last_chunk["metadata"]["section"] = chunk_metadata["section"]
                if (
                    "section_stack" not in last_chunk["metadata"]
                    and "section_stack" in chunk_metadata
                ):
                    last_chunk["metadata"]["section_stack"] = chunk_metadata[
                        "section_stack"
                    ]
                continue

        # Ensure metadata is included in the chunk
        chunk["metadata"] = chunk_metadata
        filtered_chunks.append(chunk)

    return filtered_chunks


# ============================================================================
# Helper Functions for Section-Based Chunking
# ============================================================================


def extract_doc_id(file_path: str) -> str:
    """
    Extract normalized document ID from filename.

    Examples:
        'resume--vaishak-menon--2025-09-23.md' → 'resume'
        'certificate--cka--2024-06-26.md' → 'certificate-cka'
        'term_2024_fall.md' → 'term-2024-fall'

    Args:
        file_path: Full path to document file

    Returns:
        Normalized document identifier
    """
    import os

    basename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(basename)[0]

    # Remove version dates (patterns: --YYYY-MM-DD or _YYYY_SEASON)
    # Example: "resume--vaishak-menon--2025-09-23" → "resume--vaishak-menon"
    doc_id = re.sub(r"--\d{4}-\d{2}-\d{2}$", "", name_without_ext)

    # For resume files, extract just "resume"
    if doc_id.startswith("resume--"):
        doc_id = "resume"
    elif doc_id.startswith("certificate--"):
        # Keep certificate type: "certificate--cka" → "certificate-cka"
        doc_id = doc_id.replace("--", "-", 1)
    elif doc_id.startswith("term_"):
        # "term_2024_fall" → "term-2024-fall"
        doc_id = doc_id.replace("_", "-")

    return doc_id


def slugify(text: str) -> str:
    """
    Convert section name to URL-friendly slug.

    Examples:
        'Work Experience' → 'work-experience'
        'Teaching Assistant > Responsibilities' → 'teaching-assistant-responsibilities'
        'CS 665 - Deep Learning' → 'cs-665-deep-learning'

    Args:
        text: Section name or title

    Returns:
        URL-friendly slug (lowercase, alphanumeric + hyphens)
    """
    # Lowercase
    text = text.lower()

    # Replace section separators
    text = text.replace(" > ", "-")

    # Replace spaces with hyphens
    text = text.replace(" ", "-")

    # Remove special characters (keep alphanumeric and hyphens)
    text = re.sub(r"[^a-z0-9-]", "", text)

    # Collapse multiple hyphens into one
    text = re.sub(r"-+", "-", text)

    # Remove leading/trailing hyphens
    return text.strip("-")


def extract_section_slug(section_doc_id: str) -> str:
    """
    Extract section slug from section_doc_id.

    Example:
        "resume@2025-09-23#work-experience" → "work-experience"

    Args:
        section_doc_id: Full section document ID

    Returns:
        Section slug portion after '#'
    """
    if "#" in section_doc_id:
        parts = section_doc_id.split("#")
        return parts[1] if len(parts) > 1 else "default"
    return "default"


# ============================================================================
# Section Document Splitting
# ============================================================================


def create_section_document(
    text: str,
    base_metadata: Dict[str, Any],
    section_name: str,
    section_stack: List[str],
    source_path: str,
) -> Dict[str, Any]:
    """
    Create a section document with enriched metadata.

    Args:
        text: Section text content
        base_metadata: Metadata from YAML frontmatter (inherited)
        section_name: Name of this section (e.g., "Work Experience")
        section_stack: Hierarchical section path (e.g., ["Work Experience", "Teaching Assistant"])
        source_path: Original file path

    Returns:
        Dict with 'text' and 'metadata' keys
    """
    # Extract doc_id from filename
    doc_id = extract_doc_id(source_path)

    # Generate version identifier with collision detection
    # This checks ChromaDB for existing versions and auto-increments if needed
    version = generate_version_identifier(base_metadata, doc_id)

    # Create section slug
    section_slug = slugify(section_name)

    # Build section document ID
    section_doc_id = f"{doc_id}@{version}#{section_slug}"

    # Create enriched metadata (inherit all parent metadata)
    section_metadata = {
        **base_metadata,
        "section_doc_id": section_doc_id,
        "section_name": section_name,
        "doc_id": doc_id,
        "version_identifier": version,
    }

    # Add structured section levels (section_l1, section_l2, section_l3, etc.)
    for i, section_title in enumerate(section_stack, start=1):
        section_metadata[f"section_l{i}"] = section_title

    # Also keep human-readable section path
    if section_stack:
        section_metadata["section"] = " > ".join(section_stack)

    return {"text": text.strip(), "metadata": section_metadata}


def split_into_section_documents(
    text: str,
    base_metadata: Dict[str, Any],
    source_path: str,
    split_level: int = 2,
) -> List[Dict[str, Any]]:
    """
    Split markdown text into section documents at specified header level.

    This function parses markdown headers and creates separate "virtual documents"
    for each section at the specified level. Each section inherits the parent
    metadata and adds section-specific metadata.

    Args:
        text: Markdown body text (after frontmatter extraction)
        base_metadata: Metadata from YAML frontmatter
        source_path: Original file path
        split_level: Header level to split at (1-6, default 2 for ##)

    Returns:
        List of section document dicts with 'text' and 'metadata' keys

    Example:
        Given markdown:
        ```
        # Work Experience
        ## Teaching Assistant
        Content about TA role...
        ## Infrastructure Intern
        Content about intern role...
        ```

        With split_level=2, creates 2 section documents:
        - One for "Teaching Assistant" section
        - One for "Infrastructure Intern" section
    """
    sections = []
    current_section_text = ""
    current_section_name = None
    current_section_stack = []  # Stack at the time this section started
    section_stack = []  # Global tracking of nested header hierarchy

    lines = text.split("\n")

    for line in lines:
        # Detect markdown headers (e.g., "## Teaching Assistant")
        header_match = re.match(r"^(#{1,6})\s+(.+)$", line)

        if header_match:
            level = len(header_match.group(1))  # Count number of #
            title = header_match.group(2).strip()

            # Update section stack for ALL headers
            if level == 1:
                section_stack = [title]
            elif level == 2:
                section_stack = section_stack[:1] + [title]
            elif level == 3:
                section_stack = section_stack[:2] + [title]
            elif level == 4:
                section_stack = section_stack[:3] + [title]
            elif level == 5:
                section_stack = section_stack[:4] + [title]
            elif level == 6:
                section_stack = section_stack[:5] + [title]

            # If we hit a split-level header
            if level == split_level:
                # Save previous section if it exists
                if current_section_text.strip() and current_section_name:
                    section_doc = create_section_document(
                        text=current_section_text,
                        base_metadata=base_metadata,
                        section_name=current_section_name,
                        section_stack=(
                            current_section_stack
                            if current_section_stack
                            else [current_section_name]
                        ),
                        source_path=source_path,
                    )
                    sections.append(section_doc)

                # Start new section and capture stack at this moment
                current_section_text = line + "\n"
                current_section_name = title
                current_section_stack = (
                    section_stack.copy()
                )  # Freeze the stack for this section

            else:
                # Non-split-level header - add to current section
                current_section_text += line + "\n"

        else:
            # Regular content line
            current_section_text += line + "\n"

    # Save final section
    if current_section_text.strip() and current_section_name:
        section_doc = create_section_document(
            text=current_section_text,
            base_metadata=base_metadata,
            section_name=current_section_name,
            section_stack=section_stack,
            source_path=source_path,
        )
        sections.append(section_doc)

    # If no sections were created (no split-level headers), return entire document as one section
    if not sections and text.strip():
        logger.warning(
            f"No level-{split_level} headers found in {source_path}, "
            f"treating entire document as one section"
        )
        sections.append(
            create_section_document(
                text=text,
                base_metadata=base_metadata,
                section_name="Complete Document",
                section_stack=["Complete Document"],
                source_path=source_path,
            )
        )

    logger.debug(
        f"Split {source_path} into {len(sections)} section documents "
        f"at level {split_level}"
    )

    return sections
