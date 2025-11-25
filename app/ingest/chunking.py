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
    # Initialize section_stack as list for internal use (will be converted to string on output)
    section_stack_str = metadata.get("section_stack", "")
    section_stack = section_stack_str.split(" > ") if section_stack_str else []

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
                # Convert section_stack to string for ChromaDB compatibility
                if section_stack:
                    chunk_metadata["section_stack"] = " > ".join(section_stack)

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
                # Convert section_stack to string for ChromaDB compatibility
                if section_stack:
                    chunk_metadata["section_stack"] = " > ".join(section_stack)

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
                            # Convert section_stack to string for ChromaDB compatibility
                            if section_stack:
                                chunk_metadata["section_stack"] = " > ".join(section_stack)

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
        # Convert section_stack to string for ChromaDB compatibility
        if section_stack:
            chunk_metadata["section_stack"] = " > ".join(section_stack)

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


def extract_doc_id(file_path: str) -> tuple[str, str]:
    """
    Extract normalized document ID and doc_type from filename.

    Examples:
        'resume--vaishak-menon--2025-09-23.md' → ('resume', 'resume')
        'certificate--cka--2024-06-26.md' → ('certificate-cka', 'certificate')
        'term_2024_fall.md' → ('term-2024-fall', 'term')
        'complete_transcript_analysis.md' → ('transcript-analysis', 'transcript_analysis')

    Args:
        file_path: Full path to document file

    Returns:
        Tuple of (doc_id, doc_type)
    """
    import os

    basename = os.path.basename(file_path)
    name_without_ext = os.path.splitext(basename)[0]

    # Remove version dates (patterns: --YYYY-MM-DD or _YYYY_SEASON)
    # Example: "resume--vaishak-menon--2025-09-23" → "resume--vaishak-menon"
    doc_id = re.sub(r"--\d{4}-\d{2}-\d{2}$", "", name_without_ext)

    # Determine doc_type and normalize doc_id
    if doc_id.startswith("resume--") or doc_id == "resume":
        doc_type = "resume"
        doc_id = "resume"
    elif doc_id.startswith("certificate--"):
        doc_type = "certificate"
        # Keep certificate type: "certificate--cka" → "certificate-cka"
        doc_id = doc_id.replace("--", "-", 1)
    elif doc_id.startswith("term_") or doc_id.startswith("term-"):
        doc_type = "term"
        # "term_2024_fall" → "term-2024-fall"
        doc_id = doc_id.replace("_", "-")
    elif "transcript" in doc_id.lower():
        doc_type = "transcript_analysis"
        doc_id = "transcript-analysis"
    else:
        # Default: use the filename as both
        doc_type = name_without_ext.split("--")[0].split("_")[0]
        doc_id = doc_id.replace("_", "-").replace("--", "-")

    return doc_id, doc_type


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


def enrich_section_text(
    section_text: str,
    section_name: str,
    doc_type: str,
    base_metadata: Dict[str, Any]
) -> str:
    """
    Add contextual prefix to section text for better semantic matching.

    The embeddings only see the TEXT, not metadata. This enriches the text
    with semantic context words that help match user queries.

    Args:
        section_text: Original section content
        section_name: Name of the section (e.g., "Skills", "Work Experience")
        doc_type: Document type (resume, term, etc.)
        base_metadata: Metadata dict (may contain name, etc.)

    Returns:
        Enriched text with contextual prefix
    """

    # Only enrich resume sections for now
    if doc_type != "resume":
        return section_text

    # Get person name from metadata if available
    person_name = base_metadata.get("name", "Vaishak Menon")

    # Determine section type and add appropriate context
    section_lower = section_name.lower()

    if "skill" in section_lower:
        prefix = f"{person_name}'s technical skills, expertise, and experience include:\n\n"
        suffix = "\n\nKeywords: technical skills, cloud platforms, programming experience, DevOps expertise, technologies"

    elif "experience" in section_lower or "intern" in section_lower or "assistant" in section_lower:
        # Extract company name from section_name if present
        company = "the organization"
        if "maven wave" in section_lower:
            company = "Maven Wave Partners"
        elif "alabama" in section_lower or "birmingham" in section_lower:
            company = "University of Alabama at Birmingham"

        prefix = f"Professional work experience at {company}:\n\n"
        suffix = "\n\nKeywords: work experience, employment, job, position, company, role, professional background"

    elif "project" in section_lower or "fastapi" in section_lower or "gpt" in section_lower or "music" in section_lower:
        prefix = f"Personal project built by {person_name}:\n\n"
        suffix = "\n\nKeywords: personal projects, software projects, machine learning projects, AI projects, development work"

    elif "education" in section_lower:
        prefix = f"{person_name}'s educational background:\n\n"
        suffix = "\n\nKeywords: education, degrees, graduation, university, academic background, BS, MS"

    elif "certification" in section_lower:
        prefix = f"{person_name}'s professional certifications and credentials:\n\n"
        suffix = "\n\nKeywords: certifications, credentials, professional development, certificates"

    elif "summary" in section_lower:
        prefix = f"Professional summary for {person_name}:\n\n"
        suffix = "\n\nKeywords: summary, background, overview, experience, expertise, professional profile"

    elif "header" in section_lower:
        prefix = f"Contact information for {person_name}:\n\n"
        suffix = "\n\nKeywords: contact, email, phone, location, website, profiles"

    else:
        # Default: just add person's name
        prefix = f"{person_name} - {section_name}:\n\n"
        suffix = ""

    # Combine: prefix + original text + suffix
    enriched = prefix + section_text + suffix

    return enriched


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
    # Extract doc_id and doc_type from filename
    doc_id, _ = extract_doc_id(source_path)

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
        "section_slug": section_slug,  # Add for chunk ID generation
        "doc_id": doc_id,
        "version_identifier": version,
    }

    # Add structured section levels (section_l1, section_l2, section_l3, etc.)
    for i, section_title in enumerate(section_stack, start=1):
        section_metadata[f"section_l{i}"] = section_title

    # Also keep human-readable section path (as string, not list)
    if section_stack:
        section_metadata["section"] = " > ".join(section_stack)

    # Enrich the text with contextual information for better semantic matching
    doc_type = base_metadata.get("doc_type", "")
    enriched_text = enrich_section_text(text, section_name, doc_type, base_metadata)

    return {"text": enriched_text.strip(), "metadata": section_metadata}


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

    IMPORTANT: This function also creates sections for:
    - Content before the first split-level header (preamble sections)
    - Level-1 headers that have no split-level children (standalone sections)

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
        # Professional Summary
        Summary content...
        # Work Experience
        ## Teaching Assistant
        Content about TA role...
        ## Infrastructure Intern
        Content about intern role...
        # Skills
        Skills content...
        ```

        With split_level=2, creates 4 section documents:
        - One for "Professional Summary" (level-1 with no level-2 children)
        - One for "Teaching Assistant" section
        - One for "Infrastructure Intern" section
        - One for "Skills" (level-1 with no level-2 children)
    """
    sections = []
    current_section_text = ""
    current_section_name = None
    current_section_stack = []  # Stack at the time this section started
    section_stack = []  # Global tracking of nested header hierarchy

    # Track the current level-1 header for sections that don't have split-level children
    current_l1_header = None
    current_l1_text = ""
    has_split_level_child = False

    lines = text.split("\n")

    def save_current_section():
        """Helper to save the current section if it has content."""
        nonlocal current_section_text, current_section_name, current_section_stack
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
            current_section_text = ""
            current_section_name = None
            current_section_stack = []

    def save_l1_section_if_standalone():
        """Save a level-1 section if it had no split-level children."""
        nonlocal current_l1_header, current_l1_text, has_split_level_child
        if current_l1_header and current_l1_text.strip() and not has_split_level_child:
            section_doc = create_section_document(
                text=current_l1_text,
                base_metadata=base_metadata,
                section_name=current_l1_header,
                section_stack=[current_l1_header],
                source_path=source_path,
            )
            sections.append(section_doc)
        # Reset L1 tracking
        current_l1_header = None
        current_l1_text = ""
        has_split_level_child = False

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

            # Handle level-1 headers specially
            if level == 1:
                # Save any pending split-level section
                save_current_section()
                # Save previous L1 section if it was standalone (no split-level children)
                save_l1_section_if_standalone()

                # Start tracking new L1 section
                current_l1_header = title
                current_l1_text = line + "\n"
                has_split_level_child = False

            # If we hit a split-level header
            elif level == split_level:
                # Mark that current L1 has a split-level child
                has_split_level_child = True

                # Save previous split-level section if it exists
                save_current_section()

                # Start new section and capture stack at this moment
                current_section_text = line + "\n"
                current_section_name = title
                current_section_stack = section_stack.copy()

            else:
                # Non-split-level, non-L1 header - add to current section
                if current_section_name:
                    current_section_text += line + "\n"
                elif current_l1_header:
                    current_l1_text += line + "\n"

        else:
            # Regular content line - add to appropriate section
            if current_section_name:
                # We're in a split-level section
                current_section_text += line + "\n"
            elif current_l1_header:
                # We're in a L1 section that may or may not have split-level children
                current_l1_text += line + "\n"
            # else: content before any header - typically ignored or could be added to preamble

    # Save final split-level section if exists
    save_current_section()

    # Save final L1 section if it was standalone
    save_l1_section_if_standalone()

    # If no sections were created (no headers at all), return entire document as one section
    if not sections and text.strip():
        logger.warning(
            f"No headers found in {source_path}, "
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
