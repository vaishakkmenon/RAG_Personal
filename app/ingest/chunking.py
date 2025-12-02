"""
Chunking - Text splitting with section awareness

Handles splitting text into overlapping chunks while preserving section information.
"""

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from ..settings import ingest_settings
from .metadata import generate_version_identifier

logger = logging.getLogger(__name__)


# ============================================================================
# NEW: Header-Based Chunking (Single-Stage)
# ============================================================================


def chunk_by_headers(
    text: str,
    base_metadata: Dict[str, Any],
    source_path: str,
    chunk_size: int = None,
    overlap: int = None,
    split_level: int = 2,
) -> List[Dict[str, Any]]:
    """
    Single-stage header-based chunking that includes headers with content.

    Replaces the two-stage process (split_into_section_documents + chunk_text_with_section_metadata)
    with a single pass that:
    1. Parses markdown headers at specified level (default: ##)
    2. Creates chunks with headers included
    3. Adds [Part X/Y] markers for multi-chunk sections
    4. Maintains all metadata for backward compatibility

    Args:
        text: Markdown body text (after frontmatter extraction)
        base_metadata: Metadata from YAML frontmatter
        source_path: Original file path
        chunk_size: Target chunk size in characters (default: 600)
        overlap: Overlap between consecutive chunks (default: 120)
        split_level: Header level to split at (default: 2 for ##)

    Returns:
        List of chunk dicts with 'text' and 'metadata' keys

    Example:
        Given markdown with ## headers, creates chunks like:
        - "## Section Name\n\nContent here..."
        - "## Long Section [Part 1/2]\n\nFirst part..."
        - "## Long Section [Part 2/2]\n\nSecond part..."
    """
    if chunk_size is None:
        chunk_size = ingest_settings.chunk_size
    if overlap is None:
        overlap = ingest_settings.chunk_overlap

    # Extract doc identifiers
    doc_id, doc_type = extract_doc_id(source_path)

    # Use version from base_metadata (already set by processor with content-based detection)
    version = base_metadata.get('version_identifier')
    if not version:
        # Fallback: generate if not present (should not happen in normal flow)
        version = generate_version_identifier(base_metadata, doc_id)
        logger.warning(f"version_identifier not in metadata for {source_path}, generating: {version}")

    # Parse markdown into section structures
    sections = _parse_markdown_sections(text, split_level)

    if not sections:
        logger.warning(f"No sections found in {source_path}, creating single section")
        sections = [{
            'header': '# Complete Document',
            'header_text': 'Complete Document',
            'header_level': 1,
            'content': text,
            'section_stack': ['Complete Document']
        }]

    # Create chunks for each section
    all_chunks = []
    chunk_idx = 0

    for section in sections:
        header_text = section['header']
        content = section['content']
        section_stack = section['section_stack']
        section_name = section['header_text']

        # Skip empty sections
        if not content.strip():
            logger.debug(f"Skipping empty section: {section_name}")
            continue

        # Calculate available space for content
        header_length = len(header_text)
        full_text = f"{header_text}\n\n{content}"

        # Check if section fits in single chunk
        if len(full_text) <= chunk_size:
            # Single chunk - no continuation needed
            chunk_metadata = _create_chunk_metadata(
                base_metadata=base_metadata,
                section_stack=section_stack,
                section_name=section_name,
                doc_id=doc_id,
                doc_type=doc_type,
                version=version,
                part_num=None,
                total_parts=1
            )

            all_chunks.append({
                'text': full_text.strip(),
                'metadata': chunk_metadata,
                'id': f"{doc_id}@{version}#{chunk_metadata['section_slug']}:{chunk_idx}"
            })
            chunk_idx += 1
        else:
            # Multi-chunk section - needs splitting
            # Reserve space for header + " [Part X/Y]" (max 12 chars for " [Part 99/99]")
            max_content_size = chunk_size - header_length - 15

            if max_content_size < 200:
                logger.warning(f"Header too long ({header_length} chars): {header_text[:50]}...")
                max_content_size = max(chunk_size - header_length - 15, 200)

            # Split content into parts with overlap
            content_parts = _split_content_with_overlap(content, max_content_size, overlap)
            total_parts = len(content_parts)

            for part_num, content_part in enumerate(content_parts, start=1):
                # Add continuation marker to header
                part_header = f"{header_text} [Part {part_num}/{total_parts}]"
                chunk_text = f"{part_header}\n\n{content_part}".strip()

                chunk_metadata = _create_chunk_metadata(
                    base_metadata=base_metadata,
                    section_stack=section_stack,
                    section_name=section_name,
                    doc_id=doc_id,
                    doc_type=doc_type,
                    version=version,
                    part_num=part_num,
                    total_parts=total_parts
                )

                all_chunks.append({
                    'text': chunk_text,
                    'metadata': chunk_metadata,
                    'id': f"{doc_id}@{version}#{chunk_metadata['section_slug']}:{chunk_idx}"
                })
                chunk_idx += 1

    logger.info(f"Created {len(all_chunks)} chunks from {len(sections)} sections in {source_path}")
    return all_chunks


def _parse_markdown_sections(
    text: str,
    split_level: int = 2
) -> List[Dict[str, Any]]:
    """
    Parse markdown into section structures based on header level.

    Tracks header hierarchy and creates section dictionaries for each
    section at the split level (default: ## headers).

    Args:
        text: Markdown text to parse
        split_level: Header level to split at (1-6, default: 2)

    Returns:
        List of section dicts with keys:
        - header: Raw header line (e.g., "## Work Experience")
        - header_text: Clean title (e.g., "Work Experience")
        - header_level: Integer level (e.g., 2)
        - content: Content text (string)
        - section_stack: List of hierarchical section names
    """
    sections = []
    current_section = None
    section_stack = []
    current_l1_header = None
    preamble_content = []
    in_preamble = True

    lines = text.split('\n')

    for line in lines:
        # Check for markdown header
        header_match = re.match(r'^(#{1,6})\s+(.+)$', line)

        if header_match:
            in_preamble = False
            level = len(header_match.group(1))
            title = header_match.group(2).strip()

            # Update section stack based on header level
            if level == 1:
                section_stack = [title]
                current_l1_header = title
            elif level == 2:
                if current_l1_header:
                    section_stack = [current_l1_header, title]
                else:
                    section_stack = [title]
            elif level == 3:
                if len(section_stack) >= 2:
                    section_stack = section_stack[:2] + [title]
                else:
                    section_stack = section_stack + [title]
            else:
                # Levels 4-6: append to existing stack
                if len(section_stack) >= 3:
                    section_stack = section_stack[:3] + [title]
                else:
                    section_stack = section_stack + [title]

            # Check if this is a split-level header
            if level == split_level:
                # Save previous section if exists
                if current_section is not None:
                    sections.append(current_section)

                # Start new section
                current_section = {
                    'header': line,
                    'header_text': title,
                    'header_level': level,
                    'content': '',
                    'section_stack': section_stack.copy()
                }
            elif level == 1 and split_level == 2:
                # Special case: Level-1 header when splitting at level 2
                # Save previous section if exists
                if current_section is not None:
                    sections.append(current_section)

                # This L1 might become a section if it has no L2 children
                current_section = {
                    'header': line,
                    'header_text': title,
                    'header_level': level,
                    'content': '',
                    'section_stack': section_stack.copy(),
                    'is_l1_standalone': True  # Mark for potential section creation
                }
            else:
                # Lower-level header - add to current section content
                if current_section is not None:
                    current_section['content'] += line + '\n'
        else:
            # Regular content line
            if in_preamble:
                preamble_content.append(line)
            elif current_section is not None:
                current_section['content'] += line + '\n'

    # Don't forget last section
    if current_section is not None:
        sections.append(current_section)

    # Handle preamble if exists
    if preamble_content and any(line.strip() for line in preamble_content):
        preamble_text = '\n'.join(preamble_content).strip()
        if preamble_text:
            sections.insert(0, {
                'header': '# Preamble',
                'header_text': 'Preamble',
                'header_level': 1,
                'content': preamble_text,
                'section_stack': ['Preamble']
            })

    # Clean up content (remove trailing whitespace)
    for section in sections:
        section['content'] = section['content'].strip()

    return sections


def _split_content_with_overlap(
    content: str,
    max_size: int,
    overlap: int = 120
) -> List[str]:
    """
    Split content into overlapping parts that fit within max_size.

    Splitting strategy:
    1. Try to split at paragraph boundaries (\n\n)
    2. If paragraph too large, split at sentence boundaries
    3. Maintain overlap between consecutive parts

    Args:
        content: Text content to split
        max_size: Maximum size per part in characters
        overlap: Number of characters to overlap between parts

    Returns:
        List of content strings (without headers)
    """
    if len(content) <= max_size:
        return [content]

    parts = []
    remaining = content

    while remaining:
        if len(remaining) <= max_size:
            # Last part
            parts.append(remaining)
            break

        # Take up to max_size characters
        chunk = remaining[:max_size]

        # Try to break at paragraph boundary
        if '\n\n' in chunk:
            break_point = chunk.rfind('\n\n')
            if break_point > max_size * 0.3:  # Don't break too early
                chunk = chunk[:break_point].strip()
            else:
                # Paragraph break too early, try sentence break
                if '. ' in chunk[-200:]:
                    break_point = chunk.rfind('. ', max(0, len(chunk) - 200))
                    chunk = chunk[:break_point + 1].strip()
        elif '. ' in chunk[-200:]:
            # Break at sentence boundary near end
            break_point = chunk.rfind('. ', max(0, len(chunk) - 200))
            if break_point > 0:
                chunk = chunk[:break_point + 1].strip()
        else:
            # No good break point, just split at max_size
            # Try to avoid splitting mid-word
            last_space = chunk.rfind(' ')
            if last_space > max_size * 0.8:
                chunk = chunk[:last_space].strip()

        parts.append(chunk)

        # Move forward in remaining text
        remaining = remaining[len(chunk):].strip()

        # Add overlap from current chunk to beginning of next remaining
        if remaining and overlap > 0:
            # Get last overlap characters from chunk
            overlap_text = chunk[-overlap:] if len(chunk) >= overlap else chunk
            # Only add overlap if it's not already at start of remaining
            if not remaining.startswith(overlap_text):
                remaining = overlap_text + ' ' + remaining

    return parts


def _create_chunk_metadata(
    base_metadata: Dict[str, Any],
    section_stack: List[str],
    section_name: str,
    doc_id: str,
    doc_type: str,
    version: str,
    part_num: Optional[int] = None,
    total_parts: int = 1
) -> Dict[str, Any]:
    """
    Create complete metadata dictionary for a chunk.

    Includes all metadata fields for filtering and adds fields for multi-part tracking.

    Args:
        base_metadata: Metadata inherited from frontmatter
        section_stack: Hierarchical section path (e.g., ["Work Experience", "Teaching Assistant"])
        section_name: Name of this section
        doc_id: Document identifier
        doc_type: Document type (resume, certificate, transcript_analysis, term)
        version: Version identifier
        part_num: Part number for multi-chunk sections (1-indexed, None for single)
        total_parts: Total number of parts in this section

    Returns:
        Metadata dict ready for ChromaDB (all values are str/int/float/bool, no None)
    """
    # Generate section slug from full stack
    section_slug = slugify(" > ".join(section_stack))
    section_doc_id = f"{doc_id}@{version}#{section_slug}"

    # Build metadata with all required fields
    metadata = {
        **base_metadata,  # Inherit all frontmatter metadata

        # Core identifiers
        "doc_id": doc_id,
        "doc_type": doc_type,
        "version_identifier": version,

        # Section identifiers
        "section": " > ".join(section_stack),
        "section_slug": section_slug,
        "section_doc_id": section_doc_id,
        "section_name": section_name,

        # Section stack for ChromaDB (string, not list)
        "section_stack": " > ".join(section_stack),
    }

    # Add hierarchical levels (section_l1, section_l2, etc.)
    for i, section_title in enumerate(section_stack, start=1):
        metadata[f"section_l{i}"] = section_title

    # Add multi-part tracking fields (NEW)
    if part_num is not None:
        metadata["part_num"] = part_num
        metadata["total_parts"] = total_parts
        metadata["is_multipart"] = True
    else:
        metadata["total_parts"] = 1
        metadata["is_multipart"] = False

    # Remove None values (ChromaDB doesn't accept them)
    metadata = {k: v for k, v in metadata.items() if v is not None}

    return metadata


# ============================================================================
# Helper Functions (shared by all chunking methods)
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
