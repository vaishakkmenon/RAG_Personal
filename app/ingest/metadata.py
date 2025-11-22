"""
Metadata Extraction - YAML front-matter and normalization

Handles extracting and normalizing metadata for ChromaDB compatibility.
"""

import logging
from datetime import date, datetime
from typing import Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)


def extract_frontmatter(text: str) -> Tuple[Dict, str]:
    """Extract YAML front-matter from Markdown and normalize for ChromaDB.

    ChromaDB metadata requirements:
    - Values must be: str, int, float, or bool (NOT None!)
    - No date objects, no nested dicts, no None values

    Args:
        text: Full file content

    Returns:
        Tuple of (metadata_dict, body_text)
        If no front-matter found, returns ({}, original_text)
    """
    if not text.startswith("---"):
        return {}, text

    parts = text.split("---", 2)
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
    """Read and return the full text content of a file using UTF-8 encoding.

    Args:
        path: File path to read

    Returns:
        File contents as a single string

    Raises:
        Exception: If file cannot be read
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


# ============================================================================
# Version Management Functions
# ============================================================================


def normalize_version_identifier(metadata: Dict) -> str:
    """
    Extract and normalize version identifier from various metadata fields.

    Handles different doc types with different version field names:
    - resume: version_date
    - certificate: earned
    - term: term_id
    - transcript_analysis: analysis_date

    Args:
        metadata: Document metadata dict

    Returns:
        Normalized version string (ISO date or semantic version)
    """
    # Priority order for version fields (first found wins)
    version_fields = [
        'version_date',      # Resume
        'earned',            # Certificates
        'term_id',           # Terms (e.g., "2024-fall")
        'analysis_date',     # Transcript analysis
        'updated_date',      # Generic fallback
        'created_date',      # Generic fallback
    ]

    for field in version_fields:
        if field in metadata and metadata[field]:
            value = metadata[field]

            # If already a string (like term_id: "2024-fall")
            if isinstance(value, str):
                return value

            # If date/datetime object
            if isinstance(value, (date, datetime)):
                return value.isoformat()

            # If parseable date string
            if isinstance(value, str):
                try:
                    # Try parsing as date
                    parsed = datetime.fromisoformat(value.replace('/', '-'))
                    return parsed.date().isoformat()
                except (ValueError, AttributeError):
                    # Not a date, return as-is
                    return value

    # No version found - use current date with warning
    logger.warning(
        f"No version identifier found in metadata, using current date. "
        f"Available keys: {list(metadata.keys())}"
    )
    return datetime.now().date().isoformat()


def get_existing_versions(doc_id: str, base_version: str) -> List[str]:
    """
    Query ChromaDB for existing versions of this document on this date.

    Used for same-day collision detection - finds versions like:
    - "2025-11-20"
    - "2025-11-20.v2"
    - "2025-11-20.v3"

    Args:
        doc_id: Document identifier (e.g., "resume", "certificate-cka")
        base_version: Base version string (e.g., "2025-11-20")

    Returns:
        List of existing version identifiers that start with base_version
    """
    try:
        # Import ChromaDB collection here to avoid circular imports
        from ..retrieval.store import _collection

        # Query chunks matching doc_id
        results = _collection.get(
            where={
                "doc_id": {"$eq": doc_id}
            },
            limit=10000  # Get all chunks for this doc
        )

        # Extract unique versions that match base_version
        versions = set()
        for metadata in results.get('metadatas', []):
            version = metadata.get('version_identifier', '')
            if version.startswith(base_version):
                versions.add(version)

        return sorted(versions)

    except ImportError:
        # ChromaDB not available (e.g., during testing)
        logger.debug("ChromaDB collection not available for version checking")
        return []
    except Exception as e:
        logger.warning(f"Error checking existing versions for {doc_id}: {e}")
        return []


def generate_version_identifier(metadata: Dict, doc_id: str) -> str:
    """
    Generate unique version identifier with same-day collision detection.

    Handles multiple ingestions of the same document on the same day by
    auto-incrementing a sequence number.

    Examples:
        First ingestion today:  "2025-11-20"
        Second ingestion today: "2025-11-20.v2"
        Third ingestion today:  "2025-11-20.v3"

    Args:
        metadata: Document metadata dict
        doc_id: Document identifier for collision checking

    Returns:
        Unique version string (e.g., "2025-11-20" or "2025-11-20.v2")
    """
    # Get base version from metadata
    base_version = normalize_version_identifier(metadata)

    # Check if this version already exists in ChromaDB
    existing_versions = get_existing_versions(doc_id, base_version)

    if not existing_versions:
        # First version of the day - use base version as-is
        logger.debug(f"First version for {doc_id}: {base_version}")
        return base_version

    # Same-day update detected - find next sequence number
    max_sequence = 1
    for version in existing_versions:
        if version.startswith(base_version):
            # Extract sequence number if present
            if '.v' in version:
                try:
                    seq_str = version.split('.v')[1]
                    seq = int(seq_str)
                    max_sequence = max(max_sequence, seq)
                except (IndexError, ValueError):
                    continue

    # Return next sequence
    next_sequence = max_sequence + 1
    next_version = f"{base_version}.v{next_sequence}"

    logger.info(
        f"Same-day collision detected for {doc_id}: "
        f"base={base_version}, assigned={next_version}"
    )

    return next_version
