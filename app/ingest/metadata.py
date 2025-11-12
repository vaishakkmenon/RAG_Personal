"""
Metadata Extraction - YAML front-matter and normalization

Handles extracting and normalizing metadata for ChromaDB compatibility.
"""

import logging
from datetime import date, datetime
from typing import Dict, Tuple

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
