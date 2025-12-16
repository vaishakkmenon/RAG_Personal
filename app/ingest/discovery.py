"""
File Discovery - Recursive file finding and validation

Handles finding markdown/text files within specified directories.
"""

import logging
import os
from typing import List

from fastapi import HTTPException

from app.settings import settings, ingest_settings

logger = logging.getLogger(__name__)

# Configuration
ALLOWED_EXT = ingest_settings.allowed_extensions


def find_files(base_paths: List[str]) -> List[str]:
    """Recursively find all allowed text files (.md, .txt) in the given paths.

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
            continue

        if os.path.isfile(abs_base):
            ext = os.path.splitext(abs_base)[1].lower()
            if ext in ALLOWED_EXT or ext.lstrip(".") in ALLOWED_EXT:
                files.append(abs_base)
                logger.debug(f"Found file: {abs_base}")
            else:
                logger.warning(f"Skipping {abs_base}: invalid extension {ext}")

        elif os.path.isdir(abs_base):
            for root, _, filenames in os.walk(abs_base):
                for name in filenames:
                    fp = os.path.join(root, name)
                    ext = os.path.splitext(name)[1].lower()
                    abs_fp = os.path.abspath(fp)

                    if abs_fp.startswith(base_docs_dir) and (
                        ext in ALLOWED_EXT or ext.lstrip(".") in ALLOWED_EXT
                    ):
                        files.append(abs_fp)
                        logger.debug(f"Found file: {abs_fp}")
                    else:
                        logger.debug(f"Skipping {fp}: invalid or outside docs_dir")
        else:
            logger.warning(f"Path does not exist: {abs_base}")

    if not files:
        raise HTTPException(
            status_code=400, detail=f"No valid .txt or .md files found in {base_paths}"
        )

    logger.info(f"Found {len(files)} files to process")
    return files
