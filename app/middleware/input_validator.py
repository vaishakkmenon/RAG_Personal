"""
Input validation and sanitization for security.

This module provides input sanitization and validation functions to protect
against various attack vectors including:
- Null byte injection
- Unicode homoglyph attacks
- Control character injection
- Excessive special character sequences
- Query complexity attacks
"""

import re
import unicodedata
import logging
from typing import Tuple

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from app.metrics import security_input_validation_total, security_events_total

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# Characters that should be stripped from input
BLOCKED_CHARACTERS = {
    "\x00",  # Null byte
    "\x01",
    "\x02",
    "\x03",
    "\x04",
    "\x05",
    "\x06",
    "\x07",  # Control chars
    "\x08",
    "\x0b",
    "\x0c",
    "\x0e",
    "\x0f",  # More control chars
    "\x10",
    "\x11",
    "\x12",
    "\x13",
    "\x14",
    "\x15",
    "\x16",
    "\x17",
    "\x18",
    "\x19",
    "\x1a",
    "\x1b",
    "\x1c",
    "\x1d",
    "\x1e",
    "\x1f",
    "\x7f",  # DEL
}

# Configuration
MAX_QUERY_LENGTH = 500  # Portfolio queries should be concise
MAX_WORDS = 50
MIN_UNIQUE_WORD_RATIO = 0.3  # At least 30% unique words
MAX_SPECIAL_CHAR_SEQUENCE = 10  # Max consecutive special characters


def sanitize_input(text: str) -> str:
    """
    Remove dangerous characters and normalize input.

    Args:
        text: Raw input text

    Returns:
        Sanitized text with control characters removed and Unicode normalized
    """
    if not text:
        return ""

    # Strip null bytes and control characters
    cleaned = "".join(c for c in text if c not in BLOCKED_CHARACTERS)

    # Normalize Unicode to prevent homoglyph attacks (e.g., Cyrillic 'а' vs Latin 'a')
    # NFKC normalization decomposes and then recomposes by compatibility
    cleaned = unicodedata.normalize("NFKC", cleaned)

    return cleaned.strip()


def validate_query_complexity(question: str) -> Tuple[bool, str]:
    """
    Validate that a query is reasonable in length and complexity.

    Args:
        question: The user's question

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if not question:
        if METRICS_ENABLED:
            security_input_validation_total.labels(result="failed_empty").inc()
        return False, "Query cannot be empty"

    # Length check
    if len(question) > MAX_QUERY_LENGTH:
        logger.warning(
            f"Query too long: {len(question)} chars (max {MAX_QUERY_LENGTH})"
        )
        if METRICS_ENABLED:
            security_input_validation_total.labels(result="failed_length").inc()
            security_events_total.labels(
                event_type="input_validation_failed", severity="low"
            ).inc()
        return False, f"Query too long (max {MAX_QUERY_LENGTH} characters)"

    # Word count check
    words = question.split()
    if len(words) > MAX_WORDS:
        logger.warning(f"Query too complex: {len(words)} words (max {MAX_WORDS})")
        if METRICS_ENABLED:
            security_input_validation_total.labels(result="failed_complexity").inc()
            security_events_total.labels(
                event_type="input_validation_failed", severity="low"
            ).inc()
        return False, f"Query too complex (max {MAX_WORDS} words)"

    # Detect excessive repeated tokens (potential scraping/attack)
    if len(words) >= 5:  # Only check if enough words
        unique_ratio = len(set(w.lower() for w in words)) / len(words)
        if unique_ratio < MIN_UNIQUE_WORD_RATIO:
            logger.warning(f"Suspicious query pattern: {unique_ratio:.0%} unique words")
            if METRICS_ENABLED:
                security_input_validation_total.labels(result="failed_complexity").inc()
                security_events_total.labels(
                    event_type="input_validation_failed", severity="medium"
                ).inc()
            return False, "Suspicious query pattern detected"

    # Detect excessive special character sequences
    if re.search(
        r'[!@#$%^&*()_+=\[\]{}|\\:";\'<>,.?/~`]{'
        + str(MAX_SPECIAL_CHAR_SEQUENCE + 1)
        + r",}",
        question,
    ):
        logger.warning("Excessive special characters detected")
        if METRICS_ENABLED:
            security_input_validation_total.labels(result="failed_characters").inc()
            security_events_total.labels(
                event_type="input_validation_failed", severity="medium"
            ).inc()
        return False, "Excessive special characters detected"

    if METRICS_ENABLED:
        security_input_validation_total.labels(result="passed").inc()
    return True, ""


def validate_and_sanitize(text: str) -> Tuple[str, bool, str]:
    """
    Combined validation and sanitization.

    Args:
        text: Raw input text

    Returns:
        Tuple of (sanitized_text, is_valid, error_message)
    """
    # First sanitize
    sanitized = sanitize_input(text)

    # Then validate
    is_valid, error_message = validate_query_complexity(sanitized)

    return sanitized, is_valid, error_message
