"""
Output validation and sanitization for security.

This module provides output validation functions to detect and handle:
- System prompt leakage in LLM responses
- Internal terminology exposure (RAG implementation details)
- Response sanitization before returning to users
"""

import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from app.metrics import security_output_validation_total, security_events_total

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False

# Fragments from system prompts that should NEVER appear in responses
# These indicate the LLM is leaking its instructions
LEAKED_SYSTEM_FRAGMENTS = [
    # Common system prompt patterns
    "you are an ai assistant",
    "you are a helpful",
    "your system instructions",
    "your instructions are",
    "my instructions are",
    "i am instructed to",
    "critical rules",  # Without colon to catch "critical rules include:"
    "privacy rules",
    "response format:",
    # System behavior descriptions (meta-responses)
    "the system is trained",
    "the system uses",
    "the system's primary",
    "context fidelity",
    "field inference",
    "filtering workflow",
    "context documents",
    "trained on a set of",
    # Implementation-specific
    "build_groq_prompt",
    "get_system_prompt",
    "prompt_builder",
    # Security-related
    "never reveal",
    "do not disclose",
    "keep confidential",
]

# Terms that reveal RAG/LLM implementation details
# These should be flagged but not necessarily blocked
INTERNAL_TERMINOLOGY = [
    # RAG components
    "chunk",
    "chunks",
    "chunking",
    "embedding",
    "embeddings",
    "vector store",
    "vector database",
    "retrieval",
    "retrieved context",
    "rerank",
    "reranking",
    # Specific technologies
    "chromadb",
    "chroma",
    "bge-small",
    "bge-v1.5",
    "sentence-transformers",
    "groq",
    "groq api",
    "llama-3",
    "llama3",
    "llama guard",
    # Technical parameters
    "top_k",
    "top-k",
    "similarity score",
    "distance score",
    "metadata filter",
    "doc_type",
]


def detect_prompt_leakage(response: str) -> Tuple[bool, Optional[str]]:
    """
    Check if LLM response contains system prompt fragments.

    Args:
        response: The LLM-generated response text

    Returns:
        Tuple of (is_leaked, leaked_fragment).
        If no leakage detected, returns (False, None)
    """
    if not response:
        return False, None

    response_lower = response.lower()

    for fragment in LEAKED_SYSTEM_FRAGMENTS:
        # For short fragments like "context:", require them to be at start of line
        # to avoid false positives
        if len(fragment) < 10:
            if f"\n{fragment}" in response_lower or response_lower.startswith(fragment):
                logger.warning(f"Potential prompt leakage detected: '{fragment}'")
                if METRICS_ENABLED:
                    security_output_validation_total.labels(
                        result="prompt_leakage_blocked"
                    ).inc()
                    security_events_total.labels(
                        event_type="prompt_leakage", severity="critical"
                    ).inc()
                return True, fragment
        else:
            if fragment in response_lower:
                logger.warning(f"System prompt leakage detected: '{fragment}'")
                if METRICS_ENABLED:
                    security_output_validation_total.labels(
                        result="prompt_leakage_blocked"
                    ).inc()
                    security_events_total.labels(
                        event_type="prompt_leakage", severity="critical"
                    ).inc()
                return True, fragment

    return False, None


def contains_internal_terminology(response: str) -> Tuple[bool, Optional[str]]:
    """
    Check if response accidentally mentions RAG implementation internals.

    Args:
        response: The LLM-generated response text

    Returns:
        Tuple of (contains_internal, first_term_found).
        If no internal terms found, returns (False, None)
    """
    if not response:
        return False, None

    response_lower = response.lower()

    for term in INTERNAL_TERMINOLOGY:
        if term in response_lower:
            logger.warning(f"Internal terminology detected in response: '{term}'")
            if METRICS_ENABLED:
                security_output_validation_total.labels(
                    result="internal_term_flagged"
                ).inc()
                security_events_total.labels(
                    event_type="internal_term_leaked", severity="medium"
                ).inc()
            return True, term

    return False, None


def sanitize_response(response: str, strict: bool = False) -> Tuple[str, bool]:
    """
    Apply all output validation checks and sanitize response.

    Args:
        response: The LLM-generated response text
        strict: If True, return error response on any detection.
                If False, log warnings but return original response.

    Returns:
        Tuple of (sanitized_response, had_issues).
        If strict=True and issues found, returns a safe fallback response.
    """
    if not response:
        return response, False

    had_issues = False

    # Check for system prompt leakage (always critical)
    leaked, fragment = detect_prompt_leakage(response)
    if leaked:
        had_issues = True
        if strict:
            logger.error(f"Blocked response due to prompt leakage: {fragment}")
            return (
                "I apologize, but I encountered an issue generating a response. "
                "Could you please rephrase your question about Vaishak's professional background?",
                True,
            )

    # Check for internal terminology (warning, not blocking unless strict)
    has_internal, term = contains_internal_terminology(response)
    if has_internal:
        had_issues = True
        if strict:
            logger.warning(f"Response contains internal term: {term}")
            # Don't block, just flag

    return response, had_issues


def validate_response_safety(response: str) -> Tuple[bool, str]:
    """
    Validate that a response is safe to return to users.

    This is a convenience function that combines all checks.

    Args:
        response: The LLM-generated response text

    Returns:
        Tuple of (is_safe, reason).
        If safe, reason is empty string.
    """
    # Check prompt leakage
    leaked, fragment = detect_prompt_leakage(response)
    if leaked:
        return False, f"Response contains system prompt fragment: {fragment}"

    # Internal terminology is a warning, not a blocker
    # So we still return True but log it
    has_internal, term = contains_internal_terminology(response)
    if has_internal:
        logger.info(f"Response contains internal term (allowed): {term}")

    # Track successful validation
    if METRICS_ENABLED:
        security_output_validation_total.labels(result="passed").inc()

    return True, ""
