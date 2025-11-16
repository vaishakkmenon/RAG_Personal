"""
Prompt Builder - Prompt construction and validation

Handles building and validating prompts with safety guards.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..query_router.patterns import PatternMatcher
from ..settings import QueryRouterSettings, settings
from .clarification import build_clarification_message
from .config import PromptConfig, PromptResult

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into context string with metadata injection.

    Args:
        chunks: List of chunks with 'source', 'text', and 'metadata' keys

    Returns:
        Formatted context string with metadata injected
    """

    parts: List[str] = []
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        metadata = chunk.get("metadata", {})

        # Skip if no text
        if not text:
            continue

        # Start building chunk parts
        chunk_parts = [f"[Source: {source}]"]

        # Add metadata if enabled
        if settings.metadata_injection.enabled:
            # Get document type from metadata
            doc_type = metadata.get("doc_type")

            # Get relevant metadata fields
            relevant_metadata = {}
            if doc_type in settings.metadata_injection.injection_config:
                for field in settings.metadata_injection.injection_config[doc_type]:
                    if field in metadata and metadata[field] is not None:
                        relevant_metadata[field] = metadata[field]

            # Format metadata if any
            if relevant_metadata:
                metadata_str = ", ".join(
                    f"{k.capitalize()}: {v}" for k, v in relevant_metadata.items()
                )
                chunk_parts.append(f"[Metadata: {metadata_str}]")

        # Add the actual text
        chunk_parts.append(text)

        # Add to final parts
        parts.append("\n".join(chunk_parts))

    return "\n\n".join(parts)


class PromptBuilder:
    """Handles construction and validation of prompts with safety guards."""

    def __init__(self, config: Optional[PromptConfig] = None):
        """Initialize prompt builder.

        Args:
            config: Optional PromptConfig instance
        """
        self.config = config or PromptConfig()
        # Initialize pattern matchers for ambiguity detection
        router_settings = QueryRouterSettings()
        self.tech_matcher = PatternMatcher(router_settings.technology_terms)
        self.category_matcher = PatternMatcher(router_settings.categories)
        self.question_matcher = PatternMatcher(router_settings.question_patterns)

    def _contains_technology(self, question: str) -> bool:
        """Check if question mentions specific technologies."""
        return len(self.tech_matcher.find_matches(question)) > 0

    def _matches_concrete_category(self, question: str) -> bool:
        """Check if question matches concrete categories."""
        return len(self.category_matcher.find_matches(question)) > 0

    def _is_specific_question_type(self, question: str) -> bool:
        """Check if question follows specific question patterns."""
        return len(self.question_matcher.find_matches(question)) > 0

    def is_ambiguous(self, question: str, is_structured_summary: bool = False) -> Tuple[bool, str]:
        """Check if a question is ambiguous or too vague.

        Args:
            question: The user's question
            is_structured_summary: Whether router detected structured summary intent

        Returns:
            Tuple of (is_ambiguous, reason)
        """
        question = (question or "").strip()

        # Basic validation
        if not question:
            return True, "Question cannot be empty"

        if len(question.split()) < self.config.min_question_length:
            return True, "Question is too short"

        # Structured summaries are NOT ambiguous (have clear intent + scope)
        if is_structured_summary:
            return False, ""

        # Early exit: Check for disambiguating signals
        if self._contains_technology(question):
            return False, ""  # Question mentions specific technology

        if self._matches_concrete_category(question):
            return False, ""  # Question asks about concrete category

        if self._is_specific_question_type(question):
            return False, ""  # Question follows specific pattern

        # Check for ambiguous phrases
        lower_question = question.lower()
        for phrase in self.config.ambiguous_phrases:
            if phrase in lower_question:
                return True, f"Question contains ambiguous phrase: {phrase}"

        # Check for vague questions without specifics
        first_word = lower_question.split()[0]
        if first_word in self.config.vague_question_words and len(question.split()) < 5:
            return True, "Question is too vague, please be more specific"

        return False, ""

    def build_prompt(
        self, question: str, context_chunks: List[Dict[str, Any]]
    ) -> PromptResult:
        """Build a prompt with the given question and context chunks."""
        try:
            # Format context with metadata injection
            context = _format_context(context_chunks)

            # Build prompt sections
            prompt_sections = [
                self.config.system_prompt.strip(),
                self.config.certification_guidelines.strip(),
                context,
                "QUESTION:",
                question.strip(),
                "ANSWER:",
            ]

            # Join sections with double newlines
            prompt = "\n\n".join(section for section in prompt_sections if section)

            return PromptResult(status="success", prompt=prompt, context=context)

        except Exception as e:
            logger.error(f"Error building prompt: {str(e)}")
            return PromptResult(
                status="error", message=f"Failed to build prompt: {str(e)}"
            )

    def is_refusal(self, answer: str) -> bool:
        """Check if answer is a refusal.

        Args:
            answer: The generated answer

        Returns:
            True if answer appears to be a refusal
        """
        text = (answer or "").strip().lower()
        if not text:
            return True
        if any(cue in text for cue in self.config.refusal_cues):
            return True
        if not any(ch.isalnum() for ch in text):
            return True
        return False

    def needs_clarification(self, answer: str) -> bool:
        """Check if answer needs clarification.

        Args:
            answer: The generated answer

        Returns:
            True if answer lacks clarification
        """
        text = (answer or "").strip().lower()
        if not text:
            return False
        return not any(cue in text for cue in self.config.clarification_cues)
