"""
Prompt Builder - Prompt construction and validation

Handles building and validating prompts with safety guards.
"""

import logging
from typing import Any, Dict, List, Optional

from ..settings import settings
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

    def build_prompt(
        self,
        question: str,
        context_chunks: List[Dict[str, Any]],
        keywords: Optional[List[str]] = None,
        negative_inference_hint: Optional[Dict[str, Any]] = None
    ) -> PromptResult:
        """Build a prompt with the given question and context chunks.

        Args:
            question: The user's question
            context_chunks: Retrieved context chunks
            keywords: Optional list of keywords to guide LLM focus
            negative_inference_hint: Optional hint about negative inference opportunity
                Dict with keys: 'missing_entities', 'category'

        Returns:
            PromptResult with the constructed prompt
        """
        try:
            # Format context with metadata injection
            context = _format_context(context_chunks)

            # Format question with optional keyword guidance
            question_section = question.strip()
            if keywords and len(keywords) > 0:
                keyword_str = ", ".join(keywords)
                question_section += f"\n[Focus areas: {keyword_str}]"

            # Add negative inference hint if provided
            if negative_inference_hint:
                missing = negative_inference_hint.get('missing_entities', [])
                category = negative_inference_hint.get('category', 'items')

                if missing:
                    entities_str = ", ".join(missing)
                    hint = (
                        f"\n\n[IMPORTANT INSTRUCTION: The question asks about '{entities_str}' which does not appear in the provided context. "
                        f"However, the context DOES contain a complete list of {category}. "
                        f"You MUST apply NEGATIVE INFERENCE by examining the complete list and answering 'No' with what IS present. "
                        f"Do NOT say 'I don't know' - instead, use the complete list to infer the negative answer.]"
                    )
                    question_section += hint

            # Build prompt sections
            prompt_sections = [
                self.config.system_prompt.strip(),
                self.config.certification_guidelines.strip(),
                context,
                "QUESTION:",
                question_section,  # Now includes keyword annotation and negative inference hint if present
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
