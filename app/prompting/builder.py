"""
Prompt Builder - Prompt construction and validation (OPTIMIZED)

Handles building and validating prompts with safety guards.
Uses conditional example injection for token efficiency.
"""

import logging
from typing import Any, Dict, List, Optional

from app.settings import settings
from app.prompting.config import PromptConfig, PromptResult

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into context string with numbered citations.

    Args:
        chunks: List of chunks with 'source', 'text', and 'metadata' keys.
                Each chunk will be assigned a citation index [1], [2], etc.

    Returns:
        Formatted context string with numbered source citations
    """
    import os

    parts: List[str] = []
    citation_index = 1

    for chunk in chunks:
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        metadata = chunk.get("metadata", {})

        # Skip if no text
        if not text:
            continue

        # Store citation index in chunk for later use by chat_service
        chunk["citation_index"] = citation_index

        # Use basename for cleaner citation labels
        source_basename = os.path.basename(source)

        # Start building chunk parts with numbered citation
        chunk_parts = [f"[{citation_index}] {source_basename}"]

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
        citation_index += 1

    return "\n\n".join(parts)


def _format_conversation_history(history: List[Dict[str, str]]) -> str:
    """Format conversation history for inclusion in prompt.

    Args:
        history: List of dicts with 'role' and 'content' keys
                Example: [
                    {"role": "user", "content": "What's my GPA?"},
                    {"role": "assistant", "content": "Your GPA is 4.00."}
                ]

    Returns:
        Formatted conversation history string
    """
    if not history:
        return ""

    formatted_turns = []
    for turn in history:
        role = turn.get("role", "").upper()
        content = turn.get("content", "").strip()
        if content:
            formatted_turns.append(f"{role}: {content}")

    if formatted_turns:
        return "PREVIOUS CONVERSATION:\n" + "\n\n".join(formatted_turns) + "\n"
    return ""


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
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> PromptResult:
        """Build a prompt with the given question and context chunks.

        Args:
            question: The user's question
            context_chunks: Retrieved context chunks
            keywords: Optional list of keywords to guide LLM focus
            conversation_history: Optional list of previous conversation turns

        Returns:
            PromptResult with the constructed prompt
        """
        try:
            # Format context with metadata injection
            context = _format_context(context_chunks)

            # Format conversation history
            history_text = ""
            if conversation_history:
                history_text = _format_conversation_history(conversation_history)

            # Format question with optional keyword guidance
            question_section = question.strip()
            if keywords and len(keywords) > 0:
                keyword_str = ", ".join(keywords)
                question_section += f"\n[Focus areas: {keyword_str}]"

            # OPTIMIZED: Get system prompt with conditional examples based on query
            system_prompt = self.config.get_system_prompt_with_examples(question)

            # OPTIMIZED: Get certification guidelines only if query mentions certifications
            cert_guidelines = self.config.get_certification_guidelines_with_check(
                question
            )

            # Build prompt sections
            prompt_sections = [
                system_prompt.strip(),
                cert_guidelines.strip() if cert_guidelines else None,
                history_text,
                "### CONTEXT (DO NOT OUTPUT) ###",
                context,
                "### END CONTEXT ###",
                "QUESTION:",
                question_section,
                "CRITICAL REMINDER: Use ONLY the context above. If the answer is not in the context, say 'I don't know. It isn't mentioned in the provided documents.' NEVER use external knowledge or make assumptions.",
                self.config.response_template.strip(),
                "ANSWER:",
            ]

            # Join sections with double newlines (filter out None/empty strings)
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
