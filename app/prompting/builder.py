"""
Prompt Builder - XML-based prompt construction (Version 2.0)

Simplified to use XML structure for better instruction following.
Backup of original: builder.py.backup
"""

import logging
from typing import Any, Dict, List, Optional

from app.prompting.config import PromptConfig, PromptResult, SYSTEM_PROMPT_XML

logger = logging.getLogger(__name__)


def _format_context_xml(chunks: List[Dict[str, Any]]) -> str:
    """Format chunks into XML-compatible context with numbered citations.

    Args:
        chunks: List of chunks with 'source', 'text', and 'metadata' keys.

    Returns:
        Formatted context string with [1], [2] citation markers
    """
    parts: List[str] = []
    citation_index = 1

    for chunk in chunks:
        text = (chunk.get("text") or "").strip()

        # Skip if no text
        if not text:
            continue

        # Store citation index in chunk for later use by chat_service
        chunk["citation_index"] = citation_index

        # Simple format: [N] text (no filename to reduce reference list temptation)
        parts.append(f"[{citation_index}] {text}")
        citation_index += 1

    return "\n\n".join(parts)


def _format_history_xml(history: List[Dict[str, str]]) -> str:
    """Format conversation history for XML prompt.

    Args:
        history: List of conversation turns with 'role' and 'content' keys.

    Returns:
        Formatted history string with XML tags for each turn.
    """
    if not history:
        return ""

    formatted = []
    for turn in history:
        role = turn.get("role", "user")
        content = turn.get("content", "").strip()
        if content:
            formatted.append(f"<{role}>{content}</{role}>")

    return "\n".join(formatted)


def build_xml_prompt(
    query: str,
    chunks: List[Dict[str, Any]],
    conversation_history: Optional[List[Dict[str, str]]] = None,
) -> str:
    """Build prompt using XML structure with history support.

    This is the new builder function per the optimization plan.

    Args:
        query: User's question
        chunks: Retrieved context chunks
        conversation_history: Optional list of previous conversation turns

    Returns:
        Formatted XML prompt string
    """
    formatted_context = _format_context_xml(chunks)
    formatted_history = (
        _format_history_xml(conversation_history) if conversation_history else ""
    )

    return SYSTEM_PROMPT_XML.format(
        chunks=formatted_context.strip(), history=formatted_history, query=query.strip()
    )


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

        Now uses simplified XML structure.

        Args:
            question: The user's question
            context_chunks: Retrieved context chunks
            keywords: Optional list of keywords (currently ignored in XML mode)
            conversation_history: Optional conversation history (currently ignored in XML mode)

        Returns:
            PromptResult with the constructed prompt
        """
        try:
            # Use new XML-based prompt builder with history support
            prompt = build_xml_prompt(question, context_chunks, conversation_history)

            # Format context for result (used by chat_service for sources)
            context = _format_context_xml(context_chunks)

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
