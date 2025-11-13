"""
Prompt Builder - Prompt construction and validation

Handles building and validating prompts with safety guards.
"""

import logging
from typing import Dict, List, Optional, Tuple

from ..query_router.patterns import PatternMatcher
from ..settings import QueryRouterSettings
from .clarification import build_clarification_message
from .config import PromptConfig, PromptResult

logger = logging.getLogger(__name__)


def _format_context(chunks: List[Dict[str, str]]) -> str:
    """Format chunks into context string."""
    parts: List[str] = []
    for chunk in chunks:
        source = chunk.get("source", "unknown")
        text = (chunk.get("text") or "").strip()
        if text:
            parts.append(f"[Source: {source}]\n{text}")
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

    def is_ambiguous(self, question: str) -> Tuple[bool, str]:
        """Check if a question is ambiguous or too vague.

        Args:
            question: The user's question

        Returns:
            Tuple of (is_ambiguous, reason)
        """
        question = (question or "").strip()

        # Basic validation
        if not question:
            return True, "Question cannot be empty"

        if len(question.split()) < self.config.min_question_length:
            return True, "Question is too short"

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
        self, question: str, context_chunks: List[Dict[str, str]], **kwargs
    ) -> PromptResult:
        """Build a structured prompt with validation.

        Args:
            question: The user's question
            context_chunks: List of context chunks
            **kwargs: Additional arguments (unused)

        Returns:
            PromptResult with status and prompt
        """
        stripped_question = (question or "").strip()
        ambiguous, reason = self.is_ambiguous(stripped_question)
        if ambiguous:
            return PromptResult(
                status="ambiguous",
                message=build_clarification_message(stripped_question, self.config),
            )

        context = _format_context(context_chunks)
        if not context:
            return PromptResult(
                status="no_context",
                message="No relevant context found to answer the question.",
            )

        prompt_sections = [self.config.system_prompt.strip()]

        guidelines = (self.config.certification_guidelines or "").strip()
        if guidelines:
            prompt_sections.append(guidelines)

        examples_block = self._render_examples_block()
        if examples_block:
            prompt_sections.append(examples_block)

        prompt_sections.append(
            """CONTEXT:
{context}

QUESTION: {question}

ANSWER:""".format(
                context=context,
                question=stripped_question,
            )
        )

        prompt = "\n\n".join(section for section in prompt_sections if section)

        return PromptResult(status="success", prompt=prompt, context=context)

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

    def _render_examples_block(self) -> str:
        """Render few-shot examples block.

        Returns:
            Formatted examples string or empty string
        """
        if not self.config.use_certification_examples:
            return ""

        examples = self.config.certification_examples
        if not examples:
            return ""

        lines: List[str] = ["FEW-SHOT EXAMPLES:"]
        for idx, (question, answer) in enumerate(examples, start=1):
            lines.append(f"Example {idx}:")
            lines.append(f"Q: {question}")
            lines.append("A:")
            lines.extend(f"  {line}" for line in answer.strip().splitlines())
        return "\n".join(lines)
