"""
Response Parsing Utilities

Handles model-specific output parsing, including Qwen's "Thinking Process" blocks.
Qwen 3 models can output their reasoning inside <think>...</think> tags before
the final answer. This module separates thinking from the response.

Also provides the ReasoningEffort abstraction for controlling model reasoning
at request time - the correct API-level control surface for reasoning-capable models.
"""

import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ReasoningEffort(str, Enum):
    """
    Reasoning effort levels matching DeepInfra's API.

    Controls how much reasoning the model performs at request time.
    Values match the API exactly: none, low, medium, high.

    For RAG applications:
    - NONE: Default for most RAG answers (externalized reasoning via documents)
    - LOW: Light reasoning for simple multi-step queries
    - MEDIUM: For query decomposition or agentic flows
    - HIGH: For complex multi-hop reasoning (rarely needed in RAG)

    Key insight: Reasoning is a request-time capability, not a model configuration.
    RAG already externalizes reasoning via documents, so full chain-of-thought
    wastes tokens and latency for most queries.
    """

    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


def parse_reasoning_effort(value: Optional[str]) -> ReasoningEffort:
    """Parse a string value to ReasoningEffort enum.

    Args:
        value: String like "none", "low", "medium", "high" or None

    Returns:
        ReasoningEffort enum value, defaults to NONE if invalid/None
    """
    if not value:
        return ReasoningEffort.NONE

    try:
        return ReasoningEffort(value.lower())
    except ValueError:
        logger.warning(f"Invalid reasoning_effort '{value}', defaulting to 'none'")
        return ReasoningEffort.NONE


class ChunkType(Enum):
    """Type of streaming chunk for frontend differentiation."""

    THINKING = "thinking"
    ANSWER = "answer"


@dataclass
class StreamChunk:
    """A typed chunk for streaming responses.

    Used to differentiate between thinking process and actual answer
    content during streaming, allowing frontends to display them differently.
    """

    type: ChunkType
    content: str

    def is_thinking(self) -> bool:
        return self.type == ChunkType.THINKING

    def is_answer(self) -> bool:
        return self.type == ChunkType.ANSWER


@dataclass
class ParsedResponse:
    """Container for parsed LLM response."""

    answer: str
    thinking: Optional[str] = None
    had_thinking_block: bool = False


def parse_thinking_process(llm_response: str) -> Tuple[str, str]:
    """Separate the 'Thinking Process' from the 'Final Answer'.

    Qwen 3 models in "thinking mode" output their reasoning inside
    <think>...</think> tags before producing the final answer.
    This function extracts both components.

    Args:
        llm_response: Raw LLM output that may contain <think> blocks

    Returns:
        Tuple of (thinking_content, final_answer)
        - thinking_content: Content inside <think> tags (empty string if none)
        - final_answer: Response with <think> blocks removed and cleaned
    """
    if not llm_response:
        return "", ""

    # Pattern to match content inside <think>...</think> (dotall to match newlines)
    # Also handle potential variations: <thinking>, </thinking>
    think_patterns = [
        r"<think>(.*?)</think>",
        r"<thinking>(.*?)</thinking>",
    ]

    thinking_content = ""
    cleaned_response = llm_response

    for pattern in think_patterns:
        match = re.search(pattern, cleaned_response, re.DOTALL | re.IGNORECASE)
        if match:
            # Accumulate thinking content if multiple blocks exist
            if thinking_content:
                thinking_content += "\n\n"
            thinking_content += match.group(1).strip()

            # Remove the think block from response
            cleaned_response = re.sub(
                pattern, "", cleaned_response, flags=re.DOTALL | re.IGNORECASE
            )

    # Clean up the final answer
    final_answer = cleaned_response.strip()

    # Remove any leading/trailing whitespace or newlines that were around think blocks
    final_answer = re.sub(r"^\s+", "", final_answer)
    final_answer = re.sub(r"\s+$", "", final_answer)

    return thinking_content, final_answer


def parse_llm_response(llm_response: str, log_thinking: bool = True) -> ParsedResponse:
    """Parse LLM response and optionally log thinking process.

    Higher-level function that wraps parse_thinking_process with logging
    and returns a structured result.

    Args:
        llm_response: Raw LLM output
        log_thinking: Whether to log the thinking process (default: True)

    Returns:
        ParsedResponse with separated answer and thinking
    """
    thinking, answer = parse_thinking_process(llm_response)

    had_thinking = bool(thinking)

    if had_thinking and log_thinking:
        # Log at DEBUG level - useful for development/debugging
        # Truncate long thinking processes for log readability
        log_thinking_content = (
            thinking[:500] + "..." if len(thinking) > 500 else thinking
        )
        logger.debug(f"Model Thinking Process:\n{log_thinking_content}")

    return ParsedResponse(
        answer=answer,
        thinking=thinking if thinking else None,
        had_thinking_block=had_thinking,
    )


def strip_thinking_tags(text: str) -> str:
    """Quick utility to strip thinking tags without parsing content.

    Use when you just need the clean answer and don't care about
    the thinking content.

    Args:
        text: Text potentially containing <think> blocks

    Returns:
        Text with all <think>...</think> blocks removed
    """
    if not text:
        return ""

    patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
    ]

    result = text
    for pattern in patterns:
        result = re.sub(pattern, "", result, flags=re.DOTALL | re.IGNORECASE)

    return result.strip()


@dataclass
class CitationRemapResult:
    """Result of citation remapping."""

    text: str  # Text with renumbered citations
    citation_map: Dict[int, int]  # Original -> New number mapping
    used_indices: List[int]  # Original indices in order of first appearance


def renumber_citations(text: str) -> CitationRemapResult:
    """Renumber citations based on order of first appearance.

    Transforms citations like [1], [5], [3] to [1], [2], [3] based on
    the order they first appear in the text.

    Args:
        text: Text containing citations like [1], [2], etc.

    Returns:
        CitationRemapResult with remapped text and mapping info
    """
    if not text:
        return CitationRemapResult(text="", citation_map={}, used_indices=[])

    # Find all citations in order of appearance
    citation_pattern = r"\[(\d+)\]"
    matches = list(re.finditer(citation_pattern, text))

    if not matches:
        return CitationRemapResult(text=text, citation_map={}, used_indices=[])

    # Build mapping: original number -> new sequential number
    seen_originals: List[int] = []
    for match in matches:
        orig_num = int(match.group(1))
        if orig_num not in seen_originals:
            seen_originals.append(orig_num)

    # Create mapping: {original: new_sequential}
    citation_map = {orig: idx + 1 for idx, orig in enumerate(seen_originals)}

    # Replace citations in text
    def replace_citation(match):
        orig_num = int(match.group(1))
        new_num = citation_map.get(orig_num, orig_num)
        return f"[{new_num}]"

    remapped_text = re.sub(citation_pattern, replace_citation, text)

    return CitationRemapResult(
        text=remapped_text,
        citation_map=citation_map,
        used_indices=seen_originals,
    )
