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


class StreamingMarkdownStripper:
    """Strips markdown formatting from streaming text.

    Removes:
    - **bold** and __bold__ → bold
    - *italic* and _italic_ → italic
    - ## headers → headers
    - Preserves [N] citations

    Uses buffering to handle split tokens like * arriving separately.
    """

    def __init__(self):
        self._buffer = ""

    def process(self, token: str) -> str:
        """Process token and return cleaned text."""
        self._buffer += token
        return self._extract_clean_output()

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        result = self._strip_markdown(self._buffer)
        self._buffer = ""
        return result

    def _extract_clean_output(self) -> str:
        """Extract text that's safe to emit (complete markdown patterns stripped)."""
        # Check for incomplete patterns at the end that need more tokens
        # We buffer if we see potential start of markdown that isn't complete yet

        # Check for trailing * or _ that might be start of formatting
        if self._buffer.endswith("*") or self._buffer.endswith("_"):
            # Could be start of bold/italic - wait for more
            if len(self._buffer) > 1:
                safe = self._buffer[:-1]
                self._buffer = self._buffer[-1:]
                return self._strip_markdown(safe)
            return ""

        # Check for trailing # that might be header
        if self._buffer.endswith("#"):
            if len(self._buffer) > 1:
                safe = self._buffer[:-1]
                self._buffer = self._buffer[-1:]
                return self._strip_markdown(safe)
            return ""

        # No incomplete patterns - process everything
        result = self._strip_markdown(self._buffer)
        self._buffer = ""
        return result

    def _strip_markdown(self, text: str) -> str:
        """Strip markdown formatting from text."""
        if not text:
            return text

        # Remove **bold** and __bold__
        text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
        text = re.sub(r"__([^_]+)__", r"\1", text)

        # Remove *italic* and _italic_ (but not inside words)
        # Be careful not to match _in_variable_names
        text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
        text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)

        # Remove ## headers at start of lines
        text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

        # Remove - bullets at start of lines, but keep the content
        # Convert "- item" to just "item" (let numbered lists remain)
        text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)

        return text


class StreamingCitationRemapper:
    """Remaps citations to sequential order during streaming.

    Buffers minimally to detect [N] patterns and remaps before sending to client.
    Client never sees out-of-order citations.

    Based on token analysis:
    - " [" or "[" - opening bracket (possibly with leading space)
    - "1", "3" - the digit(s)
    - "]." or "]" - closing bracket (possibly with trailing punctuation)

    Usage:
        remapper = StreamingCitationRemapper()
        for token in llm_stream:
            output = remapper.process(token)
            if output:
                yield output
        final = remapper.flush()
        if final:
            yield final
        used_sources = remapper.get_used_sources()  # Original indices in order
    """

    def __init__(self):
        self._buffer = ""
        self._used_sources: List[int] = []  # Original source numbers in citation order
        self._source_map: Dict[int, int] = {}  # Original -> Sequential

    def process(self, token: str) -> str:
        """Process token, return remapped text ready for client."""
        self._buffer += token
        return self._extract_safe_output()

    def flush(self) -> str:
        """Flush remaining buffer at end of stream."""
        # Process any complete citations in buffer
        result = self._remap_complete_citations(self._buffer)
        self._buffer = ""
        return result

    def get_used_sources(self) -> List[int]:
        """Get original source indices in order of first citation."""
        return self._used_sources.copy()

    def get_source_map(self) -> Dict[int, int]:
        """Get mapping of original -> sequential citation numbers."""
        return self._source_map.copy()

    def _extract_safe_output(self) -> str:
        """Extract text that's safe to emit (complete citations remapped)."""
        # Find the last '[' that might be start of incomplete citation
        last_bracket = self._buffer.rfind("[")

        if last_bracket == -1:
            # No bracket - emit everything
            result = self._buffer
            self._buffer = ""
            return result

        # Check if there's a ']' after the last '['
        close_after = self._buffer.find("]", last_bracket)

        if close_after != -1:
            # Citation is complete - can emit everything
            result = self._remap_complete_citations(self._buffer)
            self._buffer = ""
            return result

        # Incomplete citation - emit up to '[', keep rest buffered
        safe_part = self._buffer[:last_bracket]
        self._buffer = self._buffer[last_bracket:]
        return self._remap_complete_citations(safe_part)

    def _remap_complete_citations(self, text: str) -> str:
        """Remap all complete citations in text to sequential order."""
        if not text:
            return text

        def replace_citation(match):
            orig_num = int(match.group(1))

            if orig_num not in self._source_map:
                # First time seeing this source - assign next sequential number
                seq_num = len(self._used_sources) + 1
                self._used_sources.append(orig_num)
                self._source_map[orig_num] = seq_num

            return f"[{self._source_map[orig_num]}]"

        return re.sub(r"\[(\d+)\]", replace_citation, text)


def strip_markdown(text: str) -> str:
    """Strip markdown formatting from text (non-streaming version)."""
    if not text:
        return text

    # Remove **bold** and __bold__
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"__([^_]+)__", r"\1", text)

    # Remove *italic* and _italic_ (but not inside words)
    text = re.sub(r"(?<!\w)\*([^*]+)\*(?!\w)", r"\1", text)
    text = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", text)

    # Remove ## headers at start of lines
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)

    # Remove - bullets at start of lines
    text = re.sub(r"^\s*-\s+", "", text, flags=re.MULTILINE)

    return text


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
