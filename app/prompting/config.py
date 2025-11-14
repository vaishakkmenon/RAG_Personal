"""
Prompt Configuration - Settings and defaults

Handles configuration for prompt building and validation.
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class PromptConfig:
    """Configuration for prompt construction and validation."""

    min_question_length: int = 3
    max_context_length: int = 4000
    system_prompt: str = """You are an AI assistant that provides factual answers based on the provided context.

CRITICAL RULES:
1. Answer ONLY using information from the provided context.
2. If you CANNOT find the answer in the context, respond ONLY with: "I don't know. It isn't mentioned in the provided documents."
3. If you CAN answer from the context, provide ONLY the answer. DO NOT add "I don't know" or any refusal phrases.
4. NEVER mix an answer with a refusal. Choose one or the other, never both.

GUIDELINES:
1. Be concise and factual.
2. Use bullet points for lists.
3. Always include source references when available.
4. If a question is ambiguous or unclear, ask for clarification.
5. For numerical questions, provide exact figures from the context.
6. When answering about a specific item (e.g., a certification), focus ONLY on information about that exact item, even if other similar items are in the context.
7. For questions with multiple parts, address each part separately and ensure accuracy for each."""
    certification_guidelines: str = """CERTIFICATION RESPONSES:
- Prioritize the canonical certification name and issuer. If aliases appear in the question, clarify using the official title from context.
- When dates are available, state them concisely using both ISO (YYYY-MM-DD) and human-friendly formats, e.g. "Earned: 2024-06-26 (June 26, 2024)" and "Expires: 2028-05-26 (May 26, 2028)."
- Highlight current status when relevant and keep the answer tightly focused on the question.
- When multiple certifications are relevant, format the answer as a short bullet list (one bullet per certification).
- If no certification information is present in the context, respond with the standard refusal template."""

    # Common ambiguous phrases and patterns
    ambiguous_phrases: List[str] = field(
        default_factory=lambda: [
            "tell me about",
            "what about",
            "what do you know about",
            "background",
            "history",
            "experience",
            "skills",
            "explain",
            "how to",
            "help with",
            "what's the deal with",
        ]
    )

    # Question words that often indicate vagueness without specifics
    vague_question_words: List[str] = field(
        default_factory=lambda: ["how", "what", "why", "when", "where", "who"]
    )

    clarification_cues: tuple = (
        "clarify",
        "which specific",
        "could you clarify",
        "please specify",
        "what aspect",
    )

    refusal_cues: tuple = (
        "i don't know",
        "i do not know",
        "not mentioned",
        "couldn't find",
        "could not find",
        "cannot find",
        "not available",
        "no relevant information",
        "i'm not sure",
        "unable to find",
        "insufficient information",
    )


@dataclass
class PromptResult:
    """Result from prompt building."""

    status: str
    prompt: str = ""
    message: str = ""
    context: str = ""
