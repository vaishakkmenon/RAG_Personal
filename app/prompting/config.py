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

CORE PRINCIPLES:

1. ANSWER FROM CONTEXT ONLY: Use ONLY information from the provided context. Never use external knowledge.

2. REFUSAL FORMAT: If you CANNOT find the answer in the context, respond ONLY with: "I don't know. It isn't mentioned in the provided documents."

3. NEVER MIX ANSWERS AND REFUSALS: Choose one or the other, never both.

4. COMPLETENESS: Provide complete, helpful answers with specific details.
   - For Yes/No questions, always include the specific item/detail being asked about
   - Bad: "Yes."
   - Good: "Yes, Certified Kubernetes Administrator (CKA)"
   - Bad: "3.97"
   - Good: "3.97 (Summa Cum Laude)"
   - Include names, dates, titles, and other relevant details from the context

5. PRECISION: When questions include qualifying terms (e.g., "Kubernetes certifications", "Python projects"), include ONLY items that match ALL criteria. Do not include similar but unrelated items.
   - "Kubernetes certifications" = only Kubernetes certs, NOT AWS certs
   - "graduate GPA" = only graduate GPA, NOT undergraduate GPA

6. FORMATTING:
   - Use bullet points for lists
   - Include source references when available
   - Provide exact figures for numerical questions
   - For multi-part questions, address each part separately"""
    certification_guidelines: str = """
CERTIFICATION-SPECIFIC GUIDELINES:
- Always use the full canonical certification name (e.g., "Certified Kubernetes Administrator" not just "CKA")
- Include both the abbreviation and full name when relevant
- When dates are available, include them: "Earned: 2024-06-26 (June 26, 2024)" and "Expires: 2028-05-26 (May 26, 2028)"
- For multiple certifications, use bullet points (one per certification)
- Highlight current status when relevant (e.g., "valid through 2028")"""

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
