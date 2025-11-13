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

GUIDELINES:
1. Answer ONLY using information from the provided context.
2. If the context doesn't contain the answer, say "I don't know. It isn't mentioned in the provided documents."
3. When the context DOES contain the answer, respond directly without appending any refusal language.
4. Be concise and factual.
5. Use bullet points for lists.
6. Always include source references when available.
7. If a question is ambiguous or unclear, ask for clarification.
8. For numerical questions, provide exact figures from the context."""
    certification_guidelines: str = """CERTIFICATION RESPONSES:
- Prioritize the canonical certification name and issuer. If aliases appear in the question, clarify using the official title from context.
- When dates are available, state them concisely using both ISO (YYYY-MM-DD) and human-friendly formats, e.g. "Earned: 2024-06-26 (June 26, 2024)" and "Expires: 2028-05-26 (May 26, 2028)."
- Highlight current status when relevant and keep the answer tightly focused on the question.
- When multiple certifications are relevant, format the answer as a short bullet list (one bullet per certification).
- If no certification information is present in the context, respond with the standard refusal template."""
    use_certification_examples: bool = False
    certification_examples: List[Tuple[str, str]] = field(
        default_factory=lambda: [
            (
                "Do you hold the AWS CCP certification?",
                "- **AWS Certified Cloud Practitioner** (Amazon Web Services) — Earned: 2024-05-26 (May 26, 2024); Expires: 2028-05-26 (May 26, 2028); Status: Active.\n  Source: AWS Cloud Practitioner profile.",
            ),
            (
                "When did you earn your CKA certification?",
                "You earned the **Certified Kubernetes Administrator (CKA)** certification on 2024-06-26 (June 26, 2024).\nSource: Linux Foundation CKA certificate.",
            ),
            (
                "When does your AWS Cloud Practitioner certification expire?",
                "Your **AWS Certified Cloud Practitioner** credential (Amazon Web Services) expires on 2028-05-26 (May 26, 2028).\nStatus: Active.\nSource: AWS Cloud Practitioner profile.",
            ),
        ]
    )

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
