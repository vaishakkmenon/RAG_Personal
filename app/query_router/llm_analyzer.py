"""
LLM-based ambiguity analyzer for query routing.

Uses Groq's llama-3.1-8b-instant model to detect ambiguous queries that need clarification.
Provides structured output with confidence scores and suggested clarifications.
"""

import logging
import time
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

from ..services.llm import async_generate_with_ollama

logger = logging.getLogger(__name__)


class AmbiguityAnalysis(BaseModel):
    """Structured output from LLM ambiguity analysis."""

    is_ambiguous: bool = Field(
        description="Whether the query is too ambiguous to answer without clarification"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence score for the ambiguity decision (0.0-1.0)"
    )
    reason: str = Field(
        description="Brief explanation of why the query is or isn't ambiguous"
    )
    suggested_clarification: Optional[str] = Field(
        default=None,
        description="Suggested question to ask the user for clarification (if ambiguous)"
    )
    detected_domains: List[str] = Field(
        default_factory=list,
        description="List of detected domains: education, work, certifications, skills"
    )


# Prompt template for ambiguity detection
AMBIGUITY_DETECTION_PROMPT = """You are a query classifier. Determine if this question is too ambiguous to answer without additional context.

A question is AMBIGUOUS if:
- It uses vague terms without specifics (e.g., "What about my background?")
- It could refer to multiple different topics without clear scope
- The user needs to specify which aspect they're interested in

A question is CLEAR if:
- It asks for specific information (e.g., "What was my graduate GPA?")
- It explicitly mentions multiple domains with clear intent (e.g., "Summarize my education and work experience")
- It mentions specific topics/technologies (e.g., "Tell me about Kubernetes certifications")

Examples:

AMBIGUOUS:
- "What about my background?" → Could mean education, work, skills, or qualifications
- "Tell me about my history" → Unclear if academic history, work history, or other
- "Looking for information" → No specific topic mentioned
- "What do I know?" → Too broad, needs domain specification

CLEAR:
- "What was my graduate GPA?" → Specific academic metric
- "Summarize my education and work experience" → Two explicit domains
- "Which Kubernetes certifications do I have?" → Specific topic (Kubernetes) and type (certifications)
- "List my technical skills" → Specific domain (skills)

Question: "{question}"

Analyze this question and respond in the following format:

AMBIGUOUS: [YES or NO]
CONFIDENCE: [0.0-1.0]
REASON: [brief explanation in one sentence]
DOMAINS: [comma-separated list of detected domains from: education, work, certifications, skills]
CLARIFICATION: [if ambiguous, suggest what to ask the user; otherwise write "N/A"]

Example response:
AMBIGUOUS: YES
CONFIDENCE: 0.9
REASON: The query uses "background" without specifying education, work, or skills.
DOMAINS: education, work
CLARIFICATION: Which aspect of your background would you like to know about: education, work experience, certifications, or skills?
"""


def parse_llm_response(response: str) -> AmbiguityAnalysis:
    """Parse LLM response into structured AmbiguityAnalysis.

    Args:
        response: Raw text response from LLM

    Returns:
        Structured AmbiguityAnalysis object

    Raises:
        ValueError: If response cannot be parsed
    """
    lines = response.strip().split('\n')
    parsed = {}

    for line in lines:
        line = line.strip()
        if ':' not in line:
            continue

        key, value = line.split(':', 1)
        key = key.strip().upper()
        value = value.strip()

        if key == 'AMBIGUOUS':
            parsed['is_ambiguous'] = value.upper() in ['YES', 'TRUE', '1']
        elif key == 'CONFIDENCE':
            try:
                parsed['confidence'] = float(value)
            except ValueError:
                logger.warning(f"Failed to parse confidence: {value}")
                parsed['confidence'] = 0.5
        elif key == 'REASON':
            parsed['reason'] = value
        elif key == 'DOMAINS':
            # Parse comma-separated domains
            domains = [d.strip().lower() for d in value.split(',') if d.strip()]
            # Filter to valid domains
            valid_domains = ['education', 'work', 'certifications', 'skills']
            parsed['detected_domains'] = [d for d in domains if d in valid_domains]
        elif key == 'CLARIFICATION':
            if value.upper() not in ['N/A', 'NA', 'NONE', '']:
                parsed['suggested_clarification'] = value

    # Validate required fields
    if 'is_ambiguous' not in parsed:
        raise ValueError("Missing AMBIGUOUS field in response")
    if 'confidence' not in parsed:
        parsed['confidence'] = 0.5
    if 'reason' not in parsed:
        parsed['reason'] = "No reason provided"

    return AmbiguityAnalysis(**parsed)


async def analyze_ambiguity_with_llm(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0
) -> AmbiguityAnalysis:
    """Analyze query ambiguity using LLM.

    Args:
        question: The user's question to analyze
        context: Optional context from query analysis (technologies, categories, etc.)
        timeout: Maximum time to wait for LLM response (seconds)

    Returns:
        AmbiguityAnalysis with structured decision and metadata

    Raises:
        Exception: If LLM call fails or times out
    """
    start_time = time.time()

    try:
        # Format prompt with question
        prompt = AMBIGUITY_DETECTION_PROMPT.format(question=question)

        logger.debug(f"Analyzing ambiguity for: {question[:50]}...")

        # Generate response asynchronously with timeout
        response_text = await async_generate_with_ollama(
            prompt=prompt,
            temperature=0.0,  # Deterministic for classification
            max_tokens=150,  # Short structured response
        )

        # Parse structured response
        analysis = parse_llm_response(response_text)

        duration = time.time() - start_time
        logger.info(
            f"LLM ambiguity analysis completed in {duration:.2f}s: "
            f"ambiguous={analysis.is_ambiguous}, confidence={analysis.confidence:.2f}"
        )

        return analysis

    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"LLM ambiguity analysis failed after {duration:.2f}s: {e}")
        raise


# Synchronous wrapper for backward compatibility
def analyze_ambiguity_with_llm_sync(
    question: str,
    context: Optional[Dict[str, Any]] = None,
    timeout: float = 5.0
) -> AmbiguityAnalysis:
    """Synchronous wrapper for LLM ambiguity analysis.

    Args:
        question: The user's question to analyze
        context: Optional context from query analysis
        timeout: Maximum time to wait for LLM response

    Returns:
        AmbiguityAnalysis with structured decision
    """
    # For now, directly call the function (will use asyncio.run when async is added)
    import asyncio
    try:
        # Try to get the running event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're already in an async context, can't use asyncio.run
            # Fall back to sync implementation for now
            logger.warning("Already in async context, using direct call")
            return asyncio.create_task(analyze_ambiguity_with_llm(question, context, timeout))
        else:
            return asyncio.run(analyze_ambiguity_with_llm(question, context, timeout))
    except RuntimeError:
        # No event loop, use asyncio.run
        return asyncio.run(analyze_ambiguity_with_llm(question, context, timeout))


__all__ = [
    "AmbiguityAnalysis",
    "analyze_ambiguity_with_llm",
    "analyze_ambiguity_with_llm_sync",
    "parse_llm_response",
]
