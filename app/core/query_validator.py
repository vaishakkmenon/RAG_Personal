"""
Query Validator - Handles input validation, safety checks, and intent detection.

Encapsulates logic for:
- Query normalization
- Prompt Guard safety checks
- Chitchat detection
- Ambiguity detection
"""

import logging
from typing import List, Dict, Tuple, Optional

from app.settings import settings
from app.services.prompt_guard import get_prompt_guard

logger = logging.getLogger(__name__)


class QueryValidator:
    """Validates and analyzes user queries."""

    def __init__(self):
        pass

    def normalize_query(self, query: str) -> str:
        """
        Normalize query for consistent caching and processing.

        - Lowercase
        - Strip whitespace
        - Remove trailing punctuation
        """
        normalized = query.lower().strip()
        return normalized.rstrip("?!.")

    def check_safety(self, query: str, history: List[Dict[str, str]]) -> Dict:
        """
        Check if the input is safe using Prompt Guard.

        Returns:
            Dict with 'blocked' (bool) and details.
        """
        if not settings.prompt_guard.enabled:
            return {"blocked": False}

        guard = get_prompt_guard()
        return guard.check_input(
            user_input=query,
            conversation_history=history,
        )

    def detect_chitchat(self, query: str) -> Tuple[bool, str]:
        """
        Detect conversational/social interactions that don't need retrieval.

        Returns:
            Tuple of (is_chitchat, response_message)
        """
        q = query.strip().lower()

        # Greetings
        greetings = {
            "hello",
            "hi",
            "hey",
            "good morning",
            "good afternoon",
            "good evening",
            "howdy",
            "greetings",
        }
        if any(
            greeting == q
            or q.startswith(greeting + " ")
            or q.startswith(greeting + "!")
            for greeting in greetings
        ):
            return (
                True,
                "Hello! I can help answer questions about your background, experience, certifications, and education. What would you like to know?",
            )

        # Gratitude
        if any(
            word in q
            for word in ["thank you", "thanks", "thx", "appreciate", "grateful"]
        ):
            return True, "You're welcome! Is there anything else you'd like to know?"

        # Farewells
        farewells = {"bye", "goodbye", "see you", "later", "farewell"}
        if any(farewell in q for farewell in farewells):
            return True, "Goodbye! Feel free to come back if you have more questions."

        return False, ""

    def detect_ambiguity(
        self, query: str, history: Optional[List[Dict[str, str]]] = None
    ) -> bool:
        """
        Detect truly vague questions using simple rules.
        """
        q = query.strip()

        # Empty or just punctuation
        if not q or len(q) <= 2:
            return True

        # Remove punctuation and count words
        words = q.replace("?", "").replace(".", "").replace("!", "").strip().split()

        # With conversation context, allow very short follow-ups
        if history and len(history) > 0:
            if len(words) >= 1 and len(q) > 2:
                return False  # Trust context

        # Without context: only flag truly minimal queries
        if len(words) <= 1:
            return True

        # Let the system prompt (Rule 11) handle everything else
        return False
