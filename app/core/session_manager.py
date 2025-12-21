"""
Session Manager - Handles session creation, retrieval, and history management.

Encapsulates logic for:
- Session retrieval via SessionStore
- Rate limiting checks
- History management and truncation
- IP address extraction
"""

import logging
from datetime import datetime
from typing import Optional, List, Dict
import uuid

from fastapi import HTTPException
from app.models import ChatRequest
from app.storage import get_session_store, Session
from app.storage.utils import mask_session_id

logger = logging.getLogger(__name__)


class SessionManager:
    """Manages chat sessions and related state."""

    def __init__(self, session_store=None):
        self.session_store = session_store or get_session_store()

    def get_or_create_session(self, request: ChatRequest) -> Session:
        """Get existing session or create a new one.

        Args:
            request: The chat request object

        Returns:
            Session object
        """
        try:
            session = self.session_store.get_or_create_session(
                session_id=request.session_id,
                ip_address=self._get_client_ip(request),
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to get/create session: {e}", exc_info=True)
            # Fallback for resiliency
            session = Session(
                session_id=str(uuid.uuid4()),
                created_at=datetime.now(),
                last_accessed=datetime.now(),
            )
        return session

    def update_session(self, session: Session) -> None:
        """Persist session updates (e.g., new turns)."""
        try:
            self.session_store.update_session(session)
            logger.debug(
                f"Updated session {mask_session_id(session.session_id)} "
                f"(turns: {len(session.history)})"
            )
        except Exception as e:
            logger.error(f"Failed to update session: {e}")

    def check_rate_limit(self, session: Session) -> bool:
        """Check if session is allowed to make a request.

        Returns:
            True if allowed, False if rate limited.
        """
        return self.session_store.check_rate_limit(session)

    def _get_client_ip(self, request: ChatRequest) -> Optional[str]:
        """Extract client IP from request.

        In Phase 7, this will be upgraded to check FastAPI Request headers.
        """
        # Placeholder for future implementation
        return None

    def truncate_history(
        self, history: List[Dict[str, str]], max_tokens: int, tokenizer_func
    ) -> List[Dict[str, str]]:
        """Truncate conversation history to fit within token limits.

        Args:
            history: List of message dicts
            max_tokens: Maximum tokens allowed for history
            tokenizer_func: Function to estimate/count tokens

        Returns:
            Truncated history list
        """
        current_text = "\n".join(
            [f"{t.get('role')}: {t.get('content')}" for t in history]
        )
        current_tokens = tokenizer_func(current_text)

        if current_tokens <= max_tokens:
            return history

        logger.info(f"Truncating history: {current_tokens} > {max_tokens}")

        truncated = list(history)
        while truncated and len(truncated) > 2:
            # Remove oldest pair
            truncated.pop(0)
            if truncated:
                truncated.pop(0)

            temp_text = "\n".join(
                [f"{t.get('role')}: {t.get('content')}" for t in truncated]
            )
            if tokenizer_func(temp_text) <= max_tokens:
                break

        return truncated
