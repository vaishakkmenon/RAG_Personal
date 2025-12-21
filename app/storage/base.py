"""
Abstract base class for session storage.
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import List, Optional

from app.settings import settings
from app.storage.models import Session
from app.storage.utils import mask_session_id

logger = logging.getLogger(__name__)

# Import session metrics
try:
    from app.metrics import rag_rate_limit_violations_total

    RATE_LIMIT_METRICS_ENABLED = True
except ImportError:
    RATE_LIMIT_METRICS_ENABLED = False


class SessionStore(ABC):
    """Abstract base class for session storage backends."""

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Retrieve session by ID."""
        pass

    @abstractmethod
    def create_session(
        self,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Create new session."""
        pass

    @abstractmethod
    def update_session(self, session: Session) -> None:
        """Update existing session."""
        pass

    @abstractmethod
    def delete_session(self, session_id: str) -> None:
        """Delete session."""
        pass

    def get_or_create_session(
        self,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
    ) -> Session:
        """Get existing session or create new one.

        Args:
            session_id: Existing session ID (optional)
            ip_address: Client IP address
            user_agent: Client user agent string

        Returns:
            Session object

        Raises:
            HTTPException: If session/rate limits exceeded
        """
        from fastapi import HTTPException

        # Check global session limit before creating
        if session_id is None:
            total_sessions = self.get_session_count()
            if total_sessions >= settings.session.max_total_sessions:
                logger.error(f"Max total sessions reached: {total_sessions}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Server capacity reached. Maximum {settings.session.max_total_sessions} active sessions.",
                )

        # Get or create session
        if session_id:
            session = self.get_session(session_id)
            if session:
                session.record_request()
                self.update_session(session)  # Persist updated timestamps to Redis
                return session

        # Create new session with IP limit check
        if ip_address:
            ip_sessions = self.get_sessions_by_ip(ip_address)
            if len(ip_sessions) >= settings.session.max_sessions_per_ip:
                logger.warning(f"Max sessions per IP reached for {ip_address}")
                raise HTTPException(
                    status_code=429,
                    detail=f"Too many sessions from your IP. Maximum {settings.session.max_sessions_per_ip} sessions per IP.",
                )

        session = self.create_session(
            session_id=session_id, ip_address=ip_address, user_agent=user_agent
        )
        session.record_request()
        self.update_session(session)  # Persist initial request timestamp
        return session

    def check_rate_limit(self, session: Session) -> bool:
        """Check if session is within rate limit.

        Uses sliding window: counts requests in last hour.

        Args:
            session: Session to check

        Returns:
            True if within limit, False if exceeded
        """
        if settings.session.queries_per_hour <= 0:
            return True  # Rate limiting disabled

        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)

        # Count requests in last hour
        recent_requests = [ts for ts in session.request_timestamps if ts > one_hour_ago]

        within_limit = len(recent_requests) <= settings.session.queries_per_hour

        if not within_limit:
            logger.warning(
                f"Rate limit exceeded for session {mask_session_id(session.session_id)}: "
                f"{len(recent_requests)} requests in last hour"
            )
            # Track rate limit violation
            if RATE_LIMIT_METRICS_ENABLED:
                rag_rate_limit_violations_total.labels(
                    limit_type="queries_per_hour"
                ).inc()

        return within_limit

    @abstractmethod
    def get_session_count(self) -> int:
        """Get total number of active sessions."""
        pass

    @abstractmethod
    def get_sessions_by_ip(self, ip_address: str) -> List[Session]:
        """Get all sessions for an IP address."""
        pass
