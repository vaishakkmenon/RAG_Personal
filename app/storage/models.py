"""
Session data model.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from app.settings import settings
from app.storage.utils import estimate_tokens, truncate_history


@dataclass
class Session:
    """Session object holding conversation history and metadata."""

    session_id: str
    created_at: datetime
    last_accessed: datetime
    history: List[Dict[str, str]] = field(default_factory=list)

    # Security tracking
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_count: int = 0
    request_timestamps: List[datetime] = field(default_factory=list)

    # Resource tracking
    total_tokens_used: int = 0

    def add_turn(self, role: str, content: str) -> None:
        """Add a message to conversation history.

        Args:
            role: 'user' or 'assistant'
            content: Message content
        """
        self.history.append({"role": role, "content": content})
        self.total_tokens_used += estimate_tokens(content)

    def get_truncated_history(
        self, max_tokens: Optional[int] = None, max_turns: Optional[int] = None
    ) -> List[Dict[str, str]]:
        """Get conversation history truncated to fit budget.

        Args:
            max_tokens: Maximum tokens (default from settings)
            max_turns: Maximum turns (default from settings)

        Returns:
            Truncated history
        """
        max_tokens = max_tokens or settings.session.max_history_tokens
        max_turns = max_turns or settings.session.max_history_turns

        return truncate_history(self.history, max_tokens, max_turns)

    def record_request(self) -> None:
        """Record a new request (for rate limiting and tracking)."""
        self.request_count += 1
        self.request_timestamps.append(datetime.now())
        self.last_accessed = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary (for Redis storage)."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "history": self.history,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "request_count": self.request_count,
            "request_timestamps": [ts.isoformat() for ts in self.request_timestamps],
            "total_tokens_used": self.total_tokens_used,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Session":
        """Deserialize session from dictionary (for Redis storage)."""
        return cls(
            session_id=data["session_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            history=data.get("history", []),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            request_count=data.get("request_count", 0),
            request_timestamps=[
                datetime.fromisoformat(ts) for ts in data.get("request_timestamps", [])
            ],
            total_tokens_used=data.get("total_tokens_used", 0),
        )
