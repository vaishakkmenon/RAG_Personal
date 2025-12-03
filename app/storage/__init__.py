"""
Storage module for session management.

Architecture:
- primary/: Production-ready Redis implementation
- fallback/: High-availability in-memory implementation
- factory.py: Automatic backend selection with fallback
"""

from .base import SessionStore
from .factory import create_session_store, get_session_store
from .fallback import InMemorySessionStore
from .models import Session
from .primary import RedisSessionStore
from .utils import estimate_tokens, mask_session_id, truncate_history

__all__ = [
    "Session",
    "SessionStore",
    "InMemorySessionStore",
    "RedisSessionStore",
    "create_session_store",
    "get_session_store",
    "estimate_tokens",
    "truncate_history",
    "mask_session_id",
]
