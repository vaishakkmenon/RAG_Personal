"""
Storage module for session management.

Architecture:
- primary/: Production-ready Redis implementation
- fallback/: High-availability in-memory implementation
- factory.py: Automatic backend selection with fallback
"""

from app.storage.base import SessionStore
from app.storage.factory import create_session_store, get_session_store
from app.storage.fallback import InMemorySessionStore
from app.storage.models import Session
from app.storage.primary import RedisSessionStore
from app.storage.utils import estimate_tokens, mask_session_id, truncate_history

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
