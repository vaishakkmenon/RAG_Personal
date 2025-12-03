"""
FALLBACK Implementation: In-Memory Session Store

This package contains the high-availability fallback implementation.
Automatically activated when:
- Redis connection fails
- Redis URL not configured
- Redis library not installed
- Explicitly requested for development/testing

See memory.py for implementation details.
"""

from .memory import InMemorySessionStore

__all__ = ["InMemorySessionStore"]
