"""
PRIMARY Implementation: Redis Session Store

This package contains the production-ready Redis implementation.
Use this backend for:
- Production deployments
- Multi-instance applications
- Persistent session storage
- Distributed systems

See redis_store.py for implementation details.
"""

from .redis_store import RedisSessionStore

__all__ = ["RedisSessionStore"]
