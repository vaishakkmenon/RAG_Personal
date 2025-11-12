"""
API dependencies for Personal RAG system.

Contains FastAPI dependencies for authentication, services, etc.
"""

import logging
from typing import TYPE_CHECKING

from fastapi import Header, HTTPException, status

from ..certifications import get_registry
from ..core import ChatService
from ..settings import settings

if TYPE_CHECKING:  # pragma: no cover - import for typing only
    from ..certifications import CertificationRegistry

logger = logging.getLogger(__name__)

# Initialize services at module level for better performance
# This ensures they're created once at startup, not on every request
_cert_registry = get_registry()
_chat_service = ChatService(cert_registry=_cert_registry)

logger.info("Initialized chat service and certification registry at module load")


def check_api_key(x_api_key: str = Header(...)) -> str:
    """Verify the API key from the X-API-Key header.

    Args:
        x_api_key: Value from X-API-Key header (automatically extracted by FastAPI)

    Returns:
        The API key if valid

    Raises:
        HTTPException: If API key is missing or invalid
    """
    expected_key = settings.api_key  # From .env file

    if x_api_key != expected_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
        )

    return x_api_key


def get_chat_service() -> ChatService:
    """Get the global chat service instance.

    Returns:
        ChatService instance
    """
    return _chat_service


__all__ = ["check_api_key", "get_chat_service"]
