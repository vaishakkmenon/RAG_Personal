from fastapi import Request
from fastapi.responses import JSONResponse
import logging

logger = logging.getLogger(__name__)


class RAGException(Exception):
    """Base exception for RAG system"""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMException(RAGException):
    """LLM API errors"""

    def __init__(self, message: str):
        super().__init__(message, status_code=503)


class RetrievalException(RAGException):
    """Vector search errors"""

    def __init__(self, message: str):
        super().__init__(message, status_code=500)


class RateLimitException(RAGException):
    """Rate limit exceeded"""

    def __init__(self, message: str = "Rate limit exceeded"):
        super().__init__(message, status_code=429)


async def rag_exception_handler(request: Request, exc: RAGException):
    """Handle custom RAG exceptions"""
    logger.error(
        f"RAG Exception: {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "client_ip": request.client.host,
        },
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "detail": exc.message,
            "type": exc.__class__.__name__,
        },
    )


async def generic_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions"""
    logger.exception(
        f"Unexpected error: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Don't leak implementation details
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An internal error occurred. Please try again later.",
        },
    )
