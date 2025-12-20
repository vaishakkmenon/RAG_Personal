import uuid
import logging
import time
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class RequestTracingMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add a unique request ID to every request.
    Useful for distributed tracing and log correlation.
    """

    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        # Generate or get request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

        # Store in request state for access in endpoints
        request.state.request_id = request_id

        # Add to logger context (if supported by logging setup, here we just log it)
        # Note: Ideally this would set a contextvar for structured logging

        start_time = time.time()

        # Process request
        response = await call_next(request)

        # Calculate duration
        (time.time() - start_time) * 1000

        # Add header to response
        response.headers["X-Request-ID"] = request_id

        # Log completion (optional: move to LoggingMiddleware for centralized logging)
        # logger.info(f"Request {request_id} completed in {duration_ms:.2f}ms")

        return response
