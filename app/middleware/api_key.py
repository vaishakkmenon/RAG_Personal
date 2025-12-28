# app/middleware/api_key.py
from __future__ import annotations

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.settings import settings

import fnmatch

log = logging.getLogger(__name__)


class APIKeyMiddleware(BaseHTTPMiddleware):
    # Public paths that don't require API key authentication (Exact match)
    PUBLIC_PATHS = {
        "/health",
        "/health/detailed",
        "/health/ready",
        "/health/live",
        "/health/llm",
        "/metrics",
        "/docs",
        "/openapi.json",
        "/redoc",
        "/auth/token",
        "/auth/users/me",
        "/ingest",  # Protected by Admin Token
    }

    # Public path prefixes (Allow sub-paths)
    PUBLIC_PREFIXES = (
        "/admin",  # Protected by Admin Token
        "/debug",  # Protected by Admin Token
    )

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_PATHS or request.url.path.startswith(
            self.PUBLIC_PREFIXES
        ):
            return await call_next(request)

        # Allow OPTIONS requests (CORS preflight)
        if request.method == "OPTIONS":
            return await call_next(request)

        # Validate origin (primary defense)
        origin = request.headers.get("origin") or request.headers.get("referer", "")
        if origin:
            # Clean origin to just scheme + domain (remove trailing slash)
            origin_clean = origin.rstrip("/")

            # Check against allowed origins (supports wildcards)
            origin_valid = False
            for allowed in settings.api.cors_origins + [
                "https://deploy-preview-*.vaishakmenon.com"  # Explicit wildcard support
            ]:
                if fnmatch.fnmatch(origin_clean, allowed):
                    origin_valid = True
                    break

            if not origin_valid:
                log.warning(f"Invalid origin: {origin} for {request.url.path}")
                return JSONResponse(
                    status_code=403, content={"detail": "Origin not allowed"}
                )

        # Validate API key (secondary defense)
        valid_keys = settings.valid_api_keys
        if valid_keys:
            provided = request.headers.get("x-api-key")
            if not provided or provided not in valid_keys:
                log.error(
                    f"API Key authentication failed. Received: {provided[:4] if provided else 'None'}***"
                )
                return JSONResponse(
                    status_code=401, content={"detail": "Invalid or missing API key"}
                )

        return await call_next(request)
