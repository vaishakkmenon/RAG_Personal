# app/middleware/api_key.py
from __future__ import annotations

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.settings import settings

import fnmatch
from app.settings import settings

log = logging.getLogger(__name__)

class APIKeyMiddleware(BaseHTTPMiddleware):
    # Public paths that don't require API key authentication
    PUBLIC_PATHS = {"/health", "/health/detailed", "/health/ready", "/health/live", "/metrics", "/docs", "/openapi.json", "/redoc"}

    async def dispatch(self, request: Request, call_next) -> Response:
        # Skip authentication for public endpoints
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Validate origin (primary defense)
        origin = request.headers.get("origin") or request.headers.get("referer", "")
        if origin:
            # Clean origin to just scheme + domain (remove trailing slash)
            origin_clean = origin.rstrip("/")
            
            # Check against allowed origins (supports wildcards)
            origin_valid = False
            for allowed in settings.api.cors_origins + [
                "https://deploy-preview-*.vaishakmenon.com" # Explicit wildcard support
            ]:
                if fnmatch.fnmatch(origin_clean, allowed):
                    origin_valid = True
                    break
            
            if not origin_valid:
                log.warning(f"Invalid origin: {origin} for {request.url.path}")
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Origin not allowed"}
                )

        # Validate API key (secondary defense)
        api_key = getattr(settings, "api_key", None)
        if api_key:
            provided = request.headers.get("x-api-key")
            if provided != api_key:
                log.error(f"API Key mismatch! Expected: {api_key[:4]}***, Received: {provided[:4] if provided else 'None'}***")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"}
                )

        return await call_next(request)
