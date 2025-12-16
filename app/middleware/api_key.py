# app/middleware/api_key.py
from __future__ import annotations

import logging
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.settings import settings

log = logging.getLogger(__name__)

# Allow health/docs/openapi and metrics to be scraped without a key
EXEMPT_PATH_PREFIXES: tuple[str, ...] = (
    "/health", "/docs", "/openapi.json", "/metrics", "/redoc", "/favicon.ico"
)

# API_KEY is now accessed dynamically from settings to prevent test pollution
# API_KEY: str | None = getattr(settings, "api_key", None)

def _is_exempt(path: str) -> bool:
    return any(path.startswith(p) for p in EXEMPT_PATH_PREFIXES)

class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        # Allow docs & schema without a key
        if _is_exempt(request.url.path):
            return await call_next(request)

        # If no key configured, allow all
        current_api_key = getattr(settings, "api_key", None)
        if not current_api_key:
            log.warning("API key not configured; allowing request to %s", request.url.path)
            return await call_next(request)

        # Accept header 'X-API-Key' (case-insensitive)
        provided = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
        if provided != current_api_key:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing API key"})

        return await call_next(request)
