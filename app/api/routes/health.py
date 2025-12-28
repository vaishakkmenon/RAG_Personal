"""
Health check endpoint for Personal RAG system.
"""

import socket
import logging
from typing import Dict, Any

from fastapi import APIRouter, status
from app.settings import settings

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Basic health check",
    description="""
    Quick health check endpoint that returns system status and configuration.

    **Use Cases:**
    - Verify API is running
    - Check LLM provider configuration
    - Monitor system availability

    **Response Time:** <10ms

    **Example Response:**
    ```json
    {
      "status": "healthy",
      "provider": "groq",
      "model": "llama-3.1-8b-instant",
      "socket": "rag-api-container"
    }
    ```
    """,
    responses={
        200: {
            "description": "System is healthy",
            "content": {
                "application/json": {
                    "example": {
                        "status": "healthy",
                        "provider": "groq",
                        "model": "llama-3.1-8b-instant",
                        "socket": "hostname",
                    }
                }
            },
        }
    },
)
async def health() -> Dict[str, Any]:
    """Returns basic system health status."""
    # Get model based on provider
    if settings.llm.provider == "groq":
        model = settings.llm.groq_model
    else:
        model = "unknown"

    return {
        "status": "healthy",
        "provider": settings.llm.provider,
        "model": model,
        "socket": socket.gethostname(),
    }


@router.get("/health/detailed", status_code=status.HTTP_200_OK)
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with dependency status"""
    health_status = {"status": "healthy", "dependencies": {}}

    # Check Redis
    try:
        from app.services.response_cache import get_response_cache

        # Note: Depending on how your cache is implemented, you might need a different way to ping
        # Assuming get_response_cache returns an object with a redis client or similar
        cache = get_response_cache()
        # Verify if 'ping' is available on the client or the cache wrapper
        # If cache is our RedisResponseCache, it has ._redis
        if hasattr(cache, "_redis"):
            cache._redis.ping()
        else:
            # Fallback if interface is different, just check instantiation
            pass
        health_status["dependencies"]["redis"] = "healthy"
    except Exception as e:
        logger.warning(f"Redis health check failed: {e}")
        health_status["dependencies"]["redis"] = "degraded"
        health_status["status"] = "degraded"

    # Check ChromaDB
    try:
        from app.retrieval.vector_store import get_vector_store

        store = get_vector_store()
        # Try to heartbeat or list collections to verify connectivity
        store.heartbeat()
        health_status["dependencies"]["chromadb"] = "healthy"
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        health_status["dependencies"]["chromadb"] = "degraded"
        health_status["status"] = "degraded"

    # Check Postgres
    try:
        from app.database import check_db_connectivity

        if check_db_connectivity():
            health_status["dependencies"]["postgres"] = "healthy"
        else:
            health_status["dependencies"]["postgres"] = "unhealthy"
            health_status["status"] = "degraded"
    except Exception as e:
        logger.warning(f"Postgres health check failed: {e}")
        health_status["dependencies"]["postgres"] = "unknown"
        health_status["status"] = "degraded"

    # Check LLM (optional - skipped to avoid cost/latency)
    health_status["dependencies"]["llm"] = "not_checked"

    return health_status


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, str]:
    """Kubernetes-style readiness probe"""
    # Check if app can serve requests (e.g. DB is accessible)
    try:
        from app.retrieval.vector_store import get_vector_store

        store = get_vector_store()
        # Simple verification
        store.heartbeat()
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """Kubernetes-style liveness probe"""
    # Just check if process is alive
    return {"status": "alive"}


@router.get(
    "/health/llm",
    status_code=status.HTTP_200_OK,
    summary="LLM provider health ping",
    description="""
    Ping the LLM provider with a minimal 1-token request to check availability.

    **Use Cases:**
    - Check if primary provider (DeepInfra) is available after 429 errors
    - Frontend can poll this to know when to retry with primary provider
    - Detect cold vs hot model state

    **Response Fields:**
    - `status`: "available", "busy", or "error"
    - `provider`: Primary provider name
    - `model`: Model being used
    - `response_time_ms`: Time for the ping request
    - `is_hot`: True if response was fast (<3s), indicating model is loaded
    - `fallback_available`: Whether fallback provider is configured
    - `fallback_provider`: Name of fallback provider (if configured)

    **Cost:** ~$0.00001 per ping (1 token)

    **Example Response (available):**
    ```json
    {
      "status": "available",
      "provider": "deepinfra",
      "model": "Qwen/Qwen3-32B",
      "response_time_ms": 245,
      "is_hot": true,
      "fallback_available": true,
      "fallback_provider": "groq"
    }
    ```

    **Example Response (busy):**
    ```json
    {
      "status": "busy",
      "provider": "deepinfra",
      "model": "Qwen/Qwen3-32B",
      "error": "429: Model is currently at capacity",
      "fallback_available": true,
      "fallback_provider": "groq"
    }
    ```
    """,
)
async def llm_health_ping() -> Dict[str, Any]:
    """Ping LLM provider with minimal 1-token request to check availability."""
    import time
    from app.services.llm import get_llm_service
    from app.core.parsing import ReasoningEffort

    service = get_llm_service()
    provider_name = service.provider.provider_name
    model_name = service.provider.default_model

    result = {
        "provider": provider_name,
        "model": model_name,
        "fallback_available": service.fallback_provider is not None,
        "fallback_provider": (
            service.fallback_provider.provider_name
            if service.fallback_provider
            else None
        ),
    }

    start = time.time()
    try:
        # Minimal 1-token request - cheapest way to check availability
        await service.async_generate(
            prompt="Hi",
            max_tokens=1,
            temperature=0,
            reasoning_effort=ReasoningEffort.NONE,  # No reasoning for ping
        )
        elapsed_ms = int((time.time() - start) * 1000)

        result.update(
            {
                "status": "available",
                "response_time_ms": elapsed_ms,
                "is_hot": elapsed_ms < 3000,  # <3s = model is loaded
            }
        )
        logger.info(f"LLM ping: {provider_name} available in {elapsed_ms}ms")

    except Exception as e:
        elapsed_ms = int((time.time() - start) * 1000)
        error_msg = str(e).lower()

        if "429" in error_msg or "busy" in error_msg or "capacity" in error_msg:
            result.update(
                {
                    "status": "busy",
                    "error": str(e)[:200],
                    "response_time_ms": elapsed_ms,
                }
            )
            logger.warning(f"LLM ping: {provider_name} busy/rate-limited")
        else:
            result.update(
                {
                    "status": "error",
                    "error": str(e)[:200],
                    "response_time_ms": elapsed_ms,
                }
            )
            logger.error(f"LLM ping failed: {e}")

    return result
