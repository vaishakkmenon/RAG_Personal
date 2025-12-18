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
                        "socket": "hostname"
                    }
                }
            }
        }
    }
)
async def health() -> Dict[str, Any]:
    """Returns basic system health status."""
    # Get model based on provider
    if settings.llm.provider == "ollama":
        model = settings.llm.ollama_model
    elif settings.llm.provider == "groq":
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
    health_status = {
        "status": "healthy",
        "dependencies": {}
    }

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
        from app.retrieval.store import get_chroma_client
        client = get_chroma_client()
        # Try to heartbeat or list collections to verify connectivity
        client.heartbeat() 
        health_status["dependencies"]["chromadb"] = "healthy"
    except Exception as e:
        logger.warning(f"ChromaDB health check failed: {e}")
        health_status["dependencies"]["chromadb"] = "degraded"
        health_status["status"] = "degraded"

    # Check LLM (optional - skipped to avoid cost/latency)
    health_status["dependencies"]["llm"] = "not_checked"

    return health_status


@router.get("/health/ready", status_code=status.HTTP_200_OK)
async def readiness_check() -> Dict[str, str]:
    """Kubernetes-style readiness probe"""
    # Check if app can serve requests (e.g. DB is accessible)
    try:
        from app.retrieval.store import get_chroma_client
        client = get_chroma_client()
        # Simple verification
        client.heartbeat()
        return {"status": "ready"}
    except Exception:
        return {"status": "not_ready"}


@router.get("/health/live", status_code=status.HTTP_200_OK)
async def liveness_check() -> Dict[str, str]:
    """Kubernetes-style liveness probe"""
    # Just check if process is alive
    return {"status": "alive"}
