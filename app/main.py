"""
FastAPI RAG Chatbot - Main Application

Clean, modular application setup with middleware configuration.
"""

import logging
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import create_api_router
from app.middleware.api_key import APIKeyMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.max_size import MaxSizeMiddleware
from app.middleware.tracing import RequestTracingMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.settings import settings
from app.config_validator import validate_config
from app.logging_config import setup_logging

from app.exceptions import (
    RAGException,
    rag_exception_handler,
    generic_exception_handler,
)

# Configure structured logging (JSON in production, colored in development)
setup_logging()
logger = logging.getLogger(__name__)

# Shutdown flag for graceful shutdown
_shutdown_event = asyncio.Event()


# ------------------------------------------------------------------------------
# Lifespan event handler
# ------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle application startup and shutdown events with graceful cleanup."""
    # Startup: Validate environment configuration
    logger.info("Starting application...")
    validate_config()

    yield

    # Shutdown: Graceful cleanup
    logger.info("Initiating graceful shutdown...")

    # 1. Close Redis connection pool (session storage)
    try:
        from app.storage.factory import get_session_store

        storage = get_session_store()
        if hasattr(storage, "close"):
            await storage.close()
            logger.info("Closed session storage backend")
        elif hasattr(storage, "_client") and storage._client:
            # Redis client with connection pool
            storage._client.connection_pool.disconnect()
            logger.info("Closed Redis connection pool")
    except Exception as e:
        logger.warning(f"Error closing session storage: {e}")

    # 2. Close PostgreSQL connection pool (feedback database)
    try:
        from app.database import engine

        if engine:
            engine.dispose()
            logger.info("Closed PostgreSQL connection pool")
    except Exception as e:
        logger.warning(f"Error closing PostgreSQL pool: {e}")

    # 3. Close ChromaDB client
    try:
        from app.retrieval.vector_store import get_vector_store

        vector_store = get_vector_store()
        if vector_store and hasattr(vector_store, "_client"):
            # ChromaDB PersistentClient doesn't have an explicit close
            # but we can log that we're releasing the reference
            logger.info("Released ChromaDB client reference")
    except Exception as e:
        logger.warning(f"Error releasing ChromaDB client: {e}")

    # 4. Allow brief time for in-flight requests to complete
    logger.info("Waiting for in-flight requests to complete...")
    await asyncio.sleep(2)

    logger.info("Graceful shutdown complete")


# ------------------------------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------------------------------

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    summary=settings.api.summary,
    lifespan=lifespan,
)

# ------------------------------------------------------------------------------
# Exception Handlers
# ------------------------------------------------------------------------------

app.add_exception_handler(RAGException, rag_exception_handler)
app.add_exception_handler(Exception, generic_exception_handler)

# ------------------------------------------------------------------------------
# Middleware configuration
# ------------------------------------------------------------------------------

# CORS for local frontend
# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_origin_regex=r"https://deploy-preview-.*\.vaishakmenon\.com",  # Allow Netlify deploy previews
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "X-API-Key", "Authorization"],
    max_age=600,  # Cache preflight requests for 10 minutes
)

# Custom middleware: API key validation, request size limiting, structured logging, security headers, tracing

app.add_middleware(RequestTracingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(APIKeyMiddleware)
app.add_middleware(MaxSizeMiddleware, max_bytes=settings.max_bytes)
app.add_middleware(LoggingMiddleware)

# ------------------------------------------------------------------------------
# Prometheus metrics
# ------------------------------------------------------------------------------

Instrumentator().instrument(app).expose(app)

# ------------------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------------------

# Include all API routes
api_router = create_api_router()
app.include_router(api_router)

logger.info(f"Application started: {settings.api.title} v{settings.api.version}")
logger.info(f"LLM Provider: {settings.llm.provider}")
if settings.llm.provider == "groq":
    logger.info(f"Groq Model: {settings.llm.groq_model}")
    logger.info(
        f"Groq API Key: {'✓ Set' if settings.llm.groq_api_key else '✗ Not Set'}"
    )
