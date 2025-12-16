"""
FastAPI RAG Chatbot - Main Application

Clean, modular application setup with middleware configuration.
"""

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from app.api import create_api_router
from app.middleware.api_key import APIKeyMiddleware
from app.middleware.logging import LoggingMiddleware
from app.middleware.max_size import MaxSizeMiddleware
from app.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# FastAPI app setup
# ------------------------------------------------------------------------------

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    summary=settings.api.summary,
)

# ------------------------------------------------------------------------------
# Middleware configuration
# ------------------------------------------------------------------------------

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom middleware: API key validation, request size limiting, structured logging
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
    logger.info(f"Groq API Key: {'✓ Set' if settings.llm.groq_api_key else '✗ Not Set'}")
logger.info(f"Ollama Fallback: {settings.llm.ollama_host}")
logger.info(f"Ollama Model: {settings.llm.ollama_model}")
