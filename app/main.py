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
from app.middleware.security_headers import SecurityHeadersMiddleware
from app.settings import settings
from app.config_validator import validate_config

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

@app.on_event("startup")
async def startup_event():
    """Validate environment configuration on startup."""
    validate_config()

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

# Custom middleware: API key validation, request size limiting, structured logging
# Custom middleware: API key validation, request size limiting, structured logging, security headers
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
    logger.info(f"Groq API Key: {'✓ Set' if settings.llm.groq_api_key else '✗ Not Set'}")
logger.info(f"Ollama Fallback: {settings.llm.ollama_host}")
logger.info(f"Ollama Model: {settings.llm.ollama_model}")
