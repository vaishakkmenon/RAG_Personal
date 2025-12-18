"""
Environment variable validation for Personal RAG System.

This module validates that all required environment variables are properly
configured before the application starts. It helps catch configuration errors
early and provides clear error messages for missing or insecure settings.

Called automatically during application startup in app/main.py.
"""

import os
import sys
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ==============================================================================
# Required Environment Variables
# ==============================================================================

# Critical security variables (MUST be set)
REQUIRED_VARS = [
    "API_KEY",
    "REDIS_PASSWORD",
]

# LLM provider-specific variables
GROQ_VARS = ["LLM_GROQ_API_KEY"]

# Production-only variables (warnings if not set in production)
PRODUCTION_VARS = [
    "ALLOWED_ORIGINS",
    "SESSION_REQUIRE_HTTPS",
]

# Insecure default values that must be changed
INSECURE_DEFAULTS = {
    "API_KEY": [
        "change-me",
        "change-me-to-a-secure-random-key",
        "change-me-to-a-secure-random-key-64-chars",
    ],
    "REDIS_PASSWORD": [
        "change-me-devpassword123",
        "change-me-secure-redis-password",
    ],
    "GRAFANA_ADMIN_PASSWORD": [
        "admin123",
        "change-me-secure-grafana-password",
    ],
}

# ==============================================================================
# Documentation of All Environment Variables
# ==============================================================================

ENV_VAR_DOCUMENTATION = """
# ==============================================================================
# ENVIRONMENT VARIABLES REFERENCE
# ==============================================================================

Below is a complete list of all environment variables used by this application.
See .env.example for detailed descriptions and recommended values.

## CRITICAL (Required for application to start)
- API_KEY: API authentication key for securing endpoints
- REDIS_PASSWORD: Password for Redis authentication

## LLM Configuration
- LLM_PROVIDER: LLM provider (groq or ollama)
- LLM_GROQ_API_KEY: Groq API key (required if provider=groq)
- LLM_GROQ_MODEL: Groq model name
- LLM_OLLAMA_HOST: Ollama server URL
- LLM_OLLAMA_MODEL: Ollama model name
- LLM_OLLAMA_TIMEOUT: Ollama request timeout
- LLM_TEMPERATURE: Generation temperature
- LLM_MAX_TOKENS: Maximum tokens to generate
- LLM_NUM_CTX: Context window size

## Embeddings & Storage
- EMBED_MODEL: SentenceTransformer model for embeddings
- CHROMA_DIR: ChromaDB storage directory
- DOCS_DIR: Documents directory for ingestion
- COLLECTION_NAME: ChromaDB collection name

## Retrieval Configuration
- CHUNK_SIZE: Target chunk size in characters
- CHUNK_OVERLAP: Overlap between chunks
- TOP_K: Number of chunks to send to LLM
- MAX_DISTANCE: Maximum cosine distance for retrieval
- NULL_THRESHOLD: Distance threshold for grounding check
- RERANK: Enable hybrid reranking
- RERANK_LEX_WEIGHT: Lexical vs semantic weight
- RERANK_RETRIEVAL_K: Chunks to retrieve before reranking

## Session Management
- SESSION_STORAGE_BACKEND: Storage backend (redis or memory)
- SESSION_REDIS_URL: Redis connection URL
- SESSION_MAX_TOTAL_SESSIONS: Maximum total sessions
- SESSION_MAX_SESSIONS_PER_IP: Maximum sessions per IP
- SESSION_QUERIES_PER_HOUR: Rate limit per session
- SESSION_TTL_SECONDS: Session time-to-live
- SESSION_MAX_HISTORY_TOKENS: Max tokens for conversation history
- SESSION_MAX_HISTORY_TURNS: Max conversation turns to keep
- SESSION_REQUIRE_HTTPS: Require HTTPS in production

## Response Caching
- RESPONSE_CACHE_ENABLED: Enable response caching
- RESPONSE_CACHE_TTL_SECONDS: Cache TTL
- RESPONSE_CACHE_MAX_SIZE_MB: Maximum cache size
- RESPONSE_CACHE_PROMPT_VERSION: Version for cache invalidation

## Query Processing
- USE_ROUTER: Enable query routing
- NEGATIVE_INFERENCE_METHOD: Negative inference method
- NEGATIVE_INFERENCE_THRESHOLD: Threshold for entity existence
- QUERY_REWRITER_ENABLED: Enable query rewriting
- QUERY_REWRITER_CONFIG: Pattern configuration path
- QUERY_REWRITER_HOT_RELOAD: Enable hot-reloading
- QUERY_REWRITER_RELOAD_INTERVAL: Hot-reload interval
- QUERY_REWRITER_ANALYTICS: Enable analytics tracking
- QUERY_REWRITER_ANALYTICS_PATH: Analytics storage path
- QUERY_REWRITER_FAILED_PATH: Failed queries storage path
- QUERY_REWRITER_MAX_LATENCY: Maximum latency threshold
- QUERY_REWRITER_FAILED_THRESHOLD: Failed query threshold

## Cross-Encoder Reranking
- CROSS_ENCODER_ENABLED: Enable cross-encoder reranking
- CROSS_ENCODER_MODEL: HuggingFace model name
- CROSS_ENCODER_CACHE_DIR: Model cache directory
- CROSS_ENCODER_RETRIEVAL_K: Chunks to retrieve
- CROSS_ENCODER_TOP_K: Final chunks after reranking
- CROSS_ENCODER_MAX_LATENCY_MS: Maximum latency

## BM25 Configuration
- BM25_K1: Term frequency saturation
- BM25_B: Document length normalization
- BM25_RRF_K: Reciprocal rank fusion parameter

## Prompt Guard (Security)
- PROMPT_GUARD_ENABLED: Enable prompt injection guard
- PROMPT_GUARD_MODEL: Groq model for guard
- PROMPT_GUARD_FAIL_OPEN: Behavior on guard errors
- PROMPT_GUARD_TIMEOUT_SECONDS: API timeout
- PROMPT_GUARD_MAX_RETRIES: Maximum retries
- PROMPT_GUARD_CACHE_TTL_SECONDS: Cache TTL
- PROMPT_GUARD_CACHE_MAX_SIZE: Cache size

## HTTP & CORS
- ALLOWED_ORIGINS: Allowed CORS origins
- MAX_BYTES: Maximum request body size

## Ingestion
- INGEST_ALLOWED_EXTENSIONS: Allowed file extensions
- INGEST_MAX_FILE_SIZE: Maximum file size
- INGEST_BATCH_SIZE: Batch size for processing

## Monitoring
- GRAFANA_ADMIN_USER: Grafana admin username
- GRAFANA_ADMIN_PASSWORD: Grafana admin password
- GRAFANA_ALLOW_SIGNUP: Allow user signup
- GRAFANA_ANALYTICS_REPORTING: Enable analytics
- PROMETHEUS_RETENTION_DAYS: Data retention period

## General
- ENV: Environment type (production, development)
- LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

For complete documentation, see:
- .env.example: Template with all variables and descriptions
- app/settings.py: Settings classes with default values
"""


def validate_config() -> None:
    """
    Validate required environment variables are set and configured properly.

    This function:
    1. Checks that all critical variables are set
    2. Validates provider-specific variables
    3. Warns about insecure defaults
    4. Warns about missing production variables
    5. Exits with error code 1 if critical variables are missing

    Raises:
        SystemExit: If any required variables are missing
    """
    missing_vars: List[str] = []
    insecure_vars: List[Tuple[str, str]] = []

    # 1. Check required variables
    for var in REQUIRED_VARS:
        val = os.getenv(var)
        if not val:
            missing_vars.append(var)
        else:
            # Check for insecure defaults
            if var in INSECURE_DEFAULTS:
                if val in INSECURE_DEFAULTS[var]:
                    insecure_vars.append((var, val))

    # 2. Check provider-specific variables
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "groq":
        for var in GROQ_VARS:
            val = os.getenv(var)
            if not val:
                missing_vars.append(f"{var} (required for LLM_PROVIDER=groq)")
            elif val in ["your-groq-api-key-here", "change-me"]:
                insecure_vars.append((var, val))

    # 3. Check Grafana password (warn if default)
    grafana_pass = os.getenv("GRAFANA_ADMIN_PASSWORD", "")
    if grafana_pass and grafana_pass in INSECURE_DEFAULTS.get(
        "GRAFANA_ADMIN_PASSWORD", []
    ):
        insecure_vars.append(("GRAFANA_ADMIN_PASSWORD", grafana_pass))

    # 4. Report critical errors (missing variables)
    if missing_vars:
        logger.critical("=" * 80)
        logger.critical("STARTUP FAILED: Missing required environment variables")
        logger.critical("=" * 80)
        for var in missing_vars:
            logger.error(f"  ✗ {var} is not set")
        logger.critical("")
        logger.critical("Action required:")
        logger.critical("  1. Copy .env.example to .env")
        logger.critical("  2. Fill in all required values")
        logger.critical("  3. Generate secure keys where indicated")
        logger.critical("")
        logger.critical(
            "For documentation, run: python -c 'from app.config_validator import ENV_VAR_DOCUMENTATION; print(ENV_VAR_DOCUMENTATION)'"
        )
        logger.critical("=" * 80)
        sys.exit(1)

    # 5. Warn about insecure defaults (non-fatal)
    if insecure_vars:
        logger.warning("=" * 80)
        logger.warning("SECURITY WARNING: Using insecure default values!")
        logger.warning("=" * 80)
        for var, val in insecure_vars:
            logger.warning(f"  ⚠ {var} = '{val}' (default placeholder)")
        logger.warning("")
        logger.warning(
            "These default values are INSECURE and must be changed for production!"
        )
        logger.warning("Generate secure values:")
        logger.warning("  API_KEY: openssl rand -hex 32")
        logger.warning("  REDIS_PASSWORD: openssl rand -base64 32")
        logger.warning("  GRAFANA_ADMIN_PASSWORD: openssl rand -base64 24")
        logger.warning("=" * 80)
        logger.warning("")

    # 6. Check production-specific configuration
    env = os.getenv("ENV", "development")
    if env == "production":
        missing_prod_vars = []
        for var in PRODUCTION_VARS:
            if not os.getenv(var):
                missing_prod_vars.append(var)

        if missing_prod_vars:
            logger.warning("=" * 80)
            logger.warning(
                "PRODUCTION WARNING: Missing recommended production variables"
            )
            logger.warning("=" * 80)
            for var in missing_prod_vars:
                logger.warning(f"  ⚠ {var} is not set")
            logger.warning("")
            logger.warning(
                "These variables should be configured for production deployments."
            )
            logger.warning("See .env.example for details.")
            logger.warning("=" * 80)
            logger.warning("")

        # Warn if HTTPS is not required in production
        require_https = os.getenv("SESSION_REQUIRE_HTTPS", "false").lower()
        if require_https != "true":
            logger.warning(
                "⚠ SESSION_REQUIRE_HTTPS=false in production - cookies may be insecure!"
            )

        # Warn if using default origins
        allowed_origins = os.getenv("ALLOWED_ORIGINS", "")
        if "localhost" in allowed_origins:
            logger.warning(
                "⚠ ALLOWED_ORIGINS includes localhost in production - potential security risk!"
            )

    # 7. Success message
    logger.info("=" * 80)
    logger.info("✅ Environment configuration validated successfully")
    logger.info("=" * 80)
    logger.info(f"Environment: {env}")
    logger.info(f"LLM Provider: {provider}")
    logger.info(f"Session Backend: {os.getenv('SESSION_STORAGE_BACKEND', 'redis')}")
    logger.info(
        f"Response Caching: {'enabled' if os.getenv('RESPONSE_CACHE_ENABLED', 'true') == 'true' else 'disabled'}"
    )
    logger.info(
        f"Prompt Guard: {'enabled' if os.getenv('PROMPT_GUARD_ENABLED', 'true') == 'true' else 'disabled'}"
    )
    logger.info("=" * 80)
    logger.info("")


def print_env_documentation() -> None:
    """Print complete environment variable documentation."""
    print(ENV_VAR_DOCUMENTATION)


if __name__ == "__main__":
    # Allow running this module directly to print documentation
    print_env_documentation()
