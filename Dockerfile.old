# ============================================================================
# Personal RAG System - Production Dockerfile
# ============================================================================
# Multi-stage build for optimal image size and security
# Base: Chainguard Python (minimal, secure, non-root by default)
# ============================================================================

# ============================================================================
# Stage 1: Builder - Install dependencies and prepare virtualenv
# ============================================================================
FROM python:3.12-slim-bookworm AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 65532 -s /bin/bash nonroot

# Switch to non-root user and create directories
USER root
RUN install -d -o nonroot -g nonroot /opt/venv \
    && mkdir -p /workspace && chown -R nonroot:nonroot /workspace
USER nonroot

# Set up virtual environment
ENV VENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /workspace

# Create virtualenv
RUN python -m venv "$VENV"

# Upgrade system pip to fix CVE-2025-8869 (base image has older pip)
RUN python -m pip install --upgrade "pip>=25.3"

# Create HuggingFace cache directory with correct ownership BEFORE installing packages
RUN mkdir -p /home/nonroot/.cache/huggingface && chown -R nonroot:nonroot /home/nonroot/.cache

# Install production dependencies
# Separating this from app code allows better Docker layer caching
# Step 1: Install CPU-only PyTorch first (saves ~8GB vs GPU version)
COPY --chown=nonroot:nonroot requirements.txt .
RUN /opt/venv/bin/python -m pip install --upgrade "pip>=25.3" wheel setuptools \
    && /opt/venv/bin/python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && /opt/venv/bin/python -m pip install --no-cache-dir -r requirements.txt

# Verify critical dependencies are installed
RUN /opt/venv/bin/python -c "import fastapi; import chromadb; import redis; print('Core dependencies installed')" \
    || (echo "ERROR: Failed to import core dependencies" && exit 1)

# Download NLTK data (required for BM25 search)
# - stopwords: For filtering common words in BM25 tokenization
# - punkt: For sentence tokenization
RUN mkdir -p /home/nonroot/nltk_data \
    && /opt/venv/bin/python -c "import nltk; nltk.download('stopwords', download_dir='/home/nonroot/nltk_data'); nltk.download('punkt', download_dir='/home/nonroot/nltk_data'); nltk.download('punkt_tab', download_dir='/home/nonroot/nltk_data')" \
    && chown -R nonroot:nonroot /home/nonroot/nltk_data

# Pre-download embedding model (required for semantic search)
# This avoids download on container start and works with read-only filesystem
RUN /opt/venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')" \
    && chown -R nonroot:nonroot /home/nonroot/.cache

# Copy application code
COPY --chown=nonroot:nonroot app ./app

# Copy scripts (including docker-entrypoint.sh)
COPY --chown=nonroot:nonroot scripts ./scripts

# Create required directories
RUN mkdir -p /workspace/data/chroma /workspace/data/docs

# Make entrypoint script executable
RUN chmod +x /workspace/scripts/docker-entrypoint.sh || true

# Validate app structure
RUN test -f ./app/main.py || (echo "ERROR: app/main.py not found!" && exit 1)

# ============================================================================
# Stage 2: Test - Testing environment with dev dependencies
# ============================================================================
FROM builder AS test

# Install testing and linting tools into the same virtualenv
RUN /opt/venv/bin/python -m pip install --no-cache-dir \
    pytest==8.4.1 \
    pytest-cov==4.1.0 \
    pytest-mock==3.12.0 \
    faker==22.0.0 \
    ruff==0.12.9

# Copy test files and configuration
COPY --chown=nonroot:nonroot tests ./tests
COPY --chown=nonroot:nonroot pytest.ini ./

# Verify test structure
RUN test -d ./tests || (echo "WARNING: tests directory not found" && exit 0)

# Reset entrypoint for test stage
ENTRYPOINT []

# Default: Run unit tests (not integration tests which need services)
CMD ["/opt/venv/bin/python", "-m", "pytest", "-m", "not integration", "-v", "--tb=short"]

# ============================================================================
# Stage 3: Runtime - Minimal production image
# ============================================================================
FROM python:3.12-slim-bookworm AS runtime

# Environment configuration
ENV VENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Logging configuration
    LOG_LEVEL=INFO \
    # Security: disable pip version check in prod
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Create non-root user (if it doesn't exist)
RUN useradd -m -u 65532 -s /bin/bash nonroot || true

# Upgrade system pip to fix CVE-2025-8869 (base image has older pip)
RUN python -m pip install --upgrade "pip>=25.3"

WORKDIR /workspace

# Copy only the virtualenv with production dependencies
COPY --from=builder --chown=nonroot:nonroot /opt/venv /opt/venv

# Copy only application code (no tests, no dev files)
COPY --from=builder --chown=nonroot:nonroot /workspace/app ./app

# Copy scripts directory (includes docker-entrypoint.sh and utility scripts)
COPY --from=builder --chown=nonroot:nonroot /workspace/scripts ./scripts

# Copy NLTK data and HuggingFace cache from builder stage
COPY --from=builder --chown=nonroot:nonroot /home/nonroot/nltk_data /home/nonroot/nltk_data
COPY --from=builder --chown=nonroot:nonroot /home/nonroot/.cache /home/nonroot/.cache

# Create directories for runtime data
RUN mkdir -p /workspace/data/chroma /workspace/data/docs

# Verify application can import successfully
RUN /opt/venv/bin/python -c "from app.main import app; print('Application imports successfully')" \
    || (echo "ERROR: Failed to import application!" && exit 1)

# Run as non-root user (Chainguard default: nonroot:nonroot = 65532:65532)
USER nonroot

# Health check (verifies server is responding)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD ["/opt/venv/bin/python", "-c", "import socket; s=socket.socket(); s.settimeout(2); s.connect(('127.0.0.1', 8000)); print('✅ Health check passed')"]

# Reset entrypoint (Chainguard default would prepend 'python')
ENTRYPOINT []

# Default command: Run uvicorn with 2 workers
# Note: UVICORN_WORKERS can be overridden via environment variable
CMD ["/opt/venv/bin/python", "-m", "uvicorn", "app.main:app", \
    "--host", "0.0.0.0", \
    "--port", "8000", \
    "--workers", "2", \
    "--log-level", "info"]

# ============================================================================
# Build Information (for debugging)
# ============================================================================
# Build: docker build -t personal-rag-system .
# Test:  docker build --target test -t personal-rag-system:test .
# Run:   docker run -p 8000:8000 personal-rag-system
# ============================================================================
