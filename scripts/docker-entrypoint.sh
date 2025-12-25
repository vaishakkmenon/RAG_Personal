#!/bin/bash
# ============================================================================
# Docker Entrypoint Script for Personal RAG System
# ============================================================================
# This script runs before starting the FastAPI server to ensure all
# initialization tasks are complete:
# 1. Check/build BM25 index if needed
# 2. Verify NLTK data is accessible
# 3. Start the uvicorn server
# ============================================================================

set -e  # Exit on any error

echo "=========================================="
echo "Personal RAG System - Initialization"
echo "=========================================="

# ============================================================================
# Environment Setup
# ============================================================================
export PYTHONUNBUFFERED=1
export NLTK_DATA=/home/nonroot/nltk_data

# ============================================================================
# Verify NLTK Data
# ============================================================================
echo ""
echo "[1/4] Verifying NLTK data..."
if python -c "import nltk; nltk.data.find('corpora/stopwords')" >/dev/null 2>&1; then
    echo "✓ NLTK stopwords found"
else
    echo "⚠ NLTK stopwords not found - BM25 search may fail"
fi

if python -c "import nltk; nltk.data.find('tokenizers/punkt')" >/dev/null 2>&1; then
    echo "✓ NLTK punkt tokenizer found"
else
    echo "⚠ NLTK punkt not found - tokenization may fail"
fi

# ============================================================================
# Check/Build BM25 Index
# ============================================================================
echo ""
echo "[2/4] Checking BM25 index..."

BM25_INDEX_PATH="${CHROMA_DIR:-./data/chroma}/bm25_index.pkl"
CHROMA_DB_PATH="${CHROMA_DIR:-./data/chroma}"

# Check if ChromaDB directory exists and has data
if [ -d "$CHROMA_DB_PATH" ] && [ "$(ls -A $CHROMA_DB_PATH 2>/dev/null)" ]; then
    echo "✓ ChromaDB directory found: $CHROMA_DB_PATH"

    # Check if BM25 index exists
    if [ -f "$BM25_INDEX_PATH" ]; then
        echo "✓ BM25 index already exists: $BM25_INDEX_PATH"
        BM25_SIZE=$(du -h "$BM25_INDEX_PATH" | cut -f1)
        echo "  Index size: $BM25_SIZE"
    else
        echo "⚠ BM25 index not found: $BM25_INDEX_PATH"
        echo "  Building BM25 index from ChromaDB..."

        # Build BM25 index (non-blocking - if it fails, server still starts)
        if python scripts/build_bm25_index.py; then
            echo "✓ BM25 index built successfully"
            if [ -f "$BM25_INDEX_PATH" ]; then
                BM25_SIZE=$(du -h "$BM25_INDEX_PATH" | cut -f1)
                echo "  Index size: $BM25_SIZE"
            fi
        else
            echo "⚠ Failed to build BM25 index - hybrid search will be disabled"
            echo "  You can build it manually later with: python scripts/build_bm25_index.py"
        fi
    fi
else
    echo "⚠ ChromaDB directory empty or not found: $CHROMA_DB_PATH"
    echo "  Starting automatic ingestion..."

    if python -m scripts.ingest; then
        echo "✓ Automatic ingestion completed successfully"
    else
        echo "✗ Automatic ingestion failed"
        # We don't exit here to allow debugging, but specific error logged above
    fi
fi

# ============================================================================
# Initialize Database
# ============================================================================
echo ""
echo "[3/4] Initializing Database..."
if python -m app.init_db; then
    echo "✓ Database tables initialized"
else
    echo "⚠ Database initialization failed - check connection settings"
    # We don't exit here to allow app to start in degraded mode if DB is down,
    # but for production you might want to exit 1
fi

# ============================================================================
# Configuration Summary
# ============================================================================
echo ""
echo "[4/4] Configuration Summary"
echo "-------------------------------------------"
echo "LLM Provider:        ${LLM_PROVIDER:-groq}"
echo "Embedding Model:     ${EMBED_MODEL:-BAAI/bge-small-en-v1.5}"
echo "ChromaDB Path:       ${CHROMA_DIR:-./data/chroma}"
echo "Collection Name:     ${COLLECTION_NAME:-personal_rag}"
echo "Query Rewriting:     ${QUERY_REWRITER_ENABLED:-false}"
echo "Session Backend:     ${SESSION_STORAGE_BACKEND:-redis}"
echo "Uvicorn Workers:     ${UVICORN_WORKERS:-2}"
echo "-------------------------------------------"

# ============================================================================
# Start FastAPI Server
# ============================================================================
echo ""
echo "=========================================="
echo "Starting FastAPI Server"
echo "=========================================="
echo ""

# Execute the command passed to the container (uvicorn server)
exec "$@"
