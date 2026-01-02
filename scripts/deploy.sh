#!/bin/bash
set -e

# Deploy Script for Personal RAG System
# Usage: ./scripts/deploy.sh [--reingest]
# Run this on the VPS to update the running application.
#
# Options:
#   --reingest    Clear the knowledge base and reingest all documents

REINGEST=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --reingest)
            REINGEST=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./scripts/deploy.sh [--reingest]"
            exit 1
            ;;
    esac
done

echo "🚀 Starting Deployment..."
echo "📅 Date: $(date)"
if [ "$REINGEST" = true ]; then
    echo "📚 Reingest mode enabled - will clear and rebuild knowledge base"
fi

# 1. Check for .env file
if [ ! -f .env ]; then
    echo "❌ Error: .env file missing!"
    echo "   Please copy .env.prod to .env before deploying."
    exit 1
fi

# 2. Pull latest code
echo "📥 Pulling latest changes from git..."
# Check if we are in a git repo
if [ -d .git ]; then
    git pull
else
    echo "⚠️  Not a git repository. Skipping git pull."
fi

# 3. Pull/Build Images
# We prioritize local build for now since we don't have a remote registry set up yet
echo "🏗️  Building production images..."
docker compose -f docker-compose.prod.yml build

# 4. Restart Services
echo "🔄 Restarting services..."
# up -d --remove-orphans will recreate containers if image changed
docker compose -f docker-compose.prod.yml up -d --remove-orphans

# 5. Prune Docker System (Clean up old images to save space)
echo "🧹 Cleaning up old images..."
docker image prune -f

# 6. Verify Deployment
echo "✅ Deployment complete! Checking health..."
sleep 5
docker compose -f docker-compose.prod.yml ps

# 7. Reingest if requested
if [ "$REINGEST" = true ]; then
    echo ""
    echo "📚 Starting knowledge base reingest..."

    # Wait for API to be healthy
    echo "⏳ Waiting for API to be ready..."
    sleep 10

    # Clear ChromaDB collection
    echo "🗑️  Clearing existing knowledge base..."
    docker compose -f docker-compose.prod.yml exec -T api python scripts/clear_chromadb.py

    # Trigger reingest via API
    echo "📥 Ingesting documents..."
    docker compose -f docker-compose.prod.yml exec -T api python -c "
from app.ingest import ingest_paths
from app.settings import settings
from app.retrieval.bm25_search import BM25Index
from app.retrieval.vector_store import get_vector_store

# Ingest documents
added = ingest_paths([settings.docs_dir])
print(f'✓ Ingested {added} chunks')

# Rebuild BM25 index
if added > 0:
    vector_store = get_vector_store()
    documents = vector_store.get_all_documents()
    if documents:
        bm25_index = BM25Index(
            index_path='data/chroma/bm25_index.pkl',
            k1=settings.bm25.k1,
            b=settings.bm25.b,
        )
        bm25_index.build_index(documents)
        bm25_index.save_index()
        print(f'✓ BM25 index rebuilt with {len(documents)} documents')
"

    echo "✅ Knowledge base reingest complete!"
fi

echo ""
echo "🌐 API should be available at https://api.vaishakmenon.com"
