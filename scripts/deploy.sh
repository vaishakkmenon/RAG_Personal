#!/bin/bash

# ============================================================================
# Personal RAG System - Deployment Script
# Usage: ./scripts/deploy.sh [--no-cache]
# ============================================================================

set -e

echo "🚀 Starting deployment..."

# 1. Pull latest changes
echo "📥 Pulling latest code..."
git pull origin main || echo "⚠️  Git pull failed or already up to date."

# 2. Build images
# If --no-cache is passed, force a full rebuild (good for security updates)
if [[ "$1" == "--no-cache" ]]; then
    echo "🏗️  Building images (no cache)..."
    docker compose build --no-cache
else
    echo "🏗️  Building images..."
    docker compose build
fi

# 3. Restart services
echo "🔄 Restarting services..."
docker compose down
docker compose up -d

# 4. Check health
echo "🏥 Waiting for API to be healthy..."
# Simple wait loop (optional, or just trust Docker healthchecks)
sleep 10
docker compose ps

# 5. Cleanup
echo "🧹 Cleaning up unused images..."
docker image prune -f

echo "✅ Deployment complete!"
echo "   - API: https://localhost/docs"
echo "   - Admin: https://localhost/admin"
