#!/bin/bash
set -e

# Deploy Script for Personal RAG System
# Usage: ./scripts/deploy.sh
# Run this on the VPS to update the running application.

echo "🚀 Starting Deployment..."
echo "📅 Date: $(date)"

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

echo "🌐 API should be available at https://api.vaishakmenon.com"
