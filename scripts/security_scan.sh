#!/bin/bash
# ============================================================================
# Security Scan Script - Trivy Container Vulnerability Scanner
# ============================================================================
# Scans Docker images for HIGH and CRITICAL vulnerabilities
# 
# Usage:
#   ./scripts/security_scan.sh                        # Scans default image
#   ./scripts/security_scan.sh personal-rag-system:prod  # Custom image
#   ./scripts/security_scan.sh --full                 # Include MEDIUM severity
#
# Prerequisites:
#   - Docker must be running
#   - Internet access to pull Trivy image (first run only)
#
# Exit codes:
#   0 - No HIGH/CRITICAL vulnerabilities found
#   1 - Vulnerabilities found or scan failed
# ============================================================================

set -e

# Configuration
DEFAULT_IMAGE="personal-rag-system:prod"
TRIVY_IMAGE="aquasec/trivy:latest"

# Parse arguments
SEVERITY="HIGH,CRITICAL"
if [[ "$1" == "--full" ]]; then
    SEVERITY="MEDIUM,HIGH,CRITICAL"
    shift
fi

IMAGE_NAME="${1:-$DEFAULT_IMAGE}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo ""
echo "=============================================="
echo "🔍 Docker Image Security Scan"
echo "=============================================="
echo "Image:    $IMAGE_NAME"
echo "Severity: $SEVERITY"
echo "=============================================="
echo ""

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo " Error: Docker is not running"
    exit 1
fi

# Check if image exists
if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
    echo " Error: Image '$IMAGE_NAME' not found"
    echo ""
    echo "Build the image first:"
    echo "  docker build -f Dockerfile.prod -t $IMAGE_NAME ."
    exit 1
fi

echo "Pulling latest Trivy scanner..."
docker pull -q "$TRIVY_IMAGE"

echo ""
echo "Scanning image for vulnerabilities..."
echo ""

# Run Trivy scan
# --exit-code 1 = fail if vulnerabilities found
# --severity = filter by severity level
# --ignore-unfixed = skip vulnerabilities without available fix
docker run --rm \
    -v /var/run/docker.sock:/var/run/docker.sock \
    "$TRIVY_IMAGE" image \
    --severity "$SEVERITY" \
    --exit-code 1 \
    --ignore-unfixed \
    "$IMAGE_NAME"

SCAN_RESULT=$?

echo ""
if [ $SCAN_RESULT -eq 0 ]; then
    echo " Security scan passed - no $SEVERITY vulnerabilities found"
else
    echo " Security scan failed - vulnerabilities detected"
    echo ""
    echo "Review the vulnerabilities above and consider:"
    echo "  1. Updating base image (python:3.12-slim-bookworm)"
    echo "  2. Updating dependencies in requirements-prod.txt"
    echo "  3. Rebuilding the image after updates"
fi

exit $SCAN_RESULT
