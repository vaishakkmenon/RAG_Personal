# ============================================================================
# Security Scan Script - Trivy Container Vulnerability Scanner (Windows)
# ============================================================================
# Scans Docker images for HIGH and CRITICAL vulnerabilities
#
# Usage:
#   .\scripts\security_scan.ps1                          # Scans default image
#   .\scripts\security_scan.ps1 -ImageName "myimage:tag" # Custom image
#   .\scripts\security_scan.ps1 -Full                    # Include MEDIUM severity
#
# Prerequisites:
#   - Docker Desktop must be running
#   - Internet access to pull Trivy image (first run only)
#
# Exit codes:
#   0 - No HIGH/CRITICAL vulnerabilities found
#   1 - Vulnerabilities found or scan failed
# ============================================================================

param(
    [string]$ImageName = "personal-rag-system:prod",
    [switch]$Full
)

# Configuration
$TrivyImage = "aquasec/trivy:latest"
$Severity = if ($Full) { "MEDIUM,HIGH,CRITICAL" } else { "HIGH,CRITICAL" }

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  Docker Image Security Scan" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "Image:    $ImageName" -ForegroundColor Yellow
Write-Host "Severity: $Severity" -ForegroundColor Yellow
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
try {
    docker info 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "Docker not responding"
    }
}
catch {
    Write-Host " Error: Docker is not running" -ForegroundColor Red
    exit 1
}

# Check if image exists
$imageExists = docker image inspect $ImageName 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host " Error: Image '$ImageName' not found" -ForegroundColor Red
    Write-Host ""
    Write-Host "Build the image first:" -ForegroundColor White
    Write-Host "  docker build -f Dockerfile.prod -t $ImageName ." -ForegroundColor Gray
    exit 1
}

Write-Host "Pulling latest Trivy scanner..." -ForegroundColor White
docker pull -q $TrivyImage

Write-Host ""
Write-Host "Scanning image for vulnerabilities..." -ForegroundColor White
Write-Host ""

# Run Trivy scan
# --exit-code 1 = fail if vulnerabilities found
# --severity = filter by severity level
# --ignore-unfixed = skip vulnerabilities without available fix
# --timeout 15m = allow up to 15 minutes for large images
# --scanners vuln = only vulnerability scanning (faster, no secret scanning)
docker run --rm `
    -v /var/run/docker.sock:/var/run/docker.sock `
    $TrivyImage image `
    --severity $Severity `
    --exit-code 1 `
    --ignore-unfixed `
    --timeout 15m `
    --scanners vuln `
    $ImageName

$ScanResult = $LASTEXITCODE

Write-Host ""
if ($ScanResult -eq 0) {
    Write-Host " Security scan passed - no $Severity vulnerabilities found" -ForegroundColor Green
}
else {
    Write-Host " Security scan failed - vulnerabilities detected" -ForegroundColor Red
    Write-Host ""
    Write-Host "Review the vulnerabilities above and consider:" -ForegroundColor White
    Write-Host "  1. Updating base image (python:3.12-slim-bookworm)" -ForegroundColor Gray
    Write-Host "  2. Updating dependencies in requirements-prod.txt" -ForegroundColor Gray
    Write-Host "  3. Rebuilding the image after updates" -ForegroundColor Gray
}

exit $ScanResult
