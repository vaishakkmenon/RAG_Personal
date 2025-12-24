# Operations Runbook

Quick reference guide for operating and troubleshooting the Personal RAG System in production.

## Table of Contents

- [Quick Commands](#quick-commands)
- [Authentication](#authentication)
- [Circuit Breaker](#circuit-breaker)
- [Failure Modes & Recovery](#failure-modes--recovery)
- [Health Monitoring](#health-monitoring)
- [Disaster Recovery](#disaster-recovery)
- [Common Issues](#common-issues)
- [Maintenance Procedures](#maintenance-procedures)

---

## Quick Commands

### Deploy Update
```bash
# Production Deployment
ssh user@server
cd ~/RAG_Personal
./scripts/deploy.sh
```

### Rollback to Previous Version
```bash
git log --oneline  # Find previous commit
git checkout <commit-hash>
./scripts/deploy.sh
```

### Check Service Status
```bash
docker compose -f docker-compose.prod.yml ps
docker compose -f docker-compose.prod.yml logs api --tail=50
```

### Restart Services
```bash
# Restart specific service
docker compose -f docker-compose.prod.yml restart api

# Full restart
./scripts/deploy.sh
```

### Clear Caches
```bash
# Clear Redis cache (Session & Response Cache)
docker compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD FLUSHDB
```

### Backup
```bash
# Manual Trigger on Server
./scripts/backup.sh

# Download to Local Machine (Windows)
.\scripts\sync_backups.ps1
```

### Check Health
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check (includes Redis, ChromaDB, PostgreSQL)
curl http://localhost:8000/health/detailed
```

---

## Authentication

### Overview

The system uses two authentication methods:
1. **API Key** (`X-API-Key` header) - For chat and public endpoints
2. **JWT Token** (`Authorization: Bearer` header) - For admin endpoints

### Creating Admin Users

```bash
# Set the admin password (required)
export ADMIN_PASSWORD="your-secure-password"

# Create admin user
docker compose -f docker-compose.prod.yml run --rm api python scripts/create_admin.py
```

The script creates a user with:
- Username: `admin`
- Email: `admin@example.com` (placeholder)
- Superuser privileges enabled

### Getting a JWT Token

```bash
# Request token
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=your-secure-password"

# Response:
# {"access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...", "token_type": "bearer"}
```

### Using JWT for Admin Operations

```bash
# Store token
TOKEN="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."

# Access admin endpoints
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/admin/stats

# Trigger ingestion
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/ingest
```

### Token Configuration

Token settings are in `app/core/security_config.py`:
- **Algorithm**: HS256
- **Expiration**: 30 minutes (configurable in code)
- **Secret Key**: Set via `SECRET_KEY` environment variable

---

## Circuit Breaker

### Overview

The Groq API circuit breaker protects the system from cascading failures. It's implemented in `app/services/llm.py`.

### States

| State | Description | Behavior |
|-------|-------------|----------|
| **CLOSED** | Normal operation | Requests pass through |
| **OPEN** | Service failing | Requests rejected immediately (fail fast) |
| **HALF_OPEN** | Testing recovery | Limited requests allowed to test if service recovered |

### Transitions

- **CLOSED → OPEN**: After 5 consecutive failures
- **OPEN → HALF_OPEN**: After 30 seconds
- **HALF_OPEN → CLOSED**: After 3 successful requests
- **HALF_OPEN → OPEN**: On any failure

### Monitoring

Check circuit breaker status via Prometheus metrics:

```bash
# Current state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
curl -s http://localhost:8000/metrics | grep rag_circuit_breaker_state

# Transition count
curl -s http://localhost:8000/metrics | grep rag_circuit_breaker_transitions_total
```

### Manual Intervention

If the circuit breaker is stuck OPEN:

```bash
# Restart the API service to reset the circuit breaker
docker compose -f docker-compose.prod.yml restart api
```

---

## Failure Modes & Recovery

### Redis Failure

**Symptoms:**
- "FALLBACK ACTIVATED" in logs
- Sessions stored in memory (not persisted)

**Recovery:**
```bash
docker compose -f docker-compose.prod.yml restart redis
# Sessions will automatically switch back to Redis
```

### ChromaDB Failure

**Symptoms:**
- Retrieval errors
- Empty search results

**Recovery (Data intact):**
```bash
docker compose -f docker-compose.prod.yml restart api
```

**Recovery (Data corrupted):**
```bash
# 1. Stop services
docker compose -f docker-compose.prod.yml stop

# 2. Rotate data folder
mv data/chroma data/chroma_backup_$(date +%Y%m%d)
mkdir data/chroma

# 3. Restart and Re-ingest
./scripts/deploy.sh

# 4. Trigger ingestion (requires JWT token)
curl -X POST -H "Authorization: Bearer $TOKEN" http://localhost:8000/ingest
```

### Groq API Failure

**Symptoms:**
- LLM requests timing out
- Circuit breaker OPEN
- HTTP 503 responses

**Recovery:**
1. Check Groq status: https://status.groq.com/
2. Wait for circuit breaker recovery (30 seconds)
3. If persistent, restart API to reset circuit breaker

### PostgreSQL Failure

**Symptoms:**
- Feedback not being saved
- `/health/detailed` shows PostgreSQL unhealthy

**Recovery:**
```bash
docker compose -f docker-compose.prod.yml restart postgres
docker compose -f docker-compose.prod.yml restart api
```

---

## Health Monitoring

### Endpoints

| Endpoint | Purpose | Authentication |
|----------|---------|----------------|
| `/health` | Basic status | None |
| `/health/detailed` | Full component check | None |
| `/health/ready` | Kubernetes readiness | None |
| `/health/live` | Kubernetes liveness | None |
| `/metrics` | Prometheus metrics | Basic Auth (Caddy) |

### Key Metrics to Monitor

```bash
# Error rate
curl -s http://localhost:8000/metrics | grep http_requests_total | grep status=\"5

# LLM latency
curl -s http://localhost:8000/metrics | grep rag_llm_latency_seconds

# Cache hit rate
curl -s http://localhost:8000/metrics | grep rag_cache_hits_total

# Circuit breaker state
curl -s http://localhost:8000/metrics | grep rag_circuit_breaker_state
```

### Alerting

Alerts are configured in `monitoring/prometheus/alerts.yml`:
- **HighErrorRate**: >5% 5xx errors for 5 minutes
- **HighLatency**: P99 > 2 seconds for 5 minutes
- **InstanceDown**: Unreachable for 1 minute
- **RedisDown**: Redis unavailable for 1 minute
- **PostgresDown**: PostgreSQL unavailable for 1 minute
- **LLMErrorsHigh**: LLM error rate > 5%

---

## Disaster Recovery

### RTO/RPO

- **RTO (Recovery Time Objective)**: ~48 seconds (tested)
- **RPO (Recovery Point Objective)**: 24 hours (daily backups)

### Backup Schedule

Automated backups run daily via the backup service in docker-compose:
- PostgreSQL: SQL dump (gzip compressed)
- Redis: RDB file copy
- ChromaDB: Tar archive (gzip compressed)

### Restore Procedure

**Linux/macOS:**
```bash
# List available backups
./scripts/restore.sh --list

# Dry run (preview what will be restored)
./scripts/restore.sh --dry-run 2025-12-24

# Full restore
./scripts/restore.sh 2025-12-24
```

**Windows (PowerShell):**
```powershell
# List available backups
.\scripts\restore.ps1 -List

# Dry run
.\scripts\restore.ps1 -Date 2025-12-24 -DryRun

# Full restore
.\scripts\restore.ps1 -Date 2025-12-24
```

### Restore Verification

After restore, verify:
1. Health check: `curl http://localhost:8000/health/detailed`
2. Sample query: Test a chat query
3. Check logs: `docker compose -f docker-compose.prod.yml logs api --tail=20`

---

## Common Issues

### "Rate limit exceeded" errors

**Cause:** Groq API rate limits (28 req/min on free tier)

**Solution:**
- Wait for rate limit window to reset
- Check `/metrics` for `rag_rate_limit_remaining`
- Consider upgrading Groq tier

### "Circuit breaker open" errors

**Cause:** Multiple Groq API failures

**Solution:**
- Wait 30 seconds for HALF_OPEN state
- Check Groq status page
- Restart API if persistent: `docker compose -f docker-compose.prod.yml restart api`

### Sessions not persisting

**Cause:** Redis connection failed, using memory fallback

**Solution:**
```bash
# Check Redis status
docker compose -f docker-compose.prod.yml logs redis --tail=20

# Restart Redis
docker compose -f docker-compose.prod.yml restart redis
```

### Slow responses

**Cause:** Usually reranker model loading on first request

**Solution:**
- First request may take 10-30 seconds (model loading)
- Subsequent requests should be fast
- Check `/metrics` for latency breakdown

---

## Maintenance Procedures

### Updating Dependencies

```bash
# Update requirements
pip-compile requirements.in -o requirements.txt

# Rebuild and deploy
./scripts/deploy.sh --no-cache
```

### Rotating Secrets

```bash
# 1. Generate new secrets
openssl rand -hex 32  # API_KEY
openssl rand -hex 32  # SECRET_KEY (for JWT)

# 2. Update .env file

# 3. Restart services
./scripts/deploy.sh
```

### Clearing Old Data

```bash
# Clear response cache (Redis)
docker compose -f docker-compose.prod.yml exec redis redis-cli -a $REDIS_PASSWORD FLUSHDB

# Clear old backups (keep last 7 days)
find backups/ -type f -mtime +7 -delete
```

### Graceful Shutdown

The system handles graceful shutdown automatically:
1. Stops accepting new requests
2. Closes Redis connection pool
3. Closes PostgreSQL connection pool
4. Releases ChromaDB client
5. Waits 2 seconds for in-flight requests

To trigger graceful shutdown:
```bash
docker compose -f docker-compose.prod.yml stop api
```

---

**Last Updated:** 2025-12-24
**Version:** 2.0 (Production Ready)
