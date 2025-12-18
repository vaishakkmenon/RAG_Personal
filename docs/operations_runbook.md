# Operations Runbook

Quick reference guide for operating and troubleshooting the Personal RAG System in production.

## Table of Contents

- [Quick Commands](#quick-commands)
- [Failure Modes & Recovery](#failure-modes--recovery)
- [Health Monitoring](#health-monitoring)
- [Common Issues](#common-issues)
- [Maintenance Procedures](#maintenance-procedures)

---

## Quick Commands

### Deploy Update
```bash
ssh deploy@server
cd ~/RAG_Personal
git pull
docker compose build
docker compose up -d
docker compose logs -f api
```

### Rollback to Previous Version
```bash
git log --oneline  # Find previous commit
git checkout <commit-hash>
docker compose build
docker compose up -d
docker compose logs -f api
```

### Check Service Status
```bash
docker compose ps
docker compose logs api --tail=50
docker compose logs redis --tail=50
```

### Restart Services
```bash
# Restart specific service
docker compose restart api

# Restart all services
docker compose restart

# Full restart (recreate containers)
docker compose down && docker compose up -d
```

### Check Logs
```bash
# Follow logs
docker compose logs -f api

# Last 100 lines
docker compose logs api --tail=100

# Filter for errors
docker compose logs api | grep ERROR

# With timestamps
docker compose logs -t api
```

### Clear Caches
```bash
# Clear Redis cache
docker compose exec redis redis-cli FLUSHDB

# Clear fallback cache via API
curl -X DELETE http://localhost:8000/admin/fallback-cache \
     -H "X-API-Key: your-api-key"
```

### Backup Now
```bash
./backup.sh
```

### Check Health
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health check with dependencies
curl http://localhost:8000/health/detailed

# Readiness probe
curl http://localhost:8000/health/ready

# Liveness probe
curl http://localhost:8000/health/live
```

---

## Failure Modes & Recovery

### Redis Failure

**Symptoms:**
- Detailed health check shows Redis as "degraded"
- Warning logs: "FALLBACK ACTIVATED: Using in-memory session store"
- Sessions don't persist across API restarts

**Impact:**
- **Severity:** LOW-MEDIUM
- Sessions work but are stored in-memory only
- Session data lost on API restart
- Response caching disabled
- Application continues to function normally

**Automatic Recovery:**
- Application automatically falls back to in-memory session storage
- No manual intervention required for continued operation

**Manual Recovery Steps:**
1. Check Redis container status:
   ```bash
   docker compose ps redis
   docker compose logs redis
   ```

2. If Redis is down, restart it:
   ```bash
   docker compose restart redis
   ```

3. If Redis won't start, check:
   - Disk space: `df -h`
   - Redis password configuration in `.env`
   - Redis data volume permissions

4. Test Redis connection:
   ```bash
   docker compose exec redis redis-cli -a $REDIS_PASSWORD PING
   # Should return: PONG
   ```

5. Restart API to reconnect to Redis:
   ```bash
   docker compose restart api
   ```

6. Verify recovery:
   ```bash
   curl http://localhost:8000/health/detailed
   # Check that redis status is "healthy"
   ```

**Prevention:**
- Monitor Redis memory usage
- Set up Redis persistence (AOF enabled in docker-compose.yml)
- Regular backups of Redis data

---

### ChromaDB Failure

**Symptoms:**
- Detailed health check shows ChromaDB as "degraded"
- Error logs: "ChromaDB query failed"
- Retrieval exceptions in chat endpoints
- Fallback cache being used (if available)

**Impact:**
- **Severity:** HIGH
- New queries may fail or return cached results
- No access to vector database for semantic search
- Application may return 500 errors for uncached queries

**Automatic Recovery:**
- If query was previously cached, fallback cache provides results
- Errors logged with full context
- Graceful error messages returned to users

**Manual Recovery Steps:**
1. Check ChromaDB status:
   ```bash
   docker compose logs api | grep -i chroma
   ```

2. Check disk space (ChromaDB needs disk I/O):
   ```bash
   df -h
   ```

3. Check ChromaDB directory permissions:
   ```bash
   ls -la data/chroma
   ```

4. Try restarting the API (ChromaDB is embedded):
   ```bash
   docker compose restart api
   docker compose logs -f api
   ```

5. If ChromaDB data is corrupted:
   ```bash
   # Backup current data
   cp -r data/chroma data/chroma_backup_$(date +%Y%m%d_%H%M%S)

   # Clear and rebuild
   curl -X DELETE http://localhost:8000/admin/chromadb \
        -H "X-API-Key: your-api-key"

   # Re-ingest documents
   docker compose run --rm test python scripts/ingest.py
   ```

6. Monitor fallback cache usage:
   ```bash
   curl http://localhost:8000/admin/fallback-cache/stats \
        -H "X-API-Key: your-api-key"
   ```

**Fallback Cache Details:**
- Automatically caches successful retrievals
- LRU cache with 100 query limit
- 1-hour TTL on cached entries
- Fuzzy matching for similar queries (80% similarity threshold)
- View stats: `GET /admin/fallback-cache/stats`
- Clear cache: `DELETE /admin/fallback-cache`

**Prevention:**
- Regular backups of `data/chroma` directory
- Monitor disk space
- Test ingestion pipeline regularly
- Keep fallback cache enabled (automatic)

---

### LLM API Failure (Groq)

**Symptoms:**
- Error logs: "Groq API error"
- Warning logs: "Falling back to Ollama"
- Increased latency (Ollama is slower than Groq)

**Impact:**
- **Severity:** LOW
- Automatic fallback to Ollama
- Slower response times but fully functional
- No user-visible errors

**Automatic Recovery:**
- Retry with exponential backoff (2 attempts)
- Automatic fallback to Ollama if Groq fails
- Prometheus metrics track fallback usage

**Manual Recovery Steps:**
1. Check Groq API status:
   ```bash
   # Check if API key is valid
   curl https://api.groq.com/openai/v1/models \
        -H "Authorization: Bearer $LLM_GROQ_API_KEY"
   ```

2. Check application logs:
   ```bash
   docker compose logs api | grep -i "groq\|llm"
   ```

3. Verify Groq API key in `.env`:
   ```bash
   grep LLM_GROQ_API_KEY .env
   ```

4. If using rate-limited Groq tier:
   - Wait for rate limit to reset
   - Monitor metrics for fallback usage
   - Consider upgrading Groq tier

5. Force Ollama-only mode temporarily:
   ```bash
   # Edit .env
   LLM_PROVIDER=ollama

   # Restart
   docker compose restart api
   ```

**Monitoring Fallback:**
```bash
# Check Prometheus metrics
curl http://localhost:8000/metrics | grep llm_fallback

# Look for:
# llm_fallback_count - number of times fallback was used
# llm_retry_count - number of retries before fallback
```

**Prevention:**
- Monitor Groq API rate limits
- Keep Ollama up-to-date
- Test fallback regularly
- Consider Groq paid tier for production

---

### Ollama Failure (Both Groq and Ollama Down)

**Symptoms:**
- Error logs: "LLM generation failed"
- 503 Service Unavailable responses
- Both primary and fallback LLM unavailable

**Impact:**
- **Severity:** CRITICAL
- Cannot generate responses
- Application returns 503 errors
- Only retrieval works (search results available)

**Manual Recovery Steps:**
1. Check Ollama container:
   ```bash
   docker compose ps ollama
   docker compose logs ollama
   ```

2. Restart Ollama:
   ```bash
   docker compose restart ollama
   ```

3. Test Ollama directly:
   ```bash
   curl http://localhost:11434/api/generate -d '{
     "model": "llama3.1:8b-instruct-q4_K_M",
     "prompt": "test",
     "stream": false
   }'
   ```

4. If model not found:
   ```bash
   docker compose exec ollama ollama pull llama3.1:8b-instruct-q4_K_M
   ```

5. Check GPU/CPU resources:
   ```bash
   docker stats ollama
   ```

**Prevention:**
- Monitor Ollama container health
- Ensure sufficient resources (RAM/CPU/GPU)
- Keep Ollama model pulled and ready
- Test LLM endpoints regularly

---

### Complete System Failure

**Symptoms:**
- All services down
- Cannot access API
- Docker containers not running

**Manual Recovery Steps:**
1. Check Docker daemon:
   ```bash
   sudo systemctl status docker
   sudo systemctl start docker
   ```

2. Check disk space:
   ```bash
   df -h
   ```

3. Check Docker logs:
   ```bash
   sudo journalctl -u docker -n 100
   ```

4. Restart all services:
   ```bash
   cd ~/RAG_Personal
   docker compose down
   docker compose up -d
   ```

5. Monitor startup:
   ```bash
   docker compose logs -f
   ```

6. Verify health:
   ```bash
   curl http://localhost:8000/health/detailed
   ```

---

## Health Monitoring

### Health Check Endpoints

1. **Basic Health** - `/health`
   - Always returns 200 if API is running
   - Quick check for uptime
   ```bash
   curl http://localhost:8000/health
   ```

2. **Detailed Health** - `/health/detailed`
   - Checks all dependencies (Redis, ChromaDB)
   - Returns "degraded" if any dependency fails
   - Best for monitoring dashboards
   ```bash
   curl http://localhost:8000/health/detailed
   ```

3. **Readiness Probe** - `/health/ready`
   - Kubernetes-style readiness check
   - Checks if app can serve traffic
   - Returns "not_ready" if critical dependencies fail
   ```bash
   curl http://localhost:8000/health/ready
   ```

4. **Liveness Probe** - `/health/live`
   - Kubernetes-style liveness check
   - Always returns 200 if process is alive
   - Use for restart decisions
   ```bash
   curl http://localhost:8000/health/live
   ```

### Interpreting Health Status

**Healthy:**
```json
{
  "status": "healthy",
  "dependencies": {
    "redis": "healthy",
    "chromadb": "healthy",
    "llm": "not_checked"
  }
}
```

**Degraded (Redis Down):**
```json
{
  "status": "degraded",
  "dependencies": {
    "redis": "degraded",
    "chromadb": "healthy",
    "llm": "not_checked"
  }
}
```
- Application functional with in-memory sessions
- No immediate action required

**Degraded (ChromaDB Down):**
```json
{
  "status": "degraded",
  "dependencies": {
    "redis": "healthy",
    "chromadb": "degraded",
    "llm": "not_checked"
  }
}
```
- May return cached results or errors
- **Action Required:** Investigate ChromaDB

### Prometheus Metrics

Key metrics to monitor:

```bash
# Request rate
curl http://localhost:8000/metrics | grep rag_request_total

# Error rate
curl http://localhost:8000/metrics | grep rag_request_total | grep status=\"500\"

# LLM fallback usage
curl http://localhost:8000/metrics | grep llm_fallback_count

# Session counts
curl http://localhost:8000/metrics | grep session_
```

---

## Common Issues

### Issue: High Memory Usage

**Symptoms:**
- Docker stats shows high memory
- OOM errors in logs
- Containers restarting

**Solutions:**
1. Check which service is using memory:
   ```bash
   docker stats
   ```

2. Clear caches:
   ```bash
   docker compose exec redis redis-cli FLUSHDB
   curl -X DELETE http://localhost:8000/admin/fallback-cache
   ```

3. Restart high-memory service:
   ```bash
   docker compose restart api
   ```

4. Check for memory leaks:
   ```bash
   docker compose logs api | grep -i "memory\|oom"
   ```

---

### Issue: Slow Response Times

**Symptoms:**
- Requests taking >5 seconds
- Timeout errors
- Users complaining about latency

**Diagnostic Steps:**
1. Check if using Ollama fallback:
   ```bash
   docker compose logs api | grep -i fallback
   ```

2. Check Ollama performance:
   ```bash
   docker stats ollama
   ```

3. Check ChromaDB query times:
   ```bash
   docker compose logs api | grep "Semantic search"
   ```

4. Check Redis latency:
   ```bash
   docker compose exec redis redis-cli --latency
   ```

**Solutions:**
- Ensure Groq API is working (faster than Ollama)
- Check if cross-encoder is enabled (adds latency)
- Consider disabling HyDE for faster queries
- Monitor disk I/O for ChromaDB

---

### Issue: Cannot Connect from Frontend

**Symptoms:**
- CORS errors in browser console
- 403 Forbidden responses
- Authentication failures

**Diagnostic Steps:**
1. Check CORS configuration:
   ```bash
   grep ALLOWED_ORIGINS .env
   ```

2. Check API key:
   ```bash
   grep API_KEY .env
   ```

3. Test with curl:
   ```bash
   curl -H "Origin: https://vaishakmenon.com" \
        -H "X-API-Key: your-key" \
        http://localhost:8000/chat/simple \
        -d '{"question":"test"}'
   ```

**Solutions:**
- Verify origin is in `ALLOWED_ORIGINS`
- Check API key matches between frontend and backend
- Check Cloudflare SSL mode (should be "Full (strict)")
- Verify middleware order in `app/main.py`

---

## Maintenance Procedures

### Regular Maintenance (Weekly)

1. **Check Logs for Errors:**
   ```bash
   docker compose logs api --since 7d | grep ERROR
   ```

2. **Review Health Status:**
   ```bash
   curl http://localhost:8000/health/detailed
   ```

3. **Check Disk Usage:**
   ```bash
   df -h
   du -sh data/chroma
   ```

4. **Clean Up Old Docker Images:**
   ```bash
   docker system prune -a --volumes --force
   ```

5. **Backup ChromaDB:**
   ```bash
   tar -czf backup_chroma_$(date +%Y%m%d).tar.gz data/chroma
   ```

### Monthly Maintenance

1. **Review Prometheus Metrics:**
   - Check average response times
   - Review error rates
   - Monitor fallback usage trends

2. **Update Dependencies:**
   ```bash
   pip-audit  # Check for vulnerabilities
   pip list --outdated  # Check for updates
   ```

3. **Test Disaster Recovery:**
   - Practice restoring from backup
   - Test failover scenarios
   - Verify monitoring alerts work

4. **Review and Rotate Logs:**
   ```bash
   docker compose logs api > logs/api_$(date +%Y%m).log
   ```

### Emergency Contacts

- **System Owner:** [Your contact info]
- **On-Call:** [On-call rotation]
- **Groq Support:** [Groq support details]
- **Infrastructure:** [Cloud provider support]

---

## Appendix: Environment Variables Reference

### Critical Variables
- `API_KEY` - API authentication (MUST be secure)
- `REDIS_PASSWORD` - Redis authentication
- `LLM_GROQ_API_KEY` - Groq API access
- `ALLOWED_ORIGINS` - CORS allowed origins

### Optional Performance Tuning
- `LLM_OLLAMA_TIMEOUT` - Ollama timeout (default: 60s)
- `RESPONSE_CACHE_ENABLED` - Enable/disable caching (default: true)
- `SESSION_QUERIES_PER_HOUR` - Rate limiting (default: 20)

### Monitoring
- `GRAFANA_ADMIN_PASSWORD` - Grafana dashboard access
- `PROMETHEUS_RETENTION_DAYS` - Metrics retention (default: 15)

---

**Last Updated:** 2025-12-17
**Version:** 1.0
