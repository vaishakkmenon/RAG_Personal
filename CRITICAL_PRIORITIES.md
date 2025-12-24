# Critical Priorities for Production Robustness

**Project:** Personal RAG System
**Created:** 2025-12-23

This is a focused list of high-impact items for a robust personal production deployment.

---

## Priority 1: Test Backup Restoration

**Status:** Not done
**Impact:** Critical - backups are useless without verified restoration

**Action Items:**
- [ ] Document restoration procedure for each service:
  - PostgreSQL: `psql < backup.sql`
  - Redis: Copy RDB file to data directory
  - ChromaDB: Extract tar.gz to data directory
- [ ] Perform test restoration to a fresh environment
- [ ] Verify data integrity after restoration
- [ ] Document RTO (Recovery Time Objective) - target time to restore
- [ ] Document RPO (Recovery Point Objective) - acceptable data loss window

**Files:** `scripts/backup.sh`

---

## Priority 2: Circuit Breaker for Groq API

**Status:** Not implemented
**Impact:** High - prevents cascading failures when external API is down

**Action Items:**
- [ ] Implement circuit breaker pattern in `app/services/llm.py`
- [ ] States: CLOSED (normal) → OPEN (failing) → HALF-OPEN (testing)
- [ ] Configuration:
  - Failure threshold: 5 consecutive failures
  - Recovery timeout: 30 seconds
  - Half-open max requests: 3
- [ ] Return graceful error when circuit is open
- [ ] Add metrics: `rag_circuit_breaker_state`, `rag_circuit_breaker_trips_total`

**Reference:** `app/services/llm.py`

---

## Priority 3: Graceful Shutdown

**Status:** Placeholder only
**Impact:** Medium-High - prevents data corruption during restarts

**Action Items:**
- [ ] Implement proper shutdown in `app/main.py` lifespan context
- [ ] Close database connections cleanly:
  - PostgreSQL session pool
  - Redis connection pool
  - ChromaDB client
- [ ] Allow in-flight requests to complete (connection draining)
- [ ] Set appropriate shutdown timeout in Docker (default 10s may be too short)
- [ ] Test with `docker-compose stop` and verify no errors in logs

**Files:** `app/main.py` (lifespan function), `docker-compose.prod.yml`

---

## Priority 4: Enable Test Coverage Reporting

**Status:** Disabled in pytest.ini
**Impact:** Low effort, high visibility

**Action Items:**
- [ ] Uncomment coverage settings in `pytest.ini`
- [ ] Run: `pytest --cov=app --cov-report=html --cov-report=term`
- [ ] Review coverage report and identify gaps
- [ ] Consider adding coverage threshold (e.g., 70% minimum)

**Files:** `pytest.ini`

---

## Priority 5: Add Trivy to CI Pipeline

**Status:** Manual only
**Impact:** Low effort, catches vulnerabilities automatically

**Action Items:**
- [ ] Add Trivy scan step to `.github/workflows/ci.yml`:
```yaml
- name: Run Trivy vulnerability scanner
  uses: aquasecurity/trivy-action@master
  with:
    image-ref: 'rag-personal:test'
    format: 'table'
    exit-code: '1'
    severity: 'CRITICAL,HIGH'
```
- [ ] Configure to fail on CRITICAL/HIGH vulnerabilities
- [ ] Add exceptions file for accepted risks if needed

**Files:** `.github/workflows/ci.yml`

---

## Quick Reference

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Test backup restoration | 2-3 hours | Critical |
| 2 | Circuit breaker for Groq | 2-4 hours | High |
| 3 | Graceful shutdown | 1-2 hours | Medium-High |
| 4 | Enable coverage reporting | 10 minutes | Medium |
| 5 | Add Trivy to CI | 10 minutes | Medium |

---

## Items Deprioritized (OK for Personal Use)

These are fine to skip for a personal project:

- mTLS internal communication (single VPS)
- Blue-green/canary deployments (single instance)
- Staging environment (direct to prod is acceptable)
- Feature flags (you control all changes)
- OpenTelemetry distributed tracing (request IDs sufficient)
- Data encryption at rest (low risk for personal resume data)

---

## Completion Tracking

- [ ] Priority 1 complete
- [ ] Priority 2 complete
- [ ] Priority 3 complete
- [ ] Priority 4 complete
- [ ] Priority 5 complete

**Target:** Complete all 5 priorities before considering the system production-hardened.
