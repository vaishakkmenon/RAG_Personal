# Critical Priorities for Production Robustness

**Project:** Personal RAG System
**Created:** 2025-12-23

This is a focused list of high-impact items for a robust personal production deployment.

---

## Priority 1: Test Backup Restoration

**Status:** COMPLETE (2025-12-23)
**Impact:** Critical - backups are useless without verified restoration

**Action Items:**
- [x] Document restoration procedure for each service:
  - PostgreSQL: `psql < backup.sql`
  - Redis: Copy RDB file to data directory
  - ChromaDB: Extract tar.gz to data directory
- [x] Perform test restoration to a fresh environment
- [x] Verify data integrity after restoration
- [x] Document RTO (Recovery Time Objective) - ~48 seconds
- [ ] Document RPO (Recovery Point Objective) - 24 hours (daily backups)

**Files:** `scripts/backup.sh`, `scripts/restore.sh`, `scripts/restore.ps1`

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

**Status:** COMPLETE (2025-12-23)
**Impact:** Low effort, high visibility

**Action Items:**
- [x] Uncomment coverage settings in `pytest.ini`
- [x] Run: `pytest --cov=app --cov-report=html --cov-report=term`
- [x] Review coverage report and identify gaps
- [x] Consider adding coverage threshold (e.g., 70% minimum) - Set to 60%

**Files:** `pytest.ini`

---

## Priority 5: Add Trivy to CI Pipeline

**Status:** COMPLETE (2025-12-23)
**Impact:** Low effort, catches vulnerabilities automatically

**Action Items:**
- [x] Add Trivy scan step to `.github/workflows/ci.yml`
- [x] Configure to fail on CRITICAL/HIGH vulnerabilities
- [ ] Add exceptions file for accepted risks if needed (as needed)

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

- [x] Priority 1 complete (2025-12-23) - RTO: ~48 seconds
- [ ] Priority 2 complete
- [ ] Priority 3 complete
- [x] Priority 4 complete (2025-12-23) - Coverage threshold: 60%
- [x] Priority 5 complete (2025-12-23) - Trivy scans CRITICAL/HIGH

**Target:** Complete all 5 priorities before considering the system production-hardened.
