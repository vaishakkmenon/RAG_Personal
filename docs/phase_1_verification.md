# Phase 1 Verification Guide (Dockerized)

Run all these commands from your Windows host terminal.

## IMPORTANT: Rebuild Before Testing

Since we modified the application code, we must rebuild the test container to ensure it uses the latest code.

```bash
docker compose build test
```

## Phase 1.1: Security Headers & CORS

Verify that security headers and CORS are correctly configured.

```bash
docker compose run --rm test pytest tests/test_phase_1_1.py -v
```

**Expected Output:** All tests passed (green).

## Phase 1.2: Input Validation

Verify that invalid inputs are rejected.

```bash
docker compose run --rm test pytest tests/test_phase_1_2.py -v
```

**Expected Output:** All tests passed (green).

## Phase 1.4: Redis Authentication

Verify that Redis rejects unauthenticated connections.

1. **Apply Changes (Restart Redis)**
   ```bash
   docker compose down redis
   docker compose up -d redis
   ```

2. **Test Unauthenticated Access (Should Fail)**
   ```bash
   docker compose exec redis redis-cli PING
   ```
   **Expected Output:** `(error) NOAUTH Authentication required.`

3. **Test Authenticated Access (Should Succeed)**
   ```bash
   docker compose exec redis redis-cli -a devpassword123 PING
   ```
   **Expected Output:** `PONG`

## Phase 1.5: Dependency Security Scanning

Run the security audit tool inside the running `api` container.

1. **Ensure API Container is Running**
   ```bash
   docker compose up -d api
   ```

2. **Install and Run Audit Tool**
   ```bash
   docker compose exec api pip install pip-audit
   docker compose exec api pip-audit
   ```

**Expected Output:** A report of dependencies.
