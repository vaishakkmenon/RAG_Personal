# Dependency Update Process

## Routine Updates

1.  **Check for updates:**
    ```bash
    docker compose exec api pip list --outdated
    ```

2.  **Update specific package:**
    Update the version in `requirements.txt`.

3.  **Rebuild container:**
    ```bash
    docker compose up --build -d api
    ```

4.  **Verify:**
    Run tests to ensure no regressions.
    ```bash
    docker compose run --rm test
    ```

## Security Scanning

Run `pip-audit` to check for known vulnerabilities in current environment:

```bash
docker compose exec api pip install pip-audit
docker compose exec api pip-audit
```

If vulnerabilities are found:
1.  Identify the patched version.
2.  Update `requirements.txt` with the new version.
3.  Rebuild and test.
