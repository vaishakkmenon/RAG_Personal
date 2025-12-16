# Testing Guide

Comprehensive testing guide for the Personal RAG System.

## Overview

The testing suite includes:
- **Unit Tests**: Individual component testing (prompt guard, sessions, retrieval)
- **Integration Tests**: API endpoint testing with mocked dependencies
- **E2E Tests**: Full pipeline testing with real services
- **Coverage Reporting**: Track test coverage across codebase

## Quick Start

### Install Test Dependencies

```bash
# Install development dependencies (includes pytest, mocking tools, etc.)
pip install -r requirements-dev.txt
```

### Run All Tests

```bash
# Run all tests with coverage
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_api_endpoints.py

# Run specific test class
pytest tests/test_prompt_guard.py::TestPromptGuardSingleTurn

# Run specific test
pytest tests/test_api_endpoints.py::TestHealthEndpoint::test_health_check_success
```

### Run Tests by Marker

```bash
# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run only prompt guard tests
pytest -m prompt_guard

# Run everything except slow tests
pytest -m "not slow"

# Run multiple markers
pytest -m "unit and prompt_guard"
```

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures and test configuration
├── test_api_endpoints.py          # Integration tests for API routes
├── test_prompt_guard.py           # Unit tests for prompt injection detection
├── test_session_management.py     # Unit tests for session store
├── test_retrieval.py              # Unit tests for search and reranking
├── test_llm_service.py            # Unit tests for LLM integration
└── test_metrics.py                # Unit tests for Prometheus metrics
```

## Writing Tests

### Using Fixtures

Fixtures are defined in `tests/conftest.py` and available to all tests:

```python
def test_chat_endpoint(client, auth_headers, sample_chunks):
    """
    client: FastAPI TestClient
    auth_headers: {"X-API-Key": "test-api-key"}
    sample_chunks: List of sample retrieval chunks
    """
    response = client.post(
        "/chat/simple",
        json={"question": "What is my Python experience?"},
        headers=auth_headers,
    )
    assert response.status_code == 200
```

### Mocking External Services

Use `@patch` decorator to mock external dependencies:

```python
from unittest.mock import patch, MagicMock

@patch("app.services.llm.generate_with_ollama")
def test_llm_generation(mock_generate):
    """Test LLM generation with mocked Ollama."""
    mock_generate.return_value = "This is a test response."

    # Your test code here
    result = some_function_that_calls_llm()

    assert "test response" in result
    mock_generate.assert_called_once()
```

### Testing API Endpoints

```python
def test_endpoint_with_auth(client, auth_headers):
    """Test authenticated endpoint."""
    response = client.post(
        "/chat/simple",
        json={"question": "Test question"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
```

### Testing Error Cases

```python
def test_invalid_api_key(client):
    """Test that invalid API key is rejected."""
    response = client.post(
        "/chat/simple",
        json={"question": "Test"},
        headers={"X-API-Key": "wrong-key"},
    )

    assert response.status_code == 403
    assert "API key" in response.json()["detail"]
```

## Test Markers

Organize tests with markers (defined in `pytest.ini`):

```python
@pytest.mark.unit
def test_unit_logic():
    """Fast unit test for pure functions."""
    pass

@pytest.mark.integration
def test_api_endpoint(client):
    """Integration test requiring API."""
    pass

@pytest.mark.slow
def test_expensive_operation():
    """Slow test (>1s)."""
    pass

@pytest.mark.e2e
def test_full_pipeline():
    """End-to-end test requiring all services."""
    pass
```

## Coverage Reports

### Generate Coverage Report

```bash
# Run tests with coverage
pytest --cov=app --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
xdg-open htmlcov/index.html # Linux
```

### Coverage Targets

Aim for:
- **Overall**: 80%+ coverage
- **Critical paths**: 90%+ coverage (prompt guard, auth, retrieval)
- **API endpoints**: 85%+ coverage

### Check Coverage Threshold

```bash
# Fail if coverage below 80%
pytest --cov=app --cov-fail-under=80
```

## Continuous Integration (CI)

### GitHub Actions

Tests run automatically on every push and PR to `main`:

```yaml
# .github/workflows/ci.yml
- name: Run unit tests
  run: docker run --rm ragchatbot:test

- name: Run linter
  run: docker run --rm ragchatbot:test ruff check .
```

### Running Tests in Docker

```bash
# Build test image
docker build --target test -t ragchatbot:test .

# Run tests in container
docker run --rm ragchatbot:test
```

## Best Practices

### 1. **Isolate Tests**
- Each test should be independent
- Use fixtures for shared setup
- Don't rely on test execution order

### 2. **Mock External Dependencies**
- Mock ChromaDB, Redis, LLM APIs
- Use `@patch` for external services
- Avoid network calls in unit tests

### 3. **Test Edge Cases**
- Empty results
- Invalid inputs
- Error conditions
- Rate limiting
- Timeout scenarios

### 4. **Keep Tests Fast**
- Unit tests should run in milliseconds
- Mark slow tests with `@pytest.mark.slow`
- Use mocks to avoid expensive operations

### 5. **Use Descriptive Names**
```python
# Good
def test_prompt_guard_blocks_injection_attempt():
    pass

# Bad
def test_guard():
    pass
```

### 6. **Test One Thing Per Test**
```python
# Good - tests one behavior
def test_rate_limit_blocks_excessive_requests():
    pass

# Bad - tests multiple things
def test_session_management():
    # Tests creation, rate limiting, and expiration all in one
    pass
```

## Troubleshooting

### Test Discovery Issues

If pytest doesn't find your tests:

```bash
# Check test discovery
pytest --collect-only

# Ensure test files start with test_
mv my_tests.py test_my_component.py

# Ensure test functions start with test_
def test_my_function():  # ✓
def check_my_function():  # ✗
```

### Import Errors

If tests can't import app modules:

```bash
# Install package in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Fixture Not Found

If fixture is not available:

```bash
# Ensure conftest.py is in tests/ directory
tests/conftest.py

# Check fixture is defined and not misspelled
# Fixture: client
# Usage:   test_something(client)  ✓
# Usage:   test_something(test_client)  ✗
```

### Mock Not Working

If mocks aren't being called:

```python
# Ensure you're patching the right location
# Mock where it's USED, not where it's DEFINED

# ✓ Correct - patch where it's imported
@patch("app.api.routes.chat.search")

# ✗ Wrong - patch where it's defined
@patch("app.retrieval.search.search")
```

## Advanced Topics

### Parameterized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("question,expected_grounded", [
    ("What is my Python experience?", True),
    ("Do I have underwater basket weaving experience?", False),
    ("What courses did I take?", True),
])
def test_grounding_detection(question, expected_grounded, client, auth_headers):
    response = client.post("/chat/simple", json={"question": question}, headers=auth_headers)
    data = response.json()
    assert data["grounded"] == expected_grounded
```

### Async Tests

For async functions:

```python
import pytest

@pytest.mark.asyncio
async def test_async_function():
    result = await some_async_function()
    assert result == "expected"
```

### Fixture Scopes

Control fixture lifecycle:

```python
@pytest.fixture(scope="session")
def expensive_setup():
    """Run once for entire test session."""
    pass

@pytest.fixture(scope="function")
def per_test_setup():
    """Run before each test (default)."""
    pass
```

## References

- [Pytest Documentation](https://docs.pytest.org/)
- [FastAPI Testing](https://fastapi.tiangolo.com/tutorial/testing/)
- [unittest.mock Guide](https://docs.python.org/3/library/unittest.mock.html)
- [Coverage.py Documentation](https://coverage.readthedocs.io/)
