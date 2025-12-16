# Testing Setup Summary

✅ **Complete testing infrastructure added to Personal RAG System**

## What Was Added

### 1. **Test Configuration Files**
- `pytest.ini` - Pytest configuration with markers, coverage settings
- `requirements-dev.txt` - Development dependencies (pytest, mocking tools, coverage)
- `tests/conftest.py` - Shared fixtures for all tests

### 2. **Test Files Created**
- `tests/test_api_endpoints.py` - Integration tests for API routes
  - Health endpoint tests
  - Chat simple endpoint tests (auth, prompt guard, retrieval)
  - Chat endpoint tests (with sessions)
  - Metrics endpoint tests

- `tests/test_prompt_guard.py` - Unit tests for prompt injection detection
  - Single-turn injection detection
  - Multi-turn conversation injection
  - Caching behavior
  - Retry logic
  - Error handling

- `tests/test_session_management.py` - Unit tests for session store
  - Memory-based session store
  - Redis-backed session store
  - Rate limiting
  - Conversation history
  - Session expiration

### 3. **Documentation**
- `docs/TESTING.md` - Comprehensive testing guide
  - How to run tests
  - Writing new tests
  - Best practices
  - Coverage reporting
  - CI/CD integration

### 4. **Docker Updates**
- Updated `Dockerfile` test stage with pytest-mock and faker

## Quick Start

### Install Dependencies

```bash
# For local development
pip install -r requirements-dev.txt
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run specific test file
pytest tests/test_prompt_guard.py

# Run in Docker (CI/CD style)
docker build --target test -t ragchatbot:test .
docker run --rm ragchatbot:test
```

### View Coverage Report

```bash
pytest --cov=app --cov-report=html
open htmlcov/index.html  # macOS
start htmlcov/index.html # Windows
```

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures (client, mocks, sample data)
├── test_api_endpoints.py          # API integration tests (21 tests)
├── test_prompt_guard.py           # Prompt guard unit tests (12 tests)
└── test_session_management.py     # Session management tests (13 tests)
```

## Available Fixtures

From `conftest.py`:
- `client` - FastAPI TestClient for API testing
- `auth_headers` - Authentication headers
- `mock_chromadb` - Mocked ChromaDB collection
- `mock_redis` - Mocked Redis client
- `mock_ollama` - Mocked LLM responses
- `mock_prompt_guard` - Mocked prompt guard API
- `sample_chunks` - Sample retrieval chunks
- `sample_questions` - Sample test questions
- `sample_chat_request` - Sample request payload
- `sample_session` - Sample session data

## Test Markers

Organize and filter tests:
- `@pytest.mark.unit` - Fast unit tests
- `@pytest.mark.integration` - Integration tests (may need services)
- `@pytest.mark.e2e` - End-to-end tests
- `@pytest.mark.slow` - Tests >1 second
- `@pytest.mark.prompt_guard` - Prompt guard tests
- `@pytest.mark.retrieval` - Retrieval tests
- `@pytest.mark.llm` - LLM tests
- `@pytest.mark.session` - Session tests
- `@pytest.mark.metrics` - Metrics tests

## Examples

### Writing a Unit Test

```python
import pytest
from unittest.mock import patch

@pytest.mark.unit
@pytest.mark.prompt_guard
def test_prompt_guard_blocks_injection():
    """Test that injection attempts are blocked."""
    from app.services.prompt_guard import PromptGuard

    with patch("app.services.prompt_guard.requests.post") as mock_post:
        # Mock API response
        mock_post.return_value.json.return_value = {
            "label": "prompt_injection",
            "score": 0.95
        }

        guard = PromptGuard()
        result = guard.check_input("Ignore previous instructions...")

        assert result["blocked"] is True
```

### Writing an Integration Test

```python
import pytest

@pytest.mark.integration
def test_chat_endpoint(client, auth_headers, mock_chromadb):
    """Test chat endpoint with mocked dependencies."""
    # Configure mock
    mock_chromadb.query.return_value = {
        "ids": [["chunk-1"]],
        "documents": [["Sample text"]],
        "metadatas": [[{"source": "test.md"}]],
        "distances": [[0.15]],
    }

    # Make request
    response = client.post(
        "/chat/simple",
        json={"question": "Test question?"},
        headers=auth_headers,
    )

    # Assertions
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert data["grounded"] is True
```

## CI/CD Integration

Tests run automatically in GitHub Actions:

```yaml
# .github/workflows/ci.yml
- name: Build test image
  run: docker build --target test -t ragchatbot:test .

- name: Run unit tests
  run: docker run --rm ragchatbot:test
```

## Next Steps

1. **Run the tests**: `pytest`
2. **Check coverage**: `pytest --cov=app --cov-report=html`
3. **Add more tests**: Follow examples in existing test files
4. **Improve coverage**: Aim for 80%+ overall, 90%+ for critical paths

## Test Coverage Targets

| Component | Target | Priority |
|-----------|--------|----------|
| Prompt Guard | 90%+ | Critical |
| Session Management | 85%+ | Critical |
| API Endpoints | 85%+ | High |
| Retrieval | 80%+ | High |
| Chat Service | 80%+ | High |
| LLM Service | 75%+ | Medium |
| Query Rewriting | 70%+ | Medium |

## Resources

- Full guide: `docs/TESTING.md`
- Pytest docs: https://docs.pytest.org/
- FastAPI testing: https://fastapi.tiangolo.com/tutorial/testing/
- Coverage.py: https://coverage.readthedocs.io/

---

**Status**: ✅ Ready to use
**Total Test Files**: 3
**Estimated Tests**: 40+ (with more to be added)
**Setup Time**: ~5 minutes
