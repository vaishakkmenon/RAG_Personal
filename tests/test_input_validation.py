from app.settings import settings

# client = TestClient(app) - Removed to use fixture from conftest
# Removed global API_KEY_HEADER


def test_validation_empty_message(client):
    """Test validation catches empty message."""
    response = client.post(
        "/chat",
        json={"question": "   ", "session_id": "test-session"},
        headers={"X-API-Key": settings.api_key},
    )
    assert response.status_code == 422
    assert "Question cannot be empty" in response.text


def test_validation_excessive_repetition(client):
    """Test validation catches repetitive spam."""
    # Create a message with > 90% repetition
    # "blah " * 100 -> 100 words, only 1 unique. 1 < 100/10 (10).
    repetitive_msg = "blah " * 100

    response = client.post(
        "/chat",
        json={"question": repetitive_msg, "session_id": "test-session"},
        headers={"X-API-Key": settings.api_key},
    )
    assert response.status_code == 422
    assert "Query contains excessive repetition" in response.text


def test_validation_invalid_session_id(client):
    """Test validation catches invalid session_id characters."""
    response = client.post(
        "/chat",
        json={
            "question": "Valid question?",
            "session_id": "invalid/session/id",  # slashes not allowed
        },
        headers={"X-API-Key": settings.api_key},
    )
    assert response.status_code == 422
    # Detailed error message usually contains "string_pattern_mismatch" or similar from pydantic

    response = client.post(
        "/chat",
        json={
            "question": "Valid question?",
            "session_id": "invalid$session",  # special chars not allowed
        },
        headers={"X-API-Key": settings.api_key},
    )
    assert response.status_code == 422


def test_validation_valid_request(client):
    """Test valid request passes validation."""
    # Note: This might fail if other dependencies (like LLM) are not mocked or available,
    # but we are testing Model validation which happens BEFORE logic execution.
    # However, if validation passes, it proceeds to logic.
    # To strictly test validation, we expect it NOT to be 422.
    # It might be 500 or 503 if downstream fails, but that means validation PASSED.

    response = client.post(
        "/chat",
        json={
            "question": "This is a valid question.",
            "session_id": "valid-session_123",
        },
        headers={"X-API-Key": settings.api_key},
    )
    assert response.status_code != 422
