import pytest
from unittest.mock import MagicMock
import json
from app.main import app
from app.api.dependencies import get_chat_service

# We don't need a global client, we'll use the one from conftest or create a new one if needed for overrides


@pytest.fixture
def mock_chat_service():
    """Create a mock chat service."""
    service = MagicMock()
    return service


@pytest.fixture
def override_dependency(mock_chat_service):
    """Override the get_chat_service dependency."""
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service
    yield
    app.dependency_overrides = {}


def test_chat_stream_happy_path(
    client, mock_chat_service, override_dependency, auth_headers
):
    # Mock the generator to yield metadata then tokens
    async def mock_generator(*args, **kwargs):
        # 1. Metadata event
        metadata = {
            "sources": [
                {
                    "id": "doc1",
                    "source": "test.md",
                    "text": "content",
                    "citation_index": 1,
                    "distance": 0.1,
                }
            ],
            "grounded": True,
            "session_id": "test-session",
            "is_ambiguous": False,
        }
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"

        # 2. Token events
        yield f"event: token\ndata: {json.dumps('Hello')}\n\n"
        yield f"event: token\ndata: {json.dumps(' World')}\n\n"

        # 3. Done event
        yield "event: done\ndata: [DONE]\n\n"

    # Setup the mock service to return this generator
    mock_chat_service.handle_chat_stream.side_effect = mock_generator

    response = client.post(
        "/chat/stream", json={"question": "Test question"}, headers=auth_headers
    )

    # Verify response
    assert response.status_code == 200
    # Content-Type can be 'text/event-stream; charset=utf-8'
    assert "text/event-stream" in response.headers["content-type"]

    content = response.text
    assert "event: metadata" in content
    assert '"citation_index": 1' in content
    assert "event: token" in content
    assert "Hello" in content
    assert "event: done" in content


def test_chat_stream_ungrounded(
    client, mock_chat_service, override_dependency, auth_headers
):
    async def mock_generator(*args, **kwargs):
        # Ungrounded metadata
        metadata = {
            "sources": [],
            "grounded": False,
            "session_id": "test-session",
            "is_ambiguous": False,
        }
        yield f"event: metadata\ndata: {json.dumps(metadata)}\n\n"
        yield f"event: token\ndata: {json.dumps('I do not know.')}\n\n"
        yield "event: done\ndata: [DONE]\n\n"

    mock_chat_service.handle_chat_stream.side_effect = mock_generator

    response = client.post(
        "/chat/stream", json={"question": "Unknown question"}, headers=auth_headers
    )

    assert response.status_code == 200
    assert '"grounded": false' in response.text
    assert "I do not know." in response.text


def test_chat_stream_error_handling(
    client, mock_chat_service, override_dependency, auth_headers
):
    async def mock_generator(*args, **kwargs):
        yield f"event: error\ndata: {json.dumps({'detail': 'Simulated error'})}\n\n"

    mock_chat_service.handle_chat_stream.side_effect = mock_generator

    response = client.post(
        "/chat/stream", json={"question": "Crash me"}, headers=auth_headers
    )

    assert response.status_code == 200
    assert "event: error" in response.text
    assert "Simulated error" in response.text
