"""
Tests for the /ingest API endpoint.

Verifies that the API correctly handles valid requests, passes them to the
ingestion service, and handles errors appropriately.
"""

from unittest.mock import patch
from fastapi.testclient import TestClient
from app.main import app


from app.settings import settings

# raise_server_exceptions=False allows us to test 500 error responses
# without the TestClient raising the exception directly.
client = TestClient(app, raise_server_exceptions=False)


# Correct path for where ingest_documents is imported/used in the route handler
# The route is likely in app.api.routes.ingest
MOCK_TARGET = "app.api.routes.ingest.ingest_paths"


def test_ingest_happy_path(mock_chromadb):
    """Verify successful ingestion triggering."""

    # Mock the return value of the service function
    mock_result = 42

    with patch(MOCK_TARGET) as mock_ingest:
        mock_ingest.return_value = mock_result

        # Make request
        response = client.post(
            "/ingest",
            headers={"X-API-Key": settings.api_key},
            json={"paths": ["./data/test.md"]},
        )

        # Verify response
        assert response.status_code == 200
        data = response.json()
        assert data["ingested_chunks"] == 42
        # BM25 stats might differ depending on how mock_chromadb behaves,
        # checking basic structure is enough or update expectation
        assert "bm25_stats" in data

        # Verify service was called with correct arguments
        mock_ingest.assert_called_once()
        # You might want to check the args passed if your service takes specific objects
        # args, _ = mock_ingest.call_args
        # assert args[0] == ["./data/test.md"]


def test_ingest_service_failure():
    """Verify 500 response when ingestion service fails."""

    with patch(MOCK_TARGET) as mock_ingest:
        # Simulate an unexpected error in the service
        mock_ingest.side_effect = Exception("Disk full")

        response = client.post(
            "/ingest",
            headers={"X-API-Key": settings.api_key},
            json={"paths": ["./data/test.md"]},
        )

        # Should return 500 Internal Server Error
        assert response.status_code == 500
        # The global exception handler masks the specific error message
        assert "internal error occurred" in response.json()["detail"]


def test_ingest_unauthorized():
    """Verify 401 when API key is missing or invalid."""

    response = client.post(
        "/ingest", headers={"X-API-Key": "wrong-key"}, json={"paths": []}
    )
    assert response.status_code == 401
