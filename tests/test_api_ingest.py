"""
Tests for the /ingest API endpoint.

Verifies that the API correctly handles valid requests, passes them to the
ingestion service, and handles errors appropriately.
"""

from unittest.mock import patch


from app.settings import settings
from app.core.auth import get_current_admin_user
from app.models import User

# raise_server_exceptions=False allows us to test 500 error responses
# without the TestClient raising the exception directly.
# client = TestClient(app, raise_server_exceptions=False)


# Correct path for where ingest_documents is imported/used in the route handler
# The route is likely in app.api.routes.ingest
MOCK_TARGET = "app.api.routes.ingest.ingest_paths"


def test_ingest_happy_path(client, mock_chromadb):
    """Verify successful ingestion triggering."""

    # Mock admin user
    admin_user = User(username="admin", is_superuser=True)
    client.app.dependency_overrides[get_current_admin_user] = lambda: admin_user

    try:
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
            assert "bm25_stats" in data

            # Verify service was called
            mock_ingest.assert_called_once()
    finally:
        client.app.dependency_overrides = {}


def test_ingest_service_failure(client):
    """Verify 500 response when ingestion service fails."""

    # Mock admin user
    admin_user = User(username="admin", is_superuser=True)
    client.app.dependency_overrides[get_current_admin_user] = lambda: admin_user

    try:
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
            assert "internal error occurred" in response.json()["detail"]
    finally:
        client.app.dependency_overrides = {}


def test_ingest_unauthorized(client):
    """Verify 401 when authentication fails."""
    # Ensure no dependency override is active
    client.app.dependency_overrides = {}

    response = client.post(
        "/ingest", headers={"X-API-Key": "wrong-key"}, json={"paths": []}
    )
    assert response.status_code == 401
