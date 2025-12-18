"""
Tests for admin endpoints.

Tests administrative operations like ChromaDB management
and fallback cache management.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestAdminFallbackCache:
    """Tests for fallback cache admin endpoints."""

    def test_get_fallback_cache_stats(self, client: TestClient, auth_headers: dict):
        """Test getting fallback cache statistics."""
        response = client.get("/admin/fallback-cache/stats", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "active"
        assert "statistics" in data
        assert "cache_size" in data["statistics"]
        assert "hits" in data["statistics"]
        assert "misses" in data["statistics"]
        assert "hit_rate" in data["statistics"]
        assert "fallback_uses" in data["statistics"]

    def test_clear_fallback_cache(self, client: TestClient, auth_headers: dict):
        """Test clearing the fallback cache."""
        # First add some entries to the cache
        from app.retrieval.fallback_cache import get_fallback_cache
        cache = get_fallback_cache()
        cache.cache_results("test query 1", [{"id": "1", "text": "test"}])
        cache.cache_results("test query 2", [{"id": "2", "text": "test"}])

        # Get stats before clearing
        stats_response = client.get("/admin/fallback-cache/stats", headers=auth_headers)
        assert stats_response.status_code == 200

        # Clear cache
        response = client.delete("/admin/fallback-cache", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "entries_cleared" in data
        assert data["entries_cleared"] >= 0

        # Verify cache is empty
        stats_after = client.get("/admin/fallback-cache/stats", headers=auth_headers)
        assert stats_after.json()["statistics"]["cache_size"] == 0

    def test_cleanup_fallback_cache(self, client: TestClient, auth_headers: dict):
        """Test cleaning up expired entries from fallback cache."""
        response = client.post("/admin/fallback-cache/cleanup", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "entries_removed" in data
        assert "current_cache_size" in data
        assert data["entries_removed"] >= 0

    def test_admin_endpoints_require_auth(self, client: TestClient):
        """Test that admin endpoints require authentication."""
        # Try without auth headers
        response = client.get("/admin/fallback-cache/stats")
        assert response.status_code == 401

        response = client.delete("/admin/fallback-cache")
        assert response.status_code == 401

        response = client.post("/admin/fallback-cache/cleanup")
        assert response.status_code == 401


@pytest.mark.integration
class TestAdminChromaDB:
    """Tests for ChromaDB admin endpoints."""

    def test_get_chromadb_status(self, client: TestClient, auth_headers: dict):
        """Test getting ChromaDB status."""
        response = client.get("/admin/chromadb/status", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        # Should have status information
        assert "status" in data
        assert "path" in data
        assert "exists" in data

    @patch("app.api.routes.admin.reset_collection")
    def test_clear_chromadb(self, mock_reset, client: TestClient, auth_headers: dict):
        """Test clearing ChromaDB collection."""
        # Mock successful reset
        mock_reset.return_value = None

        response = client.delete("/admin/chromadb", headers=auth_headers)

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "success"
        assert "collection_name" in data
        mock_reset.assert_called_once()

    @patch("app.api.routes.admin.reset_collection")
    def test_clear_chromadb_handles_errors(self, mock_reset, client: TestClient, auth_headers: dict):
        """Test that ChromaDB clear handles errors gracefully."""
        # Mock error during reset
        mock_reset.side_effect = Exception("ChromaDB error")

        response = client.delete("/admin/chromadb", headers=auth_headers)

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_chromadb_endpoints_require_auth(self, client: TestClient):
        """Test that ChromaDB admin endpoints require authentication."""
        response = client.get("/admin/chromadb/status")
        assert response.status_code == 401

        response = client.delete("/admin/chromadb")
        assert response.status_code == 401


@pytest.mark.integration
class TestAdminEndpointsSecurity:
    """Security tests for admin endpoints."""

    def test_admin_endpoints_reject_invalid_api_key(self, client: TestClient):
        """Test that admin endpoints reject invalid API keys."""
        invalid_headers = {"X-API-Key": "invalid-key"}

        endpoints = [
            ("GET", "/admin/fallback-cache/stats"),
            ("DELETE", "/admin/fallback-cache"),
            ("POST", "/admin/fallback-cache/cleanup"),
            ("GET", "/admin/chromadb/status"),
            ("DELETE", "/admin/chromadb"),
        ]

        for method, endpoint in endpoints:
            if method == "GET":
                response = client.get(endpoint, headers=invalid_headers)
            elif method == "DELETE":
                response = client.delete(endpoint, headers=invalid_headers)
            elif method == "POST":
                response = client.post(endpoint, headers=invalid_headers)

            assert response.status_code == 401, f"Expected 401 for {method} {endpoint}"

    def test_admin_endpoints_require_valid_methods(self, client: TestClient, auth_headers: dict):
        """Test that admin endpoints only accept correct HTTP methods."""
        # GET on DELETE endpoint should fail
        response = client.get("/admin/fallback-cache", headers=auth_headers)
        assert response.status_code == 405  # Method Not Allowed

        # POST on GET endpoint should fail
        response = client.post("/admin/fallback-cache/stats", headers=auth_headers)
        assert response.status_code == 405  # Method Not Allowed
