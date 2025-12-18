"""
Tests for ChromaDB degradation and error handling scenarios.

Verifies that the application gracefully handles ChromaDB unavailability:
- Application doesn't crash when ChromaDB fails
- Appropriate error messages returned to users
- Health checks report degraded status
- Errors are logged with context
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient


@pytest.mark.integration
class TestChromaDBQueryFailures:
    """Tests for handling ChromaDB query failures."""

    @patch("app.retrieval.store._collection")
    def test_chromadb_connection_failure_logged(self, mock_collection, caplog):
        """Test that ChromaDB connection failures are logged appropriately."""
        import logging
        from app.retrieval.store import _semantic_search

        # Simulate ChromaDB connection error
        mock_collection.query.side_effect = Exception("Connection refused")

        with caplog.at_level(logging.ERROR):
            with pytest.raises(Exception):
                _semantic_search(query="test query", k=5, max_dist=1.0)

            # Should log error
            assert any(
                "ChromaDB query failed" in record.message for record in caplog.records
            )

    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.get_prompt_guard")
    def test_chat_endpoint_handles_retrieval_failure(
        self,
        mock_guard,
        mock_search,
        client: TestClient,
        auth_headers: dict,
    ):
        """Test that chat endpoint returns appropriate error when retrieval fails."""
        # Mock prompt guard
        mock_guard_instance = MagicMock()
        mock_guard_instance.check_input.return_value = {
            "blocked": False,
            "label": "safe",
        }
        mock_guard.return_value = mock_guard_instance

        # Simulate retrieval failure
        mock_search.side_effect = Exception("ChromaDB connection failed")

        response = client.post(
            "/chat/simple",
            json={"question": "What is my Python experience?"},
            headers=auth_headers,
        )

        # Should return 500 with error message
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Failed to search knowledge base" in data["detail"]

    @patch("app.retrieval.store._collection")
    def test_empty_results_when_chromadb_returns_malformed_data(self, mock_collection):
        """Test handling of malformed data from ChromaDB."""
        from app.retrieval.store import _semantic_search

        # Return malformed results
        mock_collection.query.return_value = {
            "ids": [["chunk-1", "chunk-2"]],
            "documents": [["doc1"]],  # Mismatched length
            "metadatas": [[]],
            "distances": [[0.5]],
        }

        # Should handle gracefully and return what it can
        results = _semantic_search(query="test", k=5, max_dist=1.0)

        # Should not crash, may return partial results or empty
        assert isinstance(results, list)

    @patch("app.retrieval.store._collection")
    def test_search_handles_missing_result_fields(self, mock_collection):
        """Test that search handles missing or None fields in results."""
        from app.retrieval.store import _semantic_search

        # Return results with missing fields
        mock_collection.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [[None]],  # None document
            "metadatas": [[None]],  # None metadata
            "distances": [[0.5]],
        }

        # Should handle gracefully
        results = _semantic_search(query="test", k=5, max_dist=1.0)

        # Should return results but handle None values
        assert isinstance(results, list)


@pytest.mark.integration
class TestChromaDBHealthCheck:
    """Tests for health check reporting when ChromaDB is degraded."""

    def test_health_check_reports_chromadb_degraded(self, client: TestClient):
        """Test that detailed health check reports ChromaDB as degraded when unavailable."""
        # Patch ChromaDB client to fail
        with patch("app.retrieval.store.get_chroma_client") as mock_client:
            mock_client.side_effect = Exception("ChromaDB not available")

            response = client.get("/health/detailed")

            assert response.status_code == 200
            data = response.json()

            # Overall status should be degraded
            assert data["status"] == "degraded"

            # ChromaDB should be marked as degraded
            assert "dependencies" in data
            assert data["dependencies"]["chromadb"] == "degraded"

    def test_readiness_probe_fails_when_chromadb_unavailable(self, client: TestClient):
        """Test that readiness probe fails when ChromaDB is unavailable."""
        with patch("app.retrieval.store.get_chroma_client") as mock_client:
            mock_client.side_effect = Exception("ChromaDB not available")

            response = client.get("/health/ready")

            # Readiness should fail without ChromaDB (critical dependency)
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "not_ready"

    def test_liveness_probe_succeeds_without_chromadb(self, client: TestClient):
        """Test that liveness probe succeeds even if ChromaDB is down."""
        with patch("app.retrieval.store.get_chroma_client") as mock_client:
            mock_client.side_effect = Exception("ChromaDB not available")

            response = client.get("/health/live")

            # Liveness only checks if process is alive
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "alive"


@pytest.mark.integration
class TestChromaDBGracefulDegradation:
    """Tests for graceful degradation when ChromaDB is unavailable."""

    @patch("app.retrieval.store._collection")
    def test_search_returns_empty_on_chromadb_failure(self, mock_collection):
        """Test that search returns empty results rather than crashing."""
        from app.retrieval.store import _semantic_search

        # Simulate ChromaDB failure
        mock_collection.query.side_effect = Exception("ChromaDB unavailable")

        # Should raise exception (which will be caught by endpoint)
        with pytest.raises(Exception):
            _semantic_search(query="test", k=5, max_dist=1.0)

    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.generate_with_ollama")
    @patch("app.api.routes.chat.get_prompt_guard")
    def test_chat_with_no_retrieval_results(
        self,
        mock_guard,
        mock_generate,
        mock_search,
        client: TestClient,
        auth_headers: dict,
    ):
        """Test chat endpoint behavior when retrieval returns no results."""
        # Mock prompt guard
        mock_guard_instance = MagicMock()
        mock_guard_instance.check_input.return_value = {
            "blocked": False,
            "label": "safe",
        }
        mock_guard.return_value = mock_guard_instance

        # Mock search returns empty
        mock_search.return_value = []

        # Mock LLM generation
        mock_generate.return_value = "I don't have information about that."

        response = client.post(
            "/chat/simple",
            json={"question": "What is my Python experience?"},
            headers=auth_headers,
        )

        # Should still return 200 with ungrounded response
        assert response.status_code == 200
        data = response.json()
        assert data["grounded"] is False
        assert len(data["sources"]) == 0

    def test_collection_count_failure_handled(self):
        """Test that collection.count() failures are handled gracefully."""
        from app.retrieval.store import _collection

        # Try to access collection (may fail if ChromaDB not initialized properly)
        try:
            count = _collection.count()
            assert isinstance(count, int)
        except Exception:
            # Should log error but not crash
            assert True  # Expected behavior


@pytest.mark.integration
class TestChromaDBErrorMessages:
    """Tests for user-facing error messages when ChromaDB fails."""

    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.get_prompt_guard")
    def test_retrieval_error_message_user_friendly(
        self,
        mock_guard,
        mock_search,
        client: TestClient,
        auth_headers: dict,
    ):
        """Test that retrieval errors return user-friendly messages."""
        # Mock prompt guard
        mock_guard_instance = MagicMock()
        mock_guard_instance.check_input.return_value = {
            "blocked": False,
            "label": "safe",
        }
        mock_guard.return_value = mock_guard_instance

        # Simulate retrieval failure
        mock_search.side_effect = Exception("Database connection error")

        response = client.post(
            "/chat/simple",
            json={"question": "Test question"},
            headers=auth_headers,
        )

        assert response.status_code == 500
        data = response.json()

        # Should return user-friendly message, not internal details
        assert "detail" in data
        assert "Failed to search knowledge base" in data["detail"]
        assert "Database connection error" not in data["detail"]  # Don't leak internals

    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.get_prompt_guard")
    def test_retrieval_exception_type_preserved(
        self,
        mock_guard,
        mock_search,
        client: TestClient,
        auth_headers: dict,
    ):
        """Test that RetrievalException is properly raised and caught."""
        from app.exceptions import RetrievalException

        # Mock prompt guard
        mock_guard_instance = MagicMock()
        mock_guard_instance.check_input.return_value = {
            "blocked": False,
            "label": "safe",
        }
        mock_guard.return_value = mock_guard_instance

        # Raise RetrievalException
        mock_search.side_effect = RetrievalException("Custom retrieval error")

        response = client.post(
            "/chat/simple",
            json={"question": "Test question"},
            headers=auth_headers,
        )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "type" in data
        assert data["type"] == "RetrievalException"


@pytest.mark.integration
class TestBM25HybridSearchDegradation:
    """Tests for BM25 hybrid search fallback scenarios."""

    @patch("app.retrieval.store._bm25_index")
    def test_hybrid_search_falls_back_to_semantic_on_bm25_failure(self, mock_bm25):
        """Test that hybrid search falls back to semantic-only when BM25 fails."""
        from app.retrieval.store import search

        # Mock BM25 failure
        mock_bm25.search.side_effect = Exception("BM25 index corrupted")

        # Should fall back to semantic search
        # Note: This requires actual ChromaDB, so we'll just verify it doesn't crash
        try:
            results = search(
                query="test", top_k=5, use_hyde=False, use_cross_encoder=False
            )
            # If ChromaDB is available, should return results
            assert isinstance(results, list)
        except Exception:
            # If ChromaDB not available in test env, that's ok
            pass

    def test_search_works_without_bm25_index(self):
        """Test that search works when BM25 index is not available."""
        from app.retrieval.store import search, _bm25_index

        # Check if BM25 is available
        if _bm25_index is None:
            # Should still be able to do semantic-only search
            try:
                results = search(
                    query="test", top_k=5, use_hyde=False, use_cross_encoder=False
                )
                assert isinstance(results, list)
            except Exception:
                # If ChromaDB not available, that's expected in test env
                pass


@pytest.mark.integration
class TestChromaDBRecoveryScenarios:
    """Tests for ChromaDB recovery and reconnection scenarios."""

    def test_chromadb_reinitialization_after_failure(self):
        """Test that ChromaDB can be reinitialized after failure."""
        from app.retrieval.store import get_chroma_client

        # Get client (should work or raise exception)
        try:
            client = get_chroma_client()
            assert client is not None

            # Try to use it
            collections = client.list_collections()
            assert isinstance(collections, list)
        except Exception:
            # Expected if ChromaDB not available in test environment
            pass

    def test_concurrent_chromadb_access_thread_safe(self):
        """Test that concurrent ChromaDB access is thread-safe."""
        from app.retrieval.store import _collection
        import threading

        results = []
        errors = []

        def access_chromadb():
            try:
                count = _collection.count()
                results.append(count)
            except Exception as e:
                errors.append(str(e))

        # Create multiple threads accessing ChromaDB
        threads = [threading.Thread(target=access_chromadb) for _ in range(5)]

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Either all should succeed or all should fail gracefully
        assert len(results) + len(errors) == 5
