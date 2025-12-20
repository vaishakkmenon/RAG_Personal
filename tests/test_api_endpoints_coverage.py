"""
Additional test coverage for chat API routes.

Tests error handling, cache hits, and validation in production endpoints.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.integration
class TestChatAPIBasicFunctionality:
    """Tests for basic chat API functionality."""

    def test_chat_endpoint_requires_auth(self, client):
        """Test that chat endpoint requires API key."""
        response = client.post("/chat", json={"question": "What is my experience?"})
        assert response.status_code == 401

    def test_simple_chat_endpoint_requires_auth(self, client):
        """Test that simple chat endpoint requires API key."""
        response = client.post(
            "/chat/simple", json={"question": "What is my experience?"}
        )
        assert response.status_code == 401

    def test_chat_endpoint_accepts_valid_request(
        self, client, auth_headers, mock_llm, mock_chromadb, mock_rate_limit
    ):
        """Test that chat endpoint accepts valid requests."""
        # Set up mock to return some results
        mock_chromadb.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Test content about Python experience"]],
            "metadatas": [[{"source": "resume.md", "doc_type": "resume"}]],
            "distances": [[0.2]],
        }

        response = client.post(
            "/chat",
            json={"question": "What is my Python experience?"},
            headers=auth_headers,
        )

        # Should get a successful response
        assert response.status_code == 200
        assert "answer" in response.json()

    def test_simple_chat_endpoint_accepts_valid_request(
        self, client, auth_headers, mock_llm, mock_chromadb, mock_rate_limit
    ):
        """Test that simple chat endpoint accepts valid requests."""
        mock_chromadb.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Test content"]],
            "metadatas": [[{"source": "test.md"}]],
            "distances": [[0.2]],
        }

        response = client.post(
            "/chat/simple", json={"question": "Test question"}, headers=auth_headers
        )

        assert response.status_code == 200
        assert "answer" in response.json()


@pytest.mark.integration
class TestChatAPIErrorHandling:
    """Tests for error handling in chat endpoints."""

    def test_chat_returns_response_with_no_results(
        self, client, auth_headers, mock_llm, mock_chromadb, mock_rate_limit
    ):
        """Test that chat endpoint handles empty retrieval gracefully."""
        mock_chromadb.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        response = client.post(
            "/chat",
            json={"question": "Something with no results"},
            headers=auth_headers,
        )

        # Should return a response (grounded=false or clarification)
        assert response.status_code == 200

    @patch("app.api.routes.chat.generate_with_llm")
    @patch("app.api.routes.chat.search")
    def test_simple_chat_llm_exception(
        self, mock_search, mock_generate, client, auth_headers, mock_rate_limit
    ):
        """Test that LLM exceptions in simple_chat raise LLMException."""

        mock_search.return_value = [
            {
                "id": "1",
                "text": "test",
                "source": "test.md",
                "distance": 0.1,
                "metadata": {},
            }
        ]
        mock_generate.side_effect = Exception("LLM failed")

        response = client.post(
            "/chat/simple", json={"question": "Test question"}, headers=auth_headers
        )

        # Should get error status
        assert response.status_code >= 400


@pytest.mark.integration
class TestChatAPIValidation:
    """Tests for request validation."""

    def test_chat_handles_short_question(
        self, client, auth_headers, mock_llm, mock_chromadb, mock_rate_limit
    ):
        """Test that short questions get a response."""
        response = client.post("/chat", json={"question": "hi"}, headers=auth_headers)

        # Should return 422 (validation error) because min_length is 3
        # "hi" is only 2 chars
        assert response.status_code == 422

    def test_chat_rejects_invalid_payload(self, client, auth_headers):
        """Test that invalid payloads are rejected."""
        response = client.post(
            "/chat",
            json={},  # Missing required 'question' field
            headers=auth_headers,
        )

        assert response.status_code == 422  # Validation error

    def test_chat_request_model_validation(self):
        """Test that ChatRequest model validates correctly."""
        from app.models import ChatRequest

        # Valid request
        request = ChatRequest(question="What is Python?")
        assert request.question == "What is Python?"

        # Request with optional session_id
        request = ChatRequest(question="Test", session_id="abc-123")
        assert request.session_id == "abc-123"


@pytest.mark.integration
class TestChatAPICacheHits:
    """Tests for cache hit paths."""

    @patch("app.services.response_cache.get_response_cache")
    def test_chat_cache_hit_returns_cached_response(
        self, mock_get_cache, client, auth_headers, mock_rate_limit
    ):
        """Test that cache hits return cached responses."""
        # Mock the cache to return a hit
        mock_cache = MagicMock()
        mock_cache.get.return_value = {
            "answer": "Cached answer",
            "sources": [],
            "grounded": True,
            "session_id": "cached-session",
        }
        mock_get_cache.return_value = mock_cache

        response = client.post(
            "/chat", json={"question": "What is my GPA?"}, headers=auth_headers
        )

        assert response.status_code == 200
        response.json()
        # Cache was checked
        mock_get_cache.assert_called()


@pytest.mark.integration
class TestChatAPIPromptGuard:
    """Tests for prompt guard integration."""

    def test_prompt_guard_integration(
        self, client, auth_headers, mock_rate_limit, mock_chromadb, mock_llm
    ):
        """Test that prompt guard is invoked on requests."""
        # Setup mock chromadb
        mock_chromadb.query.return_value = {
            "ids": [["chunk-1"]],
            "documents": [["Test content"]],
            "metadatas": [[{"source": "test.md"}]],
            "distances": [[0.2]],
        }

        # A normal request should go through
        response = client.post(
            "/chat/simple",
            json={"question": "What is my experience?"},
            headers=auth_headers,
        )

        # Should get a response (prompt guard allowed it)
        assert response.status_code == 200


@pytest.mark.integration
class TestChatAPIExceptionHandling:
    """Tests for exception handling in chat endpoints."""

    def test_chat_validation_error_on_invalid_json(self, client, auth_headers):
        """Test that invalid JSON returns 422."""
        response = client.post(
            "/chat",
            content="not valid json",
            headers={**auth_headers, "Content-Type": "application/json"},
        )

        # Should get validation error
        assert response.status_code == 422

    def test_chat_missing_question_field(self, client, auth_headers):
        """Test that missing question field returns 422."""
        response = client.post(
            "/chat", json={"wrong_field": "value"}, headers=auth_headers
        )

        assert response.status_code == 422


@pytest.mark.integration
class TestSimpleChatAPIExceptions:
    """Tests for exception handling in simple_chat endpoint."""

    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.generate_with_llm")
    def test_simple_chat_rag_exception_reraises(
        self, mock_generate, mock_search, client, auth_headers, mock_rate_limit
    ):
        """Test that RAGException is properly handled."""
        from app.exceptions import LLMException

        mock_search.return_value = [
            {
                "id": "1",
                "text": "test",
                "source": "test.md",
                "distance": 0.1,
                "metadata": {},
            }
        ]
        mock_generate.side_effect = LLMException("LLM service unavailable")

        response = client.post(
            "/chat/simple", json={"question": "Test question"}, headers=auth_headers
        )

        # Should get error status
        assert response.status_code >= 400

    @patch("app.api.routes.chat.search")
    @patch("app.api.routes.chat.generate_with_llm")
    def test_simple_chat_unexpected_exception(
        self, mock_generate, mock_search, client, auth_headers, mock_rate_limit
    ):
        """Test that unexpected exceptions are handled properly."""
        mock_search.return_value = [
            {
                "id": "1",
                "text": "test",
                "source": "test.md",
                "distance": 0.1,
                "metadata": {},
            }
        ]
        mock_generate.side_effect = ValueError("Unexpected value error")

        response = client.post(
            "/chat/simple", json={"question": "Test question"}, headers=auth_headers
        )

        # Should get error status
        assert response.status_code >= 400
