"""
Tests for custom exception classes and handlers.

Tests RAGException hierarchy and exception handlers.
"""

import pytest
from unittest.mock import MagicMock


@pytest.mark.unit
class TestRAGException:
    """Tests for RAGException base class."""

    def test_rag_exception_init(self):
        """Test RAGException initialization."""
        from app.exceptions import RAGException

        exc = RAGException("Test error", status_code=500)

        assert exc.message == "Test error"
        assert exc.status_code == 500
        assert str(exc) == "Test error"

    def test_rag_exception_default_status_code(self):
        """Test RAGException default status code."""
        from app.exceptions import RAGException

        exc = RAGException("Test error")

        assert exc.status_code == 500


@pytest.mark.unit
class TestLLMException:
    """Tests for LLMException."""

    def test_llm_exception_init(self):
        """Test LLMException initialization."""
        from app.exceptions import LLMException

        exc = LLMException("LLM failed")

        assert exc.message == "LLM failed"
        assert exc.status_code == 503


@pytest.mark.unit
class TestRetrievalException:
    """Tests for RetrievalException."""

    def test_retrieval_exception_init(self):
        """Test RetrievalException initialization."""
        from app.exceptions import RetrievalException

        exc = RetrievalException("Search failed")

        assert exc.message == "Search failed"
        assert exc.status_code == 500


@pytest.mark.unit
class TestRateLimitException:
    """Tests for RateLimitException."""

    def test_rate_limit_exception_init(self):
        """Test RateLimitException initialization."""
        from app.exceptions import RateLimitException

        exc = RateLimitException()

        assert exc.status_code == 429
        assert "rate limit" in exc.message.lower()

    def test_rate_limit_exception_custom_message(self):
        """Test RateLimitException with custom message."""
        from app.exceptions import RateLimitException

        exc = RateLimitException("Too many requests")

        assert exc.message == "Too many requests"
        assert exc.status_code == 429


@pytest.mark.unit
class TestExceptionInheritance:
    """Tests for exception hierarchy."""

    def test_llm_exception_is_rag_exception(self):
        """Test LLMException inherits from RAGException."""
        from app.exceptions import LLMException, RAGException

        exc = LLMException("test")

        assert isinstance(exc, RAGException)
        assert isinstance(exc, Exception)

    def test_retrieval_exception_is_rag_exception(self):
        """Test RetrievalException inherits from RAGException."""
        from app.exceptions import RetrievalException, RAGException

        exc = RetrievalException("test")

        assert isinstance(exc, RAGException)

    def test_rate_limit_exception_is_rag_exception(self):
        """Test RateLimitException inherits from RAGException."""
        from app.exceptions import RateLimitException, RAGException

        exc = RateLimitException()

        assert isinstance(exc, RAGException)
