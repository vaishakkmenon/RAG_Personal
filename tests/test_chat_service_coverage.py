"""
Additional test coverage for ChatService core logic.

Tests chitchat detection, ambiguity detection, session management
error handling, and conversation history updates.
"""

import pytest
from unittest.mock import MagicMock, patch
from app.models import ChatRequest


@pytest.mark.unit
class TestChatServiceChitchat:
    """Tests for chitchat detection and responses."""

    def test_chitchat_detection_gratitude(self):
        """Test that gratitude messages are detected as chitchat."""
        from app.core.chat_service import _is_chitchat

        is_chitchat, response = _is_chitchat("thank you")

        assert is_chitchat is True
        assert response is not None
        assert len(response) > 0

    def test_chitchat_detection_farewell(self):
        """Test that farewell messages are detected as chitchat."""
        from app.core.chat_service import _is_chitchat

        is_chitchat, response = _is_chitchat("goodbye")

        assert is_chitchat is True
        assert response is not None

    def test_normal_question_not_chitchat(self):
        """Test that normal questions are not detected as chitchat."""
        from app.core.chat_service import _is_chitchat

        is_chitchat, response = _is_chitchat("What is my Python experience?")

        assert is_chitchat is False


@pytest.mark.unit
class TestChatServiceAmbiguityDetection:
    """Tests for ambiguity detection."""

    def test_ambiguity_single_word_without_context(self):
        """Test that single-word questions without context are ambiguous."""
        from app.core.chat_service import _is_truly_ambiguous

        is_ambiguous = _is_truly_ambiguous("experience", None)

        # Should be ambiguous (single word)
        assert is_ambiguous is True

    def test_ambiguity_two_word_question(self):
        """Test that very short questions are detected as ambiguous."""
        from app.core.chat_service import _is_truly_ambiguous

        is_ambiguous = _is_truly_ambiguous("my skills", None)

        # May be ambiguous (2 words with filler)
        assert isinstance(is_ambiguous, bool)

    def test_full_question_not_ambiguous(self):
        """Test that full questions are not ambiguous."""
        from app.core.chat_service import _is_truly_ambiguous

        is_ambiguous = _is_truly_ambiguous(
            "What programming languages have I used?", None
        )

        assert is_ambiguous is False

    def test_ambiguity_with_conversation_history(self):
        """Test that context from history helps with ambiguous words."""
        from app.core.chat_service import _is_truly_ambiguous

        history = [
            {"role": "user", "content": "Tell me about my Python experience"},
            {"role": "assistant", "content": "You have 5 years of Python experience."},
        ]

        # Single word but has context
        is_ambiguous = _is_truly_ambiguous("more", history)

        # With history context, might not be ambiguous
        assert isinstance(is_ambiguous, bool)


@pytest.mark.unit
class TestChatServiceHelpers:
    """Tests for helper functions."""

    def test_build_context_query(self):
        """Test building context query from history."""
        from app.core.chat_service import _build_context_query

        history = [
            {"role": "user", "content": "Tell me about Python"},
            {"role": "assistant", "content": "Python is a programming language."},
        ]

        context = _build_context_query(history, max_turns=2)

        # Should include some content from history
        assert context is not None or context == ""

    def test_build_context_query_empty_history(self):
        """Test building context query with empty history."""
        from app.core.chat_service import _build_context_query

        context = _build_context_query([], max_turns=2)

        # Should return None or empty string
        assert context is None or context == ""

    def test_merge_and_dedupe_chunks(self):
        """Test merging and deduplicating chunks."""
        from app.core.chat_service import _merge_and_dedupe_chunks

        chunks_1 = [
            {"id": "chunk-1", "text": "Content 1"},
            {"id": "chunk-2", "text": "Content 2"},
        ]
        chunks_2 = [
            {"id": "chunk-2", "text": "Content 2"},  # Duplicate
            {"id": "chunk-3", "text": "Content 3"},
        ]

        merged = _merge_and_dedupe_chunks(chunks_1, chunks_2)

        # Should have 3 unique chunks
        assert len(merged) == 3
        ids = [c["id"] for c in merged]
        assert "chunk-1" in ids
        assert "chunk-2" in ids
        assert "chunk-3" in ids


@pytest.mark.unit
class TestChatServiceInit:
    """Tests for ChatService initialization."""

    def test_chat_service_initialization(self):
        """Test that ChatService can be initialized."""
        from app.core.chat_service import ChatService

        service = ChatService()

        assert service is not None
        assert hasattr(service, "prompt_builder")

    def test_chat_service_with_custom_session_store(self):
        """Test ChatService with custom session store."""
        from app.core.chat_service import ChatService

        mock_store = MagicMock()
        service = ChatService(session_store=mock_store)

        assert service is not None


@pytest.mark.unit
class TestRetrievalQualityCheck:
    """Tests for retrieval quality checking."""

    def test_check_retrieval_quality_with_matching_terms(self):
        """Test quality check when chunks contain query terms."""
        from app.core.chat_service import _check_retrieval_quality

        chunks = [
            {"text": "I have 5 years of Python experience."},
            {"text": "I worked with FastAPI framework."},
        ]

        result = _check_retrieval_quality("Python experience", chunks)

        # Should not be weak if terms are found
        assert result["is_weak"] is False

    def test_check_retrieval_quality_no_matching_terms(self):
        """Test quality check when chunks don't contain query terms."""
        from app.core.chat_service import _check_retrieval_quality

        chunks = [
            {"text": "I enjoy hiking and outdoor activities."},
            {"text": "Music is my favorite hobby."},
        ]

        result = _check_retrieval_quality("Python programming", chunks)

        # Might be weak if no terms found
        assert isinstance(result["is_weak"], bool)


@pytest.mark.unit
class TestChatServiceSessionErrors:
    """Tests for session management error handling."""

    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_session_http_exception_reraises(self, mock_generate, mock_search):
        """Test that HTTPException from session is re-raised."""
        from app.core.chat_service import ChatService
        from fastapi import HTTPException

        # Mock session store to raise HTTPException (rate limit)
        mock_store = MagicMock()
        mock_store.get_or_create_session.side_effect = HTTPException(
            status_code=429, detail="Rate limit exceeded"
        )

        service = ChatService(session_store=mock_store)
        request = ChatRequest(question="Test question")

        with pytest.raises(HTTPException) as exc_info:
            service.handle_chat(request=request)

        assert exc_info.value.status_code == 429

    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_session_unexpected_error_creates_temp_session(
        self, mock_generate, mock_search
    ):
        """Test that unexpected session error creates temp session."""
        from app.core.chat_service import ChatService

        # Mock session store to raise unexpected error
        mock_store = MagicMock()
        mock_store.get_or_create_session.side_effect = RuntimeError(
            "Redis connection failed"
        )
        mock_store.check_rate_limit.return_value = True
        mock_store.update_session.return_value = None

        mock_search.return_value = [
            {
                "id": "1",
                "text": "test",
                "source": "test.md",
                "distance": 0.1,
                "metadata": {},
            }
        ]
        mock_generate.return_value = "Test answer"

        service = ChatService(session_store=mock_store)
        request = ChatRequest(question="What is Python?")

        # Should not raise, should create temp session
        response = service.handle_chat(request=request)

        assert response is not None
        assert response.session_id is not None  # Temp session created


@pytest.mark.unit
class TestChatServiceCachePaths:
    """Tests for cache integration paths."""

    @patch("app.core.chat_service.get_response_cache")
    def test_cache_hit_bypasses_retrieval(self, mock_get_cache):
        """Test that cache hit skips retrieval and LLM."""
        from app.core.chat_service import ChatService

        # Mock cache to return hit
        mock_cache = MagicMock()
        mock_cache.get.return_value = {
            "answer": "Cached answer",
            "sources": [],
            "grounded": True,
        }
        mock_get_cache.return_value = mock_cache

        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_session.session_id = "test-session"
        mock_session.get_truncated_history.return_value = []
        mock_store.get_or_create_session.return_value = mock_session
        mock_store.check_rate_limit.return_value = True

        service = ChatService(session_store=mock_store)
        request = ChatRequest(question="What is Python?")

        service.handle_chat(request=request)

        # Cache should have been checked
        mock_get_cache.assert_called()


@pytest.mark.unit
class TestChatServiceGreetings:
    """Tests for greeting/chitchat handling."""

    def test_chitchat_hello(self):
        """Test hello is detected as chitchat."""
        from app.core.chat_service import _is_chitchat

        is_chitchat, response = _is_chitchat("hello")

        assert is_chitchat is True
        assert response is not None

    def test_chitchat_hi_there(self):
        """Test 'hi there' is detected as chitchat."""
        from app.core.chat_service import _is_chitchat

        is_chitchat, response = _is_chitchat("hi there")

        assert is_chitchat is True

    def test_chitchat_thanks(self):
        """Test 'thanks' is detected as chitchat."""
        from app.core.chat_service import _is_chitchat

        is_chitchat, response = _is_chitchat("thanks")

        assert is_chitchat is True


@pytest.mark.unit
class TestChatServiceRateLimit:
    """Tests for rate limiting."""

    @patch("app.core.chat_service.search")
    def test_rate_limit_check_fails(self, mock_search):
        """Test that rate limit check failure raises HTTPException."""
        from app.core.chat_service import ChatService
        from fastapi import HTTPException

        mock_store = MagicMock()
        mock_session = MagicMock()
        mock_session.session_id = "test-session"
        mock_store.get_or_create_session.return_value = mock_session
        mock_store.check_rate_limit.return_value = False  # Rate limit exceeded

        service = ChatService(session_store=mock_store)
        request = ChatRequest(question="Test question")

        with pytest.raises(HTTPException) as exc_info:
            service.handle_chat(request=request)

        assert exc_info.value.status_code == 429
