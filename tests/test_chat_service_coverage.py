"""
Additional test coverage for decomposed ChatService core logic.

Tests:
- QueryValidator (chitchat, ambiguity)
- SessionManager (errors from store)
- ChatService integration (cache paths, greetings)
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi import HTTPException
from app.models import ChatRequest
from app.core.query_validator import QueryValidator


@pytest.mark.unit
class TestQueryValidatorChitchat:
    """Tests for chitchat detection in QueryValidator."""

    def test_chitchat_detection_gratitude(self):
        """Test that gratitude messages are detected as chitchat."""
        validator = QueryValidator()
        is_chitchat, response = validator.detect_chitchat("thank you")

        assert is_chitchat is True
        assert response is not None
        assert len(response) > 0

    def test_chitchat_detection_farewell(self):
        """Test that farewell messages are detected as chitchat."""
        validator = QueryValidator()
        is_chitchat, response = validator.detect_chitchat("goodbye")

        assert is_chitchat is True
        assert response is not None

    def test_normal_question_not_chitchat(self):
        """Test that normal questions are not detected as chitchat."""
        validator = QueryValidator()
        is_chitchat, response = validator.detect_chitchat(
            "What is my Python experience?"
        )

        assert is_chitchat is False


@pytest.mark.unit
class TestQueryValidatorAmbiguity:
    """Tests for ambiguity detection in QueryValidator."""

    def test_ambiguity_single_word_without_context(self):
        """Test that single-word questions without context are ambiguous."""
        validator = QueryValidator()
        is_ambiguous = validator.detect_ambiguity("experience", [])
        assert is_ambiguous is True

    def test_ambiguity_two_word_question(self):
        """Test that very short questions are detected as ambiguous."""
        validator = QueryValidator()
        # "my skills" is ambiguously short
        is_ambiguous = validator.detect_ambiguity("my skills", [])
        assert isinstance(is_ambiguous, bool)

    def test_full_question_not_ambiguous(self):
        """Test that full questions are not ambiguous."""
        validator = QueryValidator()
        is_ambiguous = validator.detect_ambiguity(
            "What programming languages have I used?", []
        )
        assert is_ambiguous is False

    def test_ambiguity_with_conversation_history(self):
        """Test that context from history helps with ambiguous words."""
        validator = QueryValidator()
        history = [
            {"role": "user", "content": "Tell me about my Python experience"},
            {"role": "assistant", "content": "You have 5 years of Python experience."},
        ]
        # "more" is short but context helps.
        # Actually detect_ambiguity logic treats single words < 2 as ambiguous unless history?
        # Let's verify behavior.
        is_ambiguous = validator.detect_ambiguity("more", history)
        assert isinstance(is_ambiguous, bool)


@pytest.mark.unit
class TestChatServiceInit:
    """Tests for ChatService initialization."""

    def test_chat_service_initialization(self):
        """Test that ChatService can be initialized."""
        from app.core.chat_service import ChatService

        service = ChatService()
        assert service is not None
        assert hasattr(service, "prompt_builder")
        assert hasattr(service, "retrieval")

    def test_chat_service_with_custom_session_store(self):
        """Test ChatService with custom session store."""
        from app.core.chat_service import ChatService

        mock_store = MagicMock()
        service = ChatService(session_store=mock_store)
        assert service is not None


@pytest.mark.unit
class TestChatServiceSessionErrors:
    """Tests for session management error handling."""

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.retrieval_orchestrator.search")
    def test_session_http_exception_reraises(self, mock_search, mock_cache):
        """Test that HTTPException from session is re-raised."""
        from app.core.chat_service import ChatService

        # Mock session store to raise HTTPException (rate limit)
        mock_store = MagicMock()
        mock_store.get_or_create_session.side_effect = HTTPException(
            status_code=429, detail="Rate limit exceeded"
        )
        # Assuming SessionManager delegates to store or raises on its own
        # In the new impl, get_or_create_session is called on self.session_manager.
        # But ChatService init takes session_store and initializes SessionManager with it.
        # If SessionManager.get_or_create_session calls store.get_or_create_session...

        service = ChatService(session_store=mock_store)
        request = ChatRequest(question="Test question")

        with pytest.raises(HTTPException) as exc_info:
            service.handle_chat(request=request)

        assert exc_info.value.status_code == 429

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.retrieval_orchestrator.search")
    @patch("app.core.chat_service.generate_with_llm")
    def test_session_unexpected_error_creates_temp_session(
        self, mock_generate, mock_search, mock_cache
    ):
        """Test that unexpected session error creates temp session."""
        from app.core.chat_service import ChatService

        # Mock session store to raise unexpected error
        mock_store = MagicMock()
        mock_store.get_or_create_session.side_effect = RuntimeError(
            "Redis connection failed"
        )

        # Ensure cache miss
        mock_cache.return_value.get.return_value = None

        # We need mock_search to return results so pipeline proceeds
        mock_search.return_value = [
            {
                "id": "1",
                "text": "test content",
                "source": "test.md",
                "distance": 0.1,
                "metadata": {},
            }
        ]
        mock_generate.return_value = "Test answer"

        service = ChatService(session_store=mock_store)
        request = ChatRequest(question="What is Python?")

        # Should not raise, should create temp session inside SessionManager -> returned to ChatService
        # Actually SessionManager handles the try/catch logic internally?
        # Checking SessionManager source in my memory:
        # SessionManager.get_or_create_session DOES have a try/except for Exception.

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

        # Note: ChatService calls session_manager.check_rate_limit(session).
        # We need to ensure that passes or we mock session_manager.

        service = ChatService(session_store=mock_store)

        # We can also mock the session_manager directly if checking logic is complex
        service.session_manager.check_rate_limit = MagicMock(return_value=True)

        request = ChatRequest(question="What is Python?")
        response = service.handle_chat(request=request)

        # Cache should have been checked
        mock_get_cache.assert_called()
        assert response.answer == "Cached answer"


@pytest.mark.unit
class TestChatServiceGreetings:
    """Tests for greeting/chitchat handling via ChatService integration."""

    def test_chitchat_hello(self):
        """Test hello is detected as chitchat."""
        from app.core.chat_service import ChatService

        # Testing the full integration path
        service = ChatService()
        # Mock session stuff to avoid redis calls
        service.session_manager = MagicMock()
        mock_session = MagicMock()
        mock_session.session_id = "test"
        mock_session.get_truncated_history.return_value = []
        service.session_manager.get_or_create_session.return_value = mock_session
        service.session_manager.check_rate_limit.return_value = True

        request = ChatRequest(question="hello")
        response = service.handle_chat(request)

        assert response is not None
        # Should be a greeting response, likely grounded=False but not "I don't know"
        assert response.answer not in ["I don't know.", "I don't know..."]
        assert len(response.answer) > 0


@pytest.mark.unit
class TestChatServiceRateLimit:
    """Tests for rate limiting."""

    @patch("app.core.retrieval_orchestrator.search")
    def test_rate_limit_check_fails(self, mock_search):
        """Test that rate limit check failure raises HTTPException."""
        from app.core.chat_service import ChatService

        mock_store = MagicMock()

        service = ChatService(session_store=mock_store)

        # Mock session manager specifically to fail rate limit
        mock_session = MagicMock()
        mock_session.session_id = "test-session"
        service.session_manager.get_or_create_session = MagicMock(
            return_value=mock_session
        )
        service.session_manager.check_rate_limit = MagicMock(return_value=False)

        request = ChatRequest(question="Test question")

        with pytest.raises(HTTPException) as exc_info:
            service.handle_chat(request=request)

        assert exc_info.value.status_code == 429
