"""
Unit tests for Chat Service (Core RAG Pipeline).

Tests the main business logic:
- Retrieval → Reranking → LLM Generation → Grounding
- Cache integration
- Session/conversation history
- Error handling
- Metrics recording
"""

import pytest
from unittest.mock import MagicMock, patch
from app.core.chat_service import ChatService
from app.models import ChatRequest


@pytest.mark.unit
class TestChatServiceFullPipeline:
    """Tests for the complete RAG pipeline."""

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    @patch("app.core.chat_service.create_default_prompt_builder")
    def test_successful_chat_pipeline(
        self,
        mock_prompt_builder,
        mock_generate,
        mock_search,
        mock_cache,
        sample_chunks,
    ):
        """Test full pipeline: retrieve → rerank → generate → return."""
        from app.prompting.config import PromptResult

        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search
        mock_search.return_value = sample_chunks

        # Mock LLM
        mock_generate.return_value = "You have 5 years of Python experience."

        # Mock prompt builder - must return PromptResult with status and prompt
        mock_builder = MagicMock()
        mock_builder.build_prompt.return_value = PromptResult(
            status="success", prompt="Test prompt", message="", context=""
        )
        mock_builder.is_refusal.return_value = False
        mock_prompt_builder.return_value = mock_builder

        # Create service and execute
        service = ChatService()
        request = ChatRequest(question="What is my Python experience?")

        response = service.handle_chat(
            request=request,
            skip_route_cache=True,
        )

        # Verify response structure
        assert response.answer == "You have 5 years of Python experience."
        assert len(response.sources) == 3
        assert response.grounded is True

        # Verify search was called
        mock_search.assert_called_once()

        # Verify LLM was called
        mock_generate.assert_called_once()

        # Verify cache was updated
        mock_cache_instance.set.assert_called_once()

    @patch("app.core.chat_service.get_response_cache")
    def test_cache_hit_skips_pipeline(self, mock_cache):
        """Test that cache hit returns immediately without RAG pipeline."""
        # Mock cache (hit)
        cached_response = {
            "answer": "Cached answer",
            "sources": [],
            "grounded": True,
            "metadata": {"cached": True},
        }
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = cached_response
        mock_cache.return_value = mock_cache_instance

        service = ChatService()
        request = ChatRequest(question="What is my Python experience?")

        with patch("app.core.chat_service.search") as mock_search:
            response = service.handle_chat(request=request)

            # Verify cache was checked
            mock_cache_instance.get.assert_called_once()

            # Verify search was NOT called (cache hit)
            mock_search.assert_not_called()

            # Verify cached response was returned
            assert response.answer == "Cached answer"

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_no_results_not_grounded(
        self,
        mock_generate,
        mock_search,
        mock_cache,
    ):
        """Test that empty search results return not grounded."""
        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search (no results)
        mock_search.return_value = []

        # Mock LLM (not actually called since no chunks retrieved)
        mock_generate.return_value = "I don't have information about that."

        service = ChatService()
        request = ChatRequest(
            question="What is my underwater basket weaving experience?"
        )

        response = service.handle_chat(request=request, skip_route_cache=True)

        # Verify not grounded - returns hardcoded message when no chunks
        assert response.grounded is False
        assert len(response.sources) == 0
        # Check for the actual hardcoded message from chat_service.py line 686
        assert "couldn't find any relevant information" in response.answer.lower()

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_grounding_threshold_check(
        self,
        mock_generate,
        mock_search,
        mock_cache,
    ):
        """Test grounding check based on distance threshold."""
        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search (high distance = not relevant)
        distant_chunks = [
            {
                "id": "chunk-1",
                "source": "test.md",
                "text": "Irrelevant content",
                "distance": 0.95,  # Very high distance
                "metadata": {},
            }
        ]
        mock_search.return_value = distant_chunks

        # Mock LLM
        mock_generate.return_value = "I don't have specific information."

        service = ChatService()
        request = ChatRequest(question="Test question")

        response = service.handle_chat(
            request=request,
            null_threshold=0.6,  # Threshold lower than chunk distance
            skip_route_cache=True,
        )

        # Should not be grounded (distance > threshold)
        assert response.grounded is False


@pytest.mark.unit
class TestChatServiceConversationHistory:
    """Tests for multi-turn conversation handling."""

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_conversation_history_included(
        self,
        mock_generate,
        mock_search,
        mock_cache,
        sample_chunks,
    ):
        """Test that conversation history is included in prompt."""
        from app.prompting.config import PromptResult
        from app.storage.fallback.memory import InMemorySessionStore

        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Create real session store with conversation history
        session_store = InMemorySessionStore()
        session = session_store.get_or_create_session(None, "127.0.0.1")
        session.add_turn("user", "What is my Python experience?")
        session.add_turn("assistant", "You have 5 years.")
        session_store.update_session(session)

        # Mock search
        mock_search.return_value = sample_chunks

        # Mock LLM
        mock_generate.return_value = "You also have machine learning experience."

        # Mock prompt builder BEFORE creating service
        with patch(
            "app.core.chat_service.create_default_prompt_builder"
        ) as mock_builder_func:
            mock_builder = MagicMock()
            mock_builder.build_prompt.return_value = PromptResult(
                status="success", prompt="Prompt with history", message="", context=""
            )
            mock_builder.is_refusal.return_value = False
            mock_builder_func.return_value = mock_builder

            # Create service with real session store (now uses mocked builder)
            service = ChatService(session_store=session_store)
            request = ChatRequest(
                question="What about machine learning?",
                session_id=session.session_id,
            )

            service.handle_chat(request=request, skip_route_cache=True)

            # Verify prompt builder received conversation history
            build_call = mock_builder.build_prompt.call_args
            assert "conversation_history" in build_call.kwargs
            history = build_call.kwargs["conversation_history"]
            assert len(history) == 2
            assert history[0]["content"] == "What is my Python experience?"

    def test_rate_limit_enforced(self):
        """Test that rate limits are enforced."""
        from app.storage.fallback.memory import InMemorySessionStore

        # Create real session store and mock check_rate_limit to return False
        session_store = InMemorySessionStore()
        session = session_store.get_or_create_session(None, "127.0.0.1")

        # Mock check_rate_limit to return False (rate limited)
        original_check = session_store.check_rate_limit
        session_store.check_rate_limit = MagicMock(return_value=False)

        service = ChatService(session_store=session_store)
        request = ChatRequest(
            question="Test question",
            session_id=session.session_id,
        )

        # Should raise exception for rate limit
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            service.handle_chat(request=request)

        assert exc_info.value.status_code == 429
        assert "rate limit" in exc_info.value.detail.lower()

        # Restore original method
        session_store.check_rate_limit = original_check


@pytest.mark.unit
class TestChatServiceErrorHandling:
    """Tests for error handling in chat service."""

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    @patch("app.core.chat_service.create_default_prompt_builder")
    def test_llm_failure_graceful_degradation(
        self,
        mock_prompt_builder,
        mock_generate,
        mock_search,
        mock_cache,
        sample_chunks,
    ):
        """Test that LLM failures are handled gracefully with degraded response."""
        from app.prompting.config import PromptResult

        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search
        mock_search.return_value = sample_chunks

        # Mock LLM failure
        mock_generate.side_effect = Exception("Groq API timeout")

        # Mock prompt builder
        mock_builder = MagicMock()
        mock_builder.build_prompt.return_value = PromptResult(
            status="success", prompt="Test prompt", message="", context=""
        )
        mock_builder.is_refusal.return_value = False
        mock_prompt_builder.return_value = mock_builder

        service = ChatService()
        request = ChatRequest(question="Test question")

        # Should NOT raise - should return graceful degradation
        response = service.handle_chat(request=request, skip_route_cache=True)

        # Verify graceful degradation response
        assert "temporarily unable" in response.answer.lower()
        assert response.grounded is True  # Still grounded with sources
        assert len(response.sources) == len(sample_chunks)

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    def test_search_failure_raises_error(
        self,
        mock_search,
        mock_cache,
    ):
        """Test that search failures are properly handled."""
        from fastapi import HTTPException

        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search failure
        mock_search.side_effect = Exception("ChromaDB connection failed")

        service = ChatService()
        request = ChatRequest(question="Test question")

        # Should raise HTTPException with 500 status and "Failed to retrieve documents" detail
        with pytest.raises(HTTPException) as exc_info:
            service.handle_chat(request=request, skip_route_cache=True)

        assert exc_info.value.status_code == 500
        assert "Failed to retrieve documents" in exc_info.value.detail


@pytest.mark.unit
class TestChatServiceMetadataFiltering:
    """Tests for metadata filtering."""

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_doc_type_filter_applied(
        self,
        mock_generate,
        mock_search,
        mock_cache,
        sample_chunks,
    ):
        """Test that doc_type filter is applied to search."""
        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search
        mock_search.return_value = sample_chunks

        # Mock LLM
        mock_generate.return_value = "Test response"

        service = ChatService()
        request = ChatRequest(question="Test question")

        service.handle_chat(
            request=request,
            doc_type="resume",
            skip_route_cache=True,
        )

        # Verify search was called with metadata filter
        search_call = mock_search.call_args
        assert "metadata_filter" in search_call.kwargs
        assert search_call.kwargs["metadata_filter"]["doc_type"] == "resume"


@pytest.mark.unit
class TestChatServiceReranking:
    """Tests for reranking functionality."""

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.rerank_chunks")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_reranking_enabled(
        self,
        mock_generate,
        mock_search,
        mock_rerank,
        mock_cache,
        sample_chunks,
    ):
        """Test that reranking is applied when enabled."""
        from app.prompting.config import PromptResult

        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search
        mock_search.return_value = sample_chunks

        # Mock rerank (return same chunks for simplicity)
        mock_rerank.return_value = sample_chunks

        # Mock LLM
        mock_generate.return_value = "Test response"

        # Mock prompt builder
        with patch(
            "app.core.chat_service.create_default_prompt_builder"
        ) as mock_builder_func:
            mock_builder = MagicMock()
            mock_builder.build_prompt.return_value = PromptResult(
                status="success", prompt="Test prompt", message="", context=""
            )
            mock_builder.is_refusal.return_value = False
            mock_builder_func.return_value = mock_builder

            service = ChatService()
            request = ChatRequest(question="Test question")

            service.handle_chat(
                request=request,
                rerank=True,
                skip_route_cache=True,
            )

            # Verify rerank_chunks was called
            mock_rerank.assert_called_once()

    @patch("app.core.chat_service.get_response_cache")
    @patch("app.core.chat_service.rerank_chunks")
    @patch("app.core.chat_service.search")
    @patch("app.core.chat_service.generate_with_ollama")
    def test_reranking_disabled(
        self,
        mock_generate,
        mock_search,
        mock_rerank,
        mock_cache,
        sample_chunks,
    ):
        """Test that reranking can be disabled."""
        from app.prompting.config import PromptResult

        # Mock cache (miss)
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache.return_value = mock_cache_instance

        # Mock search
        mock_search.return_value = sample_chunks

        # Mock rerank (should not be called)
        mock_rerank.return_value = sample_chunks

        # Mock LLM
        mock_generate.return_value = "Test response"

        # Mock prompt builder
        with patch(
            "app.core.chat_service.create_default_prompt_builder"
        ) as mock_builder_func:
            mock_builder = MagicMock()
            mock_builder.build_prompt.return_value = PromptResult(
                status="success", prompt="Test prompt", message="", context=""
            )
            mock_builder.is_refusal.return_value = False
            mock_builder_func.return_value = mock_builder

            service = ChatService()
            request = ChatRequest(question="Test question")

            service.handle_chat(
                request=request,
                rerank=False,
                skip_route_cache=True,
            )

            # Verify rerank_chunks was NOT called
            mock_rerank.assert_not_called()
