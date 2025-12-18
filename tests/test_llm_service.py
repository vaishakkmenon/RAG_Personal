"""
Unit tests for LLM Service (Groq-only).

Tests:
- Groq generation
- Error handling (no fallback)
- Token counting
- Prompt formatting
- Temperature and max_tokens
- Model selection
"""

import pytest
from unittest.mock import MagicMock, patch


@pytest.mark.unit
@pytest.mark.llm
class TestGroqGeneration:
    """Tests for Groq LLM generation."""

    @patch("app.services.llm.Groq")
    def test_groq_successful_generation(self, mock_groq_class):
        """Test successful generation with Groq."""
        from app.services.llm import OllamaService
        from app.settings import settings

        # Temporarily set provider to Groq
        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            # Mock Groq client and response
            mock_groq_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[
                0
            ].message.content = "This is a test response from Groq."
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_groq_client

            # Create fresh service instance with Groq
            service = OllamaService()

            result = service.generate(
                prompt="Test prompt", model="llama-3.1-8b-instant"
            )

            assert result == "This is a test response from Groq."
            assert mock_groq_client.chat.completions.create.called
        finally:
            # Restore original settings
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.Groq")
    def test_groq_with_temperature(self, mock_groq_class):
        """Test that temperature parameter is passed to Groq."""
        from app.services.llm import OllamaService
        from app.settings import settings

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_groq_client

            service = OllamaService()
            service.generate(prompt="Test", temperature=0.7)

            # Verify temperature was passed
            call_args = mock_groq_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.7
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.Groq")
    def test_groq_with_max_tokens(self, mock_groq_class):
        """Test that max_tokens parameter is passed to Groq."""
        from app.services.llm import OllamaService
        from app.settings import settings

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_groq_client

            service = OllamaService()
            service.generate(prompt="Test", max_tokens=500)

            # Verify max_tokens was passed
            call_args = mock_groq_client.chat.completions.create.call_args
            assert call_args.kwargs["max_tokens"] == 500
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.Groq")
    def test_groq_model_selection(self, mock_groq_class):
        """Test that correct model is used."""
        from app.services.llm import OllamaService
        from app.settings import settings

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_groq_client

            service = OllamaService()
            service.generate(prompt="Test", model="llama-3.3-70b-versatile")

            # Verify correct model was requested
            call_args = mock_groq_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key


@pytest.mark.unit
@pytest.mark.llm
class TestGroqErrorHandling:
    """Tests for Groq error handling (no fallback since Ollama removed)."""

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_groq_api_timeout(self, mock_groq_class, mock_ollama_class):
        """Test handling of Groq API timeout (no fallback)."""
        from app.services.llm import OllamaService
        from app.settings import settings
        from groq import APITimeoutError
        from httpx import Request

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            # Mock Groq client to raise timeout
            mock_groq_client = MagicMock()
            mock_request = Request("POST", "https://api.groq.com/test")
            mock_groq_client.chat.completions.create.side_effect = APITimeoutError(
                request=mock_request
            )
            mock_groq_class.return_value = mock_groq_client

            # Mock Ollama fallback to also fail
            mock_ollama_client = MagicMock()
            mock_ollama_client.generate.side_effect = Exception(
                "Ollama fallback failed"
            )
            mock_ollama_class.return_value = mock_ollama_client

            service = OllamaService()

            # Should get APITimeoutError or fallback error
            with pytest.raises(Exception):  # Could be APITimeoutError or fallback error
                service.generate(prompt="Test")
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_groq_api_rate_limit(self, mock_groq_class, mock_ollama_class):
        """Test handling of Groq rate limit errors."""
        from app.services.llm import OllamaService
        from app.settings import settings
        from groq import RateLimitError
        from httpx import Request, Response

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = Response(
                429, request=Request("POST", "https://api.groq.com/test")
            )
            mock_groq_client.chat.completions.create.side_effect = RateLimitError(
                "Rate limit exceeded",
                response=mock_response,
                body={"error": "rate_limit"},
            )
            mock_groq_class.return_value = mock_groq_client

            # Mock Ollama fallback to also fail
            mock_ollama_client = MagicMock()
            mock_ollama_client.generate.side_effect = Exception(
                "Ollama fallback failed"
            )
            mock_ollama_class.return_value = mock_ollama_client

            service = OllamaService()

            with pytest.raises(Exception):
                service.generate(prompt="Test")
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_groq_api_error_500(self, mock_groq_class, mock_ollama_class):
        """Test handling of Groq server errors."""
        from app.services.llm import OllamaService
        from app.settings import settings
        from groq import InternalServerError
        from httpx import Request, Response

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = Response(
                500, request=Request("POST", "https://api.groq.com/test")
            )
            mock_groq_client.chat.completions.create.side_effect = InternalServerError(
                "Internal Server Error",
                response=mock_response,
                body={"error": "server_error"},
            )
            mock_groq_class.return_value = mock_groq_client

            # Mock Ollama fallback to also fail
            mock_ollama_client = MagicMock()
            mock_ollama_client.generate.side_effect = Exception(
                "Ollama fallback failed"
            )
            mock_ollama_class.return_value = mock_ollama_client

            service = OllamaService()

            with pytest.raises(Exception):
                service.generate(prompt="Test")
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_groq_invalid_api_key(self, mock_groq_class, mock_ollama_class):
        """Test handling of invalid API key."""
        from app.services.llm import OllamaService
        from app.settings import settings
        from groq import AuthenticationError
        from httpx import Request, Response

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = Response(
                401, request=Request("POST", "https://api.groq.com/test")
            )
            mock_groq_client.chat.completions.create.side_effect = AuthenticationError(
                "Invalid API key",
                response=mock_response,
                body={"error": "invalid_api_key"},
            )
            mock_groq_class.return_value = mock_groq_client

            # Mock Ollama fallback to also fail
            mock_ollama_client = MagicMock()
            mock_ollama_client.generate.side_effect = Exception(
                "Ollama fallback failed"
            )
            mock_ollama_class.return_value = mock_ollama_client

            service = OllamaService()

            with pytest.raises(Exception):
                service.generate(prompt="Test")
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_groq_malformed_response(self, mock_groq_class, mock_ollama_class):
        """Test handling of malformed API response."""
        from app.services.llm import OllamaService
        from app.settings import settings

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            # Mock malformed response (choices array is empty)
            mock_response = MagicMock()
            mock_response.choices = []  # Empty list will cause IndexError
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_groq_client

            # Mock Ollama fallback to also fail
            mock_ollama_client = MagicMock()
            mock_ollama_client.generate.side_effect = Exception(
                "Ollama fallback failed"
            )
            mock_ollama_class.return_value = mock_ollama_client

            service = OllamaService()

            # Should raise IndexError or fallback error
            with pytest.raises(Exception):
                service.generate(prompt="Test")
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key


@pytest.mark.unit
@pytest.mark.llm
class TestLLMPromptFormatting:
    """Tests for prompt formatting."""

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_prompt_structure(self, mock_groq_class, mock_ollama_class):
        """Test that prompts are properly structured."""
        from app.services.llm import OllamaService
        from app.settings import settings

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            mock_groq_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Test"
            mock_groq_client.chat.completions.create.return_value = mock_response
            mock_groq_class.return_value = mock_groq_client
            mock_ollama_class.return_value = MagicMock()

            prompt = """Context: Test context

Question: What is my Python experience?

Answer:"""

            service = OllamaService()
            service.generate(prompt=prompt)

            # Verify prompt was sent correctly
            call_args = mock_groq_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "Context:" in messages[0]["content"]
        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key


@pytest.mark.unit
@pytest.mark.llm
class TestLLMWrapperFunction:
    """Tests for the main generate_with_ollama wrapper (Groq-only)."""

    @patch("app.services.llm._service")
    def test_wrapper_uses_groq(self, mock_service):
        """Test that wrapper function delegates to service.generate()."""
        from app.services.llm import generate_with_ollama

        # Mock the service's generate method
        mock_service.generate.return_value = "Test response from Groq"

        result = generate_with_ollama(
            prompt="Test prompt", temperature=0.1, max_tokens=500
        )

        # Verify service.generate was called with correct parameters
        mock_service.generate.assert_called_once_with(
            prompt="Test prompt", temperature=0.1, max_tokens=500, model=None
        )
        assert result == "Test response from Groq"

    @patch("app.services.llm._service")
    def test_wrapper_passes_parameters(self, mock_service):
        """Test that wrapper passes parameters correctly."""
        from app.services.llm import generate_with_ollama

        mock_service.generate.return_value = "Test"

        generate_with_ollama(prompt="Test prompt", temperature=0.7, max_tokens=1000)

        # Verify parameters were passed
        mock_service.generate.assert_called_once_with(
            prompt="Test prompt", temperature=0.7, max_tokens=1000, model=None
        )


@pytest.mark.unit
@pytest.mark.llm
class TestLLMFallback:
    """Tests for LLM fallback mechanisms."""

    @patch("app.services.llm.ollama.Client")
    @patch("app.services.llm.Groq")
    def test_groq_to_ollama_fallback(self, mock_groq_class, mock_ollama_class):
        """Test successful fallback from Groq to Ollama."""
        from app.services.llm import OllamaService
        from app.settings import settings
        from groq import APITimeoutError
        from httpx import Request

        original_provider = settings.llm.provider
        original_key = settings.llm.groq_api_key
        settings.llm.provider = "groq"
        settings.llm.groq_api_key = "test-key"

        try:
            # Mock Groq client to raise timeout
            mock_groq_client = MagicMock()
            mock_request = Request("POST", "https://api.groq.com/test")
            mock_groq_client.chat.completions.create.side_effect = APITimeoutError(
                request=mock_request
            )
            mock_groq_class.return_value = mock_groq_client

            # Mock Ollama fallback to SUCCEED
            mock_ollama_client = MagicMock()
            mock_ollama_client.generate.return_value = {
                "response": "Fallback response from Ollama"
            }
            mock_ollama_class.return_value = mock_ollama_client

            service = OllamaService()

            # Should succeed with fallback response
            # Note: Retry logic will cause Groq to be called multiple times before fallback
            # We mock it to fail every time
            result = service.generate(prompt="Test")

            assert result == "Fallback response from Ollama"

            # Verify both were called
            assert mock_groq_client.chat.completions.create.called
            # Default retry is 2 times (max_retries=2 in usage)
            assert mock_groq_client.chat.completions.create.call_count == 2
            assert mock_ollama_client.generate.called

        finally:
            settings.llm.provider = original_provider
            settings.llm.groq_api_key = original_key
