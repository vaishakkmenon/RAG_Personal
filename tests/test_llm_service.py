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


@pytest.fixture
def mock_settings():
    """Mock settings to ensure provider is groq."""
    with patch("app.services.llm.settings") as mock_settings_obj:
        mock_llm_settings = MagicMock()
        mock_llm_settings.provider = "groq"
        mock_llm_settings.groq_api_key = "test-key"
        mock_llm_settings.groq_model = "llama-3.1-8b-instant"
        mock_llm_settings.groq_tier = "free"
        mock_llm_settings.groq_requests_per_minute = 28
        mock_llm_settings.groq_requests_per_day = 13680
        mock_llm_settings.temperature = 0.1
        mock_llm_settings.max_tokens = 1000

        mock_settings_obj.llm = mock_llm_settings
        yield mock_settings_obj


@pytest.fixture
def mock_rate_limiter():
    """Mock rate limiter with proper stats."""
    mock_limiter = MagicMock()
    mock_limiter.acquire.return_value = True
    mock_limiter.get_stats.return_value = {
        "minute_utilization": 0.5,
        "day_utilization": 0.5,
        "requests_last_minute": 14,
        "requests_per_minute_limit": 28,
        "requests_last_day": 6840,
        "requests_per_day_limit": 13680,
    }
    return mock_limiter


@pytest.mark.unit
@pytest.mark.llm
class TestGroqGeneration:
    """Tests for Groq LLM generation."""

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_successful_generation(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test successful generation with Groq."""
        from app.services.llm import GroqLLMService

        # Mock Groq client and response
        mock_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is a test response from Groq."
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_groq_client

        # Patch RateLimiter to return our mock
        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            # Create service instance with Groq
            service = GroqLLMService()

            result = service.generate(
                prompt="Test prompt", model="llama-3.1-8b-instant"
            )

            assert result == "This is a test response from Groq."
            assert mock_groq_client.chat.completions.create.called

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_with_temperature(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test that temperature parameter is passed to Groq."""
        from app.services.llm import GroqLLMService

        mock_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_groq_client

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()
            service.generate(prompt="Test", temperature=0.7)

            # Verify temperature was passed
            call_args = mock_groq_client.chat.completions.create.call_args
            assert call_args.kwargs["temperature"] == 0.7

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_with_max_tokens(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test that max_tokens parameter is passed to Groq."""
        from app.services.llm import GroqLLMService

        mock_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_groq_client

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()
            service.generate(prompt="Test", max_tokens=500)

            # Verify max_tokens was passed
            call_args = mock_groq_client.chat.completions.create.call_args
            assert call_args.kwargs["max_tokens"] == 500

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_model_selection(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test that correct model is used."""
        from app.services.llm import GroqLLMService

        mock_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_groq_client

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()
            service.generate(prompt="Test", model="llama-3.3-70b-versatile")

            # Verify correct model was requested
            call_args = mock_groq_client.chat.completions.create.call_args
            assert call_args.kwargs["model"] == "llama-3.3-70b-versatile"


@pytest.mark.unit
@pytest.mark.llm
class TestGroqErrorHandling:
    """Tests for Groq error handling (no fallback since Ollama removed)."""

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_api_timeout(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test handling of Groq API timeout (no fallback)."""
        from app.services.llm import GroqLLMService
        from groq import APITimeoutError
        from httpx import Request

        # Mock Groq client to raise timeout
        mock_groq_client = MagicMock()
        mock_request = Request("POST", "https://api.groq.com/test")
        mock_groq_client.chat.completions.create.side_effect = APITimeoutError(
            request=mock_request
        )
        mock_groq_class.return_value = mock_groq_client

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()

            # Should raise exception (no fallback)
            with pytest.raises(APITimeoutError):
                service.generate(prompt="Test")

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_api_rate_limit(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test handling of Groq rate limit errors."""
        from app.services.llm import GroqLLMService
        from groq import RateLimitError
        from httpx import Request, Response

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

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()

            with pytest.raises(RateLimitError):
                service.generate(prompt="Test")

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_api_error_500(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test handling of Groq server errors."""
        from app.services.llm import GroqLLMService
        from groq import InternalServerError
        from httpx import Request, Response

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

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()

            with pytest.raises(InternalServerError):
                service.generate(prompt="Test")

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_invalid_api_key(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test handling of invalid API key."""
        from app.services.llm import GroqLLMService
        from groq import AuthenticationError
        from httpx import Request, Response

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

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()

            with pytest.raises(AuthenticationError):
                service.generate(prompt="Test")

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_groq_malformed_response(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test handling of malformed API response."""
        from app.services.llm import GroqLLMService

        mock_groq_client = MagicMock()
        # Mock malformed response (choices array is empty)
        mock_response = MagicMock()
        mock_response.choices = []  # Empty list will cause IndexError
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_groq_client

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()

            # Should raise IndexError
            with pytest.raises(IndexError):
                service.generate(prompt="Test")


@pytest.mark.unit
@pytest.mark.llm
class TestLLMPromptFormatting:
    """Tests for prompt formatting."""

    @patch("app.services.llm.AsyncGroq")
    @patch("app.services.llm.Groq")
    def test_prompt_structure(
        self, mock_groq_class, mock_async_groq, mock_settings, mock_rate_limiter
    ):
        """Test that prompts are properly structured."""
        from app.services.llm import GroqLLMService

        mock_groq_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test"
        mock_groq_client.chat.completions.create.return_value = mock_response
        mock_groq_class.return_value = mock_groq_client

        prompt = """Context: Test context

Question: What is my Python experience?

Answer:"""

        with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
            service = GroqLLMService()
            service.generate(prompt=prompt)

            # Verify prompt was sent correctly
            call_args = mock_groq_client.chat.completions.create.call_args
            messages = call_args.kwargs["messages"]
            assert len(messages) == 1
            assert messages[0]["role"] == "user"
            assert "Context:" in messages[0]["content"]


@pytest.mark.unit
@pytest.mark.llm
class TestLLMWrapperFunction:
    """Tests for the main generate_with_llm wrapper (Groq-only)."""

    @patch("app.services.llm._service")
    def test_wrapper_uses_groq(self, mock_service):
        """Test that wrapper function delegates to service.generate()."""
        from app.services.llm import generate_with_llm

        # Mock the service's generate method
        mock_service.generate.return_value = "Test response from Groq"

        result = generate_with_llm(
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
        from app.services.llm import generate_with_llm

        mock_service.generate.return_value = "Test"

        generate_with_llm(prompt="Test prompt", temperature=0.7, max_tokens=1000)

        # Verify parameters were passed
        mock_service.generate.assert_called_once_with(
            prompt="Test prompt", temperature=0.7, max_tokens=1000, model=None
        )
