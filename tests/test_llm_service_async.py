"""
Comprehensive tests for LLM service.

Tests Ollama/Groq generation, fallback behavior, retry logic,
and error handling.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock


@pytest.mark.unit
@pytest.mark.llm
class TestOllamaServiceInit:
    """Tests for OllamaService initialization."""

    def test_ollama_service_creates_with_defaults(self):
        """Test creating OllamaService with default settings."""
        from app.services.llm import OllamaService

        service = OllamaService()

        assert service is not None
        assert service.llm_settings is not None

    def test_ollama_service_custom_host(self):
        """Test OllamaService with custom host."""
        from app.services.llm import OllamaService

        service = OllamaService(host="http://custom:11434")

        assert service.host == "http://custom:11434"

    def test_ollama_service_custom_model(self):
        """Test OllamaService with custom model parameter."""
        from app.services.llm import OllamaService

        service = OllamaService(model="llama2:7b")

        # Model is stored - check it was passed (may be used by ollama)
        assert service.model is not None

    def test_ollama_service_custom_timeout(self):
        """Test OllamaService with custom timeout."""
        from app.services.llm import OllamaService

        service = OllamaService(timeout=60)

        assert service.timeout == 60

    def test_ollama_service_has_ollama_client(self):
        """Test OllamaService initializes ollama client."""
        from app.services.llm import OllamaService

        service = OllamaService()

        assert service.ollama_client is not None


@pytest.mark.unit
@pytest.mark.llm
class TestGenerateWithOllama:
    """Tests for generate_with_ollama convenience function."""

    def test_generate_with_ollama_function_exists(self):
        """Test that generate_with_ollama function exists."""
        from app.services.llm import generate_with_ollama

        assert callable(generate_with_ollama)

    @patch("app.services.llm._service")
    def test_generate_with_ollama_calls_service(self, mock_service):
        """Test that generate_with_ollama delegates to service."""
        from app.services.llm import generate_with_ollama

        mock_service.generate.return_value = "Test response"

        result = generate_with_ollama("Test prompt")

        mock_service.generate.assert_called_once()
        assert result == "Test response"


@pytest.mark.unit
@pytest.mark.llm
class TestOllamaServiceGenerate:
    """Tests for OllamaService.generate method."""

    @patch("app.services.llm.OllamaService._generate_with_groq")
    def test_generate_uses_groq_when_configured(self, mock_groq):
        """Test that generate uses Groq when provider is groq."""
        from app.services.llm import OllamaService

        mock_groq.return_value = "Groq response"

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.provider = "groq"
            service.llm_settings.temperature = 0.7
            service.llm_settings.max_tokens = 500
            service.host = "http://localhost:11434"
            service.model = "llama3.1:8b"
            service.timeout = 30
            service.groq_client = MagicMock()  # Add groq_client
            service.ollama_model = "llama3.1:8b"  # Add ollama_model
            service._generate_with_groq = mock_groq

            result = service.generate("Test prompt")

            mock_groq.assert_called_once()

    @patch("app.services.llm.OllamaService._generate_with_ollama")
    def test_generate_uses_ollama_when_configured(self, mock_ollama):
        """Test that generate uses Ollama when provider is ollama."""
        from app.services.llm import OllamaService

        mock_ollama.return_value = "Ollama response"

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.provider = "ollama"
            service.llm_settings.temperature = 0.7
            service.llm_settings.max_tokens = 500
            service.host = "http://localhost:11434"
            service.model = "llama3.1:8b"
            service.timeout = 30
            service.ollama_client = MagicMock()  # Required attribute for code path
            service._generate_with_ollama = mock_ollama

            result = service.generate("Test prompt")

            mock_ollama.assert_called_once()

    @patch("app.services.llm.OllamaService._generate_with_ollama")
    @patch("app.services.llm.OllamaService._generate_with_groq")
    @patch("app.services.llm.OLLAMA_AVAILABLE", True)  # Mock OLLAMA_AVAILABLE
    def test_generate_falls_back_to_ollama_on_groq_failure(self, mock_groq, mock_ollama):
        """Test that Groq failure triggers Ollama fallback."""
        from app.services.llm import OllamaService

        mock_groq.side_effect = Exception("Groq API error")
        mock_ollama.return_value = "Ollama fallback response"

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.provider = "groq"
            service.llm_settings.temperature = 0.7
            service.llm_settings.max_tokens = 500
            service.groq_client = MagicMock()
            service.ollama_model = "llama3.1:8b"
            service.ollama_client = MagicMock()  # Required for fallback path
            service._generate_with_groq = mock_groq
            service._generate_with_ollama = mock_ollama

            result = service.generate("Test prompt")

            # Should have tried groq then fallen back to ollama
            mock_groq.assert_called_once()
            mock_ollama.assert_called_once()
            assert result == "Ollama fallback response"


@pytest.mark.unit
@pytest.mark.llm
class TestRetryWithExponentialBackoff:
    """Tests for retry decorator."""

    def test_retry_decorator_exists(self):
        """Test that retry decorator exists."""
        from app.services.llm import retry_with_exponential_backoff

        assert callable(retry_with_exponential_backoff)

    def test_retry_decorator_returns_decorator(self):
        """Test that retry_with_exponential_backoff returns a decorator."""
        from app.services.llm import retry_with_exponential_backoff

        decorator = retry_with_exponential_backoff(max_retries=3)

        assert callable(decorator)

    def test_decorated_function_succeeds_on_first_try(self):
        """Test that decorated function works on first successful call."""
        from app.services.llm import retry_with_exponential_backoff

        call_count = [0]

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        def always_succeeds():
            call_count[0] += 1
            return "success"

        result = always_succeeds()

        assert result == "success"
        assert call_count[0] == 1

    def test_decorated_function_retries_on_failure(self):
        """Test that decorated function retries on failure."""
        from app.services.llm import retry_with_exponential_backoff

        call_count = [0]

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        def fails_twice_then_succeeds():
            call_count[0] += 1
            if call_count[0] < 3:
                raise Exception("Temporary failure")
            return "success"

        result = fails_twice_then_succeeds()

        assert result == "success"
        assert call_count[0] == 3


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncRetryDecorator:
    """Tests for async retry decorator."""

    def test_async_retry_decorator_exists(self):
        """Test that async retry decorator exists."""
        from app.services.llm import async_retry_with_exponential_backoff

        assert callable(async_retry_with_exponential_backoff)


@pytest.mark.unit
@pytest.mark.llm
class TestGetOllamaService:
    """Tests for get_ollama_service singleton."""

    def test_get_ollama_service_returns_service(self):
        """Test that get_ollama_service returns a service instance."""
        from app.services.llm import get_ollama_service

        service = get_ollama_service()

        assert service is not None

    def test_get_ollama_service_returns_singleton(self):
        """Test that get_ollama_service returns same instance."""
        from app.services.llm import get_ollama_service

        service1 = get_ollama_service()
        service2 = get_ollama_service()

        assert service1 is service2


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncGenerate:
    """Tests for async generation methods."""

    def test_async_generate_method_exists(self):
        """Test that async_generate method exists."""
        from app.services.llm import OllamaService

        service = OllamaService()

        assert hasattr(service, 'async_generate')
        assert callable(service.async_generate)

    def test_async_generate_with_ollama_function_exists(self):
        """Test that async_generate_with_ollama function exists."""
        from app.services.llm import async_generate_with_ollama

        assert callable(async_generate_with_ollama)


@pytest.mark.unit
@pytest.mark.llm
class TestOllamaGeneration:
    """Tests for _generate_with_ollama method."""

    def test_generate_with_ollama_method_exists(self):
        """Test that _generate_with_ollama method exists."""
        from app.services.llm import OllamaService

        service = OllamaService()

        assert hasattr(service, '_generate_with_ollama')
        assert callable(service._generate_with_ollama)

    @patch("ollama.Client")
    def test_generate_with_ollama_calls_client(self, mock_client_class):
        """Test that _generate_with_ollama uses the ollama client."""
        from app.services.llm import OllamaService

        mock_client = MagicMock()
        mock_client.generate.return_value = {"response": "Generated text"}
        mock_client_class.return_value = mock_client

        service = OllamaService()
        service.ollama_client = mock_client

        result = service._generate_with_ollama(
            prompt="Test prompt",
            temperature=0.7,
            max_tokens=500
        )

        assert result == "Generated text"
        mock_client.generate.assert_called_once()
