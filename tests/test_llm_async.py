"""
Async tests for LLM service.

Tests async generation methods for Groq and Ollama with proper async mocking.
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncGenerate:
    """Tests for async_generate method."""

    @pytest.mark.asyncio
    async def test_async_generate_uses_ollama_by_default(self):
        """Test async_generate uses Ollama when provider is ollama."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.provider = "ollama"
            service.llm_settings.temperature = 0.7
            service.llm_settings.max_tokens = 500
            service.async_groq_client = None
            service.ollama_client = MagicMock()  # Required attribute
            service.ollama_model = "llama3.1:8b"

            # Mock the async method
            async def mock_ollama_async(*args, **kwargs):
                return "Ollama async response"

            service._generate_with_ollama_async = mock_ollama_async

            result = await service.async_generate(
                prompt="Test prompt",
                temperature=0.7,
                max_tokens=100,
                model=None
            )

            assert result == "Ollama async response"

    @pytest.mark.asyncio
    async def test_async_generate_uses_groq_when_configured(self):
        """Test async_generate uses Groq when provider is groq."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.provider = "groq"
            service.llm_settings.temperature = 0.7
            service.llm_settings.max_tokens = 500
            service.async_groq_client = MagicMock()  # Not None
            service.ollama_model = "llama3.1:8b"

            # Mock the async method
            async def mock_groq_async(*args, **kwargs):
                return "Groq async response"

            service._generate_with_groq_async = mock_groq_async

            result = await service.async_generate(
                prompt="Test prompt",
                temperature=0.7,
                max_tokens=100,
                model=None
            )

            assert result == "Groq async response"

    @pytest.mark.asyncio
    @patch("app.services.llm.OLLAMA_AVAILABLE", True)  # Mock OLLAMA_AVAILABLE
    async def test_async_generate_fallback_on_groq_failure(self):
        """Test async_generate falls back to Ollama when Groq fails."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.provider = "groq"
            service.llm_settings.temperature = 0.7
            service.llm_settings.max_tokens = 500
            service.async_groq_client = MagicMock()
            service.ollama_client = MagicMock()  # Required for fallback path
            service.ollama_model = "llama3.1:8b"

            # Mock Groq to fail
            async def mock_groq_async_fail(*args, **kwargs):
                raise Exception("Groq API timeout")

            # Mock Ollama to succeed
            async def mock_ollama_async(*args, **kwargs):
                return "Ollama fallback response"

            service._generate_with_groq_async = mock_groq_async_fail
            service._generate_with_ollama_async = mock_ollama_async

            result = await service.async_generate(
                prompt="Test prompt",
                temperature=0.7,
                max_tokens=100,
                model=None
            )

            assert result == "Ollama fallback response"


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncGroqGeneration:
    """Tests for _generate_with_groq_async method."""

    @pytest.mark.asyncio
    async def test_groq_async_returns_response(self):
        """Test _generate_with_groq_async returns generated text."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.groq_model = "llama-3.1-8b-instant"
            service.rate_limiter = None

            # Mock async Groq client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Generated async response"

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            service.async_groq_client = mock_client

            result = await service._generate_with_groq_async(
                prompt="Test prompt",
                temperature=0.7,
                max_tokens=100,
                model=None
            )

            assert result == "Generated async response"

    @pytest.mark.asyncio
    async def test_groq_async_with_rate_limiter(self):
        """Test _generate_with_groq_async respects rate limiter."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.groq_model = "llama-3.1-8b-instant"

            # Mock rate limiter
            mock_rate_limiter = MagicMock()
            mock_rate_limiter.acquire.return_value = True
            mock_rate_limiter.get_stats.return_value = {
                "minute_utilization": 0.5,
                "day_utilization": 0.3,
                "requests_last_minute": 10,
                "requests_per_minute_limit": 28,
                "requests_last_day": 100,
                "requests_per_day_limit": 1000
            }
            service.rate_limiter = mock_rate_limiter

            # Mock async Groq client
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = "Rate limited response"

            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
            service.async_groq_client = mock_client

            # Mock event loop for rate limiter
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=True)

                result = await service._generate_with_groq_async(
                    prompt="Test prompt",
                    temperature=0.7,
                    max_tokens=100,
                    model=None
                )

            assert result == "Rate limited response"

    @pytest.mark.asyncio
    async def test_groq_async_rate_limit_timeout(self):
        """Test _generate_with_groq_async handles rate limiter timeout."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.groq_model = "llama-3.1-8b-instant"

            # Mock rate limiter to return False (timeout)
            mock_rate_limiter = MagicMock()
            mock_rate_limiter.acquire.return_value = False
            service.rate_limiter = mock_rate_limiter

            service.async_groq_client = AsyncMock()

            # Mock event loop for rate limiter
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_loop.return_value.run_in_executor = AsyncMock(return_value=False)

                with pytest.raises(Exception) as exc_info:
                    await service._generate_with_groq_async(
                        prompt="Test prompt",
                        temperature=0.7,
                        max_tokens=100,
                        model=None
                    )

                assert "Rate limiter timeout" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_groq_async_api_error(self):
        """Test _generate_with_groq_async handles API errors."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.llm_settings = MagicMock()
            service.llm_settings.groq_model = "llama-3.1-8b-instant"
            service.rate_limiter = None

            # Mock async Groq client to raise error
            mock_client = AsyncMock()
            mock_client.chat.completions.create = AsyncMock(
                side_effect=Exception("Groq API error")
            )
            service.async_groq_client = mock_client

            with pytest.raises(Exception) as exc_info:
                await service._generate_with_groq_async(
                    prompt="Test prompt",
                    temperature=0.7,
                    max_tokens=100,
                    model=None
                )

            assert "Groq API error" in str(exc_info.value)


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncOllamaGeneration:
    """Tests for _generate_with_ollama_async method."""

    @pytest.mark.asyncio
    async def test_ollama_async_wraps_sync(self):
        """Test _generate_with_ollama_async wraps sync method."""
        from app.services.llm import OllamaService

        with patch.object(OllamaService, '__init__', lambda x, **kw: None):
            service = OllamaService()
            service.ollama_model = "llama3.1:8b"
            service.num_ctx = 4096

            # Mock the sync method
            service._generate_with_ollama = MagicMock(return_value="Sync ollama response")

            # Mock event loop
            with patch('asyncio.get_event_loop') as mock_loop:
                mock_executor = AsyncMock(return_value="Sync ollama response")
                mock_loop.return_value.run_in_executor = mock_executor

                result = await service._generate_with_ollama_async(
                    prompt="Test prompt",
                    temperature=0.7,
                    max_tokens=100,
                    model=None
                )

            assert result == "Sync ollama response"


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncGenerateConvenienceFunction:
    """Tests for async_generate_with_ollama convenience function."""

    @pytest.mark.asyncio
    async def test_async_generate_with_ollama_calls_service(self):
        """Test async_generate_with_ollama delegates to service."""
        from app.services.llm import async_generate_with_ollama

        with patch('app.services.llm.get_ollama_service') as mock_get_service:
            mock_service = MagicMock()
            mock_service.async_generate = AsyncMock(return_value="Async convenience response")
            mock_get_service.return_value = mock_service

            result = await async_generate_with_ollama(
                prompt="Test prompt",
                temperature=0.5,
                max_tokens=200,
                model="custom-model"
            )

            assert result == "Async convenience response"
            mock_service.async_generate.assert_called_once_with(
                prompt="Test prompt",
                temperature=0.5,
                max_tokens=200,
                model="custom-model"
            )


@pytest.mark.unit
@pytest.mark.llm
class TestAsyncRetryDecorator:
    """Tests for async_retry_with_exponential_backoff decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_decorator_succeeds_first_try(self):
        """Test decorator doesn't retry when function succeeds."""
        from app.services.llm import async_retry_with_exponential_backoff

        call_count = 0

        @async_retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await successful_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_async_retry_decorator_retries_on_failure(self):
        """Test decorator retries on failure."""
        from app.services.llm import async_retry_with_exponential_backoff

        call_count = 0

        @async_retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        async def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success after retry"

        result = await fail_then_succeed()

        assert result == "success after retry"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_async_retry_decorator_gives_up_after_max_retries(self):
        """Test decorator gives up after max retries."""
        from app.services.llm import async_retry_with_exponential_backoff

        call_count = 0

        @async_retry_with_exponential_backoff(max_retries=2, base_delay=0.01)
        async def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")

        with pytest.raises(Exception) as exc_info:
            await always_fails()

        assert "Persistent failure" in str(exc_info.value)
        assert call_count == 2


@pytest.mark.unit
@pytest.mark.llm
class TestSyncRetryDecorator:
    """Tests for sync retry_with_exponential_backoff decorator."""

    def test_sync_retry_decorator_succeeds_first_try(self):
        """Test sync decorator doesn't retry when function succeeds."""
        from app.services.llm import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()

        assert result == "success"
        assert call_count == 1

    def test_sync_retry_decorator_retries_on_failure(self):
        """Test sync decorator retries on failure."""
        from app.services.llm import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_retries=3, base_delay=0.01)
        def fail_then_succeed():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise Exception("Temporary failure")
            return "success after retry"

        result = fail_then_succeed()

        assert result == "success after retry"
        assert call_count == 2

    def test_sync_retry_decorator_gives_up_after_max_retries(self):
        """Test sync decorator gives up after max retries."""
        from app.services.llm import retry_with_exponential_backoff

        call_count = 0

        @retry_with_exponential_backoff(max_retries=2, base_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise Exception("Persistent failure")

        with pytest.raises(Exception) as exc_info:
            always_fails()

        assert "Persistent failure" in str(exc_info.value)
        assert call_count == 2
