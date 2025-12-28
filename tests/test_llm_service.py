"""
Unit tests for LLM Service (Provider Abstraction).

Tests:
- Provider-based generation (via abstraction layer)
- Error handling
- Circuit breaker pattern
- Rate limiting
- Dynamic provider switching
- Streaming with thinking support
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def mock_settings():
    """Mock settings for LLM configuration."""
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
        mock_llm_settings.deepinfra_api_key = "test-deepinfra-key"
        mock_llm_settings.deepinfra_model = "Qwen/Qwen3-32B"

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


@pytest.fixture
def mock_provider():
    """Mock LLM provider with async methods."""
    provider = MagicMock()
    provider.provider_name = "groq"
    provider.default_model = "llama-3.1-8b-instant"
    provider.generate = AsyncMock(return_value="This is a test response from Groq.")
    provider.generate_stream = AsyncMock()
    provider.generate_stream_with_thinking = AsyncMock()
    return provider


@pytest.mark.unit
@pytest.mark.llm
class TestLLMServiceGeneration:
    """Tests for LLM service generation using provider abstraction."""

    @pytest.mark.asyncio
    async def test_successful_generation(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test successful async generation with provider."""
        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()
                result = await service.async_generate(prompt="Test prompt")

                assert result == "This is a test response from Groq."
                mock_provider.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_generation_with_temperature(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test that temperature parameter is passed to provider."""
        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()
                await service.async_generate(prompt="Test", temperature=0.7)

                # Verify temperature was passed
                call_kwargs = mock_provider.generate.call_args.kwargs
                assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_generation_with_max_tokens(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test that max_tokens parameter is passed to provider."""
        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()
                await service.async_generate(prompt="Test", max_tokens=500)

                # Verify max_tokens was passed
                call_kwargs = mock_provider.generate.call_args.kwargs
                assert call_kwargs["max_tokens"] == 500

    @pytest.mark.asyncio
    async def test_model_selection(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test that correct model is used."""
        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()
                await service.async_generate(
                    prompt="Test", model="llama-3.3-70b-versatile"
                )

                # Verify correct model was requested
                call_kwargs = mock_provider.generate.call_args.kwargs
                assert call_kwargs["model"] == "llama-3.3-70b-versatile"


@pytest.mark.unit
@pytest.mark.llm
class TestLLMServiceErrorHandling:
    """Tests for error handling in LLM service."""

    @pytest.mark.asyncio
    async def test_api_timeout(self, mock_settings, mock_rate_limiter, mock_provider):
        """Test handling of API timeout."""
        mock_provider.generate = AsyncMock(side_effect=TimeoutError("API timeout"))

        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()

                with pytest.raises(TimeoutError):
                    await service.async_generate(prompt="Test")

    @pytest.mark.asyncio
    async def test_api_error(self, mock_settings, mock_rate_limiter, mock_provider):
        """Test handling of generic API error."""
        mock_provider.generate = AsyncMock(
            side_effect=Exception("API error: 500 Internal Server Error")
        )

        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()

                with pytest.raises(Exception) as exc:
                    await service.async_generate(prompt="Test")
                assert "API error" in str(exc.value)

    @pytest.mark.asyncio
    async def test_authentication_error(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test handling of authentication error."""
        mock_provider.generate = AsyncMock(
            side_effect=Exception("Authentication error: Invalid API key")
        )

        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                service = LLMService()

                with pytest.raises(Exception) as exc:
                    await service.async_generate(prompt="Test")
                assert "Authentication" in str(exc.value)


@pytest.mark.unit
@pytest.mark.llm
class TestCircuitBreaker:
    """Tests for circuit breaker functionality."""

    def test_circuit_breaker_opens_after_failures(self, mock_settings):
        """Test that circuit breaker opens after consecutive failures."""
        from app.services.llm import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=3, name="test")

        # Record failures up to threshold
        for _ in range(3):
            cb.record_failure()

        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_allows_request_when_closed(self, mock_settings):
        """Test that requests are allowed when circuit is closed."""
        from app.services.llm import CircuitBreaker

        cb = CircuitBreaker(name="test")

        assert cb.allow_request() is True

    def test_circuit_breaker_blocks_request_when_open(self, mock_settings):
        """Test that requests are blocked when circuit is open."""
        from app.services.llm import CircuitBreaker, CircuitState

        cb = CircuitBreaker(failure_threshold=1, name="test")
        cb.record_failure()  # Opens the circuit

        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_circuit_breaker_resets_on_success(self, mock_settings):
        """Test that success resets failure count."""
        from app.services.llm import CircuitBreaker

        cb = CircuitBreaker(failure_threshold=3, name="test")

        # Record some failures
        cb.record_failure()
        cb.record_failure()

        # Success should reset the count
        cb.record_success()

        # Should still be closed and able to tolerate more failures
        assert cb.allow_request() is True


@pytest.mark.unit
@pytest.mark.llm
class TestLLMPromptFormatting:
    """Tests for prompt formatting."""

    @pytest.mark.asyncio
    async def test_prompt_structure(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test that prompts are properly structured."""
        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                from app.services.llm import LLMService

                prompt = """Context: Test context

Question: What is my Python experience?

Answer:"""

                service = LLMService()
                await service.async_generate(prompt=prompt)

                # Verify prompt was sent correctly
                call_kwargs = mock_provider.generate.call_args.kwargs
                assert "prompt" in call_kwargs
                assert "Context:" in call_kwargs["prompt"]


@pytest.mark.unit
@pytest.mark.llm
class TestLLMWrapperFunction:
    """Tests for the convenience wrapper functions."""

    @pytest.mark.asyncio
    async def test_async_wrapper_function(
        self, mock_settings, mock_rate_limiter, mock_provider
    ):
        """Test async convenience function for generation."""
        with patch("app.services.llm.get_provider", return_value=mock_provider):
            with patch("app.services.llm.RateLimiter", return_value=mock_rate_limiter):
                # Reset the global service instance
                import app.services.llm as llm_module

                llm_module._service = None

                from app.services.llm import async_generate_with_llm

                result = await async_generate_with_llm(
                    prompt="Test prompt", temperature=0.1, max_tokens=500
                )

                assert result == "This is a test response from Groq."


@pytest.mark.unit
@pytest.mark.llm
class TestDynamicProviderSwitching:
    """Tests for dynamic provider switching based on model name."""

    @pytest.mark.asyncio
    async def test_qwen_shorthand_uses_deepinfra(
        self, mock_settings, mock_rate_limiter
    ):
        """Test that 'qwen' shorthand switches to DeepInfra provider."""
        mock_groq_provider = MagicMock()
        mock_groq_provider.provider_name = "groq"
        mock_groq_provider.default_model = "llama-3.1-8b-instant"

        mock_deepinfra_provider = MagicMock()
        mock_deepinfra_provider.provider_name = "deepinfra"
        mock_deepinfra_provider.default_model = "Qwen/Qwen3-32B"
        mock_deepinfra_provider.generate = AsyncMock(return_value="Qwen response")

        with patch("app.services.llm.get_provider", return_value=mock_groq_provider):
            with patch(
                "app.services.llm.get_llm_provider",
                return_value=mock_deepinfra_provider,
            ):
                with patch(
                    "app.services.llm.RateLimiter", return_value=mock_rate_limiter
                ):
                    from app.services.llm import LLMService
                    import app.services.llm as llm_module

                    # Clear dynamic providers cache
                    llm_module._dynamic_providers = {}

                    service = LLMService()
                    result = await service.async_generate(prompt="Test", model="qwen")

                    assert result == "Qwen response"
                    mock_deepinfra_provider.generate.assert_called_once()


@pytest.mark.unit
@pytest.mark.llm
class TestRetryDecorator:
    """Tests for async retry decorator."""

    @pytest.mark.asyncio
    async def test_async_retry_decorator_backoff(self):
        """Test that async retry decorator backs off correctly."""
        from app.services.llm import async_retry_with_exponential_backoff

        mock_func = MagicMock()
        mock_func.__name__ = "mock_func"
        mock_func.side_effect = [Exception("Fail 1"), "Success"]

        async def async_wrapper(*args, **kwargs):
            return mock_func(*args, **kwargs)

        async_wrapper.__name__ = "async_wrapper"

        with patch("asyncio.sleep") as mock_sleep:
            decorated = async_retry_with_exponential_backoff(max_retries=3)(
                async_wrapper
            )
            result = await decorated()

            assert result == "Success"
            assert mock_func.call_count == 2
            mock_sleep.assert_called_with(1)

    @pytest.mark.asyncio
    async def test_async_retry_max_retries_exceeded(self):
        """Test that decorator raises exception after max retries."""
        from app.services.llm import async_retry_with_exponential_backoff

        mock_func = MagicMock()
        mock_func.__name__ = "mock_func"
        mock_func.side_effect = Exception("Persistent Fail")

        async def async_wrapper(*args, **kwargs):
            return mock_func(*args, **kwargs)

        async_wrapper.__name__ = "async_wrapper"

        with patch("asyncio.sleep"):
            decorated = async_retry_with_exponential_backoff(max_retries=2)(
                async_wrapper
            )
            with pytest.raises(Exception) as exc:
                await decorated()
            assert "Persistent Fail" in str(exc.value)
            assert mock_func.call_count == 2
