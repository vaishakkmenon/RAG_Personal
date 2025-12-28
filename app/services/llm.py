"""
LLM service for Personal RAG system.

Handles LLM client interactions with multiple providers (Groq, DeepInfra).
Includes rate limiting and circuit breaker pattern for resilience.

Version 2.0: Uses provider abstraction for multi-provider support.
Version 2.1: Added streaming with thinking support for frontend display.
"""

import logging
import time
import threading
from enum import Enum
from typing import Optional, AsyncIterator
import asyncio
from functools import wraps

from app.settings import settings
from app.services.rate_limiter import RateLimiter
from app.llm import get_provider, get_llm_provider, resolve_model, LLMProvider
from app.core.parsing import StreamChunk, ChunkType, ReasoningEffort

logger = logging.getLogger(__name__)


# Cache for dynamically created providers
_dynamic_providers: dict = {}

# Optional metrics import
try:
    from app.metrics import (
        rag_llm_request_total,
        rag_llm_latency_seconds,
        rag_circuit_breaker_state,
        rag_circuit_breaker_transitions_total,
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for LLM service")


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation, requests pass through
    OPEN = "open"  # Failing, requests are rejected immediately
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerOpen(Exception):
    """Exception raised when circuit breaker is open."""

    pass


class CircuitBreaker:
    """
    Circuit breaker pattern implementation for external API resilience.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, reject requests immediately (fail fast)
    - HALF_OPEN: Allow limited requests to test if service recovered

    Transitions:
    - CLOSED -> OPEN: After `failure_threshold` consecutive failures
    - OPEN -> HALF_OPEN: After `recovery_timeout` seconds
    - HALF_OPEN -> CLOSED: After `half_open_successes` successful requests
    - HALF_OPEN -> OPEN: On any failure
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_requests: int = 3,
        name: str = "default",
    ):
        """
        Initialize circuit breaker.

        Args:
            failure_threshold: Consecutive failures before opening circuit
            recovery_timeout: Seconds to wait before trying again (OPEN -> HALF_OPEN)
            half_open_max_requests: Successful requests needed to close circuit
            name: Name for logging and metrics
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_requests = half_open_max_requests
        self.name = name

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[float] = None
        self._lock = threading.Lock()

        logger.info(
            f"Circuit breaker '{name}' initialized: "
            f"threshold={failure_threshold}, timeout={recovery_timeout}s"
        )

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, transitioning if needed."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                # Check if we should transition to HALF_OPEN
                if self._last_failure_time is not None:
                    elapsed = time.time() - self._last_failure_time
                    if elapsed >= self.recovery_timeout:
                        self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: CircuitState) -> None:
        """Transition to a new state (must hold lock)."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        logger.warning(
            f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}"
        )

        if METRICS_ENABLED:
            rag_circuit_breaker_state.labels(name=self.name).set(
                {"closed": 0, "open": 1, "half_open": 0.5}[new_state.value]
            )
            rag_circuit_breaker_transitions_total.labels(
                name=self.name, from_state=old_state.value, to_state=new_state.value
            ).inc()

    def allow_request(self) -> bool:
        """Check if request should be allowed through."""
        current_state = self.state  # This may trigger OPEN -> HALF_OPEN

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            return False
        else:  # HALF_OPEN
            return True

    def record_success(self) -> None:
        """Record a successful request."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.half_open_max_requests:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed request."""
        with self._lock:
            self._last_failure_time = time.time()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open immediately opens circuit
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def get_stats(self) -> dict:
        """Get circuit breaker statistics."""
        with self._lock:
            return {
                "state": self._state.value,
                "failure_count": self._failure_count,
                "success_count": self._success_count,
                "last_failure_time": self._last_failure_time,
                "failure_threshold": self.failure_threshold,
                "recovery_timeout": self.recovery_timeout,
            }


def async_retry_with_exponential_backoff(
    max_retries=3, base_delay=1, max_delay=10, exponential_base=2
):
    """Retry decorator with exponential backoff for asynchronous functions"""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed in {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)

        return wrapper

    return decorator


class LLMService:
    """Service for interacting with LLM providers (Groq, DeepInfra, etc.)."""

    def __init__(self):
        """Initialize LLM service with configured provider."""
        self.llm_settings = settings.llm
        self.provider: LLMProvider = get_provider()

        # Initialize rate limiter (for Groq tier management)
        # Note: DeepInfra has different rate limits, but we use conservative defaults
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.llm_settings.groq_requests_per_minute,
            requests_per_day=self.llm_settings.groq_requests_per_day,
        )

        # Initialize circuit breaker for resilience
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0,
            half_open_max_requests=3,
            name=f"{self.provider.provider_name}_api",
        )

        logger.info(
            f"Initialized LLM service - Provider: {self.provider.provider_name}, "
            f"Model: {self.provider.default_model}"
        )

    def _get_provider_for_model(self, model: Optional[str]) -> tuple[LLMProvider, str]:
        """Get the appropriate provider for a model name.

        Supports model shorthand names (e.g., "groq", "qwen", "deepinfra")
        and automatically selects the correct provider.

        Args:
            model: Model name, shorthand, or None for default

        Returns:
            Tuple of (provider, resolved_model_name)
        """
        if not model:
            return self.provider, self.provider.default_model

        # Resolve model shorthand to provider and full model name
        provider_name, resolved_model = resolve_model(model)

        # Check if we need a different provider than the default
        if provider_name != self.provider.provider_name:
            # Get or create the alternate provider
            global _dynamic_providers
            if provider_name not in _dynamic_providers:
                logger.info(f"Creating dynamic provider for: {provider_name}")
                _dynamic_providers[provider_name] = get_llm_provider(provider_name)
            provider = _dynamic_providers[provider_name]
        else:
            provider = self.provider

        # Use resolved model name or provider default
        final_model = resolved_model or provider.default_model
        return provider, final_model

    async def async_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
    ) -> str:
        """Generate text asynchronously using configured provider.

        Args:
            prompt: The input prompt to generate text from
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name
            reasoning_effort: Control reasoning depth (OFF for fastest RAG responses)

        Returns:
            Generated text response

        Raises:
            Exception: If provider API call fails
        """
        # Use settings defaults if not provided
        temp = temperature if temperature is not None else self.llm_settings.temperature
        tokens = max_tokens if max_tokens is not None else self.llm_settings.max_tokens

        return await self._generate_with_provider(
            prompt, temp, tokens, model, reasoning_effort
        )

    @async_retry_with_exponential_backoff(max_retries=2)
    async def _generate_with_provider(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
    ) -> str:
        """Generate text using provider with rate limiting and circuit breaker.

        Supports dynamic provider switching based on model name (e.g., "qwen" -> DeepInfra).

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override (supports shorthand like "groq", "qwen", "deepinfra")
            reasoning_effort: Control reasoning depth (OFF for fastest RAG responses)

        Returns:
            Generated text

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
            Exception: If provider API call fails
        """
        # Get the appropriate provider for this model
        provider, model_name = self._get_provider_for_model(model)
        provider_name = provider.provider_name

        # Check circuit breaker before making request
        if not self.circuit_breaker.allow_request():
            logger.warning(
                f"Circuit breaker OPEN for {provider_name} API - rejecting request. "
                f"Will retry after {self.circuit_breaker.recovery_timeout}s"
            )
            raise CircuitBreakerOpen(
                f"{provider_name} API circuit breaker is open. Service is temporarily unavailable."
            )

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None, lambda: self.rate_limiter.acquire(timeout=60)
            )
            if not acquired:
                raise Exception(
                    f"Rate limiter timeout - {provider_name} request quota exceeded. Please try again later."
                )

            # Log rate limit status
            stats = self.rate_limiter.get_stats()
            if stats["minute_utilization"] > 0.8 or stats["day_utilization"] > 0.8:
                logger.warning(
                    f"High rate limit usage: "
                    f"{stats['requests_last_minute']}/{stats['requests_per_minute_limit']} req/min, "
                    f"{stats['requests_last_day']}/{stats['requests_per_day_limit']} req/day"
                )

        start = time.time()
        try:
            logger.debug(
                f"Calling {provider_name} API with model: {model_name}, reasoning_effort: {reasoning_effort.value}"
            )

            generated_text = await provider.generate(
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            )

            # Record success with circuit breaker
            self.circuit_breaker.record_success()

            # Metrics
            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"{provider_name}:{model_name}"
                ).inc()

            duration = time.time() - start
            logger.info(
                f"{provider_name} generation successful: {len(generated_text)} chars in {duration:.2f}s"
            )

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"{provider_name}:{model_name}"
                ).observe(duration)

            return generated_text

        except Exception as e:
            duration = time.time() - start

            # Record failure with circuit breaker
            self.circuit_breaker.record_failure()

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"{provider_name}:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"{provider_name}:{model_name}"
                ).observe(duration)

            # Check for specific error types
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                logger.warning(f"{provider_name} rate limit exceeded: {e}")
            elif (
                "api_key" in error_msg.lower() or "authentication" in error_msg.lower()
            ):
                logger.error(f"{provider_name} authentication error: {e}")
            else:
                logger.error(f"{provider_name} API error: {e}")

            raise

    async def async_generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
    ) -> AsyncIterator[str]:
        """Generate text stream asynchronously using configured provider.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name
            reasoning_effort: Control reasoning depth (OFF for fastest RAG responses)

        Yields:
            Generated text chunks
        """
        # Use settings defaults if not provided
        temp = temperature if temperature is not None else self.llm_settings.temperature
        tokens = max_tokens if max_tokens is not None else self.llm_settings.max_tokens

        async for chunk in self._generate_stream_with_provider(
            prompt, temp, tokens, model, reasoning_effort
        ):
            yield chunk

    async def _generate_stream_with_provider(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
    ) -> AsyncIterator[str]:
        """Generate text stream using provider with rate limiting and circuit breaker.

        Supports dynamic provider switching based on model name.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override (supports shorthand like "groq", "qwen", "deepinfra")
            reasoning_effort: Control reasoning depth (OFF for fastest RAG responses)

        Yields:
            Generated text chunks

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
        """
        # Get the appropriate provider for this model
        provider, model_name = self._get_provider_for_model(model)
        provider_name = provider.provider_name

        # Check circuit breaker before making request
        if not self.circuit_breaker.allow_request():
            logger.warning(
                f"Circuit breaker OPEN for {provider_name} API - rejecting request. "
                f"Will retry after {self.circuit_breaker.recovery_timeout}s"
            )
            raise CircuitBreakerOpen(
                f"{provider_name} API circuit breaker is open. Service is temporarily unavailable."
            )

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None, lambda: self.rate_limiter.acquire(timeout=60)
            )
            if not acquired:
                raise Exception(
                    f"Rate limiter timeout - {provider_name} request quota exceeded. Please try again later."
                )

            # Log rate limit status
            stats = self.rate_limiter.get_stats()
            if stats["minute_utilization"] > 0.8 or stats["day_utilization"] > 0.8:
                logger.warning(
                    f"High rate limit usage: "
                    f"{stats['requests_last_minute']}/{stats['requests_per_minute_limit']} req/min, "
                    f"{stats['requests_last_day']}/{stats['requests_per_day_limit']} req/day"
                )

        start = time.time()
        try:
            logger.debug(
                f"Calling {provider_name} API stream with model: {model_name}, reasoning_effort: {reasoning_effort.value}"
            )

            # Metrics (start)
            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"{provider_name}:{model_name}"
                ).inc()

            async for chunk in provider.generate_stream(
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ):
                yield chunk

            # Record success with circuit breaker (stream completed successfully)
            self.circuit_breaker.record_success()

            duration = time.time() - start
            logger.info(f"{provider_name} stream finished in {duration:.2f}s")

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"{provider_name}:{model_name}"
                ).observe(duration)

        except Exception as e:
            duration = time.time() - start

            # Record failure with circuit breaker
            self.circuit_breaker.record_failure()

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"{provider_name}:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"{provider_name}:{model_name}"
                ).observe(duration)

            # Check for specific error types
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                logger.warning(f"{provider_name} rate limit exceeded: {e}")
            elif (
                "api_key" in error_msg.lower() or "authentication" in error_msg.lower()
            ):
                logger.error(f"{provider_name} authentication error: {e}")
            else:
                logger.error(f"{provider_name} API error: {e}")

            raise

    async def async_generate_stream_with_thinking(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
    ) -> AsyncIterator[StreamChunk]:
        """Generate text stream with thinking process separated.

        This method yields typed StreamChunk objects that differentiate between
        the model's thinking process (<think> blocks) and the actual answer,
        allowing frontends to display them differently.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name
            reasoning_effort: Control reasoning depth. When not OFF, model may
                produce <think>...</think> blocks that are parsed and streamed.

        Yields:
            StreamChunk objects with type=THINKING or type=ANSWER
        """
        temp = temperature if temperature is not None else self.llm_settings.temperature
        tokens = max_tokens if max_tokens is not None else self.llm_settings.max_tokens

        async for chunk in self._generate_stream_with_thinking_provider(
            prompt, temp, tokens, model, reasoning_effort
        ):
            yield chunk

    async def _generate_stream_with_thinking_provider(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
    ) -> AsyncIterator[StreamChunk]:
        """Generate typed stream using provider with rate limiting and circuit breaker.

        Supports dynamic provider switching based on model name.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override (supports shorthand like "groq", "qwen", "deepinfra")
            reasoning_effort: Control reasoning depth. When not OFF, model may
                produce <think>...</think> blocks that are parsed and streamed.

        Yields:
            StreamChunk objects with type=THINKING or type=ANSWER

        Raises:
            CircuitBreakerOpen: If circuit breaker is open
        """
        # Get the appropriate provider for this model
        provider, model_name = self._get_provider_for_model(model)
        provider_name = provider.provider_name

        # Check circuit breaker before making request
        if not self.circuit_breaker.allow_request():
            logger.warning(
                f"Circuit breaker OPEN for {provider_name} API - rejecting request. "
                f"Will retry after {self.circuit_breaker.recovery_timeout}s"
            )
            raise CircuitBreakerOpen(
                f"{provider_name} API circuit breaker is open. Service is temporarily unavailable."
            )

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None, lambda: self.rate_limiter.acquire(timeout=60)
            )
            if not acquired:
                raise Exception(
                    f"Rate limiter timeout - {provider_name} request quota exceeded. Please try again later."
                )

        start = time.time()
        try:
            logger.debug(
                f"Calling {provider_name} API stream (with thinking) model: {model_name}, reasoning_effort: {reasoning_effort.value}"
            )

            # Metrics (start)
            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"{provider_name}:{model_name}"
                ).inc()

            async for chunk in provider.generate_stream_with_thinking(
                prompt=prompt,
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                reasoning_effort=reasoning_effort,
            ):
                yield chunk

            # Record success with circuit breaker
            self.circuit_breaker.record_success()

            duration = time.time() - start
            logger.info(f"{provider_name} thinking stream finished in {duration:.2f}s")

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"{provider_name}:{model_name}"
                ).observe(duration)

        except Exception as e:
            duration = time.time() - start
            self.circuit_breaker.record_failure()

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"{provider_name}:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"{provider_name}:{model_name}"
                ).observe(duration)

            logger.error(f"{provider_name} API error: {e}")
            raise


# Backward compatibility aliases
GroqLLMService = LLMService  # Alias for existing code


# Global service instance (lazy initialization)
_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get the global LLM service instance (lazy initialization)."""
    global _service
    if _service is None:
        _service = LLMService()
    return _service


def generate_with_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
) -> str:
    """Synchronous convenience function for generating text with LLM.

    Note: This runs the async generate in an event loop for backward compatibility.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override
        reasoning_effort: Control reasoning depth (NONE for fastest RAG responses)

    Returns:
        Generated text response
    """
    service = get_llm_service()

    # Run async generate in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    service.async_generate(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        model=model,
                        reasoning_effort=reasoning_effort,
                    ),
                )
                return future.result(timeout=120)
        else:
            return loop.run_until_complete(
                service.async_generate(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            )
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(
            service.async_generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                model=model,
                reasoning_effort=reasoning_effort,
            )
        )


async def async_generate_with_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
) -> str:
    """Async convenience function for generating text with LLM.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override
        reasoning_effort: Control reasoning depth (NONE for fastest RAG responses)

    Returns:
        Generated text response
    """
    service = get_llm_service()
    return await service.async_generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        reasoning_effort=reasoning_effort,
    )


async def async_generate_stream_with_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
) -> AsyncIterator[str]:
    """Async convenience function for streaming text with LLM.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override
        reasoning_effort: Control reasoning depth (NONE for fastest RAG responses)

    Yields:
        Generated text chunks
    """
    service = get_llm_service()
    async for chunk in service.async_generate_stream(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        reasoning_effort=reasoning_effort,
    ):
        yield chunk


async def async_generate_stream_with_thinking(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
    reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
) -> AsyncIterator[StreamChunk]:
    """Async convenience function for streaming with thinking process separated.

    This function yields typed StreamChunk objects that differentiate between
    the model's thinking process (<think> blocks) and the actual answer,
    allowing frontends to display them differently.

    SSE Event Types:
    - StreamChunk(type=THINKING, content="...") -> Send as "event: thinking"
    - StreamChunk(type=ANSWER, content="...") -> Send as "event: token"

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override
        reasoning_effort: Control reasoning depth. When not OFF, model may
            produce <think>...</think> blocks that are parsed and streamed.

    Yields:
        StreamChunk objects with type=THINKING or type=ANSWER
    """
    service = get_llm_service()
    async for chunk in service.async_generate_stream_with_thinking(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
        reasoning_effort=reasoning_effort,
    ):
        yield chunk


__all__ = [
    "LLMService",
    "GroqLLMService",  # Backward compatibility
    "get_llm_service",
    "generate_with_llm",  # Sync version for backward compatibility
    "async_generate_with_llm",
    "async_generate_stream_with_llm",
    "async_generate_stream_with_thinking",  # Streaming with thinking support
    "StreamChunk",  # For type hints
    "ChunkType",  # For type checking
    "ReasoningEffort",  # For controlling reasoning depth
    "CircuitBreakerOpen",
    "CircuitBreaker",
]
