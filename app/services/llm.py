"""
LLM service for Personal RAG system.

Handles LLM client interactions with Groq API (cloud, fast, free tier).
Includes rate limiting for Groq to stay within free tier limits.
"""

import logging
import time
from typing import Optional, AsyncIterator
import asyncio
from functools import wraps

from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion

from app.settings import settings
from app.services.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from app.metrics import (
        rag_llm_request_total,
        rag_llm_latency_seconds,
    )

    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for LLM service")


def retry_with_exponential_backoff(
    max_retries=3, base_delay=1, max_delay=10, exponential_base=2
):
    """Retry decorator with exponential backoff for synchronous functions"""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = min(base_delay * (exponential_base**attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed in {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)

        return wrapper

    return decorator


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


class GroqLLMService:
    """Service for interacting with Groq LLM API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """Initialize Groq LLM service.

        Args:
            api_key: Groq API key (defaults to settings)
            model: Default model name (defaults to settings)
        """
        # LLM settings
        self.llm_settings = settings.llm

        # Validate that provider is groq
        if self.llm_settings.provider != "groq":
            raise ValueError(
                f"GroqLLMService only supports provider='groq', got '{self.llm_settings.provider}'"
            )

        # Groq settings
        self.api_key = api_key or self.llm_settings.groq_api_key
        self.model = model or self.llm_settings.groq_model

        if not self.api_key:
            raise ValueError(
                "Groq API key is required. Set LLM_GROQ_API_KEY environment variable."
            )

        # Initialize Groq clients
        try:
            self.groq_client = Groq(api_key=self.api_key)
            self.async_groq_client = AsyncGroq(api_key=self.api_key)

            # Initialize rate limiter based on configured Groq tier
            # Tier limits (from settings):
            # - Free: 30 rpm, 14,400 rpd (8B) or 1,000 rpd (70B)
            # - Developer: 300-1000 rpm, 50,000-500,000 rpd
            # - Enterprise: Custom limits
            self.rate_limiter = RateLimiter(
                requests_per_minute=self.llm_settings.groq_requests_per_minute,
                requests_per_day=self.llm_settings.groq_requests_per_day,
            )

            logger.info(
                f"Initialized Groq client with model: {self.model} "
                f"(tier: {self.llm_settings.groq_tier})"
            )
            logger.info(
                f"Rate limiter: {self.llm_settings.groq_requests_per_minute} req/min, "
                f"{self.llm_settings.groq_requests_per_day} req/day"
            )
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {e}")
            raise

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using Groq API.

        Args:
            prompt: The input prompt to generate text from
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name

        Returns:
            Generated text response

        Raises:
            Exception: If Groq API call fails
        """
        # Use settings defaults if not provided
        temperature = (
            temperature if temperature is not None else self.llm_settings.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None else self.llm_settings.max_tokens
        )

        logger.debug("Generating with Groq")
        return self._generate_with_groq(prompt, temperature, max_tokens, model)

    @retry_with_exponential_backoff(max_retries=2)
    def _generate_with_groq(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using Groq API with rate limiting.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override

        Returns:
            Generated text

        Raises:
            Exception: If Groq API call fails
        """
        model_name = model or self.model

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            acquired = self.rate_limiter.acquire(timeout=60)
            if not acquired:
                raise Exception(
                    "Rate limiter timeout - Groq request quota exceeded. Please try again later."
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
            logger.debug(f"Calling Groq API with model: {model_name}")

            response: ChatCompletion = self.groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            generated_text = response.choices[0].message.content

            # Metrics
            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"groq:{model_name}"
                ).inc()

            duration = time.time() - start
            logger.info(
                f"Groq generation successful: {len(generated_text)} chars in {duration:.2f}s"
            )

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"groq:{model_name}"
                ).observe(duration)

            return generated_text

        except Exception as e:
            duration = time.time() - start

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"groq:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"groq:{model_name}"
                ).observe(duration)

            # Check for specific error types
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                logger.warning(f"Groq rate limit exceeded: {e}")
            elif (
                "api_key" in error_msg.lower() or "authentication" in error_msg.lower()
            ):
                logger.error(f"Groq authentication error: {e}")
            else:
                logger.error(f"Groq API error: {e}")

            raise

    async def async_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text asynchronously using Groq API.

        Args:
            prompt: The input prompt to generate text from
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name

        Returns:
            Generated text response

        Raises:
            Exception: If Groq API call fails
        """
        # Use settings defaults if not provided
        temperature = (
            temperature if temperature is not None else self.llm_settings.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None else self.llm_settings.max_tokens
        )

        logger.debug("Generating with Groq (async)")
        return await self._generate_with_groq_async(
            prompt, temperature, max_tokens, model
        )

    @async_retry_with_exponential_backoff(max_retries=2)
    async def _generate_with_groq_async(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Generate text asynchronously using Groq API with rate limiting.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override

        Returns:
            Generated text

        Raises:
            Exception: If Groq API call fails
        """
        model_name = model or self.model

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            # Run synchronous rate limiter in executor to not block
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None, lambda: self.rate_limiter.acquire(timeout=60)
            )
            if not acquired:
                raise Exception(
                    "Rate limiter timeout - Groq request quota exceeded. Please try again later."
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
            logger.debug(f"Calling Groq API async with model: {model_name}")

            response: ChatCompletion = (
                await self.async_groq_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            )

            generated_text = response.choices[0].message.content

            # Metrics
            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"groq:{model_name}"
                ).inc()

            duration = time.time() - start
            logger.info(
                f"Groq async generation successful: {len(generated_text)} chars in {duration:.2f}s"
            )

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"groq:{model_name}"
                ).observe(duration)

            return generated_text

        except Exception as e:
            duration = time.time() - start

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"groq:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"groq:{model_name}"
                ).observe(duration)

            # Check for specific error types
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                logger.warning(f"Groq rate limit exceeded: {e}")
            elif (
                "api_key" in error_msg.lower() or "authentication" in error_msg.lower()
            ):
                logger.error(f"Groq authentication error: {e}")
            else:
                logger.error(f"Groq API error: {e}")

            raise

    async def async_generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Generate text stream asynchronously using Groq API.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name

        Yields:
            Generated text chunks
        """
        # Use settings defaults if not provided
        temperature = (
            temperature if temperature is not None else self.llm_settings.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None else self.llm_settings.max_tokens
        )

        logger.debug("Generating with Groq (async stream)")
        async for chunk in self._generate_with_groq_async_stream(
            prompt, temperature, max_tokens, model
        ):
            yield chunk

    async def _generate_with_groq_async_stream(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Generate text stream asynchronously using Groq API with rate limiting.

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override

        Yields:
            Generated text chunks
        """
        model_name = model or self.model

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            # Run synchronous rate limiter in executor to not block
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None, lambda: self.rate_limiter.acquire(timeout=60)
            )
            if not acquired:
                raise Exception(
                    "Rate limiter timeout - Groq request quota exceeded. Please try again later."
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
            logger.debug(f"Calling Groq API async stream with model: {model_name}")

            stream = await self.async_groq_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            # Metrics (start)
            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"groq:{model_name}"
                ).inc()

            async for chunk in stream:
                content = chunk.choices[0].delta.content
                if content:
                    yield content

            duration = time.time() - start
            logger.info(f"Groq async stream finished in {duration:.2f}s")

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"groq:{model_name}"
                ).observe(duration)

        except Exception as e:
            duration = time.time() - start

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"groq:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"groq:{model_name}"
                ).observe(duration)

            # Check for specific error types
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                logger.warning(f"Groq rate limit exceeded: {e}")
            elif (
                "api_key" in error_msg.lower() or "authentication" in error_msg.lower()
            ):
                logger.error(f"Groq authentication error: {e}")
            else:
                logger.error(f"Groq API error: {e}")

            raise


# Global service instance (lazy initialization)
_service: GroqLLMService = None


def get_llm_service() -> GroqLLMService:
    """Get the global LLM service instance (lazy initialization)."""
    global _service
    if _service is None:
        _service = GroqLLMService()
        logger.info(
            f"Initialized global LLM service - Provider: {_service.llm_settings.provider}, Model: {_service.model}"
        )
    return _service


def generate_with_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Convenience function for generating text with Groq LLM.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override

    Returns:
        Generated text response
    """
    service = get_llm_service()
    return service.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


async def async_generate_with_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Async convenience function for generating text with Groq LLM.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override

    Returns:
        Generated text response
    """
    service = get_llm_service()
    return await service.async_generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


async def async_generate_stream_with_llm(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> AsyncIterator[str]:
    """Async convenience function for streaming text with Groq LLM.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override

    Yields:
        Generated text chunks
    """
    service = get_llm_service()
    async for chunk in service.async_generate_stream(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    ):
        yield chunk


__all__ = [
    "GroqLLMService",
    "get_llm_service",
    "generate_with_llm",
    "async_generate_with_llm",
    "async_generate_stream_with_llm",
]
