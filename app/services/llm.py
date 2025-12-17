"""
LLM service for Personal RAG system.

Handles LLM client interactions with support for multiple providers:
- Groq API (cloud, fast, free tier)
- Ollama (local, self-hosted)

Automatically falls back to Ollama if Groq fails.
Includes rate limiting for Groq to stay within free tier limits.
"""

import logging
import time
from typing import Optional
import asyncio
from functools import wraps

import ollama
from groq import Groq, AsyncGroq
from groq.types.chat import ChatCompletion

from app.settings import settings
from app.services.rate_limiter import RateLimiter, NoOpRateLimiter

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from app.metrics import (
        rag_llm_request_total,
        rag_llm_latency_seconds,
        rag_fallback_operations_total,
    )
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for LLM service")


def retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1,
    max_delay=10,
    exponential_base=2
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
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed in {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
        return wrapper
    return decorator


def async_retry_with_exponential_backoff(
    max_retries=3,
    base_delay=1,
    max_delay=10,
    exponential_base=2
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
                    delay = min(base_delay * (exponential_base ** attempt), max_delay)
                    logger.warning(
                        f"Attempt {attempt + 1}/{max_retries} failed in {func.__name__}: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    await asyncio.sleep(delay)
        return wrapper
    return decorator



class OllamaService:
    """Service for interacting with LLM providers (Groq or Ollama)."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ):
        """Initialize LLM service with provider support.

        Args:
            host: Ollama API host URL (defaults to settings)
            model: Default model name (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
            num_ctx: Context window size (defaults to settings)
        """
        # LLM settings
        self.llm_settings = settings.llm

        # Ollama settings (with backward compatibility)
        self.host = host or self.llm_settings.ollama_host
        self.ollama_model = model or self.llm_settings.ollama_model
        self.timeout = timeout or self.llm_settings.ollama_timeout
        self.num_ctx = num_ctx or self.llm_settings.num_ctx

        # For backward compatibility, set self.model based on provider
        if self.llm_settings.provider == "groq":
            self.model = self.llm_settings.groq_model
        else:
            self.model = self.ollama_model

        # Initialize Ollama client (always available as fallback)
        self.ollama_client = ollama.Client(host=self.host, timeout=self.timeout)

        # Initialize Groq client and rate limiter if configured
        self.groq_client = None
        self.async_groq_client = None
        self.rate_limiter = None
        if self.llm_settings.provider == "groq" and self.llm_settings.groq_api_key:
            try:
                self.groq_client = Groq(api_key=self.llm_settings.groq_api_key)
                self.async_groq_client = AsyncGroq(api_key=self.llm_settings.groq_api_key)

                # Initialize rate limiter for Groq free tier
                # 8B model: 30 req/min, 14,400 req/day
                # 70B model: 30 req/min, 1,000 req/day
                requests_per_day = 1000 if "70b" in self.llm_settings.groq_model.lower() else 14400
                self.rate_limiter = RateLimiter(
                    requests_per_minute=28,  # 30 limit, use 28 for safety margin
                    requests_per_day=int(requests_per_day * 0.95)  # 5% safety margin
                )

                logger.info(
                    f"Initialized Groq client with model: {self.llm_settings.groq_model}"
                )
                logger.info(
                    f"Rate limiter: 28 req/min, {int(requests_per_day * 0.95)} req/day"
                )
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")
                logger.info("Will use Ollama as fallback")
        else:
            # No rate limiting for Ollama
            self.rate_limiter = NoOpRateLimiter()

        # Log initialization
        if self.llm_settings.provider == "groq":
            logger.info(
                f"LLM Service initialized - Provider: groq, "
                f"Model: {self.llm_settings.groq_model}, "
                f"Fallback: {self.host} ({self.ollama_model})"
            )
        else:
            logger.info(
                f"LLM Service initialized - Provider: ollama, "
                f"Host: {self.host}, Model: {self.ollama_model}"
            )

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using configured LLM provider with automatic fallback.

        Args:
            prompt: The input prompt to generate text from
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name

        Returns:
            Generated text response

        Raises:
            Exception: If both providers fail
        """
        # Use settings defaults if not provided
        temperature = (
            temperature if temperature is not None
            else self.llm_settings.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None
            else self.llm_settings.max_tokens
        )

        # Try primary provider
        try:
            if self.llm_settings.provider == "groq" and self.groq_client:
                logger.debug("Attempting generation with Groq")
                return self._generate_with_groq(
                    prompt, temperature, max_tokens, model
                )
            else:
                logger.debug("Generating with Ollama")
                return self._generate_with_ollama(
                    prompt, temperature, max_tokens, model
                )
        except Exception as e:
            # Log the error
            logger.error(
                f"LLM generation failed with {self.llm_settings.provider}: {e}"
            )

            # Fallback to Ollama if we were trying Groq
            if self.llm_settings.provider == "groq":
                logger.warning("Falling back to Ollama due to Groq failure")

                # Track fallback operation
                if METRICS_ENABLED:
                    rag_fallback_operations_total.labels(
                        from_service="groq",
                        to_service="ollama"
                    ).inc()

                try:
                    return self._generate_with_ollama(
                        prompt, temperature, max_tokens, model
                    )
                except Exception as fallback_error:
                    logger.error(f"Ollama fallback also failed: {fallback_error}")
                    raise fallback_error
            else:
                # Already using Ollama, can't fallback further
                raise

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
        if not self.groq_client:
            raise ValueError("Groq client not initialized")

        model_name = model or self.llm_settings.groq_model

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            acquired = self.rate_limiter.acquire(timeout=60)
            if not acquired:
                logger.warning("Rate limiter timeout - falling back to Ollama")
                raise Exception("Rate limiter timeout")

            # Log rate limit status
            stats = self.rate_limiter.get_stats()
            if stats['minute_utilization'] > 0.8 or stats['day_utilization'] > 0.8:
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
                messages=[
                    {"role": "user", "content": prompt}
                ],
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
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                logger.error(f"Groq authentication error: {e}")
            else:
                logger.error(f"Groq API error: {e}")

            raise

    def _generate_with_ollama(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using Ollama (local LLM).

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override

        Returns:
            Generated text

        Raises:
            Exception: If Ollama generation fails
        """
        model_name = model or self.ollama_model

        start = time.time()
        try:
            logger.debug(f"Calling Ollama with model: {model_name}")

            response = self.ollama_client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                },
            )

            generated_text = response["response"]

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="success", model=f"ollama:{model_name}"
                ).inc()

            duration = time.time() - start
            logger.info(
                f"Ollama generation successful: {len(generated_text)} chars in {duration:.2f}s"
            )

            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=f"ollama:{model_name}"
                ).observe(duration)

            return generated_text

        except Exception as e:
            duration = time.time() - start

            if METRICS_ENABLED:
                rag_llm_request_total.labels(
                    status="error", model=f"ollama:{model_name}"
                ).inc()
                rag_llm_latency_seconds.labels(
                    status="error", model=f"ollama:{model_name}"
                ).observe(duration)

            logger.error(f"Ollama generation failed: {e}")
            raise

    async def async_generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text asynchronously using configured LLM provider with automatic fallback.

        Args:
            prompt: The input prompt to generate text from
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default)
            model: Optional override for the model name

        Returns:
            Generated text response

        Raises:
            Exception: If both providers fail
        """
        # Use settings defaults if not provided
        temperature = (
            temperature if temperature is not None
            else self.llm_settings.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None
            else self.llm_settings.max_tokens
        )

        # Try primary provider
        try:
            if self.llm_settings.provider == "groq" and self.async_groq_client:
                logger.debug("Attempting async generation with Groq")
                return await self._generate_with_groq_async(
                    prompt, temperature, max_tokens, model
                )
            else:
                logger.debug("Generating with Ollama (async wrapper)")
                return await self._generate_with_ollama_async(
                    prompt, temperature, max_tokens, model
                )
        except Exception as e:
            # Log the error
            logger.error(
                f"Async LLM generation failed with {self.llm_settings.provider}: {e}"
            )

            # Fallback to Ollama if we were trying Groq
            if self.llm_settings.provider == "groq":
                logger.warning("Falling back to Ollama due to Groq failure")

                # Track fallback operation
                if METRICS_ENABLED:
                    rag_fallback_operations_total.labels(
                        from_service="groq",
                        to_service="ollama"
                    ).inc()

                try:
                    return await self._generate_with_ollama_async(
                        prompt, temperature, max_tokens, model
                    )
                except Exception as fallback_error:
                    logger.error(f"Ollama fallback also failed: {fallback_error}")
                    raise fallback_error
            else:
                # Already using Ollama, can't fallback further
                raise

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
        if not self.async_groq_client:
            raise ValueError("Async Groq client not initialized")

        model_name = model or self.llm_settings.groq_model

        # Wait for rate limiter (blocks if at limit)
        if self.rate_limiter:
            logger.debug("Waiting for rate limiter...")
            # Run synchronous rate limiter in executor to not block
            loop = asyncio.get_event_loop()
            acquired = await loop.run_in_executor(
                None, lambda: self.rate_limiter.acquire(timeout=60)
            )
            if not acquired:
                logger.warning("Rate limiter timeout - falling back to Ollama")
                raise Exception("Rate limiter timeout")

            # Log rate limit status
            stats = self.rate_limiter.get_stats()
            if stats['minute_utilization'] > 0.8 or stats['day_utilization'] > 0.8:
                logger.warning(
                    f"High rate limit usage: "
                    f"{stats['requests_last_minute']}/{stats['requests_per_minute_limit']} req/min, "
                    f"{stats['requests_last_day']}/{stats['requests_per_day_limit']} req/day"
                )

        start = time.time()
        try:
            logger.debug(f"Calling Groq API async with model: {model_name}")

            response: ChatCompletion = await self.async_groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
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
            elif "api_key" in error_msg.lower() or "authentication" in error_msg.lower():
                logger.error(f"Groq authentication error: {e}")
            else:
                logger.error(f"Groq API error: {e}")

            raise

    async def _generate_with_ollama_async(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int,
        model: Optional[str] = None,
    ) -> str:
        """Generate text asynchronously using Ollama (wraps sync client in executor).

        Args:
            prompt: The input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            model: Optional model override

        Returns:
            Generated text

        Raises:
            Exception: If Ollama generation fails
        """
        # Ollama client doesn't have async support yet, so wrap in executor
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self._generate_with_ollama(prompt, temperature, max_tokens, model)
        )


# Initialize global service instance at module level for better performance
_service: OllamaService = OllamaService()
logger.info(
    f"Initialized global LLM service - Provider: {_service.llm_settings.provider}"
)


def get_ollama_service() -> OllamaService:
    """Get the global LLM service instance."""
    return _service


def generate_with_ollama(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Convenience function for generating text with configured LLM provider.

    This function maintains backward compatibility while supporting multiple providers.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override

    Returns:
        Generated text response
    """
    service = get_ollama_service()
    return service.generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


async def async_generate_with_ollama(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Async convenience function for generating text with configured LLM provider.

    Args:
        prompt: The input prompt to generate text from
        temperature: Sampling temperature
        max_tokens: Maximum number of tokens to generate
        model: Optional model override

    Returns:
        Generated text response
    """
    service = get_ollama_service()
    return await service.async_generate(
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        model=model,
    )


__all__ = [
    "OllamaService",
    "get_ollama_service",
    "generate_with_ollama",
    "async_generate_with_ollama",
]
