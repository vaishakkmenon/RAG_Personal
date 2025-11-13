"""
LLM service for Personal RAG system.

Handles Ollama client interactions and text generation.
"""

import logging
import time
from typing import Optional

import ollama

from ..settings import settings

logger = logging.getLogger(__name__)

# Optional metrics import
try:
    from ..metrics import rag_llm_request_total, rag_llm_latency_seconds
    METRICS_ENABLED = True
except ImportError:
    METRICS_ENABLED = False
    logger.debug("Metrics not available for LLM service")


class OllamaService:
    """Service for interacting with Ollama LLM."""

    def __init__(
        self,
        host: Optional[str] = None,
        model: Optional[str] = None,
        timeout: Optional[int] = None,
        num_ctx: Optional[int] = None,
    ):
        """Initialize Ollama service.

        Args:
            host: Ollama API host URL (defaults to settings)
            model: Default model name (defaults to settings)
            timeout: Request timeout in seconds (defaults to settings)
            num_ctx: Context window size (defaults to settings)
        """
        self.host = host or settings.ollama_host
        self.model = model or settings.ollama_model
        self.timeout = timeout or settings.ollama_timeout
        self.num_ctx = num_ctx or settings.num_ctx

        self.client = ollama.Client(host=self.host, timeout=self.timeout)
        logger.info(f"Initialized Ollama service: {self.host}, model={self.model}")

    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        model: Optional[str] = None,
    ) -> str:
        """Generate text using Ollama with configurable parameters.

        Args:
            prompt: The input prompt to generate text from
            temperature: Sampling temperature (None uses default from settings)
            max_tokens: Maximum number of tokens to generate (None uses default from settings)
            model: Optional override for the Ollama model name

        Returns:
            Generated text response

        Raises:
            Exception: If generation fails
        """
        temperature = (
            temperature if temperature is not None else settings.retrieval.temperature
        )
        max_tokens = (
            max_tokens if max_tokens is not None else settings.retrieval.max_tokens
        )
        model_name = model or self.model

        start = time.time()
        try:
            response = self.client.generate(
                model=model_name,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": self.num_ctx,
                },
            )

            if METRICS_ENABLED:
                rag_llm_request_total.labels(status="success", model=model_name).inc()

            return response["response"]

        except Exception as e:
            if METRICS_ENABLED:
                rag_llm_request_total.labels(status="error", model=model_name).inc()
            logger.error(f"Ollama generation failed: {e}")
            raise

        finally:
            duration = time.time() - start
            if METRICS_ENABLED:
                rag_llm_latency_seconds.labels(
                    status="success", model=model_name
                ).observe(duration)
            logger.debug(f"LLM generation took {duration:.2f}s")


# Initialize global service instance at module level for better performance
_service: OllamaService = OllamaService()
logger.info(f"Initialized Ollama service: {_service.host}, model={_service.model}")


def get_ollama_service() -> OllamaService:
    """Get the global Ollama service instance."""
    return _service


def generate_with_ollama(
    prompt: str,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    """Convenience function for generating text with Ollama.

    This is a backward-compatible wrapper around the OllamaService.

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


__all__ = ["OllamaService", "get_ollama_service", "generate_with_ollama"]
