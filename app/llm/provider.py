"""
LLM Provider Abstraction Layer

Abstract base class for LLM providers to enable multi-provider support.
Allows switching between Groq, DeepInfra, Cerebras, etc.
"""

from abc import ABC, abstractmethod
from typing import AsyncIterator, TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.parsing import StreamChunk


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs,
    ) -> str:
        """Generate text from a prompt.

        Args:
            prompt: The prompt to generate from
            model: Model name (provider-specific, uses default if None)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text response
        """
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text from a prompt.

        Args:
            prompt: The prompt to generate from
            model: Model name (provider-specific, uses default if None)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Generated text chunks (answer only, thinking filtered out)
        """
        pass

    async def generate_stream_with_thinking(
        self,
        prompt: str,
        model: str = None,
        temperature: float = 0.0,
        max_tokens: int = 1000,
        **kwargs,
    ) -> AsyncIterator["StreamChunk"]:
        """Stream generated text with thinking process separated.

        This method yields typed chunks that differentiate between
        the model's thinking process and the actual answer, allowing
        frontends to display them differently (e.g., collapsible thinking).

        Args:
            prompt: The prompt to generate from
            model: Model name (provider-specific, uses default if None)
            temperature: Sampling temperature (0.0-2.0)
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            StreamChunk objects with type=THINKING or type=ANSWER
        """
        # Default implementation: just yield everything as ANSWER
        # Providers with thinking support (like DeepInfra+Qwen) override this
        from app.core.parsing import StreamChunk, ChunkType

        async for chunk in self.generate_stream(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs,
        ):
            yield StreamChunk(type=ChunkType.ANSWER, content=chunk)

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name for logging/metrics."""
        pass

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model for this provider."""
        pass
