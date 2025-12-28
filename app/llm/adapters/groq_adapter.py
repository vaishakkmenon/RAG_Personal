"""
Groq LLM Provider Adapter

Wraps the Groq API for use with the LLM provider abstraction.

Note: Groq models (Llama) do not support reasoning_effort parameter.
The parameter is accepted for interface compatibility but ignored.
"""

import logging
from typing import AsyncIterator

from groq import AsyncGroq

from app.core.parsing import ReasoningEffort
from app.llm.provider import LLMProvider
from app.settings import settings

logger = logging.getLogger(__name__)


class GroqProvider(LLMProvider):
    """Groq API provider implementation."""

    def __init__(self, api_key: str = None, model: str = None):
        """Initialize Groq provider.

        Args:
            api_key: Groq API key (uses settings if None)
            model: Default model to use (uses settings if None)
        """
        self._api_key = api_key or settings.llm.groq_api_key
        self._model = model or settings.llm.groq_model
        self._client = AsyncGroq(api_key=self._api_key)

    @property
    def provider_name(self) -> str:
        return "groq"

    @property
    def default_model(self) -> str:
        return self._model

    async def generate(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs,
    ) -> str:
        """Generate text from a prompt using Groq.

        Args:
            prompt: The prompt to generate from
            model: Model name (uses default if None)
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens (uses settings if None)
            reasoning_effort: Ignored - Groq/Llama doesn't support reasoning modes

        Returns:
            Generated text response
        """
        # Note: reasoning_effort is ignored for Groq - Llama models don't support it
        target_model = model or self._model
        temp = temperature if temperature is not None else settings.llm.temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens

        try:
            response = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=target_model,
                temperature=temp,
                max_tokens=tokens,
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Stream generated text from Groq.

        Args:
            prompt: The prompt to generate from
            model: Model name (uses default if None)
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens (uses settings if None)
            reasoning_effort: Ignored - Groq/Llama doesn't support reasoning modes

        Yields:
            Generated text chunks
        """
        # Note: reasoning_effort is ignored for Groq - Llama models don't support it
        target_model = model or self._model
        temp = temperature if temperature is not None else settings.llm.temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens

        try:
            stream = await self._client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=target_model,
                temperature=temp,
                max_tokens=tokens,
                stream=True,
            )

            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            logger.error(f"Groq streaming error: {e}")
            raise
