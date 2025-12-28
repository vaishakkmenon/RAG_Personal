"""
DeepInfra LLM Provider Adapter

Wraps the DeepInfra API for use with the LLM provider abstraction.
Default model: Qwen/Qwen3-32B-Instruct
Selected Qwen 3 32B because input cost ($0.08/1M) matches the 14B model,
offering superior reasoning for RAG at effectively the same price.

Reasoning Control:
- Uses DeepInfra's `reasoning_effort` API parameter (not template-level flags)
- When reasoning_effort="none": No <think> blocks, fastest response
- When reasoning_effort="low/medium/high": Model produces <think>...</think> blocks
- The <think> blocks are parsed and can be streamed separately for frontend display
"""

import json
import logging
from typing import AsyncIterator

import aiohttp

from app.core.parsing import (
    parse_llm_response,
    StreamChunk,
    ChunkType,
    ReasoningEffort,
)
from app.llm.provider import LLMProvider
from app.settings import settings

logger = logging.getLogger(__name__)


class DeepInfraProvider(LLMProvider):
    """DeepInfra API provider implementation."""

    BASE_URL = "https://api.deepinfra.com/v1/openai/chat/completions"

    def __init__(self, api_key: str = None, model: str = None):
        """Initialize DeepInfra provider.

        Args:
            api_key: DeepInfra API key (uses settings if None)
            model: Default model to use (uses settings if None)
        """
        self._api_key = api_key or settings.llm.deepinfra_api_key
        self._model = model or settings.llm.deepinfra_model

        if not self._api_key:
            logger.warning("DeepInfra API key not set - provider will fail on requests")

    @property
    def provider_name(self) -> str:
        return "deepinfra"

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
        """Generate text from a prompt using DeepInfra.

        Args:
            prompt: The prompt to generate from
            model: Model name (uses default if None)
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens (uses settings if None)
            reasoning_effort: Controls reasoning depth via DeepInfra's API.
                NONE (default): No reasoning, fastest response
                LOW/MEDIUM/HIGH: Increasing reasoning with <think> blocks

        Returns:
            Generated text response
        """
        target_model = model or self._model
        temp = temperature if temperature is not None else settings.llm.temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens,
            "reasoning_effort": reasoning_effort.value,  # Use enum value directly
        }

        logger.info(
            f"DeepInfra request: model={target_model}, reasoning_effort={reasoning_effort.value}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=60),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"DeepInfra API error {resp.status}: {error_text}"
                        )

                    data = await resp.json()
                    raw_content = data["choices"][0]["message"]["content"] or ""

                    # Parse thinking blocks from Qwen responses
                    parsed = parse_llm_response(raw_content, log_thinking=True)

                    if parsed.had_thinking_block:
                        logger.info(
                            f"[DeepInfra] Parsed thinking block "
                            f"({len(parsed.thinking or '')} chars) from response"
                        )

                    return parsed.answer
        except aiohttp.ClientError as e:
            logger.error(f"DeepInfra connection error: {e}")
            raise
        except Exception as e:
            logger.error(f"DeepInfra generation error: {e}")
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
        """Stream generated text from DeepInfra.

        Handles Qwen's <think>...</think> blocks by buffering until the thinking
        section ends, then streaming the actual answer.

        Args:
            prompt: The prompt to generate from
            model: Model name (uses default if None)
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens (uses settings if None)
            reasoning_effort: Controls reasoning depth via DeepInfra's API.
                NONE (default): No reasoning, no <think> blocks, fastest
                LOW/MEDIUM/HIGH: May produce <think> blocks (filtered here)

        Yields:
            Generated text chunks (thinking blocks are filtered out)
        """
        target_model = model or self._model
        temp = temperature if temperature is not None else settings.llm.temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
            "reasoning_effort": reasoning_effort.value,  # Use enum value directly
        }

        logger.info(
            f"DeepInfra stream: model={target_model}, reasoning_effort={reasoning_effort.value}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"DeepInfra API error {resp.status}: {error_text}"
                        )

                    # State machine for handling <think> blocks in streaming
                    in_thinking_block = False
                    buffer = ""
                    thinking_logged = False

                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("choices") and data["choices"][0].get(
                                    "delta"
                                ):
                                    content = data["choices"][0]["delta"].get(
                                        "content", ""
                                    )
                                    if not content:
                                        continue

                                    buffer += content

                                    # Check for start of thinking block
                                    if (
                                        "<think>" in buffer.lower()
                                        and not in_thinking_block
                                    ):
                                        in_thinking_block = True
                                        # Yield any content before <think>
                                        think_start = buffer.lower().find("<think>")
                                        if think_start > 0:
                                            yield buffer[:think_start]
                                        buffer = buffer[think_start:]

                                    # Check for end of thinking block
                                    if (
                                        in_thinking_block
                                        and "</think>" in buffer.lower()
                                    ):
                                        in_thinking_block = False
                                        think_end = buffer.lower().find(
                                            "</think>"
                                        ) + len("</think>")

                                        if not thinking_logged:
                                            logger.debug(
                                                f"[DeepInfra Stream] Filtered thinking block "
                                                f"({think_end} chars)"
                                            )
                                            thinking_logged = True

                                        # Keep only content after </think>
                                        buffer = buffer[think_end:].lstrip()
                                        if buffer:
                                            yield buffer
                                            buffer = ""

                                    # If not in thinking block, yield accumulated buffer
                                    elif not in_thinking_block and buffer:
                                        yield buffer
                                        buffer = ""

                            except json.JSONDecodeError:
                                continue

                    # Yield any remaining buffer (shouldn't happen normally)
                    if buffer and not in_thinking_block:
                        yield buffer

        except aiohttp.ClientError as e:
            logger.error(f"DeepInfra streaming error: {e}")
            raise
        except Exception as e:
            logger.error(f"DeepInfra streaming error: {e}")
            raise

    async def generate_stream_with_thinking(
        self,
        prompt: str,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        reasoning_effort: ReasoningEffort = ReasoningEffort.NONE,
        **kwargs,
    ) -> AsyncIterator[StreamChunk]:
        """Stream generated text with thinking process as typed chunks.

        This method yields StreamChunk objects that differentiate between
        thinking process and answer content, allowing frontends to display
        them differently (e.g., collapsible thinking section).

        Args:
            prompt: The prompt to generate from
            model: Model name (uses default if None)
            temperature: Sampling temperature (uses settings if None)
            max_tokens: Maximum tokens (uses settings if None)
            reasoning_effort: Controls reasoning depth via DeepInfra's API.
                NONE: No <think> blocks produced
                LOW/MEDIUM/HIGH: Model produces <think> blocks, parsed here

        Yields:
            StreamChunk objects with type=THINKING or type=ANSWER
        """
        target_model = model or self._model
        temp = temperature if temperature is not None else settings.llm.temperature
        tokens = max_tokens if max_tokens is not None else settings.llm.max_tokens

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }

        payload = {
            "model": target_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temp,
            "max_tokens": tokens,
            "stream": True,
            "reasoning_effort": reasoning_effort.value,  # Use enum value directly
        }

        logger.info(
            f"DeepInfra thinking stream: model={target_model}, reasoning_effort={reasoning_effort.value}"
        )

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.BASE_URL,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120),
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        raise Exception(
                            f"DeepInfra API error {resp.status}: {error_text}"
                        )

                    # State machine for typed streaming
                    in_thinking_block = False
                    buffer = ""

                    async for line in resp.content:
                        line = line.decode("utf-8").strip()
                        if not line or line == "data: [DONE]":
                            continue
                        if line.startswith("data: "):
                            try:
                                data = json.loads(line[6:])
                                if data.get("choices") and data["choices"][0].get(
                                    "delta"
                                ):
                                    content = data["choices"][0]["delta"].get(
                                        "content", ""
                                    )
                                    if not content:
                                        continue

                                    buffer += content

                                    # Check for start of thinking block
                                    if (
                                        "<think>" in buffer.lower()
                                        and not in_thinking_block
                                    ):
                                        in_thinking_block = True
                                        # Yield any content before <think> as ANSWER
                                        think_start = buffer.lower().find("<think>")
                                        if think_start > 0:
                                            yield StreamChunk(
                                                type=ChunkType.ANSWER,
                                                content=buffer[:think_start],
                                            )
                                        # Remove everything up to and including <think>
                                        buffer = buffer[think_start + len("<think>") :]

                                    # Check for end of thinking block
                                    if (
                                        in_thinking_block
                                        and "</think>" in buffer.lower()
                                    ):
                                        in_thinking_block = False
                                        think_end = buffer.lower().find("</think>")

                                        # Yield thinking content
                                        thinking_content = buffer[:think_end].strip()
                                        if thinking_content:
                                            yield StreamChunk(
                                                type=ChunkType.THINKING,
                                                content=thinking_content,
                                            )

                                        # Keep only content after </think>
                                        buffer = buffer[
                                            think_end + len("</think>") :
                                        ].lstrip()
                                        if buffer:
                                            yield StreamChunk(
                                                type=ChunkType.ANSWER, content=buffer
                                            )
                                            buffer = ""

                                    # If in thinking block, accumulate (will yield when block closes)
                                    elif in_thinking_block:
                                        # Yield thinking tokens as they come for real-time display
                                        if len(buffer) > 50:  # Batch for efficiency
                                            yield StreamChunk(
                                                type=ChunkType.THINKING, content=buffer
                                            )
                                            buffer = ""

                                    # If not in thinking block, yield as answer
                                    elif not in_thinking_block and buffer:
                                        yield StreamChunk(
                                            type=ChunkType.ANSWER, content=buffer
                                        )
                                        buffer = ""

                            except json.JSONDecodeError:
                                continue

                    # Yield any remaining buffer
                    if buffer:
                        chunk_type = (
                            ChunkType.THINKING
                            if in_thinking_block
                            else ChunkType.ANSWER
                        )
                        yield StreamChunk(type=chunk_type, content=buffer)

        except aiohttp.ClientError as e:
            logger.error(f"DeepInfra streaming error: {e}")
            raise
        except Exception as e:
            logger.error(f"DeepInfra streaming error: {e}")
            raise
