"""
LLM Package - Multi-provider LLM abstraction layer.

Provides a unified interface for different LLM providers (Groq, DeepInfra, etc.)
to enable easy switching between backends.

Usage:
    from app.llm import get_provider, resolve_model

    provider = get_provider()
    response = await provider.generate("Your prompt here")

    # Model shorthand resolution
    provider_name, model_name = resolve_model("qwen")  # -> ("deepinfra", "Qwen/Qwen3-32B")
"""

from app.llm.provider import LLMProvider
from app.llm.factory import (
    get_provider,
    get_llm_provider,
    reset_provider,
    resolve_model,
    MODEL_ALIASES,
)

__all__ = [
    "LLMProvider",
    "get_provider",
    "get_llm_provider",
    "reset_provider",
    "resolve_model",
    "MODEL_ALIASES",
]
