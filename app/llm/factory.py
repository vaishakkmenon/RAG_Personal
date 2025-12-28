"""
LLM Provider Factory

Factory function to get the configured LLM provider based on settings.
Supports model shorthand names and automatic provider detection.
"""

import logging
from typing import Optional, Tuple

from app.llm.provider import LLMProvider
from app.settings import settings

logger = logging.getLogger(__name__)


# Model shorthand mappings
MODEL_ALIASES = {
    # Shorthand -> (provider, full_model_name)
    "groq": ("groq", None),  # None means use default from settings
    "llama": ("groq", None),
    "deepinfra": ("deepinfra", None),
    "qwen": ("deepinfra", "Qwen/Qwen3-32B"),
    "qwen3": ("deepinfra", "Qwen/Qwen3-32B"),
    "qwen3-32b": ("deepinfra", "Qwen/Qwen3-32B"),
}


def resolve_model(model_name: Optional[str]) -> Tuple[str, Optional[str]]:
    """Resolve a model name/shorthand to provider and full model name.

    Args:
        model_name: Model shorthand (e.g., "groq", "qwen") or full model name

    Returns:
        Tuple of (provider_name, model_name)
        - provider_name: "groq" or "deepinfra"
        - model_name: Full model name or None (use provider default)

    Examples:
        >>> resolve_model("groq")
        ("groq", None)
        >>> resolve_model("qwen")
        ("deepinfra", "Qwen/Qwen3-32B")
        >>> resolve_model("Qwen/Qwen3-32B")
        ("deepinfra", "Qwen/Qwen3-32B")
    """
    if not model_name:
        # Use default provider
        return settings.llm.provider, None

    model_lower = model_name.lower()

    # Check aliases first
    if model_lower in MODEL_ALIASES:
        return MODEL_ALIASES[model_lower]

    # Try to detect provider from full model name
    if "qwen" in model_lower:
        return "deepinfra", model_name
    elif "llama" in model_lower or "mixtral" in model_lower:
        return "groq", model_name
    else:
        # Default to configured provider with the given model name
        return settings.llm.provider, model_name


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Get an LLM provider instance.

    Args:
        provider_name: Override provider (uses settings.llm.provider if None)

    Returns:
        Configured LLMProvider instance

    Raises:
        ValueError: If provider is unknown
    """
    # Import here to avoid circular imports
    from app.llm.adapters.groq_adapter import GroqProvider
    from app.llm.adapters.deepinfra_adapter import DeepInfraProvider

    provider = (provider_name or settings.llm.provider).lower()

    if provider == "groq":
        logger.info(f"Initializing Groq provider with model: {settings.llm.groq_model}")
        return GroqProvider()
    elif provider == "deepinfra":
        logger.info(
            f"Initializing DeepInfra provider with model: {settings.llm.deepinfra_model}"
        )
        return DeepInfraProvider()
    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            f"Supported providers: 'groq', 'deepinfra'"
        )


# Singleton instance (lazy-loaded)
_provider_instance: Optional[LLMProvider] = None


def get_provider() -> LLMProvider:
    """Get the singleton LLM provider instance.

    Uses lazy initialization to avoid loading providers at import time.

    Returns:
        The configured LLMProvider instance
    """
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = get_llm_provider()
    return _provider_instance


def reset_provider():
    """Reset the singleton provider (for testing or reconfiguration)."""
    global _provider_instance
    _provider_instance = None
