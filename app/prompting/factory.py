"""
Prompt Factory - Builder instance creation

Provides factory functions for creating prompt builder instances.
"""

from .builder import PromptBuilder
from .config import PromptConfig


def create_default_prompt_builder() -> PromptBuilder:
    """Create a pre-configured prompt builder with default settings.

    Returns:
        PromptBuilder instance with default configuration
    """
    return PromptBuilder()
