"""
LLM provider module containing implementations for various language model providers.

This module provides a unified interface for different LLM providers including OpenAI,
Anthropic, and mock providers for testing purposes.
"""

from enum import Enum

from pyagentic.llm._openai import OpenAIProvider
from pyagentic.llm._anthropic import AnthropicProvider
from pyagentic.llm._mock import _MockProvider


__all__ = ["OpenAIProvider", "AnthropicProvider"]


class LLMProviders(Enum):
    """
    Enumeration of available LLM provider implementations.

    Provides easy access to different language model providers that can be used
    with agents for text generation and tool calling.
    """

    OPENAI = OpenAIProvider
    ANTHROPIC = AnthropicProvider
    _MOCK = _MockProvider
