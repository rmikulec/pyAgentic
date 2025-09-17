from enum import Enum

from pyagentic.llm._openai import OpenAIProvider
from pyagentic.llm._anthropic import AnthropicProvider
from pyagentic.llm._mock import _MockProvider


class LLMProviders(Enum):
    OPENAI = OpenAIProvider
    ANTHROPIC = AnthropicProvider
    _MOCK = _MockProvider
