from enum import Enum

from pyagentic.llm._openai import OpenAIProvider
from pyagentic.llm._anthropic import AnthropicProvider


class LLMProviders(Enum):
    OPENAI = OpenAIProvider
    ANTHROPIC = AnthropicProvider
