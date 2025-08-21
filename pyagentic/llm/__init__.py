from enum import Enum

from pyagentic.llm._openai import OpenAIBackend
from pyagentic.llm._anthropic import AnthropicBackend


class LLMBackends(Enum):
    OPENAI = OpenAIBackend
    ANTHROPIC = AnthropicBackend
