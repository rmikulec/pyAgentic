from enum import Enum

from pyagentic.llm.openai import OpenAIBackend


class LLMBackends(Enum):
    OPENAI = OpenAIBackend
