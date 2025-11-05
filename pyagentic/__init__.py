"""
PyAgentic: A declarative framework for building LLM-powered agents.

PyAgentic provides a clean, type-safe way to build agentic systems with:
  - Tool calling: Let LLMs invoke Python functions
  - Stateful agents: Maintain conversation and application state
  - Multi-agent systems: Link agents together for complex workflows
  - Provider flexibility: Support for OpenAI, Anthropic, and custom providers
  - Observability: Built-in tracing with BasicTracer and LangfuseTracer

Quick Start:
    ```python
    from pyagentic import BaseAgent, tool

    class MyAgent(BaseAgent):
        __system_message__ = "You are a helpful assistant"

        @tool("Calculate the sum of two numbers")
        def add(self, a: int, b: int) -> str:
            return str(a + b)

    # Create and run the agent
    agent = MyAgent(model="openai::gpt-4o", api_key="your-key")
    response = await agent.run("What is 5 + 3?")
    print(response.final_output)  # The LLM will call the add tool
    ```

Main Components:
    - BaseAgent: Base class for defining agents
    - tool: Decorator for marking methods as LLM-callable tools
    - State: Type annotation for persistent state fields
    - Link: Type annotation for linking agents together
    - spec: Configuration factory for state, params, and agent links
    - ref: Dynamic state reference system for tool parameters
    - AgentExtension: Base class for creating reusable agent mixins
"""

from pyagentic._base._spec import spec
from pyagentic._base._agent._agent import BaseAgent, AgentExtension
from pyagentic._base._agent._agent_linking import Link
from pyagentic._base._tool import tool

from pyagentic._base._state import State
from pyagentic._base._ref import ref

__all__ = ["BaseAgent", "AgentExtension", "tool", "spec", "State", "Link", "ref"]
