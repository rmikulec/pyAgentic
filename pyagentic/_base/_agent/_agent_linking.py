from typing import Any, TypeVar, Generic
from dataclasses import dataclass

from pydantic import BaseModel

from pyagentic._base._info import AgentInfo
from pyagentic._base._agent._agent import BaseAgent

T = TypeVar("T")


class Link(Generic[T]):
    """
    Type annotation for linking other agents as callable tools.

    Linked agents appear as tools to the parent agent, allowing complex multi-agent
    workflows. When the LLM calls a linked agent, that agent runs its full execution
    cycle and returns results to the parent.

    Args:
        T: An agent class (subclass of BaseAgent) to link

    Example:
        ```python
        class ResearchAgent(BaseAgent):
            __system_message__ = "You research topics deeply"
            __description__ = "Research agent for gathering detailed information"

        class OrchestratorAgent(BaseAgent):
            __system_message__ = "You coordinate research tasks"

            # Link research agent as a tool
            researcher: Link[ResearchAgent]

            # Conditional linking (agent only available when condition is true)
            expert: Link[ExpertAgent] = spec.AgentLink(
                condition=lambda self: self.state.needs_expert
            )
        ```
    """

    def __class_getitem__(cls, item):
        """
        Creates a generic Link type for a given agent class.

        Args:
            item: The agent class to link

        Returns:
            type: Special marker type that the metaclass can detect and process
        """
        # Return a special marker type that metaclass can detect
        return type(
            f"Link[{item.__name__}]",
            (),
            {"__origin__": Link, "__args__": (item,), "__linked_agent__": item},
        )


@dataclass
class _LinkedAgentDefinition:
    """
    Internal definition for linked agent configuration.
    """
    agent: BaseAgent
    info: AgentInfo = None
