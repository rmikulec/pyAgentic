"""
Agent Spaces - Multi-agent conversations with shared message queues.

Enables multiple agents to communicate asynchronously in a shared space,
creating dynamic group conversations.
"""

from ._base import AgentSpace, SpaceAware, SpaceParticipant
from ._messages import SpaceMessage, ConversationContext
from ._queue import MessageQueue

__all__ = [
    "AgentSpace",
    "SpaceAware",
    "SpaceParticipant",
    "SpaceMessage",
    "ConversationContext",
    "MessageQueue",
]
