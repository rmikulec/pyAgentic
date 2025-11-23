"""
Message types for agent space communication.
"""

from datetime import datetime
from typing import Any, Optional
from pydantic import BaseModel, Field
from uuid import uuid4


class SpaceMessage(BaseModel):
    """
    Message wrapper for agent space communication.

    Each message contains content, metadata about the sender, timestamp,
    and optional targeting information for directed communication.
    """

    id: str = Field(default_factory=lambda: str(uuid4()))
    content: str = Field(..., description="The message content")
    sender: Optional[str] = Field(None, description="ID or name of the sending agent")
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional message metadata"
    )
    target: Optional[str] = Field(
        None, description="Target agent ID for whisper/directed messages"
    )
    is_broadcast: bool = Field(
        default=True, description="Whether this message is broadcast to all agents"
    )

    def to_user_message(self) -> str:
        """Convert to a formatted user message string for LLM context."""
        sender_prefix = f"[{self.sender}]" if self.sender else "[Unknown]"
        return f"{sender_prefix}: {self.content}"

    def is_for_agent(self, agent_id: str) -> bool:
        """
        Check if this message is intended for a specific agent.

        Args:
            agent_id: The agent ID to check against

        Returns:
            bool: True if the message is for this agent (broadcast or targeted)
        """
        if self.is_broadcast:
            return True
        return self.target == agent_id


class ConversationContext(BaseModel):
    """
    Shared conversation context for space agents.

    Maintains the history of messages in the space conversation.
    """

    messages: list[SpaceMessage] = Field(default_factory=list)
    max_history: int = Field(default=100, description="Maximum number of messages to retain")

    def add_message(self, message: SpaceMessage) -> None:
        """Add a message to the conversation history."""
        self.messages.append(message)

        # Trim history if needed
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]

    def get_recent_messages(self, limit: int = 10) -> list[SpaceMessage]:
        """Get the most recent N messages."""
        return self.messages[-limit:]

    def get_messages_from_agent(self, agent_id: str) -> list[SpaceMessage]:
        """Get all messages from a specific agent."""
        return [msg for msg in self.messages if msg.sender == agent_id]
