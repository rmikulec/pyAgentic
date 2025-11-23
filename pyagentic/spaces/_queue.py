"""
Message queue implementation for agent spaces.
"""

import asyncio
from typing import Optional

from ._messages import SpaceMessage


class MessageQueue:
    """
    Async message queue with broadcast support for agent spaces.

    Supports both broadcast (all agents see messages) and directed
    (whisper) messaging patterns.
    """

    def __init__(self, max_size: int = 0):
        """
        Initialize the message queue.

        Args:
            max_size: Maximum queue size (0 = unlimited)
        """
        self._queues: dict[str, asyncio.Queue] = {}
        self._max_size = max_size
        self._lock = asyncio.Lock()
        self._broadcast_enabled = True

    async def register_agent(self, agent_id: str) -> None:
        """
        Register an agent to receive messages.

        Args:
            agent_id: Unique identifier for the agent
        """
        async with self._lock:
            if agent_id not in self._queues:
                self._queues[agent_id] = asyncio.Queue(maxsize=self._max_size)

    async def unregister_agent(self, agent_id: str) -> None:
        """
        Unregister an agent from receiving messages.

        Args:
            agent_id: Unique identifier for the agent
        """
        async with self._lock:
            if agent_id in self._queues:
                del self._queues[agent_id]

    async def push(self, message: SpaceMessage) -> None:
        """
        Push a message to the queue.

        If the message is broadcast, it's delivered to all registered agents.
        If targeted (whisper), it's delivered only to the specific agent.

        Args:
            message: The message to push
        """
        async with self._lock:
            if message.is_broadcast and self._broadcast_enabled:
                # Broadcast to all agents except the sender
                for agent_id, queue in self._queues.items():
                    if agent_id != message.sender:
                        await queue.put(message)
            elif message.target:
                # Directed message (whisper)
                if message.target in self._queues:
                    await self._queues[message.target].put(message)
            else:
                # Default: broadcast to all
                for agent_id, queue in self._queues.items():
                    if agent_id != message.sender:
                        await queue.put(message)

    async def pull(self, agent_id: str, timeout: Optional[float] = None) -> Optional[SpaceMessage]:
        """
        Pull a message from an agent's queue.

        Args:
            agent_id: The agent pulling the message
            timeout: Optional timeout in seconds (None = block forever)

        Returns:
            SpaceMessage or None if timeout occurred
        """
        if agent_id not in self._queues:
            await self.register_agent(agent_id)

        queue = self._queues[agent_id]

        try:
            if timeout is not None:
                return await asyncio.wait_for(queue.get(), timeout=timeout)
            else:
                return await queue.get()
        except asyncio.TimeoutError:
            return None

    def get_queue_size(self, agent_id: str) -> int:
        """Get the current size of an agent's queue."""
        if agent_id not in self._queues:
            return 0
        return self._queues[agent_id].qsize()

    def get_total_queue_sizes(self) -> dict[str, int]:
        """Get queue sizes for all agents."""
        return {agent_id: queue.qsize() for agent_id, queue in self._queues.items()}

    async def clear_agent_queue(self, agent_id: str) -> None:
        """Clear all messages from an agent's queue."""
        if agent_id in self._queues:
            async with self._lock:
                self._queues[agent_id] = asyncio.Queue(maxsize=self._max_size)

    async def clear_all(self) -> None:
        """Clear all messages from all agent queues."""
        async with self._lock:
            for agent_id in list(self._queues.keys()):
                self._queues[agent_id] = asyncio.Queue(maxsize=self._max_size)
