import asyncio
from dataclasses import dataclass, field
from typing import Optional
from pydantic import BaseModel, Field

from pyagentic import BaseAgent, AgentExtension, State, spec, tool
from ._messages import SpaceMessage, ConversationContext
from ._queue import MessageQueue


class SpaceParticipant(BaseModel):
    name: str
    description: str


class SpaceAware(AgentExtension):
    your_name: State[str] = spec.State(default="<insert-name>", access="hidden")
    space_name: State[str] = spec.State(default="<insert-name>", access="hidden")
    participants: State[list[SpaceParticipant]] = spec.State(default_factory=list, access="hidden")

    def set_your_name(self, name: str):
        self.your_name = name

    def set_space_name(self, name: str):
        self.space_name = name

    def add_participant(self, name, description):
        self.participants.append(SpaceParticipant(name=name, description=description))


@dataclass
class AgentSpace:
    name: str
    agents: dict[str, BaseAgent]
    context_window: int = 10
    max_queue_size: int = 0

    # Internal state
    _queue: MessageQueue = field(default=None, init=False, repr=False)
    _context: ConversationContext = field(default=None, init=False, repr=False)
    _tasks: list[asyncio.Task] = field(default_factory=list, init=False, repr=False)
    _running: bool = field(default=False, init=False, repr=False)

    def __post_init__(self):
        """Initialize queue and context after dataclass init."""
        self._queue = MessageQueue(max_size=self.max_queue_size)
        self._context = ConversationContext()

    def initialize(self):
        """Initialize space-aware agents with participant information."""
        for name, agent in self.agents.items():
            if isinstance(agent, SpaceAware):
                agent.set_space_name(self.name)
                agent.set_your_name(name)

                for participant_name, participant_agent in self.agents.items():
                    if participant_name == name:
                        continue

                    agent.add_participant(
                        name=participant_name,
                        description=participant_agent.state.description
                        or participant_agent.__class__.__name__,
                    )
            else:
                raise TypeError(f"All agents must have the `SpaceAware` extension: {name}")

    async def start(self, max_iterations: Optional[int] = None):
        """
        Start all agent workers in the space.

        This initializes the agents with space context and starts worker tasks
        that will pull messages from the queue and process them.

        Args:
            max_iterations: Maximum number of messages each agent should process.
                           None means run indefinitely until stopped.
        """
        if self._running:
            raise RuntimeError("AgentSpace is already running")

        # Initialize agents with space context
        self.initialize()

        self._running = True

        # Register all agents with the message queue
        for agent_name in self.agents.keys():
            await self._queue.register_agent(agent_name)

        # Start a worker task for each agent
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(self._agent_worker(agent, agent_name, max_iterations))
            self._tasks.append(task)

    async def send_message(self, content: str, sender: str = "system"):
        """
        Send a message to the space to initiate or continue the conversation.

        This broadcasts the message to all agents in the space.

        Args:
            content: The message content
            sender: The name of the sender (defaults to "system")
        """
        if not self._running:
            raise RuntimeError("AgentSpace is not running. Call start() first.")

        message = SpaceMessage(content=content, sender=sender, is_broadcast=True)
        self._context.add_message(message)
        await self._queue.push(message)

    async def stop(self, timeout: float = 5.0):
        """
        Gracefully stop all agent workers.

        Args:
            timeout: Maximum time to wait for workers to stop (in seconds)
        """
        if not self._running:
            return

        self._running = False

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()

        # Wait for cancellation with timeout
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._tasks, return_exceptions=True), timeout=timeout
            )
        except asyncio.TimeoutError:
            pass  # Tasks didn't finish in time, but they're cancelled

        self._tasks.clear()

    def show_conversation(
        self, limit: Optional[int] = None, include_timestamps: bool = False
    ) -> str:
        """
        Display the conversation history in a readable format.

        Args:
            limit: Maximum number of messages to show (None = all messages)
            include_timestamps: Whether to include timestamps in the output

        Returns:
            Formatted conversation string
        """
        messages = (
            self._context.messages if limit is None else self._context.get_recent_messages(limit)
        )

        if not messages:
            return "No messages in conversation yet."

        lines = []
        lines.append(f"=== Conversation: {self.name} ===")
        lines.append("")

        for msg in messages:
            if include_timestamps:
                timestamp = msg.timestamp.strftime("%H:%M:%S")
                lines.append(f"[{timestamp}] {msg.sender}: {msg.content}")
            else:
                lines.append(f"{msg.sender}: {msg.content}")
            lines.append("")  # Blank line between messages

        return "\n".join(lines)

    async def _agent_worker(
        self, agent: BaseAgent, agent_name: str, max_iterations: Optional[int] = None
    ):
        """
        Worker loop for a single agent.

        Continuously pulls messages from the queue, formats them with conversation
        context, calls the agent to generate a response, and pushes the response
        back to the queue.

        Args:
            agent: The agent instance to run
            agent_name: The agent's name in the space
            max_iterations: Maximum number of iterations (None = infinite)
        """
        iteration = 0

        try:
            while self._running and (max_iterations is None or iteration < max_iterations):
                # Pull message from this agent's queue with timeout
                message = await self._queue.pull(agent_name, timeout=1.0)

                if message is None:
                    # Timeout - check if we should continue
                    continue

                # Skip if this agent was the last to speak (avoid back-to-back responses)
                recent_messages = self._context.get_recent_messages(limit=3)
                if recent_messages and recent_messages[-1].sender == agent_name:
                    continue

                # Format input with conversation context
                formatted_input = f'{message.sender} said: "{message.content}"\n\nRespond only if you have something valuable to contribute to the conversation.'

                # Call the agent (it doesn't know it's in a space!)
                response = await agent(user_input=formatted_input)

                # Extract the agent's response text
                response_text = response.final_output

                if response_text == "":
                    continue

                if isinstance(response_text, str):
                    final_text = response_text
                else:
                    # Handle structured output
                    final_text = str(response_text)

                # Broadcast response to other agents
                response_message = SpaceMessage(
                    content=final_text, sender=agent_name, is_broadcast=True
                )
                self._context.add_message(response_message)
                await self._queue.push(response_message)

                iteration += 1

        except asyncio.CancelledError:
            # Worker was cancelled during shutdown
            pass
        except Exception as e:
            # Log error but don't crash the whole space
            print(f"Error in agent worker {agent_name}: {e}")
            # TODO: Integrate with tracer for proper error logging
