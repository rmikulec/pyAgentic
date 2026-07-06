"""
Prewritten message-context policies for managing agent context growth.

Attach these to an agent via the `__message_policies__` class attribute:

    class MyAgent(BaseAgent):
        __system_message__ = "..."
        __message_policies__ = [
            ToolOutputClipPolicy(max_chars=8000),
            ToolEvictionPolicy(keep_last_n=5),
            CompactionPolicy(max_input_tokens=80_000),
        ]

All policies here are stateless (config-only): policy instances are shared
across agent instances and forks, so anything per-agent is derived from the
message list itself.
"""

from pyagentic.models.llm import (
    AgentResultMessage,
    CompactionSummaryMessage,
    Message,
    ToolCallMessage,
    ToolResultMessage,
    UserMessage,
)
from pyagentic.policies._events import AppendEvent, CompileEvent
from pyagentic.policies._policy import Policy

DEFAULT_COMPACTION_PROMPT = (
    "You are a conversation summarizer. Summarize the following agent conversation "
    "transcript into a concise brief that preserves: key facts and figures, user "
    "goals and preferences, decisions made, important tool results, and any open "
    "tasks or unanswered questions. Write it as context notes for the agent to "
    "continue the conversation, not as prose for the user."
)


class ToolOutputClipPolicy(Policy):
    """
    Clips oversized tool results before they enter the message context.

    Runs at append time, so a huge tool output never occupies context. The raw,
    unclipped message is still recorded in the agent's raw history.
    """

    def __init__(self, max_chars: int = 8000, suffix: str = "\n…[output clipped]"):
        """
        Args:
            max_chars (int): Maximum content length for a tool result message.
            suffix (str): Marker appended to clipped content so the model knows
                it is seeing a truncated output.
        """
        self.max_chars = max_chars
        self.suffix = suffix

    def on_append(self, event: AppendEvent, item: Message) -> Message | None:
        if (
            isinstance(item, ToolResultMessage)
            and item.content
            and len(item.content) > self.max_chars
        ):
            return item.model_copy(update={"content": item.content[: self.max_chars] + self.suffix})
        return None


class ToolEvictionPolicy(Policy):
    """
    Evicts old tool results from the context, keeping only the most recent N.

    Evicted results have their content replaced with a stub — the message and its
    tool_call_id survive, since providers reject histories with orphaned
    call/result pairs. Idempotent: already-stubbed results are left alone.
    """

    def __init__(
        self,
        keep_last_n: int = 5,
        stub: str = "[tool result evicted to save context]",
        include_agent_results: bool = True,
    ):
        """
        Args:
            keep_last_n (int): Number of most-recent tool results to keep intact.
            stub (str): Replacement content for evicted results.
            include_agent_results (bool): Whether linked-agent results are also
                subject to eviction.
        """
        self.keep_last_n = keep_last_n
        self.stub = stub
        self.include_agent_results = include_agent_results

    def _matches(self, message: Message) -> bool:
        """Whether a message is a tool result this policy manages."""
        if not isinstance(message, ToolResultMessage):
            return False
        if not self.include_agent_results and isinstance(message, AgentResultMessage):
            return False
        return True

    async def on_compile(self, event: CompileEvent, items: list) -> list | None:
        results = [m for m in items if self._matches(m)]
        to_evict = results[: -self.keep_last_n] if self.keep_last_n > 0 else results
        evict_ids = {id(m) for m in to_evict}

        changed = False
        out = []
        for message in items:
            if id(message) in evict_ids and message.content != self.stub:
                out.append(message.model_copy(update={"content": self.stub}))
                changed = True
            else:
                out.append(message)
        return out if changed else None


class SlidingWindowPolicy(Policy):
    """
    Bounds the context to the most recent `max_messages` messages, dropping from
    the front. The cut is advanced past leading tool results so no result
    survives without the tool call that produced it.
    """

    def __init__(self, max_messages: int = 50):
        """
        Args:
            max_messages (int): Maximum number of messages to keep in context.
        """
        self.max_messages = max_messages

    async def on_compile(self, event: CompileEvent, items: list) -> list | None:
        if len(items) <= self.max_messages:
            return None
        cut = len(items) - self.max_messages
        # Never let the window start with results whose calls were dropped
        while cut < len(items) and isinstance(items[cut], ToolResultMessage):
            cut += 1
        return items[cut:]


class CompactionPolicy(Policy):
    """
    Summarizes older history into a single CompactionSummaryMessage when the
    context grows past a token threshold.

    The trigger is the previous inference's reported input tokens (no tokenizer
    dependency); when usage is unavailable, a chars/4 estimate is used. Because
    the compiled context is written back, compaction fires once per threshold
    crossing rather than re-summarizing every turn.
    """

    def __init__(
        self,
        max_input_tokens: int = 100_000,
        keep_recent: int = 10,
        summary_prompt: str = DEFAULT_COMPACTION_PROMPT,
    ):
        """
        Args:
            max_input_tokens (int): Input-token threshold that triggers compaction.
            keep_recent (int): Number of most-recent messages kept verbatim.
            summary_prompt (str): System prompt for the summarization call.
        """
        self.max_input_tokens = max_input_tokens
        self.keep_recent = keep_recent
        self.summary_prompt = summary_prompt

    def _should_compact(self, event: CompileEvent, items: list) -> bool:
        """Whether the context has crossed the token threshold."""
        if event.last_usage is not None:
            return event.last_usage.input_tokens > self.max_input_tokens
        estimate = sum(len(m.content or "") for m in items) // 4
        return estimate > self.max_input_tokens

    def _render_transcript(self, items: list) -> str:
        """Render messages into a plain-text transcript for the summarizer."""
        lines = []
        for message in items:
            if isinstance(message, ToolCallMessage):
                lines.append(f"[tool call] {message.name}({message.arguments})")
            elif isinstance(message, ToolResultMessage):
                lines.append(f"[tool result: {message.name}] {message.content}")
            else:
                lines.append(f"{message.role}: {message.content or ''}")
        return "\n".join(lines)

    async def on_compile(self, event: CompileEvent, items: list) -> list | None:
        if len(items) <= self.keep_recent or not self._should_compact(event, items):
            return None

        cut = len(items) - self.keep_recent
        # Never split a tool call from its result across the compaction boundary
        while cut < len(items) and isinstance(items[cut], ToolResultMessage):
            cut += 1
        head, tail = items[:cut], items[cut:]
        if not head:
            return None

        # Imported lazily to avoid a circular import at package load time
        from pyagentic._base._agent._agent_state import _AgentState

        temp_state = _AgentState(instructions=self.summary_prompt)
        temp_state.add_message(UserMessage(content=self._render_transcript(head)))
        response = await event.provider.generate(state=temp_state)
        summary = response.text or ""

        return [
            CompactionSummaryMessage(
                content=f"[Summary of {len(head)} earlier messages]\n{summary}",
                compacted_count=len(head),
            )
        ] + tail
