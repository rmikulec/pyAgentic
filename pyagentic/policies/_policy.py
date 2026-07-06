from typing import Protocol, TypeVar, Any
from pyagentic.policies._events import AppendEvent, CompileEvent, GetEvent, SetEvent

T = TypeVar("T")


class Policy(Protocol[T]):
    """
    Protocol for implementing state policies that intercept state operations.

    Policies allow you to add custom logic when state fields are accessed or modified,
    such as validation, transformation, logging, or side effects. Policies are attached
    to state fields using spec.State(policies=[...]), or to the agent's message
    context via the `__message_policies__` class attribute.

    Methods:
        on_get: Synchronous handler for read operations (runs immediately)
        on_set: Synchronous handler for write operations (runs immediately).
            Also fires when a list-valued field is mutated in place
            (item assignment, deletion, remove, pop, clear).
        on_append: Synchronous handler for items appended to a list-valued
            field or the message context (runs immediately, before insertion)
        on_compile: Async handler run right before each LLM inference on the
            whole list (message context or list-valued field). May transform
            the list; the result is written back. This is where eviction,
            windowing, and compaction live.
        background_get: Async handler for read operations (runs after response)
        background_set: Async handler for write operations (runs after response)
        background_append: Async handler for appends (fire-and-forget)

    Each method receives:
        - event: Context metadata (field name, timestamp, previous value for SET,
            provider/usage/system message for COMPILE)
        - value: The current value being processed

    Each method may:
        - Return a transformed value of the same type `T`
        - Return None to indicate no change
        - Raise an exception to block or veto the operation (for on_append,
            a raise skips inserting the item)

    Policy instances are attached at class-definition time and shared across all
    agent instances and forks — keep them stateless/config-only. Anything
    per-agent must be derived from the event or the list contents themselves.

    Example:
        ```python
        from pyagentic.policies import Policy
        from pyagentic.policies._events import SetEvent, GetEvent

        class ValidationPolicy(Policy[int]):
            \"\"\"Ensure counter is never negative\"\"\"

            def on_set(self, event: SetEvent, value: int) -> int:
                if value < 0:
                    raise ValueError("Counter cannot be negative")
                return value

            def on_get(self, event: GetEvent, value: int) -> int:
                return value  # Pass through unchanged

            async def background_set(self, event: SetEvent, value: int) -> int:
                # Could log to database asynchronously
                print(f"Counter set to {value}")
                return None  # No transformation

            async def background_get(self, event: GetEvent, value: int) -> int:
                return None  # No transformation

        # Use in agent
        class CounterAgent(BaseAgent):
            __system_message__ = "You manage a counter"
            counter: State[int] = spec.State(default=0, policies=[ValidationPolicy()])
        ```
    """

    def on_get(self, event: GetEvent, value: T) -> T | None: ...
    async def background_get(self, event: GetEvent, value: T) -> T | None: ...
    def on_set(self, event: SetEvent, value: T) -> T | None: ...
    async def background_set(self, event: SetEvent, value: T) -> T | None: ...
    def on_append(self, event: AppendEvent, item: Any) -> Any | None: ...
    async def background_append(self, event: AppendEvent, item: Any) -> None: ...
    async def on_compile(self, event: CompileEvent, items: list) -> list | None: ...
