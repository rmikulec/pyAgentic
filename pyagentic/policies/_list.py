"""
Observable list that routes mutations through a state's policy pipeline.

Used for list-valued state fields with attached policies and for the agent's
message context, so `agent.my_list.append(x)` triggers policies the same way
`agent.my_field = x` does.
"""

import asyncio

from pyagentic.policies._events import AppendEvent, SetEvent

# Sentinel returned by the append pipeline when a policy vetoed the item
_VETOED = object()


class PolicyList(list):
    """
    A list subclass bound to an agent state and field name that fires policy
    events on mutation.

    Appends (`append`, `extend`, `insert`, `+=`) run each policy's `on_append`
    synchronously — a policy may transform the item, return None to keep it
    unchanged, or raise to veto the insertion — then fire `background_append`
    asynchronously. Other mutations (`__setitem__`, `__delitem__`, `remove`,
    `pop`, `clear`) run the whole list through the `on_set` pipeline after the
    mutation is applied.
    """

    def __init__(self, iterable=(), *, state=None, name: str = None):
        """
        Initialize the policy list.

        Args:
            iterable: Initial contents (not run through policies).
            state (_AgentState): The owning agent state, used to resolve policies.
            name (str): The state field name (or "messages" for the message context)
                policies are registered under.
        """
        super().__init__(iterable)
        self._state = state
        self._name = name

    def _policies(self) -> list:
        """Resolve the policies attached to this list's field, or an empty list."""
        if self._state is None or self._name is None:
            return []
        return self._state.get_policies(self._name) or []

    def _transform_append(self, item):
        """Run the sync on_append pipeline; returns the final item or _VETOED."""
        policies = self._policies()
        if not policies:
            return item

        value = item
        for policy in policies:
            handler = getattr(policy, "on_append", None)
            if handler is None:
                continue
            event = AppendEvent(name=self._name, value=value)
            try:
                new_value = handler(event, value)
                if new_value is not None:
                    value = new_value
            except Exception as e:
                print(f"[PolicyError] {policy.__class__.__name__}.on_append vetoed append: {e}")
                return _VETOED
        return value

    def _fire_background_append(self, item):
        """Schedule background_append handlers on the running loop, if any."""
        policies = [p for p in self._policies() if getattr(p, "background_append", None)]
        if not policies:
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running event loop (sync context) — skip background handlers
            return

        async def _dispatch():
            for policy in policies:
                event = AppendEvent(name=self._name, value=item)
                try:
                    await policy.background_append(event, item)
                except Exception as e:
                    print(
                        f"[PolicyError] {policy.__class__.__name__}.background_append "
                        f"failed: {e}"
                    )

        loop.create_task(_dispatch())

    def _run_set_pipeline(self, previous: list):
        """Run the on_set pipeline over the full list after an in-place mutation."""
        policies = self._policies()
        if not policies:
            return

        value = list(self)
        event = SetEvent(name=self._name, previous=previous, value=value)
        for policy in policies:
            handler = getattr(policy, "on_set", None)
            if handler is None:
                continue
            try:
                new_value = handler(event, value)
                if new_value is not None:
                    value = new_value
            except Exception as e:
                print(f"[PolicyError] {policy.__class__.__name__}.on_set failed: {e}")

        if value != list(self):
            self._set_contents(value)

    def _set_contents(self, items):
        """Replace contents without triggering any policy pipeline."""
        list.clear(self)
        list.extend(self, items)

    # -- append-family mutations -------------------------------------------------

    def append(self, item):
        value = self._transform_append(item)
        if value is _VETOED:
            return
        super().append(value)
        self._fire_background_append(value)

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def insert(self, index, item):
        value = self._transform_append(item)
        if value is _VETOED:
            return
        super().insert(index, value)
        self._fire_background_append(value)

    def __iadd__(self, other):
        self.extend(other)
        return self

    # -- set-family mutations ----------------------------------------------------

    def __setitem__(self, index, value):
        previous = list(self)
        super().__setitem__(index, value)
        self._run_set_pipeline(previous)

    def __delitem__(self, index):
        previous = list(self)
        super().__delitem__(index)
        self._run_set_pipeline(previous)

    def remove(self, item):
        previous = list(self)
        super().remove(item)
        self._run_set_pipeline(previous)

    def pop(self, index=-1):
        previous = list(self)
        result = super().pop(index)
        self._run_set_pipeline(previous)
        return result

    def clear(self):
        previous = list(self)
        super().clear()
        self._run_set_pipeline(previous)
