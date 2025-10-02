from __future__ import annotations
from typing import Any, Dict, Type, Optional
import functools
import inspect
from pydantic import BaseModel, Field, computed_field, PrivateAttr

# If you already have this, reuse it
from pyagentic.models.llm import Message


# Optional: use your existing exception class if you prefer
class InvalidStateRef(Exception):
    pass


class BaseState(BaseModel):
    """
    Base class for agent state. Extend this in your own `State` models.

    - `instructions` and `input_template` support Python `.format(**data)` using
      the *serialized* model (including @computed_field values).
    - `_messages` is excluded from serialization.
    """

    instructions: Optional[str] = None
    input_template: Optional[str] = None
    _messages: list[Message] = PrivateAttr(default_factory=list)

    model_config = {
        "validate_assignment": True,
    }

    # ----- Rendering helpers -----

    def _template_data(self) -> Dict[str, Any]:
        # Include computed fields so templates can reference them
        return self.model_dump()

    @property
    def system_message(self) -> str:
        tmpl = self.instructions or ""
        return tmpl.format(**self._template_data())

    @property
    def messages(self) -> list[Message]:
        # Always re-insert fresh system message at the front
        result = self._messages.copy()
        result.insert(0, Message(role="system", content=self.system_message))
        return result

    def add_user_message(self, message: str) -> None:
        if self.input_template:
            data = self._template_data()
            data["user_message"] = message
            content = self.input_template.format(**data)
        else:
            content = message
        self._messages.append(Message(role="user", content=content))

    def get(self, path: str) -> Any:
        """
        Resolve dot-paths into the state (e.g., "profile.name").
        Final value is returned as-is; use @computed_field for derived values.
        """
        val: Any = self
        for part in path.split("."):
            if isinstance(val, dict):
                val = val[part]
            else:
                try:
                    val = getattr(val, part)
                except AttributeError as e:
                    raise InvalidStateRef(path) from e
        # If someone stored a callable, don't auto-call; keep predictable.
        return val
