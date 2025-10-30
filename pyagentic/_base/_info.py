from typing import Any, Callable
from pydantic import BaseModel


class _SpecInfo(BaseModel):
    default: Any = None
    default_factory: Callable = None

    def get_default(self):
        if self.default_factory:
            return self.default_factory()
        elif self.default:
            return self.default
        else:
            raise ValueError(
                f"Invalid Info Supplied: `default` or `default_factory`should be given"
            )


class AgentInfo(_SpecInfo):
    """Descriptor for State field configuration"""

    condition: Callable


class StateInfo(_SpecInfo):
    """Descriptor for State field configuration"""

    persist: bool = False
    include_in_templates: bool | set[str] = True
    redact_fields: set[str] = None


class ParamInfo(_SpecInfo):
    """
    Declare metadata for parameters in tool declarations and/or Parameter declarations.

    Attributes:
        description (str | None): A human-readable description of the parameter.
        required (bool): Whether this parameter must be provided by the user.
        default (Any): The default value to use if none is provided.
        values (list[str]): values to limit the input of this parameter. If used, the
            agent is forced to use on the the values in the list.

    State-Ready Attributes:
        These attributes can be given a `StateRef` to link them to any state items in
        the agent.

         - description
         - default
         - values
    """

    description: str = None
    required: bool = False
    values: list[str] = None
