from typing import Any, Callable, Self
from pydantic import BaseModel

from pyagentic._base._ref import RefNode

type MaybeRef[T] = T | RefNode


class _SpecInfo(BaseModel):
    default: Any | None = None
    default_factory: Callable | None = None

    def get_default(self):
        if self.default_factory:
            return self.default_factory()
        elif self.default:
            return self.default
        else:
            return None

    def resolve(self, agent_reference: dict) -> Self:
        attrs: dict[str, Any] = {}

        # walk actual model fields, not the dumped / serialized version
        for name in self.model_fields:
            value = getattr(self, name)

            if isinstance(value, RefNode):
                resolved = value.resolve(agent_reference)
                attrs[name] = resolved
            else:
                attrs[name] = value

        # rebuild same class with resolved attrs
        return self.__class__.model_validate(attrs)


class AgentInfo(_SpecInfo):
    """Descriptor for State field configuration"""

    condition: MaybeRef[Callable]


class StateInfo(_SpecInfo):
    """Descriptor for State field configuration"""

    persist: MaybeRef[bool] = False
    include_in_templates: MaybeRef[bool | set[str]] = True
    redact_fields: MaybeRef[set[str]] = None


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

    description: MaybeRef[str] | None = None
    required: MaybeRef[bool] | None = False
    values: MaybeRef[list[str]] | None = None
