from typing import Any, Callable, Self
from dataclasses import dataclass

from pyagentic._base._ref import RefNode
from pyagentic.policies._policy import Policy

type MaybeRef[T] = T | RefNode


@dataclass
class _SpecInfo:
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
        for name, value in self.__dict__.items():
            value = getattr(self, name)

            if isinstance(value, RefNode):
                resolved = value.resolve(agent_reference)
                attrs[name] = resolved
            else:
                attrs[name] = value

        # rebuild same class with resolved attrs
        return self.__class__(**attrs)


@dataclass
class AgentInfo(_SpecInfo):
    """
    Descriptor for configuring linked agent fields.
    """

    condition: MaybeRef[Callable] | None = None


@dataclass
class StateInfo(_SpecInfo):
    """
    Descriptor for configuring State field metadata and policies.
    """

    policies: list[Policy] | None = None


@dataclass
class ParamInfo(_SpecInfo):
    """
    Declares metadata for parameters in tool declarations and/or Parameter declarations.

    Attributes:
        description (str | None): A human-readable description of the parameter
        required (bool): Whether this parameter must be provided by the user
        default (Any): The default value to use if none is provided
        values (list[str]): Values to limit the input of this parameter. If used, the
            agent is forced to use one of the values in the list

    State-Ready Attributes:
        These attributes can be given a StateRef to link them to any state items in
        the agent:
          - description
          - default
          - values
    """

    description: MaybeRef[str] | None = None
    required: MaybeRef[bool] | None = False
    values: MaybeRef[list[str]] | None = None
