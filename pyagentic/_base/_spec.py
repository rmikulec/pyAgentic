from typing import Any, Callable

from pyagentic._base._info import StateInfo, ParamInfo, AgentInfo
from pyagentic.policies._policy import Policy


class spec:
    @staticmethod
    def State(
        default: Any = None,
        default_factory: Callable = None,
        policies: list[Policy] = None,
        **kwargs,
    ) -> StateInfo:
        return StateInfo(
            default=default, default_factory=default_factory, policies=policies, **kwargs
        )

    @staticmethod
    def Param(
        description: str = None,
        required: bool = False,
        default: Any = None,
        default_factory: Callable = None,
        values: Any = None,
    ) -> ParamInfo:
        return ParamInfo(
            description=description,
            required=required,
            default=default,
            default_factory=default_factory,
            values=values,
        )

    @staticmethod
    def AgentLink(
        default: Any = None, default_factory: Callable = None, condition: Callable = None
    ) -> AgentInfo:
        return AgentInfo(default=default, default_factory=default_factory, condition=condition)
