from typing import Any, Callable, Literal

from pyagentic._base._info import StateInfo, ParamInfo, AgentInfo, MCPInfo, MaybeRef
from pyagentic.policies._policy import Policy


class spec:
    """
    Factory class for creating configuration descriptors for agent fields.

    The `spec` object provides methods to configure state fields, tool parameters,
    and linked agents with advanced options like defaults, policies, and conditions.

    Methods:
        - spec.State(): Configure agent state fields
        - spec.Param(): Configure tool parameters with validation
        - spec.AgentLink(): Configure linked agent fields

    Example:
        ```python
        from pyagentic import BaseAgent, State, spec, tool
        from pydantic import BaseModel

        class UserProfile(BaseModel):
            name: str
            email: str

        class MyAgent(BaseAgent):
            __system_message__ = "You are a helpful assistant"

            # State with default factory
            profile: State[UserProfile] = spec.State(
                default_factory=lambda: UserProfile(name="Guest", email="")
            )

            # State with policies
            logs: State[list] = spec.State(
                default_factory=list,
                policies=[LoggingPolicy()]
            )

            @tool("Update user profile")
            def update_profile(
                self,
                name: str = spec.Param(description="User's full name", required=True),
                email: str = spec.Param(description="User's email address")
            ) -> str:
                self.state.profile.name = name
                if email:
                    self.state.profile.email = email
                return f"Updated profile for {name}"
        ```
    """

    @staticmethod
    def State(
        default: Any = None,
        default_factory: Callable = None,
        policies: list[Policy] = None,
        access: Literal["read", "write", "readwrite", "hidden"] = "read",
        description: str | None = None,
        get_description: str | None = None,
        set_description: str | None = None,
        **kwargs,
    ) -> StateInfo:
        """
        Creates a StateInfo descriptor for configuring agent state fields.

        Args:
            default (Any, optional): The default value for the state field
            default_factory (Callable, optional): A factory function to generate the default value
            policies (list[Policy], optional): List of policies to apply to this state field
            **kwargs: Additional keyword arguments passed to StateInfo

        Returns:
            StateInfo: A configured StateInfo descriptor
        """
        return StateInfo(
            default=default,
            default_factory=default_factory,
            policies=policies,
            access=access,
            description=description,
            get_description=get_description,
            set_description=set_description,
            **kwargs,
        )

    @staticmethod
    def Param(
        description: str = None,
        required: bool = False,
        default: Any = None,
        default_factory: Callable = None,
        values: Any = None,
    ) -> ParamInfo:
        """
        Creates a ParamInfo descriptor for configuring tool parameters.

        Args:
            description (str, optional): A human-readable description of the parameter
            required (bool): Whether this parameter must be provided. Defaults to False
            default (Any, optional): The default value for the parameter
            default_factory (Callable, optional): A factory function to generate the default value
            values (Any, optional): List of valid values to constrain the parameter

        Returns:
            ParamInfo: A configured ParamInfo descriptor
        """
        return ParamInfo(
            description=description,
            required=required,
            default=default,
            default_factory=default_factory,
            values=values,
        )

    @staticmethod
    def AgentLink(
        default: Any = None,
        default_factory: Callable = None,
        condition: Callable = None,
        phases: list[str] | None = None,
    ) -> AgentInfo:
        """
        Creates an AgentInfo descriptor for configuring linked agent fields.

        Args:
            default (Any, optional): The default agent instance
            default_factory (Callable, optional): A factory function to generate the default agent
            condition (Callable, optional): A callable determining when this agent link is active
            phases (list[str], optional): A list of phases of when this agent will be available.
                When None, will show for all phases. Defaults to None.

        Returns:
            AgentInfo: A configured AgentInfo descriptor
        """
        return AgentInfo(
            default=default, default_factory=default_factory, condition=condition, phases=phases
        )

    @staticmethod
    def MCPLink(
        server: MaybeRef[Any] = None,
        *,
        args: list[MaybeRef[str]] | None = None,
        tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        prefix: bool | str = True,
        condition: Callable | None = None,
        phases: list[str] | None = None,
        description: str | None = None,
    ) -> MCPInfo:
        """Creates an MCPInfo descriptor for configuring MCP server connections.

        Auto-detection of transport is handled at runtime:
          - ``str`` starting with ``http://`` or ``https://`` → streamable HTTP
          - ``str`` + ``args`` → stdio subprocess
          - ``FastMCP`` object → in-process

        Both ``server`` and ``args`` entries may be StateRefs (e.g.
        ``ref.self.root``); they are resolved against agent state when the
        MCP connection is established.

        Args:
            server (Any): URL string, command string, or FastMCP server object.
            args (list[str], optional): Arguments for stdio subprocess mode.
            tools (list[str], optional): Whitelist of tool names to expose.
            exclude_tools (list[str], optional): Blacklist of tool names to hide.
            prefix (bool | str): Whether to prefix tool names with the field
                name. ``True`` uses the field name, a string uses that value,
                ``False`` disables prefixing. Defaults to True.
            condition (Callable, optional): Callable determining when this MCP
                connection is active.
            phases (list[str], optional): Phases during which this MCP link
                is available.
            description (str, optional): Human-readable description.

        Returns:
            MCPInfo: A configured MCPInfo descriptor.
        """
        return MCPInfo(
            server=server,
            args=args,
            tools=tools,
            exclude_tools=exclude_tools,
            prefix=prefix,
            condition=condition,
            phases=phases,
            description=description,
        )
