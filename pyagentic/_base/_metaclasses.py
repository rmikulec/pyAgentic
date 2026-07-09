import copy
import inspect
import threading
import warnings
from typing import dataclass_transform, TypeVar, Mapping, Type, Any, get_type_hints
from types import MappingProxyType
from collections import ChainMap
from c3linearize import linearize
from typeguard import check_type, TypeCheckError
from pydantic import BaseModel, Field, create_model

from pyagentic._base._info import _SpecInfo, ParamInfo, AgentInfo, MCPInfo
from pyagentic._base._validation import _AgentConstructionValidator
from pyagentic._base._exceptions import InstructionsNotDeclared, UnexpectedStateItemType
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic._base._tool import _ToolDefinition, tool
from pyagentic._base._state import State, StateInfo, _StateDefinition
from pyagentic._base._agent._agent_linking import Link, _LinkedAgentDefinition
from pyagentic._base._depends import Depends
from pyagentic._base._mcp import MCPLink, _MCPDefinition

from pyagentic.models.response import AgentResponse, ErrorResponse, ToolResponse
from pyagentic.models.llm import LLMResponse

from pyagentic._utils._typing import analyze_type


# Placeholder class for Agent type annotation
# Can't import actual agent as it would cause a circular import error
class Agent:
    pass


@dataclass_transform(field_specifiers=(_SpecInfo,))
class AgentMeta(type):
    """
    Metaclass that applies only to Agent subclasses:
      - Ensures __instructions__ (or the deprecated __system_message__) was declared
      - Collects @tool definitions and State field attributes
      - Initializes class __tool_defs__ and __state_defs__
      - Dynamically injects an __init__ signature based on class __annotations__
    """

    __BaseAgent__ = None
    _lock = threading.RLock()

    @staticmethod
    def _inherited_namespace_from_bases(bases: tuple[type, ...]) -> dict[str, object]:
        """
        Builds the inherited (raw) namespace you'd see via MRO lookup for a class with `bases`.
        Returns a dict where earlier bases in the MRO win.

        Args:
            bases (tuple[type, ...]): Tuple of base classes

        Returns:
            dict[str, object]: Combined namespace from all bases in MRO order
        """
        # Build a graph: any hashable node -> list of parents
        # Use a sentinel NEW for the (not-yet-created) class
        NEW = object()
        graph = {NEW: list(bases)}

        # Add all reachable base classes and their parents
        stack = list(bases)
        seen = set()
        while stack:
            cls = stack.pop()
            if cls in seen or cls is object:
                continue
            seen.add(cls)
            parents = [b for b in cls.__bases__ if b is not object]
            graph[cls] = parents
            stack.extend(parents)

        # Use C3 linearization starting from NEW
        order = linearize(graph)[NEW]
        mro_bases = [c for c in order if isinstance(c, type) and c is not object]

        # Chain the raw class dicts in MRO precedence (leftmost wins)
        return dict(ChainMap(*(vars(c) for c in mro_bases)))

    @staticmethod
    def _extract_tool_defs(namespace) -> Mapping[str, _ToolDefinition]:
        """
        Extracts tool definitions from a given namespace.

        Any method with the `@tool` decorator will be attached to the `__tool_defs__` class
            attribute.

        Args:
            namespace (dict): The class namespace to search

        Returns:
            Mapping[str, _ToolDefinition]: Immutable mapping of tool names to definitions
        """
        tools: dict[str, _ToolDefinition] = {}
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, "__tool_def__"):
                tools[attr_name] = attr_value.__tool_def__
        return MappingProxyType(tools)

    @staticmethod
    def _extract_annotations(namespace, bases, cls=None) -> dict[str, TypeVar]:
        """
        Extracts all annotations from current class and all its parent classes. Combines them
            into one dictionary, with class order respected (subclasses override parent classes).

        Args:
            namespace (dict): The class namespace containing __annotations__
            bases (tuple): The base classes
            cls (type, optional): The created class object. In Python 3.14+, deferred
                annotations may only be available on the class, not in the namespace.

        Returns:
            dict[str, TypeVar]: Combined annotations from all classes in hierarchy
        """
        annotations = {}
        for base in reversed(bases):
            if hasattr(base, "__annotations__"):
                for name, type_ in base.__annotations__.items():
                    if not name.startswith("__"):
                        annotations[name] = type_
        for name, type_ in namespace.get("__annotations__", {}).items():
            if not name.startswith("__"):
                annotations[name] = type_
        # Python 3.14+ deferred annotations: annotations may not be in namespace
        # but are available on the created class object
        if cls is not None:
            for name, type_ in getattr(cls, "__annotations__", {}).items():
                if not name.startswith("__") and name not in annotations:
                    annotations[name] = type_
        return annotations

    @staticmethod
    def _extract_state_defs(annotations, namespace) -> Mapping[str, _StateDefinition]:
        """
        Extracts state field definitions from annotations and namespace.

        Looks for State[T] type annotations and pairs them with StateInfo descriptors
        from the namespace if present.

        Args:
            annotations (dict): Combined annotations from the class hierarchy
            namespace (dict): The class namespace

        Returns:
            Mapping[str, _StateDefinition]: Immutable mapping of state field names to definitions
        """
        state_attributes: dict[str, _StateDefinition] = {}

        for attr_name, attr_type in annotations.items():
            # Check if it's a State[T] generic
            if hasattr(attr_type, "__origin__") and attr_type.__origin__ is State:
                state_model = attr_type.__state_model__

                # Check if there's a StateInfo descriptor in namespace
                descriptor = namespace.get(attr_name)
                if isinstance(descriptor, StateInfo):
                    state_info = descriptor
                else:
                    state_info = StateInfo(default=None)

                state_attributes[attr_name] = _StateDefinition(model=state_model, info=state_info)

        return MappingProxyType(state_attributes)

    @staticmethod
    def _generate_state_tools(
        cls, state_defs: Mapping[str, "_StateDefinition"]
    ) -> Mapping[str, _ToolDefinition]:
        """
        Dynamically generate `get_<state>` and `set_<state>` tools
        based on each state's privileges and attach them to the agent class.
        """

        state_tool_defs: dict[str, _ToolDefinition] = {}

        def make_getter(name: str):
            @tool(f"Get the current value of '{name}'.")
            def _getter(self) -> str:
                """Return the current value of the given state."""
                return str(getattr(self, name))

            return _getter

        def make_setter(name: str):
            @tool(f"Set a new value for '{name}'.")
            def _setter(self, value: Any) -> str:
                """Update the state value."""
                setattr(self, name, value)
                return f"'{name}' updated to: {value!r}"

            return _setter

        for state_name, state_def in state_defs.items():
            privilege = state_def.info.access

            # --- Getter ---
            if privilege in ("read", "readwrite"):
                getter_name = f"get_{state_name}"
                getter_func = make_getter(state_name)
                setattr(cls, getter_name, getter_func)

                if not state_def.info.get_description:
                    description = f"Get the current value of '{state_name}'."
                else:
                    description = state_def.info.get_description

                state_tool_defs[getter_name] = _ToolDefinition(
                    name=getter_name,
                    description=description,
                    parameters={},  # no parameters for getter
                    return_type=str,  # getter returns str(value)
                )

            # --- Setter ---
            if privilege in ("write", "readwrite"):
                setter_name = f"set_{state_name}"
                setter_func = make_setter(state_name)
                setattr(cls, setter_name, setter_func)

                if not state_def.info.set_description:
                    description = f"Set a new value for '{state_name}'."
                else:
                    description = state_def.info.set_description

                state_tool_defs[setter_name] = _ToolDefinition(
                    name=setter_name,
                    description=description,
                    parameters={
                        "value": (state_def.model, ParamInfo(default=state_def.info.get_default()))
                    },
                    return_type=state_def.model,
                )

        return MappingProxyType(state_tool_defs)

    @staticmethod
    def _extract_linked_agents(
        annotations, namespace, Agent
    ) -> Mapping[str, _LinkedAgentDefinition]:
        """
        Extracts linked agent fields from annotations and pairs with AgentInfo.

        Looks for annotations where the type is a subclass of Agent or Link[Agent].
        Pairs each linked agent with its AgentInfo descriptor if present.

        Args:
            annotations (dict): Combined annotations from the class hierarchy
            namespace (dict): The class namespace
            Agent (type): The base Agent class to check against

        Returns:
            Mapping[str, _LinkedAgentDefinition]: Immutable mapping of agent field names to definitions
        """
        linked_agents: dict[str, _LinkedAgentDefinition] = {}

        for attr_name, attr_type in annotations.items():
            agent_class = None

            # Check if it's a Link[T] generic (similar to State[T])
            if hasattr(attr_type, "__origin__") and attr_type.__origin__ is Link:
                type_info = analyze_type(attr_type.__linked_agent__, Agent)
                if type_info.is_subclass:
                    agent_class = attr_type.__linked_agent__
            else:
                # Check if it's a direct agent type annotation
                type_info = analyze_type(attr_type, Agent)
                if type_info.has_forward_ref:
                    msg = (
                        f"Forward reference for agents are unsupported: '{attr_name}': {attr_type!r}. "
                        "Make sure the forward ref was not used for an agent, or a TypeError may occur"
                    )
                    warnings.warn(msg, RuntimeWarning, stacklevel=2)
                    raise TypeError(msg)
                elif type_info.is_subclass:
                    agent_class = attr_type

            # If we found an agent class, pair it with AgentInfo
            if agent_class is not None:
                # Check if there's an AgentInfo descriptor in namespace
                descriptor = namespace.get(attr_name)
                if isinstance(descriptor, AgentInfo):
                    agent_info = descriptor
                else:
                    agent_info = AgentInfo(default=None)

                linked_agents[attr_name] = _LinkedAgentDefinition(
                    agent=agent_class, info=agent_info
                )

        return MappingProxyType(linked_agents)

    @staticmethod
    def _extract_mcp_defs(
        annotations, namespace
    ) -> Mapping[str, _MCPDefinition]:
        """Extracts MCP server definitions from annotations and namespace.

        Looks for ``MCPLink`` type annotations and pairs them with ``MCPInfo``
        descriptors from the namespace.

        Args:
            annotations (dict): Combined annotations from the class hierarchy.
            namespace (dict): The class namespace.

        Returns:
            Mapping[str, _MCPDefinition]: Immutable mapping of field names to
                MCP definitions.
        """
        mcp_defs: dict[str, _MCPDefinition] = {}

        for attr_name, attr_type in annotations.items():
            if attr_type is MCPLink:
                descriptor = namespace.get(attr_name)
                if isinstance(descriptor, MCPInfo):
                    mcp_info = descriptor
                else:
                    mcp_info = MCPInfo()

                mcp_defs[attr_name] = _MCPDefinition(
                    field_name=attr_name, info=mcp_info
                )

        return MappingProxyType(mcp_defs)

    @staticmethod
    def _extract_dependencies(annotations, namespace) -> Mapping[str, type]:
        """Extracts dependency-injected fields declared with ``Depends[T]``.

        Looks for annotations whose marker has ``__origin__ is Depends`` and
        records the wrapped dependency type. These fields are injected at serve
        time (by type) rather than supplied by clients, so they are excluded from
        the generated construct model.

        Args:
            annotations (dict): Combined annotations from the class hierarchy.
            namespace (dict): The class namespace.

        Returns:
            Mapping[str, type]: Immutable mapping of field name to dependency type.
        """
        dependencies: dict[str, type] = {}

        for attr_name, attr_type in annotations.items():
            if getattr(attr_type, "__origin__", None) is Depends:
                dependencies[attr_name] = attr_type.__dependency_type__

        return MappingProxyType(dependencies)

    @staticmethod
    def _build_construct_model(cls) -> Type[BaseModel]:
        """Build a recursive Pydantic model mirroring the agent's constructor.

        The construct model is the serializable construction contract for the
        agent: it carries the data a client must supply to instantiate one, in a
        way that mirrors writing the construction in Python.

          - State fields → the raw per-field model, required when the StateInfo
            has no default/default_factory, otherwise optional.
          - Linked agents → that agent's own ``__construct_model__`` (nested),
            optional when the AgentLink provides a default, otherwise required.
          - ``model`` / ``api_key`` → optional scalars for provider selection.

        ``Depends`` and MCP fields are excluded — they are injected/configured
        server-side, not provided by clients.

        Args:
            cls: The agent class whose constructor to mirror.

        Returns:
            Type[BaseModel]: A dynamically created Pydantic model.
        """
        from typing import Optional
        from pydantic import ConfigDict

        fields: dict[str, Any] = {}

        # State fields: type is the raw per-field model (e.g. UserProfile).
        # Check for a default's *existence* without invoking factories.
        for name, state_def in cls.__state_defs__.items():
            info = state_def.info
            if info.default_factory is not None:
                fields[name] = (state_def.model, Field(default_factory=info.default_factory))
            elif info.default is not None:
                fields[name] = (state_def.model, Field(default=info.default))
            else:
                fields[name] = (state_def.model, Field(...))

        # Linked agents: nest each child's construct model. A link is optional
        # when its AgentLink declares any default (don't call the factory here).
        for name, linked_def in cls.__linked_agents__.items():
            child_model = linked_def.agent.__construct_model__
            info = linked_def.info
            has_default = info is not None and (
                info.default is not None or info.default_factory is not None
            )
            if has_default:
                fields[name] = (Optional[child_model], Field(default=None))
            else:
                fields[name] = (child_model, Field(...))

        # Provider selection scalars.
        fields["model"] = (Optional[str], Field(default=None))
        fields["api_key"] = (Optional[str], Field(default=None))

        return create_model(
            f"{cls.__name__}Construct",
            __config__=ConfigDict(arbitrary_types_allowed=True),
            **fields,
        )

    @staticmethod
    def _build_request_model(cls) -> Type[BaseModel]:
        """Build a Pydantic request model from the agent's __call__ signature.

        Inspects the ``__call__`` method to determine the input contract.
        For the default BaseAgent.__call__(user_input: str) this produces a
        model with a single ``user_input`` field.  When a subclass overrides
        ``__call__`` with structured parameters the model mirrors them exactly.

        Args:
            cls: The agent class whose __call__ to inspect.

        Returns:
            Type[BaseModel]: A dynamically created Pydantic model.
        """
        sig = inspect.signature(cls.__call__)
        hints = get_type_hints(cls.__call__)
        hints.pop("return", None)
        hints.pop("self", None)

        fields: dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            annotation = hints.get(name, str)
            if param.default is inspect.Parameter.empty:
                fields[name] = (annotation, Field(...))
            else:
                fields[name] = (annotation, Field(default=param.default))

        return create_model(f"{cls.__name__}Request", **fields)

    @staticmethod
    def _build_stream_event_model(
        agent_name: str,
        tool_response_models: list[Type[ToolResponse]],
        ResponseModel: Type[AgentResponse],
    ) -> Type[BaseModel]:
        """Build a discriminated-union model for SSE stream events.

        Each event kind that ``step()`` can yield gets its own typed wrapper
        with a ``Literal`` event discriminator, so the resulting union is
        fully described in the JSON Schema.

        Args:
            agent_name: Name of the agent class (used for model naming).
            tool_response_models: Per-tool response Pydantic models.
            ResponseModel: The agent's predetermined response model.

        Returns:
            Type[BaseModel]: A Union model of all possible stream events.
        """
        from typing import Literal, Union

        # LLM inference event
        LLMEvent = create_model(
            f"{agent_name}LLMEvent",
            event=(Literal["llm_response"], "llm_response"),
            data=(LLMResponse, ...),
        )

        # Tool response event — typed to the exact tool response variants, plus the
        # base ToolResponse so runtime-discovered (MCP) tool results validate and
        # ErrorResponse so a failed tool call validates. Mirrors the union built by
        # AgentResponse.from_agent_class; without it, an MCP tool result (a base
        # ToolResponse) fails validation when wrapped as a ToolEvent in the jobs backend.
        if tool_response_models:
            ToolData = Union[tuple([*tool_response_models, ToolResponse, ErrorResponse])]
        else:
            ToolData = Union[ToolResponse, ErrorResponse]
        ToolEvent = create_model(
            f"{agent_name}ToolEvent",
            event=(Literal["tool_response"], "tool_response"),
            data=(ToolData, ...),
        )

        # Final agent response event
        AgentEvent = create_model(
            f"{agent_name}AgentEvent",
            event=(Literal["agent_response"], "agent_response"),
            data=(ResponseModel, ...),
        )

        StreamEvent = create_model(
            f"{agent_name}StreamEvent",
            __base__=BaseModel,
            root=(Union[LLMEvent, ToolEvent, AgentEvent], ...),
        )
        # Store the individual event models for direct access
        StreamEvent.__llm_event__ = LLMEvent
        StreamEvent.__tool_event__ = ToolEvent
        StreamEvent.__agent_event__ = AgentEvent

        return StreamEvent

    @staticmethod
    def _build_init_signature(agent_cls: Type[Agent]) -> inspect.Signature:
        """
        Builds __init__ signature with all non-default (required) params
        before any defaulted (optional) params.

        Args:
            agent_cls (Type[Agent]): The agent class being constructed

        Returns:
            inspect.Signature: The constructed __init__ signature
        """
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)

        # Group parameters by type for proper ordering
        required: list[inspect.Parameter] = []  # Required parameters (no default)
        optional: list[inspect.Parameter] = []  # Optional parameters (with default)
        agents: list[inspect.Parameter] = []  # Linked agents (always optional, go last)

        for field_name, field_type in agent_cls.__annotations__.items():
            # State fields are always optional (have defaults or default_factory)
            if field_name in agent_cls.__state_defs__:
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=agent_cls.__state_defs__[field_name].info.default,
                    annotation=field_type,
                )
                optional.append(param)
            # Linked agents are optional by default (can be None)
            elif field_name in agent_cls.__linked_agents__:
                # Get default from AgentInfo if available
                linked_def = agent_cls.__linked_agents__[field_name]
                default = linked_def.info.get_default() if linked_def.info else None
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_type,
                )
                agents.append(param)
            # MCP fields are config-only, not constructor args
            elif field_name in agent_cls.__mcp_defs__:
                continue
            # Dependency-injected fields are optional (default None); they are
            # supplied at serve time or passed explicitly for direct/test use.
            elif field_name in agent_cls.__dependencies__:
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=field_type,
                )
                optional.append(param)
            # Other annotated fields (like model, api_key, tracer, etc.)
            else:
                default = getattr(agent_cls, field_name, inspect._empty)
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=default,
                    annotation=field_type,
                )

                if default is inspect._empty:
                    required.append(param)
                else:
                    optional.append(param)

        return inspect.Signature([self_param, *required, *optional, *agents])

    @staticmethod
    def _build_init(sig):
        """
        Builds the __init__ function for the class. This init will automatically have user-defined
            state fields as arguments, allowing for easy initialization of agents for a variety
            of different tasks.

        Args:
            sig (inspect.Signature): The signature to attach to the __init__

        Returns:
            Callable: The constructed __init__ function
        """

        def __init__(self, *args, **kwargs):
            compiled = {}
            # Process all state field definitions
            for name, definition in self.__state_defs__.items():
                # Add all state fields to compiled dict, validating types as we go
                if name in kwargs:
                    val = kwargs[name]
                    try:
                        check_type(val, definition.model)
                    except TypeCheckError:
                        raise UnexpectedStateItemType(
                            name=name, expected=definition.model, recieved=type(val)
                        )
                    compiled[name] = val
                else:
                    compiled[name] = definition.info.get_default()

            # Create the state object with instructions and state fields
            self.state = self.__state_class__(
                instructions=self.__instructions__,
                parent_instructions=self.__parent_instructions__,
                input_template=self.__input_template__,
                **compiled,
            )

            # Collect linked agent instances from kwargs or use defaults
            for agent_name, linked_def in self.__linked_agents__.items():
                if agent_name in kwargs:
                    agent_instance = kwargs[agent_name]
                else:
                    # Use default or default_factory from AgentInfo
                    agent_instance = linked_def.info.get_default() if linked_def.info else None
                compiled[agent_name] = agent_instance

            # Bind all arguments to signature and set as instance attributes
            bound = sig.bind(self, *args, **(kwargs | compiled))
            for name, val in list(bound.arguments.items())[1:]:  # Skip 'self'
                if name in self.__state_defs__:
                    continue  # State fields already set on state object
                setattr(self, name, val)

            # Capture the construction recipe so the agent can fork() fresh,
            # isolated copies for non-shared linked-agent calls: a deep snapshot
            # of the construct-time state, plus the shared linked templates,
            # dependencies, and provider config to re-pass.
            self.__initial_state_values__ = {
                name: copy.deepcopy(compiled[name]) for name in self.__state_defs__
            }
            construct_args = {name: compiled[name] for name in self.__linked_agents__}
            for name in (*self.__dependencies__, "model", "api_key", "max_call_depth"):
                if name in bound.arguments:
                    construct_args[name] = bound.arguments[name]
            self.__construct_args__ = construct_args

            # Call post-initialization hook
            self.__post_init__()

        __init__.__signature__ = sig
        __init__.__annotations__ = {
            p.name: p.annotation for p in sig.parameters.values() if p.name != "self"
        }
        return __init__

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        Creates a new Agent subclass with all the necessary class attributes and behavior.

        This metaclass automatically sets up:
          - __tool_defs__: Dictionary holding all tool definitions registered by @tool
          - __state_defs__: Dictionary holding state field definitions
          - __tool_response_models__: Dictionary holding Pydantic response models for each tool
          - __response_model__: The response model of the current agent being built

        Inheritance is respected in MRO order. Tools, state attributes, and linked agents
        can all be inherited from other agents or mixins.
        __instructions__ is inherited from the nearest ancestor when not declared. When a
        subclass declares its own __instructions__, the ancestor chain is recorded on
        __parent_instructions__ so the template can embed the parent's rendered
        instructions via `{{ super }}`.

        Args:
            name (str): Name of the new class
            bases (tuple): Base classes
            namespace (dict): Class namespace
            **kwargs: Additional keyword arguments

        Returns:
            type: The newly created Agent subclass

        Raises:
            InstructionsNotDeclared: If neither __instructions__ nor the deprecated
                __system_message__ is defined in the class
        """

        # Create an inherited namespace by combining all bases in MRO order
        # Uses c3linearize to determine the order, allowing users to extend other Agents
        # and/or any mixins. Mixins are classes that do not extend Agent, but can offer
        # Agent attributes like tools, state fields, and/or linked agents
        inherited_namespace = mcs._inherited_namespace_from_bases(bases)

        # Declare the new Agent subclass
        # If this is the base agent being declared (usually on import), then the initialization
        # of tools, state fields, etc. will be skipped, and this class will be stored in the
        # metaclass for future use. All other Agent subclasses will have __abstract_base__
        # marked as False. Since system message is not inherited, an exception is raised if
        # the user does not supply one
        with mcs._lock:
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)
            # If this is the base Agent, store it and return early
            if namespace.get("__abstract_base__", False):
                mcs.__BaseAgent__ = cls
                return cls
            cls.__abstract_base__ = False
            # Resolve instructions; __system_message__ is the deprecated spelling
            # and normalizes onto __instructions__
            declared = None
            if "__instructions__" in namespace:
                declared = namespace["__instructions__"]
            elif "__system_message__" in namespace:
                warnings.warn(
                    f"`__system_message__` on {name} is deprecated; "
                    "declare `__instructions__` instead",
                    DeprecationWarning,
                    stacklevel=2,
                )
                declared = namespace["__system_message__"]

            inherited_instructions = inherited_namespace.get(
                "__instructions__", inherited_namespace.get("__system_message__")
            )
            if declared is not None:
                cls.__instructions__ = declared
                # When overriding an ancestor's instructions, record the ancestor
                # chain (oldest first) so the template can embed the parent's
                # rendered instructions via `{{ super }}`
                if inherited_instructions is not None:
                    cls.__parent_instructions__ = (
                        *inherited_namespace.get("__parent_instructions__", ()),
                        inherited_instructions,
                    )
                else:
                    cls.__parent_instructions__ = ()
            elif inherited_instructions is not None:
                # Instructions are inherited from the nearest ancestor;
                # __parent_instructions__ resolves to the declaring ancestor's chain
                cls.__instructions__ = inherited_instructions
            else:
                raise InstructionsNotDeclared()
            # Keep the deprecated attribute readable for backwards compatibility
            cls.__system_message__ = cls.__instructions__

        # Extract and attach Agent attributes:
        #
        # __tool_defs__: Tool definitions from any method marked with @tool decorator,
        #     extracted from both current namespace and inherited namespace
        #
        # __annotations__: Python annotations extracted from namespace and inherited namespace
        #
        # __state_defs__: State field definitions from annotations of type State[T], paired
        #     with StateInfo descriptors from the namespace. Extracted from both current and
        #     inherited namespaces, using annotations to get the type information
        #
        # __linked_agents__: Linked agents extracted from annotations where the type is a
        #     subclass of Agent. These don't use namespace defaults, only annotations
        tool_defs = mcs._extract_tool_defs(inherited_namespace | namespace)
        annotations = mcs._extract_annotations(inherited_namespace | namespace, bases, cls=cls)
        tool_defs = mcs._extract_tool_defs(inherited_namespace | namespace)
        state_defs = mcs._extract_state_defs(annotations, inherited_namespace | namespace)
        state_tool_defs = mcs._generate_state_tools(cls, state_defs)
        tool_defs = MappingProxyType({**tool_defs, **state_tool_defs})
        linked_agents = mcs._extract_linked_agents(
            annotations, inherited_namespace | namespace, mcs.__BaseAgent__
        )
        mcp_defs = mcs._extract_mcp_defs(annotations, inherited_namespace | namespace)
        dependencies = mcs._extract_dependencies(annotations, inherited_namespace | namespace)
        with mcs._lock:
            cls.__tool_defs__ = tool_defs
            cls.__annotations__ = annotations
            cls.__state_defs__ = state_defs
            cls.__linked_agents__ = linked_agents
            cls.__mcp_defs__ = mcp_defs
            cls.__dependencies__ = dependencies

        # Create response models at class declaration time, giving the agent a predetermined
        # output structure. This allows developers to know exactly what the output of the
        # agent will be before even creating an instance.
        #
        # __tool_response_models__: All tools have their own Pydantic response model built
        #     from their ToolDefinition. Stored on the agent so it can create instances of
        #     the tool response after calling the tool.
        #
        # __response_model__: The final Pydantic response model of the agent, constructed
        #     using the tool response models and linked agent response models
        # Build tool response models for each tool
        tool_response_models = {
            tool_name: ToolResponse.from_tool_def(tool_def)
            for tool_name, tool_def in cls.__tool_defs__.items()
        }
        tool_response_model_list = list(tool_response_models.values())
        linked_agent_response_model_list = [
            linked_def.agent.__response_model__ for linked_def in cls.__linked_agents__.values()
        ]

        # Create a Pydantic model for the agent's state
        if "messages" in cls.__state_defs__:
            raise ValueError(
                f"Agent '{cls.__name__}' declares a state field named 'messages', which is "
                "reserved for the message context (it collides with the `messages` property "
                "and the message-policy registry). Rename the field."
            )
        StateClass = _AgentState.make_state_model(
            name=cls.__name__, state_definitions=cls.__state_defs__
        )
        # Attach policies to the state class for runtime policy enforcement.
        # Message policies (from __message_policies__) register under the
        # reserved "messages" key, alongside per-field state policies.
        StateClass.__policies__ = {
            name: def_.info.policies for name, def_ in cls.__state_defs__.items()
        }
        StateClass.__policies__["messages"] = tuple(
            (inherited_namespace | namespace).get("__message_policies__") or ()
        )

        # Create the final agent response model
        ResponseModel = AgentResponse.from_agent_class(
            agent_name=cls.__name__,
            tool_response_models=tool_response_model_list,
            linked_agents_response_models=linked_agent_response_model_list,
            ResponseFormat=cls.__response_format__,
            StateClass=StateClass,
        )
        # Build a Pydantic request model from the agent's __call__ signature.
        # This is the predetermined input contract for the agent, mirroring how
        # __response_model__ is the predetermined output contract.
        RequestModel = mcs._build_request_model(cls)

        # Build a typed model describing all possible SSE stream events.
        # This gives the streaming endpoint a fully described JSON Schema.
        StreamEventModel = mcs._build_stream_event_model(
            agent_name=cls.__name__,
            tool_response_models=tool_response_model_list,
            ResponseModel=ResponseModel,
        )

        # Build the recursive construct model mirroring the agent's constructor.
        # Linked agents' construct models already exist because their classes are
        # defined before this one (same guarantee relied on for __response_model__).
        ConstructModel = mcs._build_construct_model(cls)

        with mcs._lock:
            cls.__tool_response_models__ = MappingProxyType(tool_response_models)
            cls.__response_model__ = ResponseModel
            cls.__request_model__ = RequestModel
            cls.__construct_model__ = ConstructModel
            cls.__stream_event_model__ = StreamEventModel
            cls.__state_class__ = StateClass

        # Build the custom __init__ method
        #
        # The base init just accepts *args and **kwargs. This is replaced with a new init
        # that has a specific signature combining state fields, linked agents, and other
        # dataclass fields in the following order:
        #   1. required: Any annotated field with no default value in the namespace
        #   2. optional: State fields (always have defaults) and other fields with defaults
        #   3. linked agents: Always optional (default to None) and come last
        #
        # The new init function creates an AgentState instance with the state field values,
        # attaches it to self.state, then attaches linked agent instances and other fields
        # as instance attributes
        sig = mcs._build_init_signature(cls)
        __init__ = mcs._build_init(sig)
        with mcs._lock:
            cls.__init__ = __init__

        # Validation is commented out but can be enabled for additional runtime checks
        # _AgentConstructionValidator(cls).validate()

        return cls
