import inspect
import threading
import warnings
from typing import dataclass_transform, TypeVar, Mapping, Type
from types import MappingProxyType
from collections import ChainMap
from c3linearize import linearize
from typeguard import check_type, TypeCheckError

from pyagentic._base._info import _SpecInfo
from pyagentic._base._validation import _AgentConstructionValidator
from pyagentic._base._exceptions import SystemMessageNotDeclared, UnexpectedStateItemType
from pyagentic._base._agent._agent_state import _AgentState
from pyagentic._base._tool import _ToolDefinition
from pyagentic._base._state import State, StateInfo, _StateDefinition

from pyagentic.models.response import AgentResponse, ToolResponse

from pyagentic._utils._typing import analyze_type


# Placeholder class for Agent type annotation
# Can't import actual agent as it would cause a circular import error
class Agent:
    pass


@dataclass_transform(field_specifiers=(_SpecInfo,))
class AgentMeta(type):
    """
    Metaclass that applies only to Agent subclasses:
      - Ensures __system_message__ was declared
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
    def _extract_annotations(namespace, bases) -> dict[str, TypeVar]:
        """
        Extracts all annotations from current class and all its parent classes. Combines them
            into one dictionary, with class order respected (subclasses override parent classes).

        Args:
            namespace (dict): The class namespace containing __annotations__
            bases (tuple): The base classes

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
    def _extract_linked_agents(annotations, Agent) -> Mapping[str, "Agent"]:
        """
        Extracts linked agent fields from annotations.

        Looks for annotations where the type is a subclass of Agent (or Link[Agent]).
        These will be available as callable tools to the LLM.

        Args:
            annotations (dict): Combined annotations from the class hierarchy
            Agent (type): The base Agent class to check against

        Returns:
            Mapping[str, Agent]: Immutable mapping of agent field names to agent types
        """
        linked_agents: dict[str, "Agent"] = {}
        for attr_name, attr_type in annotations.items():
            type_info = analyze_type(attr_type, Agent)
            if type_info.has_forward_ref:
                msg = (
                    f"Forward reference for agents are unsupported: '{attr_name}': {attr_type!r}. "
                    "Make sure the forward ref was not used for an agent, or a TypeError may occur"
                )
                warnings.warn(msg, RuntimeWarning, stacklevel=2)
            elif type_info.is_subclass:
                linked_agents[attr_name] = attr_type

        return MappingProxyType(linked_agents)

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
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=field_type,
                )
                agents.append(param)
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

            # Create the state object with system message and state fields
            self.state = self.__state_class__(
                instructions=self.__system_message__,
                input_template=self.__input_template__,
                **compiled,
            )

            # Collect linked agent instances from kwargs
            for agent_name in self.__linked_agents__.keys():
                agent_instance = kwargs.get(agent_name, None)
                compiled[agent_name] = agent_instance

            # Bind all arguments to signature and set as instance attributes
            bound = sig.bind(self, *args, **(kwargs | compiled))
            for name, val in list(bound.arguments.items())[1:]:  # Skip 'self'
                if name in self.__state_defs__:
                    continue  # State fields already set on state object
                setattr(self, name, val)

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
        __system_message__ and __input_template__ are *not* inherited.

        Args:
            name (str): Name of the new class
            bases (tuple): Base classes
            namespace (dict): Class namespace
            **kwargs: Additional keyword arguments

        Returns:
            type: The newly created Agent subclass

        Raises:
            SystemMessageNotDeclared: If __system_message__ is not defined in the class
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
            # Verify system message is set (not inherited)
            if "__system_message__" not in namespace:
                raise SystemMessageNotDeclared()

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
        annotations = mcs._extract_annotations(inherited_namespace | namespace, bases)
        state_defs = mcs._extract_state_defs(annotations, inherited_namespace | namespace)
        linked_agents = mcs._extract_linked_agents(annotations, mcs.__BaseAgent__)
        with mcs._lock:
            cls.__tool_defs__ = tool_defs
            cls.__annotations__ = annotations
            cls.__state_defs__ = state_defs
            cls.__linked_agents__ = linked_agents

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
            agent.__response_model__ for agent in cls.__linked_agents__.values()
        ]

        # Create a Pydantic model for the agent's state
        StateClass = _AgentState.make_state_model(
            name=cls.__name__, state_definitions=cls.__state_defs__
        )
        # Attach policies to the state class for runtime policy enforcement
        StateClass.__policies__ = {
            name: def_.info.policies for name, def_ in cls.__state_defs__.items()
        }

        # Create the final agent response model
        ResponseModel = AgentResponse.from_agent_class(
            agent_name=cls.__name__,
            tool_response_models=tool_response_model_list,
            linked_agents_response_models=linked_agent_response_model_list,
            ResponseFormat=cls.__response_format__,
            StateClass=StateClass,
        )
        with mcs._lock:
            cls.__tool_response_models__ = MappingProxyType(tool_response_models)
            cls.__response_model__ = ResponseModel
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
