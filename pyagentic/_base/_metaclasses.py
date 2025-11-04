import inspect
import threading
import warnings
from typing import dataclass_transform, TypeVar, Mapping, Type, Any
from types import MappingProxyType
from collections import ChainMap
from c3linearize import linearize
from typeguard import check_type, TypeCheckError
from pydantic import BaseModel

from pyagentic._base._info import _SpecInfo
from pyagentic._base._validation import _AgentConstructionValidator
from pyagentic._base._exceptions import SystemMessageNotDeclared, UnexpectedStateItemType
from pyagentic._base._agent_state import _AgentState
from pyagentic._base._tool import _ToolDefinition, tool
from pyagentic._base._state import State, StateInfo, _StateDefinition
from pyagentic._base._params import ParamInfo

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
      - Ensures @system_message was declared
      - Collects @tool definitions and StateItem attributes
      - Initializes class __tool_defs__ and __state_items__
      - Dynamically injects an __init__ signature based on class __annotations__
    """

    __BaseAgent__ = None
    _lock = threading.RLock()

    @staticmethod
    def _inherited_namespace_from_bases(bases: tuple[type, ...]) -> dict[str, object]:
        """
        Build the inherited (raw) namespace you'd see via MRO lookup for a class with `bases`.
        Returns a dict where earlier bases in the MRO win.
        """
        # Build a graph: any hashable node -> list of parents.
        # We'll use a sentinel NEW for the (not-yet-created) class.
        NEW = object()
        graph = {NEW: list(bases)}

        # Add all reachable base classes and their parents.
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

        # C3 linearize starting from NEW
        order = linearize(graph)[NEW]
        mro_bases = [c for c in order if isinstance(c, type) and c is not object]

        # Chain the raw class dicts in MRO precedence (leftmost wins)
        return dict(ChainMap(*(vars(c) for c in mro_bases)))

    @staticmethod
    def _extract_tool_defs(namespace) -> Mapping[str, _ToolDefinition]:
        """
        Extracts tool definitions from a given namespace

        Any method with the `@tool` descriptor will be attached to the `__tool_defs__` class
            attribute
        """
        tools: dict[str, _ToolDefinition] = {}
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, "__tool_def__"):
                tools[attr_name] = attr_value.__tool_def__
        return MappingProxyType(tools)

    @staticmethod
    def _extract_annotations(namespace, bases) -> dict[str, TypeVar]:
        """
        Extracts all annotations from current class and all its subclasses. Combines them into
            one dictionary, with class order respected (subclasses overide parent classes.)
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
        state_attributes: dict[str, _StateDefinition] = {}

        for attr_name, attr_type in annotations.items():
            # Check if it's a State[T] generic
            if hasattr(attr_type, "__origin__") and attr_type.__origin__ is State:
                state_model = attr_type.__state_model__

                # Check if there's a descriptor in namespace
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
            privilege = state_def.info.privledge

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
                    return_type=state_def.model,
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
    def _extract_linked_agents(annotations, Agent) -> Mapping[str, "Agent"]:
        """
        Extracts any class field from annotations and namespace where the value is that of
            `StateItem`, these will later be appeneded to the agents state. This will return
            both the type and the user defined state item.
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
        Build __init__ signature with all non-default (required) params
        before any defaulted (optional) params.
        """
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)

        required: list[inspect.Parameter] = []
        optional: list[inspect.Parameter] = []
        agents: list[inspect.Parameter] = []  # Agents go last in signature for better order

        for field_name, field_type in agent_cls.__annotations__.items():
            if field_name in agent_cls.__state_defs__:
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=agent_cls.__state_defs__[field_name].info.default,
                    annotation=field_type,
                )
                optional.append(param)
            elif field_name in agent_cls.__linked_agents__:
                # Treat linked agents as optional by default
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=field_type,
                )
                agents.append(param)
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
        Builds the init function for the class. This init will automatically have user-defined
            state items as arguements, allow for easy initialization of agents for a variety
            of different tasks.
        """

        def __init__(self, *args, **kwargs):
            compiled = {}
            for name, definition in self.__state_defs__.items():
                # Skip compted states, this validaiton will happen with the validator
                #   using a dry run with supplied default values
                # Add all StateItems to the kwargs, checking type as it goes
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

            self.state = self.__state_class__(
                instructions=self.__system_message__,
                input_template=self.__input_template__,
                **compiled,
            )

            for agent_name in self.__linked_agents__.keys():
                agent_instance = kwargs.get(agent_name, None)
                compiled[agent_name] = agent_instance

            bound = sig.bind(self, *args, **(kwargs | compiled))
            # Add all other arguements to instance
            for name, val in list(bound.arguments.items())[1:]:  # skip 'self'
                if name in self.__state_defs__:
                    continue
                setattr(self, name, val)

            self.__post_init__()

        __init__.__signature__ = sig
        __init__.__annotations__ = {
            p.name: p.annotation for p in sig.parameters.values() if p.name != "self"
        }
        return __init__

    def __new__(mcs, name, bases, namespace, **kwargs):
        """
        This metaclass is attached to the Agent base class, so that when a new subclass of Agent
            is created, then this class will automatically set up class variables that define
            the functionality of the agent.

            - __tool_defs__: dictionary holding all tool defintions registered by @tool
            - __state_attrs__: dictionary holding tuple of type and item for all attributes
                that are either have a default of StateItem or use @computed_state
            - __tool_response_models__: dictionary holding pydantic response models for each tool
            - __response_model__: The response model of the current agent that is being built

        Inhertance is repected in MRO order. Tools, state attributes, computed states and
            linked agents can all be inherited, from other agents or mixins.
            __system_message__ and __input_template__ are *not* inherited
        """

        """
        Create an inherited namespace by combining all bases in MRO order.
        This uses c3linearize to determine the order, allowing uses to extend other Agents
            and / or any mixins.
        Mixins are classes that do not extend Agent, but can offer Agent attributes, like
            tools, state items, and/or linked agents
        """
        inherited_namespace = mcs._inherited_namespace_from_bases(bases)

        """
        Declare the new Agent subclass.
        If this is the base agent being declared (usually on import), then the initializtion of
            tools, state items, etc.. will be skipped, and this class will be stored in the meta
            for future use.
        All other Agent subclasses will have __abstract_base__ marked as False, so that future
            implementions "know" it is not the base.
        Since system message is not inherited, an exception is raised if the user does not
            supply one
        """
        with mcs._lock:
            cls = super().__new__(mcs, name, bases, namespace, **kwargs)
            # If it is a base Agent, then return
            if namespace.get("__abstract_base__", False):
                mcs.__BaseAgent__ = cls
                return cls
            cls.__abstract_base__ = False
            # Verify system message is set
            if "__system_message__" not in namespace:
                raise SystemMessageNotDeclared()

        """
        Extract and attach Agent attributes

        __tool_defs__: Tool definitions (from any method marked with an @tool decorator) are
            extracted from both the current namespace and the MRO ordered inherited namespace

        __annotations__: Python annotations are extracted using the namespace and the inherited
            namespace

        __state_attrs__: State attributes (from any class attribute with a StateItem in
            the namespace, or any method marked with a @computed_state decorator) are extracted
            from both the current namespace and the inherited namespace. This also needs the
            classes annotations, in order to attach the annotation to the state attribute for
            later validation

        __linked_agents__: Linked agents work a bit differently, since they cannot have default
            values, the namespaces are not used. Instead, it relies on the MRO ordered annotations
            to build a dict of any agents are are linked.
        """
        annotations = mcs._extract_annotations(inherited_namespace | namespace, bases)
        tool_defs = mcs._extract_tool_defs(inherited_namespace | namespace)
        state_defs = mcs._extract_state_defs(annotations, inherited_namespace | namespace)
        state_tool_defs = mcs._generate_state_tools(cls, state_defs)
        tool_defs = MappingProxyType({**tool_defs, **state_tool_defs})
        linked_agents = mcs._extract_linked_agents(annotations, mcs.__BaseAgent__)
        with mcs._lock:
            cls.__tool_defs__ = tool_defs
            cls.__annotations__ = annotations
            cls.__state_defs__ = state_defs
            cls.__linked_agents__ = linked_agents

        """
        Create response models. Response models are created on class declaration to give the agent
            a predetermined output. This allows developers to know exactly what the output of the
            agent will be, before even creating an instance of the agent.

        __tool_response_models__: All tools have their own pydantic response model, these need
            to be build using their Tool Definition. This needs to be stored on the agent, so that
            it can create instances of the tool response after calling the tool.

        __response_model__: The final pydantic response model of the agent. This is constructed
            using the tool definition models, and any response model of linked agents.
        """
        tool_response_models = {
            tool_name: ToolResponse.from_tool_def(tool_def)
            for tool_name, tool_def in cls.__tool_defs__.items()
        }
        tool_response_model_list = list(tool_response_models.values())
        linked_agent_response_model_list = [
            agent.__response_model__ for agent in cls.__linked_agents__.values()
        ]
        StateClass = _AgentState.make_state_model(
            name=cls.__name__, state_definitions=cls.__state_defs__
        )
        StateClass.__policies__ = {
            name: def_.info.policies for name, def_ in cls.__state_defs__.items()
        }

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

        """
        Build the new init

        The base init just accepts *args and **kwargs, this is changed by building a new init
            signature. The new signature combines any state attributes, agents, and any other
            dataclass field in the following order:
                1. required: any dataclass field with no value in the namespace
                2. optional: mostly state attributes, can include dataclass fields with values in
                    the namespace
                3. linked agents: These come last in order to keep a clear order in the init. They
                    all default to None. So if the user does not supply a linked agent, then the
                    parent agent will ignore it.

        The new init function then creates a new AgentState class, using the state attributes
            as its attributes. It loads in all the state items in it using the specified defaults
            and attaches it to the agent.
            After that it attaches any linked agents to the parent agent.
        """
        sig = mcs._build_init_signature(cls)
        __init__ = mcs._build_init(sig)
        with mcs._lock:
            cls.__init__ = __init__

        """
        Validate and return
        """
        # _AgentConstructionValidator(cls).validate()
        return cls
