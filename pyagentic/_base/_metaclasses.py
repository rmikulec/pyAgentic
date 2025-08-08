import inspect
from typing import dataclass_transform, TypeVar
from collections import ChainMap
from c3linearize import linearize
from typeguard import check_type, TypeCheckError

from pyagentic._base._validation import _AgentConstructionValidator
from pyagentic._base._exceptions import SystemMessageNotDeclared, UnexpectedContextItemType
from pyagentic._base._context import _AgentContext, ContextItem, computed_context
from pyagentic._base._tool import _ToolDefinition

from pyagentic.models.response import AgentResponse, ToolResponse

from pyagentic._utils._typing import analyze_type


# Placeholder class for Agent type annotation
# Can't import actual agent as it would cause a circular import error
class Agent:
    pass


@dataclass_transform(field_specifiers=(ContextItem,))
class AgentMeta(type):
    """
    Metaclass that applies only to Agent subclasses:
      - Ensures @system_message was declared
      - Collects @tool definitions and ContextItem attributes
      - Initializes class __tool_defs__ and __context_items__
      - Dynamically injects an __init__ signature based on class __annotations__
    """

    __BaseAgent__ = None

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
    def _extract_tool_defs(namespace) -> dict[str, _ToolDefinition]:
        """
        Extracts tool definitions from a given namespace

        Any method with the `@tool` descriptor will be attached to the `__tool_defs__` class
            attribute
        """
        tools: dict[str, _ToolDefinition] = {}
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, "__tool_def__"):
                tools[attr_name] = attr_value.__tool_def__
        return tools

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
    def _extract_context_attrs(annotations, namespace) -> dict[str, tuple[TypeVar, ContextItem]]:
        """
        Extracts any class field from annotations and namespace where the value is that of
            `ContextItem`, these will later be appeneded to the agents context. This will return
            both the type and the user defined context item.
        """
        context_attrs: dict[str, tuple[TypeVar, ContextItem]] = {}
        for attr_name, attr_type in annotations.items():
            default = namespace.get(attr_name, None)
            if isinstance(default, ContextItem):
                context_attrs[attr_name] = (attr_type, default)

        for name, value in namespace.items():
            if getattr(value, "_is_context", False):
                context_attrs[name] = (computed_context, value)
        return context_attrs

    @staticmethod
    def _extract_linked_agents(annotations, Agent) -> dict[str, "Agent"]:
        """
        Extracts any class field from annotations and namespace where the value is that of
            `ContextItem`, these will later be appeneded to the agents context. This will return
            both the type and the user defined context item.
        """
        linked_agents: dict[str, "Agent"] = {}
        for attr_name, attr_type in annotations.items():
            type_info = analyze_type(attr_type, Agent)
            if type_info.is_subclass:
                linked_agents[attr_name] = attr_type

        return linked_agents

    @staticmethod
    def _build_init_signature(cls) -> inspect.Signature:
        """
        Build __init__ signature with all non-default (required) params
        before any defaulted (optional) params.
        """
        self_param = inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)

        required: list[inspect.Parameter] = []
        optional: list[inspect.Parameter] = []
        agents: list[inspect.Parameter] = []  # Agents go last in signature for better order

        for field_name, field_type in cls.__annotations__.items():
            if field_name in cls.__context_attrs__:
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=cls.__context_attrs__[field_name][1].get_default_value(),
                    annotation=field_type,
                )
                optional.append(param)
            elif field_name in cls.__linked_agents__:
                # Treat linked agents as optional by default
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=None,
                    annotation=field_type,
                )
                agents.append(param)
            else:
                param = inspect.Parameter(
                    field_name,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    default=getattr(cls, field_name, inspect._empty),
                    annotation=field_type,
                )
                required.append(param)

        return inspect.Signature([self_param, *required, *optional, *agents])

    @staticmethod
    def _build_init(sig):
        """
        Builds the init function for the class. This init will automatically have user-defined
            context items as arguements, allow for easy initialization of agents for a variety
            of different tasks.
        """

        def __init__(self, *args, **kwargs):

            # -------- ContextClass Construction --------------------
            ContextClass = _AgentContext.make_ctx_class(
                name=self.__class__.__name__, ctx_map=self.__context_attrs__
            )

            compiled = {}
            for attr_name, (attr_type, attr_default) in self.__context_attrs__.items():
                # Skip compted contexts, this validaiton will happen with the validator
                #   using a dry run with supplied default values
                if attr_type == computed_context:
                    continue
                # Add all ContextItems to the kwargs, checking type as it goes
                if attr_name in kwargs:
                    val = kwargs[attr_name]
                    try:
                        check_type(val, attr_type)
                    except TypeCheckError:
                        raise UnexpectedContextItemType(
                            name=attr_name, expected=attr_type, recieved=type(val)
                        )
                    compiled[attr_name] = val
                else:
                    compiled[attr_name] = attr_default.get_default_value()

            self.context = ContextClass(
                instructions=self.__system_message__,
                input_template=self.__input_template__,
                **compiled,
            )
            # ------------- Retrieve Linked Agents -------------------
            for agent_name in self.__linked_agents__.keys():
                agent_instance = kwargs.get(agent_name, None)
                compiled[agent_name] = agent_instance

            bound = sig.bind(self, *args, **kwargs)

            # Add all other arguements to instance
            for name, val in list(bound.arguments.items())[1:]:  # skip 'self'
                if name in self.__context_attrs__:
                    pass
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
            - __context_attrs__: dictionary holding tuple of type and item for all attributes
                that are either have a default of ContextItem or use @computed_context
            - __tool_response_models__: dictionary holding pydantic response models for each tool
            - __response_model__: The response model of the current agent that is being built

        Inhertance is repected in MRO order. Tools, context attributes, computed contexts and
            linked agents can all be inherited, from other agents or mixins.
            __system_message__ and __input_template__ are *not* inherited
        """

        """
        Create an inherited namespace by combining all bases in MRO order.
        This uses c3linearize to determine the order, allowing uses to extend other Agents
            and / or any mixins.
        Mixins are classes that do not extend Agent, but can offer Agent attributes, like
            tools, context items, and/or linked agents
        """
        inherited_namespace = mcs._inherited_namespace_from_bases(bases)

        """
        Declare the new Agent subclass.
        If this is the base agent being declared (usually on import), then the initializtion of
            tools, context items, etc.. will be skipped, and this class will be stored in the meta
            for future use.
        All other Agent subclasses will have __abstract_base__ marked as False, so that future
            implementions "know" it is not the base.
        Since system message is not inherited, an exception is raised if the user does not
            supply one
        """
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

        __context_attrs__: Context attributes (from any class attribute with a ContextItem in
            the namespace, or any method marked with a @computed_context decorator) are extracted
            from both the current namespace and the inherited namespace. This also needs the
            classes annotations, in order to attach the annotation to the context attribute for
            later validation

        __linked_agents__: Linked agents work a bit differently, since they cannot have default
            values, the namespaces are not used. Instead, it relies on the MRO ordered annotations
            to build a dict of any agents are are linked.
        """
        cls.__tool_defs__ = mcs._extract_tool_defs(inherited_namespace | namespace)
        cls.__annotations__ = mcs._extract_annotations(inherited_namespace | namespace, bases)
        cls.__context_attrs__ = mcs._extract_context_attrs(
            cls.__annotations__, inherited_namespace | namespace
        )
        cls.__linked_agents__ = mcs._extract_linked_agents(cls.__annotations__, mcs.__BaseAgent__)

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
        cls.__tool_response_models__ = {
            tool_name: ToolResponse.from_tool_def(tool_def)
            for tool_name, tool_def in cls.__tool_defs__.items()
        }
        tool_response_models = list(cls.__tool_response_models__.values())
        linked_agent_response_models = [
            agent.__response_model__ for agent in cls.__linked_agents__.values()
        ]
        cls.__response_model__ = AgentResponse.from_tool_defs(
            agent_name=cls.__name__,
            tool_response_models=tool_response_models,
            linked_agents_response_models=linked_agent_response_models,
        )

        """
        Build the new init

        The base init just accepts *args and **kwargs, this is changed by building a new init
            signature. The new signature combines any context attributes, agents, and any other
            dataclass field in the following order:
                1. required: any dataclass field with no value in the namespace
                2. optional: mostly context attributes, can include dataclass fields with values in
                    the namespace
                3. linked agents: These come last in order to keep a clear order in the init. They
                    all default to None. So if the user does not supply a linked agent, then the
                    parent agent will ignore it.

        The new init function then creates a new AgentContext class, using the context attributes
            as its attributes. It loads in all the context items in it using the specified defaults
            and attaches it to the agent.
            After that it attaches any linked agents to the parent agent.
        """
        sig = mcs._build_init_signature(cls)
        cls.__init__ = mcs._build_init(sig)

        """
        Validate and return
        """
        _AgentConstructionValidator(cls).validate()
        return cls
