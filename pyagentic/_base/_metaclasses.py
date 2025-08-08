import inspect
from typing import dataclass_transform, TypeVar
from typeguard import check_type, TypeCheckError

from pyagentic._base._validation import _AgentConstructionValidator
from pyagentic._base._exceptions import SystemMessageNotDeclared, UnexpectedContextItemType
from pyagentic._base._context import _AgentContext, ContextItem, computed_context
from pyagentic._base._tool import _ToolDefinition

from pyagentic.models.response import AgentResponse, ToolResponse

from pyagentic._utils._typing import analyze_type


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

    @staticmethod
    def _extract_tool_defs(namespace) -> dict[str, _ToolDefinition]:
        """
        Extracts tool definitions from a given namespace

        Any method with the `@tool` descriptor will be attached to the `__tool_defs__` class
            attribute
        """
        tools = {}
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, "__tool_def__"):
                tools[attr_name] = attr_value.__tool_def__
        return tools

    @staticmethod
    def _extract_annotations(bases, namespace) -> dict[str, TypeVar]:
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
        context_attrs = {}
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
        linked_agents = {}
        for attr_name, attr_type in annotations.items():
            type_info = analyze_type(attr_type, Agent)
            if type_info.is_subclass:
                linked_agents[attr_name] = attr_type

        return linked_agents

    @staticmethod
    def _build_init_signature(cls) -> inspect.Signature:
        """
        Builds the signature for the classes __init__, injecting any context item attributes
            defined by the user properly into the inits signature. This allows IDE's to be able
            to recognize user-defined class attributes when initializing a class.
        """
        params = [inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)]
        for field_name, field_type in cls.__annotations__.items():
            if field_name in cls.__context_attrs__:
                default_val = cls.__context_attrs__[field_name][1].get_default_value()
            elif field_name in cls.__linked_agents__:
                default_val = None
            else:
                default_val = getattr(cls, field_name, inspect._empty)
            param = inspect.Parameter(
                field_name,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=default_val,
                annotation=field_type,
            )
            params.append(param)
        return inspect.Signature(params)

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

                # Not sure if there is a better way, but setting model = "validation"
                #   bypasses checking agent in args. This is so the Validator is able to create
                #   a dummy agent wihout having to worry about creating a chain of linked agents
                if not agent_instance and kwargs["model"] != "validation":
                    raise AttributeError(f"Linked Agent {agent_name} not found")
                compiled[agent_name] = agent_instance

            bound = sig.bind(self, *args, **kwargs)

            # Add all other arguements to instance
            for name, val in list(bound.arguments.items())[1:]:  # skip 'self'
                if name in self.__context_attrs__:
                    pass
                setattr(self, name, val)

            self.__post_init__()

        __init__.__signature__ = sig  # type: ignore
        return __init__

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create a new Agent class
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)
        # If it is a base Agent, then return
        if namespace.get("__abstract_base__", False):
            return cls
        # Verify system message is set
        if "__system_message__" not in namespace:
            raise SystemMessageNotDeclared()
        # Attach tool definitions
        cls.__tool_defs__ = mcs._extract_tool_defs(namespace)
        # Attach new annotations
        cls.__annotations__ = mcs._extract_annotations(bases, namespace)
        # Attach context attributes (ContextItems and computed_context)
        cls.__context_attrs__ = mcs._extract_context_attrs(cls.__annotations__, namespace)
        # Create tool response models
        cls.__tool_response_models__ = {
            tool_name: ToolResponse.from_tool_def(tool_def)
            for tool_name, tool_def in cls.__tool_defs__.items()
        }
        # Attach linked agents
        cls.__linked_agents__ = mcs._extract_linked_agents(cls.__annotations__, cls.__bases__[0])
        # Create final Agent response model, using the tool response models
        tool_response_models = list(cls.__tool_response_models__.values())
        linked_agent_response_models = [
            agent.__response_model__ for agent in cls.__linked_agents__.values()
        ]
        cls.__response_model__ = AgentResponse.from_tool_defs(
            agent_name=cls.__name__,
            tool_response_models=tool_response_models,
            linked_agents_response_models=linked_agent_response_models,
        )
        # Build the new init
        sig = mcs._build_init_signature(cls)
        cls.__init__ = mcs._build_init(sig)

        # Validate agent
        _AgentConstructionValidator(cls).validate()
        return cls
