import inspect
from typing import dataclass_transform, TypeVar
from typeguard import check_type

from pyagentic._base._exceptions import SystemMessageNotDeclared, UnexpectedContextItemType
from pyagentic._base._context import _AgentContext, ContextItem, computed_context
from pyagentic._base._tool import _ToolDefinition


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

        def __init__(self, *args, **kwargs):  # type: ignore
            ContextClass = _AgentContext.make_ctx_class(
                name=self.__class__.__name__, ctx_map=self.__context_attrs__
            )

            context_kwargs = {}
            for attr_name, (attr_type, attr_default) in self.__context_attrs__.items():
                if attr_type == computed_context:
                    continue
                if attr_name in kwargs:
                    val = kwargs[attr_name]
                    if (not check_type(val, attr_type)):
                        raise UnexpectedContextItemType(
                            name=attr_name, expected=attr_type, recieved=type(val)
                        )
                    context_kwargs[attr_name] = val
                else:
                    context_kwargs[attr_name] = attr_default.get_default_value()

            self.context = ContextClass(
                instructions=self.__system_message__,
                input_template=self.__input_template__,
                **context_kwargs,
            )

            bound = sig.bind(self, *args, **kwargs)
            for name, val in list(bound.arguments.items())[1:]:  # skip 'self'
                if name in self.__context_attrs__:
                    pass
                setattr(self, name, val)

            self.__post_init__()

        __init__.__signature__ = sig  # type: ignore
        return __init__

    def __new__(mcs, name, bases, namespace, **kwargs):
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        if namespace.get("__abstract_base__", False):
            return cls

        if "__system_message__" not in namespace:
            raise SystemMessageNotDeclared()

        cls.__tool_defs__ = mcs._extract_tool_defs(namespace)

        cls.__annotations__ = mcs._extract_annotations(bases, namespace)

        cls.__context_attrs__ = mcs._extract_context_attrs(cls.__annotations__, namespace)

        sig = mcs._build_init_signature(cls)

        cls.__init__ = mcs._build_init(sig)
        return cls
