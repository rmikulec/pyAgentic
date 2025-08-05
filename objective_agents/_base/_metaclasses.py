import inspect
from typing import dataclass_transform

from objective_agents._base._exceptions import SystemMessageNotDeclared, UnexpectedContextItemType
from objective_agents._base._context import AgentContext, ContextItem


@dataclass_transform(field_specifiers=(ContextItem,))
class AgentMeta(type):
    """
    Metaclass that applies only to Agent subclasses:
      - Ensures @system_message was declared
      - Collects @tool methods and ContextItem attributes
      - Initializes class __tools__ and __context__
      - Dynamically injects an __init__ signature based on class __annotations__
    """

    @staticmethod
    def _extract_tool_defs(namespace):
        tools = {}
        for attr_name, attr_value in namespace.items():
            if hasattr(attr_value, "__tool_def__"):
                tools[attr_name] = attr_value.__tool_def__
        return tools

    @staticmethod
    def _extract_annotations(bases, namespace):
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
    def _extract_context_attrs(annotations, namespace):
        context_attrs = {}
        for attr_name, attr_type in annotations.items():
            default = namespace.get(attr_name, None)
            if isinstance(default, ContextItem):
                context_attrs[attr_name] = (attr_type, default)
        return context_attrs

    @staticmethod
    def _build_init_signature(cls):
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
        def __init__(self, *args, **kwargs):  # type: ignore
            self.context = AgentContext(self.__system_message__)

            for attr_name, (attr_type, attr_default) in self.__context_attrs__.items():
                if attr_name in kwargs:
                    val = kwargs[attr_name]
                    if (not isinstance(val, attr_type)) or not (
                        issubclass(val.__class__, attr_type)
                    ):
                        raise UnexpectedContextItemType(
                            name=attr_name, expected=attr_type, recieved=type(val)
                        )
                    self.context.add(attr_name, val)
                else:
                    self.context.add(attr_name, attr_default.get_default_value())

            bound = sig.bind(self, *args, **kwargs)
            for name, val in list(bound.arguments.items())[1:]:  # skip 'self'
                if name in self.__context_attrs__:
                    pass
                setattr(self, name, val)

            self.__post_init__()

        # Attach the signature for IDEs and checkers
        __init__.__signature__ = sig  # type: ignore
        return __init__

    def __new__(mcs, name, bases, namespace, **kwargs):
        # Create the class first (so Agent exists)
        cls = super().__new__(mcs, name, bases, namespace, **kwargs)

        # Skip generation on the abstract base itself
        if namespace.get("__abstract_base__", False):
            return cls

        # 1. Validate system_message decorator
        if "__system_message__" not in namespace:
            raise SystemMessageNotDeclared()

        cls.__tool_defs__ = mcs._extract_tool_defs(namespace)

        cls.__annotations__ = mcs._extract_annotations(bases, namespace)

        cls.__context_attrs__ = mcs._extract_context_attrs(cls.__annotations__, namespace)

        sig = mcs._build_init_signature(cls)
        # 5. Dynamically build __init__ signature

        cls.__init__ = mcs._build_init(sig)
        return cls
