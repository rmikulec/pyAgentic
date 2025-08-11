from typing import get_origin, get_args, Any, Optional, ForwardRef
from dataclasses import dataclass
from enum import Enum


PRIMITIVES = (bool, str, int, float, type(None))


def is_primitive(type_: Any) -> bool:
    """
    Helper function to check if a type is a python primitive
    """
    return type_ in PRIMITIVES


class TypeCategory(Enum):
    PRIMITIVE = "primitive"
    LIST_PRIMITIVE = "list_primitive"
    SUBCLASS = "subclass"
    LIST_SUBCLASS = "list_subclass"
    UNSUPPORTED = "unsupported"


@dataclass
class TypeInfo:
    """Normalized information about a type"""

    category: TypeCategory
    base_type: type
    inner_type: Optional[type] = None  # For list types

    @property
    def is_list(self) -> bool:
        return self.category in [TypeCategory.LIST_PRIMITIVE, TypeCategory.LIST_SUBCLASS]

    @property
    def is_subclass(self) -> bool:
        return self.category in [TypeCategory.SUBCLASS, TypeCategory.LIST_SUBCLASS]

    @property
    def effective_type(self) -> type:
        """Returns the type to work with (inner type for lists, base type otherwise)"""
        return self.inner_type if self.is_list else self.base_type

    @property
    def has_forward_ref(self) -> bool:
        """
        True if either base_type or inner_type is a forward reference
        (e.g., a string annotation or typing.ForwardRef). Safe across Python versions.
        """

        def _is_forward_ref(t) -> bool:
            if t is None:
                return False
            # Strings from deferred annotations / forward refs
            if isinstance(t, str):
                return True
            # typing.ForwardRef in 3.8+ (internal shape has varied, so duck-type too)
            if isinstance(t, ForwardRef):
                return True
            # Fallback: anything that looks like a ForwardRef (duck-typing)
            return hasattr(t, "__forward_arg__")  # covers older/private ForwardRef variants

        return _is_forward_ref(self.base_type) or _is_forward_ref(self.inner_type)


def analyze_type(type_: type, base_class: type) -> TypeInfo:
    """
    Analyze a type and return normalized information about it.

    Args:
        type_: The type to analyze
        is_primitive_func: Function to check if a type is primitive
        param_base_class: Base class for Param types (e.g., Param)
    """
    origin = get_origin(type_)

    try:
        if origin == list:
            inner_type = get_args(type_)[0]
            if is_primitive(inner_type):
                return TypeInfo(TypeCategory.LIST_PRIMITIVE, type_, inner_type)
            elif issubclass(inner_type, base_class):
                return TypeInfo(TypeCategory.LIST_SUBCLASS, type_, inner_type)
            else:
                return TypeInfo(TypeCategory.UNSUPPORTED, type_, inner_type)

        elif is_primitive(type_):
            return TypeInfo(TypeCategory.PRIMITIVE, type_)

        elif issubclass(type_, base_class):
            return TypeInfo(TypeCategory.SUBCLASS, type_)

        else:
            return TypeInfo(TypeCategory.UNSUPPORTED, type_)
    except TypeError:
        return TypeInfo(TypeCategory.UNSUPPORTED, type_)
