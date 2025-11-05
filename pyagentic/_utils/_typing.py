from typing import get_origin, get_args, Any, Optional, ForwardRef
from dataclasses import dataclass
from enum import Enum


PRIMITIVES = (bool, str, int, float, type(None))


def is_primitive(type_: Any) -> bool:
    """
    Helper function to check if a type is a Python primitive.

    Args:
        type_ (Any): The type to check

    Returns:
        bool: True if the type is a primitive, False otherwise
    """
    return type_ in PRIMITIVES


class TypeCategory(Enum):
    """
    Enumeration of type categories for parameter analysis.
    """
    PRIMITIVE = "primitive"
    LIST_PRIMITIVE = "list_primitive"
    SUBCLASS = "subclass"
    LIST_SUBCLASS = "list_subclass"
    UNSUPPORTED = "unsupported"


@dataclass
class TypeInfo:
    """
    Normalized information about a type, including category and inner types.
    """

    category: TypeCategory
    base_type: type
    inner_type: Optional[type] = None  # For list types

    @property
    def is_list(self) -> bool:
        """
        Returns whether the type is a list type.

        Returns:
            bool: True if the category is a list type
        """
        return self.category in [TypeCategory.LIST_PRIMITIVE, TypeCategory.LIST_SUBCLASS]

    @property
    def is_subclass(self) -> bool:
        """
        Returns whether the type is a subclass of a base class.

        Returns:
            bool: True if the category is a subclass type
        """
        return self.category in [TypeCategory.SUBCLASS, TypeCategory.LIST_SUBCLASS]

    @property
    def effective_type(self) -> type:
        """
        Returns the type to work with (inner type for lists, base type otherwise).
        """
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
    Analyzes a type and returns normalized information about it.

    Args:
        type_: The type to analyze
        base_class: Base class for checking subclass relationships
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
