import warnings
import functools


def deprecated(reason: str = "", since: str | None = None, removed: str | None = None):
    """
    Decorator to mark functions or classes as deprecated.
    Emits a DeprecationWarning when used.

    Example:
        @deprecated("Use `new_func` instead.", since="0.10", removed="0.12")
        def old_func(...): ...
    """

    def decorator(obj):
        message = f"{obj.__name__} is deprecated"
        if since:
            message += f" since {since}"
        if reason:
            message += f"; {reason}"
        if removed:
            message += f". It will be removed in {removed}."
        else:
            message += "."

        if isinstance(obj, type):  # decorating a class
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return orig_init(self, *args, **kwargs)

            obj.__init__ = new_init
            return obj

        else:  # decorating a function

            @functools.wraps(obj)
            def wrapper(*args, **kwargs):
                warnings.warn(message, DeprecationWarning, stacklevel=2)
                return obj(*args, **kwargs)

            return wrapper

    return decorator
