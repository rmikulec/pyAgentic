from dataclasses import dataclass, fields, replace
from typing import Annotated, TypeVar, get_origin, get_args, Any, Self

from typeguard import check_type, TypeCheckError

from pyagentic._base._context import _AgentContext, ContextRef
from pyagentic._base._exceptions import InvalidContextRefMismatchTyping


class _CtxMarker:
    pass


T = TypeVar("T")
MaybeContext = Annotated[T, _CtxMarker()]


@dataclass
class ContextualMixin:
    """
    Class to be extended if any of the properties in the class may use a `ContextRef`. Gives the
    subclass access to `resolve`, which allows a context to be passed in to backfill any
    context-ready properties
    """

    def resolve(self, ctx: _AgentContext) -> Self:
        updates: dict[str, Any] = {}
        for f in fields(self):
            tp = f.type
            # only look at Annotated[...] with our marker
            if get_origin(tp) is Annotated and any(
                isinstance(m, _CtxMarker) for m in get_args(tp)[1:]
            ):

                raw = getattr(self, f.name)
                if isinstance(raw, ContextRef):
                    value = ctx.get(raw.path)
                    # expected type is the first Annotated arg
                    expected_type = get_args(tp)[0]
                    try:
                        check_type(value, expected_type)
                    except TypeCheckError:
                        raise InvalidContextRefMismatchTyping(
                            ref_path=raw.path,
                            field_name=f.name,
                            recieved_type=type(value),
                            expected_type=expected_type,
                        )
                    updates[f.name] = value
                    # return a new instance with those fields replaced
        return replace(self, **updates)
