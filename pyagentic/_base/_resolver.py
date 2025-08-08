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

    @classmethod
    def _get_maybe_context(cls):
        maybe_contexts = []
        for field in fields(cls):
            type_ = field.type
            # only look at Annotated[...] with our marker
            if get_origin(type_) is Annotated and any(
                isinstance(m, _CtxMarker) for m in get_args(type_)[1:]
            ):
                maybe_contexts.append(field)
        return maybe_contexts

    def resolve(self, ctx: _AgentContext) -> Self:
        updates: dict[str, Any] = {}
        for field in self._get_maybe_context():
            raw = getattr(self, field.name)
            if isinstance(raw, ContextRef):
                value = ctx.get(raw.path)
                # expected type is the first Annotated arg
                expected_type = get_args(field.type)[0]
                try:
                    check_type(value, expected_type)
                except TypeCheckError:
                    raise InvalidContextRefMismatchTyping(
                        ref_path=raw.path,
                        field_name=field.name,
                        recieved_type=type(value),
                        expected_type=expected_type,
                    )
                updates[field.name] = value
        # return a new instance with those fields replaced
        return replace(self, **updates)
