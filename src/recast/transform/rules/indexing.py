"""Fortran subscripts, described once and shifted once.

The pipeline this came from worked out what a subscript meant and emitted the
Python for it in the same pass, so the two could not be told apart and neither
could be tested without an emitter. Splitting them costs nothing: a plan names
the positions and the amount each one shifts by, and refers to the source
nodes rather than rendering them.

Three rules live here, and each exists because of a way the naive shift is
wrong.

*An inclusive range becomes an exclusive one.* Fortran's ``a(lo:hi)`` includes
``hi``; a zero-based slice stops before its bound. Subtracting the origin from
both ends and forgetting the ``+1`` loses the last element of every slice --
which changes an answer without changing a shape, so nothing downstream trips
on it.

*A literal only folds while it stays whitelisted.* ``a(3)`` may become
``a[2]``, but ``a(17)`` must not become ``a[16]``: the zero-literal rule says
every non-structural number in the output is a named constant, and quietly
introducing 16 in a subscript would put back exactly what hoisting removed.

*A shift off a declared lower bound is not a shift by one.* ``real :: q(0:n)``
indexes from zero already, and ``q(i)`` is ``q[i]``, not ``q[i-1]``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from recast.fortran._parse import f03
from recast.transform.rules import NoRule

UNIT_ORIGIN = "1"
"""What a dimension's lower bound is when the declaration does not say."""


class Kind(StrEnum):
    """What one subscript position is."""

    INDEX = "index"
    """A single element. Drops a rank."""

    RANGE = "range"
    """A triplet ``lo:hi`` or ``lo:hi:step``. Keeps a rank."""

    VECTOR = "vector"
    """An integer array used as a subscript. Keeps a rank, gathers."""


@dataclass(frozen=True)
class Position:
    """One subscript position, described but not yet written down."""

    kind: Kind
    origin: str = UNIT_ORIGIN
    """The dimension's declared lower bound, as it appeared in the source.

    ``"1"`` for the usual case. Anything else is text the backend has to
    render, because a bound may be an expression over other arguments.
    """

    index: Any | None = None
    """``Kind.INDEX`` and ``Kind.VECTOR``: the subscript expression."""

    lower: Any | None = None
    upper: Any | None = None
    step: Any | None = None
    """``Kind.RANGE``: the three parts, any of which the source may omit."""

    @property
    def shifts_by_one(self) -> bool:
        """Whether the shift is the plain one, which most callers can inline."""
        return self.origin == UNIT_ORIGIN


def describe(
    arglist: Any,
    dims: list[dict[str, Any]] | None,
    *,
    rank_of: Callable[[Any], int],
) -> tuple[Position, ...]:
    """Describe a subscript list against the dimensions it indexes.

    ``rank_of`` decides a vector subscript from a scalar one -- an integer
    array used as a subscript looks exactly like a scalar until something can
    say what rank it has. Passing it in rather than taking a ``Semantics``
    keeps this readable by anything that can answer the one question.

    Refuses rather than approximating, in the three cases where Fortran can say
    something a zero-based slice cannot say back.
    """
    items = _items(arglist)
    positions = []
    for axis, item in enumerate(items):
        origin = _origin(dims, axis)
        if isinstance(item, f03.Subscript_Triplet):
            positions.append(_range(item, origin))
        elif isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)):
            raise NoRule("a keyword argument is not a subscript")
        elif rank_of(item) > 0:
            if origin != UNIT_ORIGIN:
                raise NoRule(
                    f"vector subscript on a dimension based at {origin!r}; "
                    "gathering and re-basing at once has no single form"
                )
            positions.append(Position(Kind.VECTOR, origin, index=item))
        else:
            positions.append(Position(Kind.INDEX, origin, index=item))
    return tuple(positions)


def _items(arglist: Any) -> list[Any]:
    if arglist is None:
        return []
    return list(arglist.children) if hasattr(arglist, "children") else [arglist]


def _origin(dims: list[dict[str, Any]] | None, axis: int) -> str:
    if dims is None or axis >= len(dims):
        return UNIT_ORIGIN
    lower = dims[axis].get("lb")
    if lower in (None, "", ":"):
        return UNIT_ORIGIN
    return str(lower)


def _range(triplet: Any, origin: str) -> Position:
    lower, upper, step = triplet.children
    if step is not None and _is_negative(step):
        # Counting down, the loop stops *below* its Fortran end, and the
        # zero-based stop for reaching element 1 is off the end of the axis --
        # there is no index that means "one before the start". Expressible
        # only when both ends are written out and the axis starts at one.
        if lower is None or upper is None or origin != UNIT_ORIGIN:
            raise NoRule("negative stride with an implied or re-based bound")
    return Position(Kind.RANGE, origin, lower=lower, upper=upper, step=step)


def _is_negative(step: Any) -> bool:
    text = str(step).replace(" ", "").lstrip("(")
    return text.startswith("-")


def fold(position: Position, whitelist: frozenset[str]) -> int | None:
    """The zero-based value of a literal subscript, when writing it is allowed.

    ``None`` means the backend has to emit the shift rather than its result:
    either the subscript is not a literal, or folding it would put a number in
    the output that the zero-literal rule had just taken out. ``a(3)`` folds to
    ``2``; ``a(17)`` does not fold, because 16 is not a constant anyone named.
    """
    if position.kind is not Kind.INDEX or not position.shifts_by_one:
        return None
    if not isinstance(position.index, f03.Int_Literal_Constant):
        return None
    folded = int(str(position.index).split("_")[0]) - 1
    return folded if str(folded) in whitelist else None


@dataclass(frozen=True)
class Association:
    """How a scalar element actual reaches an array formal.

    Fortran lets ``call f(arr(1, k))`` bind to a formal declared ``real :: a(:)``
    -- the callee sees contiguous memory starting at that element. There is no
    such thing in an array language, so the actual has to be rewritten, and how
    depends on where the element sits.
    """

    whole_leading_axes: int
    """How many leading axes the formal takes entire.

    Non-zero only when the element sits at the lower bound of those axes, which
    is the common case (``arr(1, k)``) and the one with a cheap answer: take
    the whole of the leading axes and index the rest. Zero means the general
    form -- flatten in column-major order, offset, reshape -- which is correct
    everywhere and worth avoiding where it is not needed.
    """

    trailing: tuple[Any, ...] = ()
    """The subscripts of the axes the formal does not span."""


def associate(
    subscripts: list[Any],
    actual_dims: list[dict[str, Any]],
    formal_rank: int,
) -> Association:
    """Work out how a sequence-associated actual maps onto its formal.

    Refuses on a rank mismatch rather than guessing which axes line up: a wrong
    answer here passes the callee a differently shaped view of the right memory,
    which produces numbers rather than an error.
    """
    if len(subscripts) != len(actual_dims):
        raise NoRule("sequence association with a subscript per dimension mismatch")
    if formal_rank > len(actual_dims):
        raise NoRule("sequence association onto a formal of higher rank")

    first_scalar = None
    for axis, item in enumerate(subscripts):
        if isinstance(item, f03.Subscript_Triplet):
            first_scalar = None
        elif first_scalar is None:
            first_scalar = axis
    if first_scalar is None:
        raise NoRule("sequence association with no scalar subscript")

    at_lower_bound = (
        first_scalar == 0
        and isinstance(subscripts[0], f03.Int_Literal_Constant)
        and str(subscripts[0]).split("_")[0] == str(actual_dims[0].get("lb", UNIT_ORIGIN))
    )
    if at_lower_bound and formal_rank <= len(actual_dims) - first_scalar:
        return Association(
            whole_leading_axes=formal_rank,
            trailing=tuple(subscripts[formal_rank:]),
        )
    return Association(whole_leading_axes=0, trailing=tuple(subscripts))
