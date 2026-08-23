"""The seam for a call whose meaning is a framework's, not the language's.

Some calls are neither translatable nor stubbable. ``call phys_getopts(
a_out=x)`` reads a namelist option into ``x``: the translation has to emit
an assignment, from somewhere, and what that somewhere is is a fact about
the framework the source was written against. A fixed-string stub cannot
say it, because the answer depends on the call's own arguments.

So the engine takes callables. A domain package supplies
``config["call_transforms"] = {name: transform}``; the transform is handed
a ``CallSite`` and returns the lines to emit, or raises
``recast.transform.rules.NoRule`` to refuse the way any rule does. The
engine knows the shape and nothing else -- which frameworks exist, what
their calls mean, and what to call in their place all stay outside it.

Consulted before the stub tables and before any resolution against this
module or its companions, which is the order the pipeline this came from
uses: a call listed here is answered here.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

__all__ = ["CallSite", "CallTransform"]


@dataclass(frozen=True)
class CallSite:
    """One ``call name(...)``, with what a transform needs to answer it."""

    name: str
    """The callee, lower-cased."""

    indent: int
    """Nesting level of the statement; ``pad`` is the spelling of it."""

    actuals: Sequence[Any]
    """Every actual in source order, keyword ones included, as parsed nodes.

    Positional indexing is deliberate: these calls are usually specified by
    position ("argument 4 is the output"), and Fortran lets a keyword appear
    among them without changing which position an argument is at.
    """

    keywords: Mapping[str, Any]
    """The keyword actuals by name, lower-cased, as parsed nodes."""

    render: Callable[[Any], str]
    """A parsed node -> the expression this backend spells it as."""

    holds_handle: Callable[[str], None]
    """Say that an emitted name now holds an opaque handle.

    A framework that hands out registrations gives Fortran an integer index,
    and Fortran tests it with ``idx > 0`` for "is it registered". A
    transform that assigns something other than an index -- a dictionary
    key, a name -- says so here, and the test comes out as the presence
    question it is rather than as arithmetic on a string.
    """

    @property
    def pad(self) -> str:
        return "    " * self.indent

    @property
    def positional(self) -> tuple[Any, ...]:
        """The actuals written without a keyword, in order.

        Several of these calls are told apart by how many they have -- the
        same name means three different things at one, two and three.
        """
        from recast.fortran._parse import f03

        return tuple(a for a in self.actuals if not isinstance(a, f03.Actual_Arg_Spec))

    def value(self, position: int) -> str:
        """Actual at ``position``, rendered, keyword or not."""
        return self.render(self.node(position))

    @staticmethod
    def bare(actual: Any) -> Any:
        """An actual with its keyword stripped, if it has one."""
        from recast.fortran._parse import f03

        if isinstance(actual, f03.Actual_Arg_Spec):
            return actual.children[1]
        return actual

    def node(self, position: int) -> Any:
        return self.bare(self.actuals[position])


class CallTransform(Protocol):
    """What a domain package registers against a callee name."""

    def __call__(self, site: CallSite) -> list[str]:
        """The lines to emit, or raise ``NoRule`` to refuse."""
        ...
