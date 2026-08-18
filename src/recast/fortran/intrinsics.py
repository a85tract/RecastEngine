"""Fortran intrinsic procedure names.

Extracted from the translator's ``INTRINSIC_MAP`` and ``REDUCTION_MAP``, which
paired each name with the NumPy or ``math`` call it becomes. Only the names are
a Fortran fact; what they become is a property of a target language, and moves
with the Transform that has one.

The read/write-set analysis needed exactly the membership half -- it asked "is
this name an intrinsic" three times and never once asked what it maps to -- so
splitting here removes the analysis's dependency on a 2,883-line emitter and
leaves both halves usable on their own.

Not the full F2008 intrinsic list. This is what the sources being modernized
actually call, and a name missing from it is reported as an unresolved
reference rather than silently treated as a variable read, so the failure mode
of an omission is a question rather than a wrong answer.
"""

from __future__ import annotations

ELEMENTAL = frozenset(
    {
        "abs",
        "acos",
        "adjustl",
        "aimag",
        "aint",
        "alog",
        "alog10",
        "amax0",
        "amin0",
        "anint",
        "asin",
        "atan",
        "atan2",
        "c_loc",
        "ceiling",
        "char",
        "cmplx",
        "conjg",
        "cos",
        "cosh",
        "dabs",
        "datan",
        "dble",
        "dcos",
        "dexp",
        "dim",
        "dlog",
        "dlog10",
        "dmax1",
        "dmin1",
        "dsin",
        "dsqrt",
        "epsilon",
        "erf",
        "erfc",
        "exp",
        "float",
        "floor",
        "gamma",
        "huge",
        "iabs",
        "iachar",
        "iand",
        "ichar",
        "ieor",
        "index",
        "int",
        "ior",
        "is_iostat_end",
        "ishft",
        "isign",
        "isnan",
        "kind",
        "lbound",
        "len",
        "len_trim",
        "log",
        "log10",
        "max",
        "max0",
        "min",
        "min0",
        "mod",
        "modulo",
        "mvbits",
        "nint",
        "precision",
        "real",
        "scan",
        "shape",
        "sign",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
        "tiny",
        "transfer",
        "trim",
    }
)
"""Applied per element, or an inquiry that answers about one object."""

TRANSFORMATIONAL = frozenset(
    {
        "all",
        "any",
        "count",
        "dot_product",
        "matmul",
        "maxval",
        "minval",
        "product",
        "size",
        "sum",
        "ubound",
    }
)
"""Collapse or reshape an array. Kept apart from ``ELEMENTAL`` because a
Transform has to decide their result rank, and a read/write analysis has to
know that their argument is read whole rather than at one index."""

STATE_QUERY = frozenset({"allocated", "associated", "present", "merge"})
"""Answer about a variable's status rather than its value.

``present(x)`` counts as a read of ``x`` on both sides of the cross-check --
the Fortran asks whether the argument was supplied, and the translation asks
whether it is ``None``, and a gate that saw one and not the other would report
a spurious mismatch on every optional argument.
"""

ALL = ELEMENTAL | TRANSFORMATIONAL | STATE_QUERY
"""Every name this frontend recognises as an intrinsic rather than a symbol."""
