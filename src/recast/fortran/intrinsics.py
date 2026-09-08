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
        "achar",
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
        "is_nan",
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
        "radix",
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
        # Collapse an array to a scalar or a lower rank.
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
        # Reshape a whole array rather than collapse it; the result's rank
        # is the emitter's business (``vocabulary.ARRAY_TRANSFORM`` spells
        # each one by its arguments). They were kept out of this set on
        # purpose while no bit-exact gate had run over a use of one; ELM's
        # PhotosynthesisMod (readParams reshapes its parameter tables) is
        # that run, 93 blocks matching with them counted as intrinsics.
        "reshape",
        "spread",
        "pack",
        "unpack",
        "transpose",
        "cshift",
        "eoshift",
        "maxloc",
        "minloc",
        "lbound",
    }
)
"""Operate on an array as a whole rather than per element.

Kept apart from ``ELEMENTAL`` because a Transform has to work out their result
rank, and a read/write analysis has to know their argument is read entire
rather than at one index.
"""

LOCATION = frozenset({"maxloc", "minloc"})
"""Report *where* an array's extreme value is, not what it is.

Kept apart from ``TRANSFORMATIONAL`` because their result is a rank-1 position
vector unless DIM is given, so this is not a set a rank query may answer 0
for; the read/write analysis needs only the membership. Named here because
``minloc(a2(i:), 1)`` is a call: counting the name as a variable read makes
every block holding one disagree with a translation that spells it
``np.argmin``, which is what failed ``iargsort`` and ``rargsort`` of the
corpus's sorting module.
"""

RESHAPING = frozenset({"spread"})
"""Rearrange an array into another array rather than collapsing it.

Kept apart from ``TRANSFORMATIONAL`` for the reason ``LOCATION`` is: the
result is an array, so this is not a set a rank query may answer 0 for, and
the read/write analysis needs only the membership. Named here because
``spread(x, 1, size(y))`` is a call: counting the name as a variable read
makes every block holding one disagree with a translation that spells it
``np.repeat``, which is what failed both blocks of ``meshgrid`` in the
corpus's mesh module.

Only ``spread``, for the same reason ``LOCATION`` holds only the two
locators: the emitter reshapes with ``cshift``, ``eoshift``, ``pack``,
``reshape``, ``transpose`` and ``unpack`` as well, and those sites are still
the divergence this frontend deliberately keeps -- the read as a variable is
the answer a bit-exact gate has been run against, and no translation has yet
been checked against the tidier one. A name moves here when one is.
"""

STATE_QUERY = frozenset({"allocated", "associated", "present", "merge"})
"""Answer about a variable's status rather than its value.

``present(x)`` counts as a read of ``x`` on both sides of the cross-check --
the Fortran asks whether the argument was supplied, and the translation asks
whether it is ``None``, and a gate that saw one and not the other would report
a spurious mismatch on every optional argument.
"""

SUBROUTINE = frozenset(
    {
        "cpu_time",
        "date_and_time",
        "execute_command_line",
        "get_command",
        "get_command_argument",
        "get_environment_variable",
        "move_alloc",
        "mvbits",
        "random_number",
        "random_seed",
        "system_clock",
    }
)
"""Intrinsics invoked by ``call``, not in an expression.

Kept apart because every one of them *writes* an argument, which is what
makes them the wrong thing to stub away: the pipeline this was migrated from
renders ``call random_number(x)`` as ``pass``, and ``x`` then keeps whatever
it held while the read/write gate is told nothing happened. Naming them here
lets a call to one refuse as the intrinsic it is rather than as somebody
else's missing library.
"""

ALL = ELEMENTAL | TRANSFORMATIONAL | STATE_QUERY | LOCATION | RESHAPING
"""Every name this frontend recognises as an intrinsic rather than a symbol."""
