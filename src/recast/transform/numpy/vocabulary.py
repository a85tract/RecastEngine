"""How this backend spells Fortran.

Name tables, and nothing else. They were 210 lines at the top of a 2,883-line
module, mixed in with regexes the frontend already owns and with 131 stubs for
a framework the engine has never heard of.

The split that matters here is between the two halves of each entry. That
``sqrt`` is a Fortran intrinsic is a fact about the source and lives in
``recast.fortran.intrinsics``; that it becomes ``_f_sqrt`` is a property of
this target and lives here. The read/write analysis needed only the first half
and had to import the whole emitter to get it.

Two tables rather than one, because an intrinsic applied to an array is not
the same call as one applied to a scalar: ``exp`` of a scalar is ``math.exp``,
correctly rounded and matching the Fortran; ``exp`` of an array is ``_f_vexp``,
which loops rather than letting NumPy take a SIMD path that differs by an ULP.
"""

from __future__ import annotations

import keyword

from recast.fortran.constants import WHITELIST_INT, WHITELIST_REAL

__all__ = [
    "ARITH_OPS",
    "ARRAY_TRANSFORM",
    "ELEMENTAL_ARRAY",
    "ELEMENTAL_SCALAR",
    "LOGICAL_OPS",
    "REDUCTIONS",
    "RELATIONAL_OPS",
    "RESERVED",
    "WHITELIST_INT",
    "WHITELIST_REAL",
    "pysafe",
]

ELEMENTAL_SCALAR: dict[str, str] = {
    "abs": "abs",
    "acos": "math.acos",
    "adjustl": "_f_adjustl",
    "aimag": "np.imag",
    "aint": "np.trunc",
    "alog": "math.log",
    "alog10": "math.log10",
    "amax0": "max",
    "amin0": "min",
    "anint": "np.round",
    "asin": "math.asin",
    "atan": "math.atan",
    "atan2": "math.atan2",
    "c_loc": "_f_c_loc",
    "ceiling": "math.ceil",
    "char": "chr",
    "cmplx": "complex",
    "conjg": "np.conj",
    "cos": "math.cos",
    "cosh": "math.cosh",
    "dabs": "abs",
    "datan": "math.atan",
    "dble": "np.float64",
    "dcos": "math.cos",
    "dexp": "math.exp",
    "dim": "_f_dim",
    "dlog": "math.log",
    "dlog10": "math.log10",
    "dmax1": "max",
    "dmin1": "min",
    "dsin": "math.sin",
    "dsqrt": "_f_sqrt",
    "epsilon": "_f_epsilon",
    "erf": "math.erf",
    "erfc": "math.erfc",
    "exp": "math.exp",
    "float": "np.float64",
    "floor": "math.floor",
    "gamma": "math.gamma",
    "huge": "_f_huge",
    "iabs": "abs",
    "iachar": "ord",
    "iand": "_f_iand",
    "ichar": "ord",
    "ieor": "_f_ieor",
    "index": "_f_index",
    "int": "int",
    "ior": "_f_ior",
    "is_iostat_end": "_f_is_iostat_end",
    "isign": "_f_sign",
    "isnan": "np.isnan",
    "is_nan": "np.isnan",
    "ishft": "_f_ishft",
    "kind": "_f_kind",
    "lbound": "_f_lbound",
    "len": "len",
    "len_trim": "_f_len_trim",
    "log": "math.log",
    "log10": "math.log10",
    "max": "_f_max",
    "max0": "max",
    "min": "_f_min",
    "min0": "min",
    "mod": "_f_mod",
    "modulo": "_f_modulo",
    "mvbits": "_f_mvbits",
    "nint": "_f_nint",
    "precision": "_f_precision",
    "real": "np.float64",
    "scan": "_f_scan",
    "shape": "np.shape",
    "sign": "_f_sign",
    "sin": "math.sin",
    "sinh": "math.sinh",
    "sqrt": "_f_sqrt",
    "tan": "math.tan",
    "tanh": "math.tanh",
    "tiny": "_f_tiny",
    "transfer": "_f_transfer",
    "trim": "_f_trim",
}
"""Intrinsic -> what to call on a scalar argument.

Where the entry is a ``_f_`` name, the plain Python spelling is wrong: ``mod``
is not ``%``, ``nint`` is not ``round``, ``sign`` is not ``copysign``. See
``runtime``.
"""

ELEMENTAL_ARRAY: dict[str, str] = {
    "abs": "np.abs",
    "aint": "np.trunc",
    "anint": "np.round",
    "ceiling": "_f_vceil",
    "cos": "np.cos",
    "dble": "np.float64",
    "erf": "_f_verf",
    "erfc": "_f_verfc",
    "exp": "_f_vexp",
    "float": "np.float64",
    "floor": "_f_vfloor",
    "isnan": "np.isnan",
    "is_nan": "np.isnan",
    "log": "_f_vlog",
    "log10": "_f_vlog10",
    "max": "_f_vmax",
    "min": "_f_vmin",
    "real": "np.float64",
    "sign": "np.copysign",
    "sin": "np.sin",
    "sqrt": "np.sqrt",
    "tanh": "np.tanh",
}
"""Intrinsic -> what to call on an array argument.

``exp``, ``log`` and ``log10`` route through the runtime rather than NumPy:
npy_math takes a SIMD path that differs from glibc by an ULP, and an ULP is
the difference between a bit-exact gate passing and failing.
"""

REDUCTIONS: dict[str, str] = {
    "all": "np.all",
    "any": "np.any",
    "count": "np.count_nonzero",
    "dot_product": "_f_vdot",
    "matmul": "np.matmul",
    "maxval": "np.max",
    "minval": "np.min",
    "product": "np.prod",
    "size": "np.size",
    "sum": "_f_vsum",
    # With unit lower bounds -- which every translated array has -- the upper
    # bound and the extent are the same number.
    "ubound": "np.size",
}
"""Intrinsics that collapse an array. ``dot_product`` and ``sum`` are runtime
shims because NumPy's are pairwise or BLAS and round differently from a
left-to-right sum: ``np.sum`` over ELM's ten soil layers put the hydraulic
kernel 75 ULP off the recording, and a sequential fold put it at zero."""

ARRAY_TRANSFORM: frozenset[str] = frozenset(
    {"cshift", "eoshift", "maxloc", "minloc", "pack", "reshape", "spread", "transpose", "unpack"}
)
"""Intrinsics that reshape rather than collapse. Named as a set because each
one's emission depends on its arguments, so there is no single spelling."""

RELATIONAL_OPS: dict[str, str] = {
    "==": "==",
    ".EQ.": "==",
    "/=": "!=",
    ".NE.": "!=",
    "<": "<",
    ".LT.": "<",
    "<=": "<=",
    ".LE.": "<=",
    ">": ">",
    ".GT.": ">",
    ">=": ">=",
    ".GE.": ">=",
}
"""Both spellings of every comparison. Fortran kept the F77 forms and real code uses
them interchangeably, sometimes in the same expression."""

LOGICAL_OPS: dict[str, str] = {
    ".AND.": "and",
    ".OR.": "or",
    ".EQV.": "==",
    ".NEQV.": "!=",
}

ARITH_OPS: dict[str, str] = {"+": "+", "-": "-", "*": "*", "/": "/", "**": "**"}
"""Spelled identically, and listed anyway: an operator absent from this table
is one the emitter has no rule for, which is the answer it needs."""

RESERVED: frozenset[str] = frozenset({"_re", "copy", "math", "mp", "np", "os"})
"""Module aliases the emitted file uses itself.

A Fortran dummy argument named ``np`` would shadow NumPy in the translation,
so it is renamed. Declared here rather than assumed by the verifier: the
mangling is this backend's, and a backend that spelled it differently would
otherwise be reported as a mismatch on every affected variable.
"""


def pysafe(name: str) -> str:
    """Rename a Fortran symbol that would collide in the emitted file.

    A trailing underscore, PEP 8's convention, for Python keywords -- Fortran
    has variables called ``in``, ``is`` and ``lambda`` -- and for the module
    aliases in ``RESERVED``. The read/write cross-check strips it back, which
    is why the two have to agree on exactly this rule.
    """
    return name + "_" if keyword.iskeyword(name) or name in RESERVED else name
