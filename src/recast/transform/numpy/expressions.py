"""Fortran expressions, written as NumPy.

The layer where the four earlier slices meet: ``semantics`` says what an
expression is, ``vocabulary`` says how this backend spells it, ``runtime``
supplies the shims for the places the plain spelling is wrong, and
``rules.indexing`` says what happens to a subscript. Nothing here re-derives
any of that.

Most of it is unremarkable -- an operator becomes an operator, a call becomes
a call. What is worth reading is where it is not, and every one of those is a
place where the obvious translation runs and returns the wrong number:

* ``/`` between two integers truncates in Fortran and floors in Python.
* ``x**3`` is expanded to repeated multiplication or left as a ``pow`` call
  depending on which compiler produced the reference binary, and the two do
  not round identically.
* An intrinsic over constant arguments was evaluated while the reference was
  being compiled, at a precision no run-time library reproduces.
* ``min`` and ``max`` fold left, and their NaN behaviour is asymmetric.
* An intrinsic applied to an array is a different call from the same intrinsic
  applied to a scalar, because NumPy's array path differs from libm by an ULP.

Anything without a rule raises ``NoRule`` rather than being approximated. The
Transform turns that into a deferred site, which is a normal result: a partial
Candidate with an honest list of what it could not do is what the agent layer
consumes next.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03, walk
from recast.fortran.frontend import INTRINSIC_MODULES
from recast.fortran.interface import CONFLICTING_BOUNDS, emit_name
from recast.fortran.intrinsics import STANDARD_FUNCTIONS
from recast.fortran.rwset import captured, kept_from_entry
from recast.fortran.semantics import F77_SPECIFIC_TO_GENERIC, Semantics, Unanalyzable
from recast.transform.numpy.names import USE_STATEMENT, Names
from recast.transform.numpy.vocabulary import (
    ARITH_OPS,
    ARRAY_TRANSFORM,
    ELEMENTAL_ARRAY,
    ELEMENTAL_SCALAR,
    LOGICAL_OPS,
    REDUCTIONS,
    RELATIONAL_OPS,
    WHITELIST_INT,
    pysafe,
)
from recast.transform.profiles import Profile
from recast.transform.rules import NoRule, indexing
from recast.transform.rules.indexing import Kind

DERIVED_TYPE = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)", re.IGNORECASE)
"""A dummy or module variable of derived type, as the frontend spells its dtype."""

EXTENT = re.compile(r"(?:SIZE|UBOUND)\(\s*(\w+)\s*(?:,\s*((?:dim\s*=\s*)?\d+)\s*)?\)", re.I)
"""``size(a)``, ``size(a, 2)``, ``ubound(a, dim=2)`` inside a declared bound."""

DIM_KEYWORD = re.compile(r"dim\s*=\s*", re.I)

BOUND_TOKENS = re.compile(
    rf"{EXTENT.pattern}|[A-Za-z_]\w*\s*%\s*[A-Za-z_]\w*|[A-Za-z_]\w*|\d+|[()+\-*/, ]",
    re.I,
)
"""What a declared bound is allowed to be made of. Bound texts are simple by
construction; anything richer refuses the statement that needed the bound.

``SIZE``/``UBOUND`` leads the alternation because an inquiry is one token
here, comma and all: ``2*size(c,2)`` is arithmetic *over* an extent, and a
tokenizer that took ``size`` for a plain name would stop at the comma it is
not allowed to contain."""

__all__ = ["REFUSED", "Expressions", "Remote", "function_outputs"]

REFUSED = (NoRule, Unanalyzable)
"""The two ways a rule declines: no rule for the construct, or the semantics
layer could not answer a question the rule needed answered."""

CONSTANT_FOLDED = frozenset(
    {
        "acos",
        "asin",
        "atan",
        "cos",
        "cosh",
        "exp",
        "gamma",
        "log",
        "log10",
        "sin",
        "sinh",
        "sqrt",
        "tan",
        "tanh",
    }
)
"""Intrinsics a compiler evaluates while compiling, given constant arguments.

Correctly rounded there, and therefore matching neither libgfortran nor glibc
at run time -- ``gamma(1.8)`` differs from both. Only consulted under a profile
that says the reference compiler does this.
"""

KIND_CONVERSIONS = frozenset(
    {
        "aint",
        "anint",
        "ceiling",
        "cmplx",
        "dble",
        "float",
        "floor",
        "int",
        "nint",
        "real",
    }
)
"""Conversions that take an optional KIND, in any of its spellings.

``cmplx`` is the one whose second positional argument is a value -- the
imaginary part -- so its kind is third. Every other member's kind is second,
and a ``kind=`` keyword drops the tail whichever member it is.
"""

BIT_INTRINSICS = frozenset({"iand", "ior", "ieor", "ishft"})
"""Bit operations whose result depends on the operand's width, which the
emitter passes (``Expressions._bit_arguments``)."""

MAPPED_OVER_ARRAYS = frozenset(
    {"acos", "asin", "atan", "atan2", "cosh", "dim", "gamma", "mod", "sinh", "tan"}
)
"""Elemental intrinsics whose only spelling is a scalar one that an array
breaks -- a ``math`` function, or a runtime helper that calls ``int`` or
``max`` on its operands -- so over arrays they are mapped per element. The
others without an array spelling either take arrays as they are (``np.imag``,
``_f_modulo``), convert elementwise in the runtime (``_f_int``, ``_f_nint``),
or are character and inquiry functions a substring or a whole array reaches
with scalar meaning."""

INTEGER_CONVERSIONS = frozenset({"int", "nint", "floor", "ceiling"})
"""Conversions whose KIND decides the result's *range*, not its precision,
so dropping it changes the number (``_integer_conversion``)."""

KIND_BYTES = {"int32": 4, "int64": 8, "float32": 4, "float64": 8}
"""A resolved kind's dtype -> the kind number gfortran gives it: its width in
bytes, whichever type the kind parameter was written for."""

_INTEGER_KIND_DTYPES = {4: "int32", 8: "int64"}
"""The INTEGER kinds a dtype here spells. gfortran has 1, 2 and 16 as well,
and none of them is an ``int32``."""


def _selected_int_kind(digits: int) -> int:
    """gfortran's ``selected_int_kind(r)``: the smallest kind holding every
    integer of ``r`` decimal digits, -1 where none does. Not "4 unless it
    needs 8": ``selected_int_kind(4)`` is the 2-byte kind, whose HUGE is
    32767."""
    for kind, most in ((1, 2), (2, 4), (4, 9), (8, 18), (16, 38)):
        if digits <= most:
            return kind
    return -1


INQUIRY_DTYPES = frozenset(
    {"int32", "int64", "float32", "float64", "complex64", "complex128", "bool", "str"}
)
"""The dtypes a kind inquiry is answered for: every kind the frontend
resolves a declaration to."""

_NUMERIC = frozenset({"int32", "int64", "float32", "float64"})
_REAL = frozenset({"float32", "float64"})

KIND_INQUIRIES: dict[str, frozenset[str]] = {
    "kind": INQUIRY_DTYPES,
    "huge": _NUMERIC,
    "tiny": _REAL,
    "epsilon": _REAL,
    "precision": _REAL | {"complex64", "complex128"},
    "range": _NUMERIC | {"complex64", "complex128"},
    "digits": _NUMERIC,
    "maxexponent": _REAL,
    "minexponent": _REAL,
    "bit_size": frozenset({"int32", "int64"}),
}
"""Inquiries about a kind rather than a value, and the dtypes each is defined
for (F2018 16.9: HUGE takes an integer or a real, TINY and EPSILON a real,
PRECISION a real or a complex, RANGE any of the three, DIGITS an integer or
a real, MAXEXPONENT and MINEXPONENT a real, BIT_SIZE an integer). Answered from the
declaration (``Expressions._kind_inquiry``)."""


NON_NEGATIVE_INQUIRIES = frozenset({"count", "index", "len", "len_trim", "scan", "size", "verify"})
"""Intrinsics whose answer is never below zero: a section's stop edge that is
one of them cannot count from the end of the axis."""

MAX_EXPANDED_POWER = 16
"""Beyond this, expanding ``x**n`` to multiplications stops being worth reading
and starts being a place for a transcription error. Refused instead."""

FOLDED_REAL_EXPONENTS = frozenset({2})
"""Whole-number *real* exponents the reference compiler folds into
multiplication rather than a ``pow`` call. Only 2: GCC folds ``pow(x, 2.0)``
unconditionally, because ``x*x`` is what the call must return, and expands the
others only under ``-funsafe-math-optimizations``."""


_LEADING_NAME = re.compile(r"^[\s(+-]*([A-Za-z_]\w*)")
"""The first identifier of a rendered argument: the name a subscript or a
sign is applied to, which is what a declaration can be looked up for."""


def _without_kind(name: str, arguments: list[str]) -> list[str]:
    """A conversion's arguments, with the KIND dropped.

    Three cases, and the third is why this is not one membership test: a
    ``kind=`` keyword names itself whichever position it is in; ``cmplx``'s
    second positional argument is the imaginary part and stays, its third is
    the kind; every other conversion's second is the kind. Python's
    ``complex`` takes two arguments, so a kind passed through is a TypeError
    at the first call rather than anything visible here.
    """
    if name not in KIND_CONVERSIONS:
        return arguments
    if len(arguments) == 2:
        if any("kind=" in a.lower() for a in arguments[1:]):
            return arguments[:1]
        return arguments if name == "cmplx" else arguments[:1]
    if name == "cmplx" and len(arguments) == 3:
        return arguments[:2]
    return arguments


class UnknownReference(NoRule):
    """``name(...)`` that is no procedure here and no intrinsic.

    Raised so the caller can fall back to reading it as a subscript, which
    is what such a reference nearly always is -- a variable this file
    use-imports from a module whose dimensions it never saw.
    """

    def __init__(self, name: str) -> None:
        super().__init__(f"unknown function or array reference {name!r}")


@dataclass(frozen=True)
class Remote:
    """A procedure that lives in a sibling translated module."""

    alias: str
    """The emitted import alias, e.g. ``_mgu``."""

    name: str
    """What it is called there, which a use-rename may make different."""


def function_outputs(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The OUT/INOUT dummies a function hands back beside its result.

    A Fortran function may change its arguments -- SLSQP's ``linmin`` drives
    a reverse-communication line search through ``mode`` and eighteen
    INOUT scalars -- and a translation that returned the result alone kept
    the search state at zero on every call, bit-exact in ``x`` and wrong in
    everything the next call would read. A function with a mandatory
    OUT/INOUT dummy therefore returns ``(result, *outputs)`` the way a
    subroutine returns its outputs (``Statements.returned_value``), and a
    reference to it is only translatable as the whole of an assignment,
    where the statement layer unpacks that tuple (``Statements._call``).
    Every OUT/INOUT dummy is in the tuple, optional ones included, so the
    unpacking is the subroutine's. Empty for a function whose only
    OUT/INOUT dummies are optional: those are dropped from the call and it
    stays a plain expression, as it always was.
    """
    outputs = [a for a in record.get("args") or () if a.get("intent") in ("OUT", "INOUT")]
    if record.get("kind") != "function" or not any(not a.get("optional") for a in outputs):
        return []
    return outputs


REAL_ARGUMENT_INTRINSICS = frozenset(
    {
        "nint",
        "_f_nint",
        "rint",
        "int",
        "_f_int",
        "floor",
        "_f_floor",
        "ceiling",
        "_f_ceiling",
        "ceil",
        "idint",
        "idnint",
        "ifix",
        "trunc",
    }
)
"""Integer-result intrinsics whose argument is a REAL expression: a quotient
inside one is real division however integer the result."""


HOISTED_REAL = re.compile(r"^F_\d")
"""A real literal the renderer hoisted to a name (``3600.0_r8`` ->
``F_3600P0``), as real as the literal it stands for."""


class _IntegerDivision(ast.NodeTransformer):
    """``a / b`` in an integer expression is Fortran integer division.

    Only where both operands are integer-valued: a float literal on either
    side, a name the caller knows is real (``real_names``), or a position
    inside the argument of a real-taking intrinsic (``nint(secs / 3600.0)``)
    is real division, and stays Python's ``/`` (#59).
    """

    def __init__(self, real_names: frozenset[str] = frozenset()) -> None:
        self.real_names = real_names
        self.real_depth = 0

    def _real(self, node: ast.AST) -> bool:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Constant) and isinstance(inner.value, float):
                return True
            if isinstance(inner, ast.Name) and (
                inner.id.lower() in self.real_names or HOISTED_REAL.match(inner.id)
            ):
                return True
        return False

    def visit_Call(self, node: ast.Call) -> ast.AST:
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
        if str(name).lower() in REAL_ARGUMENT_INTRINSICS:
            self.real_depth += 1
            self.generic_visit(node)
            self.real_depth -= 1
            return node
        self.generic_visit(node)
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if (
            isinstance(node.op, ast.Div)
            and self.real_depth == 0
            and not self._real(node.left)
            and not self._real(node.right)
        ):
            return ast.Call(
                func=ast.Name(id="_f_int_div", ctx=ast.Load()),
                args=[node.left, node.right],
                keywords=[],
            )
        return node


def _integer_divisions(text: str, real_names: frozenset[str] = frozenset()) -> str:
    """A rendered integer expression with each integer ``/`` made the integer
    division it is.

    A declared extent is an integer expression, so ``(n+1)*(n+2)/2`` -- the
    packed triangle SLSQP hands ``slsqpb`` as ``l`` -- truncates in Fortran.
    Rendered with Python's ``/`` it was a float, and the slice it sized the
    workspace view with refused it ("slice indices must be integers").
    ``_f_int_div`` is what the statement layer already spells the operator
    as, so a bound rounds the way the body does. An integer *parameter's*
    initializer is not an integer expression throughout -- only its result
    is -- so a quotient with a real operand, or under ``nint``, is left the
    real division it is; ``real_names`` are the parameters the caller knows
    to be real.
    """
    if "/" not in text:
        return text
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        return text
    return ast.unparse(_IntegerDivision(real_names).visit(tree).body)


@dataclass
class Expressions:
    """Render Fortran expressions for one subprogram."""

    semantics: Semantics
    names: Names
    profile: Profile

    externals: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Procedures with an audited shim in the externals module."""

    remotes: dict[str, Remote] = field(default_factory=dict)
    """Local name -> where it actually lives, for companion modules."""

    kind_map: dict[str, str] = field(default_factory=dict)
    """Kind parameter name (lower case) -> the real dtype it names, from the
    unit's interface record; what spells ``cmplx(x, kind = core_rknd)``."""

    type_bound: frozenset[str] = frozenset()
    """Component names that are type-bound procedures: ``obj%method(args)``
    is a call, where every other subscripted component is an array."""

    handles: set[str] = field(default_factory=set)
    """Emitted names whose value is an opaque handle, not a number.

    A framework that hands out registrations gives Fortran an integer index
    and gets tested with ``idx > 0`` for "is it registered". A translation
    that represents the registration as something else -- a dictionary key,
    say -- has to answer that test as the presence question it is. Which
    names those are is a fact about the framework, so a domain package's
    call transform says so (``CallSite.holds_handle``) and a function it
    names in ``handle_producers`` says so for what it returns.
    """

    handle_producers: frozenset[str] = frozenset()
    """Functions whose result is a handle, so assigning from one makes the
    target one too."""

    function_transforms: dict[str, Any] = field(default_factory=dict)
    assumed_scalar: set[str] = field(default_factory=set)
    """Names whose rank the semantics could not settle and that a logical
    operator therefore spelled as scalars; cleared per block by the renderer
    and written into the block report."""
    """Function name -> a domain package's answer for it, given the rendered
    arguments.

    The reference-side twin of ``Statements.call_transforms``. A fixed-string
    stub cannot answer ``dycore_is('LR')`` or ``rad_cnst_get_spec_idx(m, s)``:
    the answer depends on what was passed. Consulted before the stub table,
    and before this file's own procedures, as the pipeline consults its own.
    """

    stubs: dict[str, str] = field(default_factory=dict)
    """Framework function -> the text that stands in for it.

    A call into a framework the translation does not carry -- a model's history
    buffer answering whether a field is active, its unit manager handing out a
    file unit -- has an answer that is a property of the framework, not of the
    language. So it is supplied, like ``intent_overrides`` and ``externals``,
    and the domain package that knows the framework ships the table. Without one the
    call is refused and becomes a deferred site, which is the honest outcome:
    the engine genuinely does not know what ``hist_fld_active`` returns.

    Consulted only for references fparser read as structure constructors,
    which is where the pipeline consults its copy. A plainly-parsed reference
    to a stubbed name refuses even when the table has an answer -- the wider
    placement this module briefly had turned a refusal the pipeline hands to
    a human into a fabricated constant.
    """

    intrinsics: dict[str, dict[str, str]] = field(default_factory=dict)
    """Spellings that replace this backend's own, as ``{"scalar": {...},
    "array": {...}}`` keyed by intrinsic name -- and ``"**"`` for the power
    operator, which is not an intrinsic but is lowered the same way.

    A reference binary linked against a maths library that is not the system
    one -- Intel's libimf under ``ifx``, whose ``exp`` and ``pow`` are an ULP
    from glibc's on some arguments -- computes different numbers, and a
    translation held to it has to call the same library. Which library, and
    what to call it, is a fact about the build rather than about Fortran, so
    it arrives as configuration like the stub tables do; the package that
    knows the build ships the binding.
    """

    statement_functions: frozenset[str] = frozenset()
    allocated_bounds: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    """Array -> the bounds its ``allocate`` gave it, when they are not the
    declared ones. Owned by the statement layer, which is where allocation is
    seen; read here so a subscript shifts by the bound in force."""

    elemental: bool = False
    """Whether the enclosing subprogram is ELEMENTAL.

    Its body is written at scalar rank but runs over array actuals, so the
    intrinsics inside it have to take the array spelling anyway.
    """

    vector_boolean: bool = False
    """Whether a boolean here is a mask rather than a scalar, which decides
    between ``and`` and ``&``. Set by the statement layer around a WHERE."""

    # -- entry point ----------------------------------------------------------

    def render(self, node: Any) -> str:
        """One expression, as Python source text."""
        if isinstance(node, f03.Name):
            return self.names.symbol(str(node))
        if isinstance(node, (f03.Real_Literal_Constant, f03.Int_Literal_Constant)):
            return self.names.literal(node)
        if isinstance(node, f03.Char_Literal_Constant):
            return repr(str(node)[1:-1])
        if isinstance(node, f03.Logical_Literal_Constant):
            return "True" if ".TRUE." in str(node).upper() else "False"
        if isinstance(node, (f03.Hex_Constant, f03.Octal_Constant, f03.Binary_Constant)):
            spelled = str(node).upper()
            base = {"Z": "0x", "O": "0o"}.get(spelled[0], "0b")
            return base + spelled.split("'")[1]
        if isinstance(node, f03.Ac_Implied_Do):
            return self._implied_do(node)
        if isinstance(node, f03.Subscript_Triplet):
            return self._triplet(node)
        if isinstance(node, f03.Complex_Literal_Constant):
            # Not through the literal table: a complex literal's two halves are
            # written where they are read, and the zero-literal rule hoists
            # reals, not pairs of them.
            return f"complex({_complex_half(node.children[0])}, {_complex_half(node.children[1])})"
        if isinstance(node, f03.Parenthesis):
            return f"({self.render(node.children[1])})"
        if isinstance(node, f03.Data_Ref):
            return self._data_ref(node)
        if isinstance(node, f03.Array_Constructor):
            items = node.children[1]
            values = items.children if hasattr(items, "children") else [items]
            # An implied-do is already the whole sequence, not one element of
            # it: nesting its comprehension inside the brackets gives shape
            # ``(1, n)`` where Fortran says ``(n,)``, and every later index
            # into it is off by a dimension. Alone it *is* the constructor;
            # beside other elements it is spliced.
            if len(values) == 1 and isinstance(values[0], f03.Ac_Implied_Do):
                return f"np.array({self.render(values[0])})"
            rendered = [
                f"*{self.render(v)}" if isinstance(v, f03.Ac_Implied_Do) else self.render(v)
                for v in values
            ]
            return f"np.array([{', '.join(rendered)}])"
        if isinstance(node, (f03.Intrinsic_Function_Reference, f03.Part_Ref)):
            return self.reference(node)
        if isinstance(node, f03.And_Operand):
            return self._not(node)
        if isinstance(node, f03.Structure_Constructor):
            return self._structure_constructor(node)
        children = getattr(node, "children", None)
        if children and len(children) == 2 and isinstance(children[0], str):
            if children[0] in ("+", "-"):
                return f"({children[0]}{self.render(children[1])})"
        if children and len(children) == 3 and isinstance(children[1], str):
            return self.binary(children[0], children[1], children[2])
        raise NoRule(f"no expression rule for {type(node).__name__}: {node}")

    # -- operators ------------------------------------------------------------

    @property
    def scalar_table(self) -> dict[str, str]:
        """Intrinsic -> spelling on a scalar argument."""
        return {**ELEMENTAL_SCALAR, **self.intrinsics.get("scalar", {})}

    @property
    def array_table(self) -> dict[str, str]:
        """Intrinsic -> spelling on an array argument."""
        return {**ELEMENTAL_ARRAY, **self.intrinsics.get("array", {})}

    def binary(self, left: Any, operator: str, right: Any) -> str:
        """A binary operation, with the three that are not what they look like."""
        spelling = operator.upper()
        rendered_left, rendered_right = self.render(left), self.render(right)

        if spelling == "/" and self.semantics.is_integer(left) and self.semantics.is_integer(right):
            # Fortran truncates toward zero; Python's `//` floors. They agree
            # only when the operands share a sign.
            return f"_f_int_div({rendered_left}, {rendered_right})"

        if spelling == "**":
            power = self._power(rendered_left, rendered_right, left, right)
            if power is not None:
                return power

        if spelling == "//":
            return f"({rendered_left} + {rendered_right})"  # character concatenation

        if spelling in ARITH_OPS:
            return f"({rendered_left} {ARITH_OPS[spelling]} {rendered_right})"

        if spelling in RELATIONAL_OPS:
            return self._comparison(spelling, left, right, rendered_left, rendered_right)

        if spelling in LOGICAL_OPS:
            if spelling in (".AND.", ".OR.") and (
                self.vector_boolean or self._array_valued(left) or self._array_valued(right)
            ):
                return f"({rendered_left} {'&' if spelling == '.AND.' else '|'} {rendered_right})"
            return f"({rendered_left} {LOGICAL_OPS[spelling]} {rendered_right})"

        raise NoRule(f"operator {operator!r}")

    def _implied_do(self, node: Any) -> str:
        """``(expr, i = lo, hi [, step])`` inside an array constructor: a
        comprehension, because the loop is the constructor's own."""
        values, control = node.children
        variable = pysafe(str(control.children[0]).lower())
        bounds = list(control.children[1])
        low, high = self.render(bounds[0]), self.render(bounds[1])
        step = self.render(bounds[2]) if len(bounds) > 2 else None
        items = values.children if hasattr(values, "children") else [values]
        body = ", ".join(self.render(item) for item in items)
        span = f"range({low}, {high} + 1, {step})" if step else f"range({low}, {high} + 1)"
        return f"[{body} for {variable} in {span}]"

    def _power(self, left: str, right: str, left_node: Any, right_node: Any) -> str | None:
        """``x**n``, which the reference compiler may have lowered two ways."""
        exponent = self.semantics.integer_literal(right_node)
        if self._integer_power(left_node, right_node) and (exponent is None or exponent < 0):
            # INTEGER ** INTEGER is integer arithmetic throughout: ``2**(-1)``
            # is 1/2 in integer division, zero (F2018 10.1.5.2.2). Python's
            # ``2 ** -1`` is 0.5, and so is gfortran's expansion ``1.0 /
            # (x*x)`` that ``expand_power`` writes for a REAL base
            # (FNP-D0012). A non-negative literal exponent is exact in
            # Python already and keeps its spelling; a negative one, or one
            # whose sign only the run knows, goes through the runtime.
            return f"_f_ipow({left}, {right if exponent is None else exponent})"
        if exponent is None:
            exponent = self._folded_real_exponent(right_node)
        if exponent is not None and exponent != 0:
            if not self.profile.int_pow_expand:
                return f"({left} ** {exponent})"
            if abs(exponent) > MAX_EXPANDED_POWER:
                raise NoRule(f"integer power {exponent} is too large to expand")
            return expand_power(left, exponent)
        try:
            over_arrays = self.semantics.rank(left_node) > 0 or self.semantics.rank(right_node) > 0
        except Unanalyzable:
            over_arrays = False
        if over_arrays or self.elemental:
            spelling = self.intrinsics.get("array", {}).get("**", "_f_vpow")
            return f"{spelling}({left}, {right})"
        scalar = self.intrinsics.get("scalar", {}).get("**")
        if scalar:
            return f"{scalar}({left}, {right})"
        if self.profile.int_pow_expand and self._integer_exponent(left_node, right_node):
            # ``(xe(i)-x0)**(j-1)``: the exponent is an integer and the
            # emitter cannot see which one, so ``expand_power`` above had
            # nothing to expand and what was left was Python's ``**`` -- a
            # ``pow`` call the reference binary never makes (``_f_powi``).
            return f"_f_powi({left}, {right})"
        return None

    def _folded_real_exponent(self, node: Any) -> int | None:
        """A real literal exponent the reference compiler folds, not calls.

        ``uaf(p)**2._r8`` is a real power, so nothing above reads it as a
        literal exponent and what is emitted is a ``pow`` call -- which NumPy
        answers one ULP away from ``x*x`` often enough to fail a bit-exact
        gate (2 points of 321,888 on a day of ELM's CanopyFluxes). GCC folds
        ``pow(x, 2.0)`` into the multiplication unconditionally, because that
        is what the call must return anyway; every other whole-number real
        exponent it expands only under ``-funsafe-math-optimizations``, so
        those stay the call the reference binary makes.
        """
        if not self.profile.int_pow_expand:
            return None
        value = self.semantics.integral_real_literal(node)
        return value if value in FOLDED_REAL_EXPONENTS else None

    def _integer_power(self, left_node: Any, right_node: Any) -> bool:
        """An INTEGER raised to an INTEGER, which is not a ``pow`` at all."""
        try:
            return self.semantics.is_integer(left_node) and self.semantics.is_integer(right_node)
        except Unanalyzable:
            return False

    def _integer_exponent(self, left_node: Any, right_node: Any) -> bool:
        """A real raised to an integer, which is the case ``powi`` covers.

        An integer base is left alone: Fortran's integer power is its own
        arithmetic (``2**(-1)`` is zero, not a half), which ``_f_ipow``
        spells above.
        """
        try:
            return self.semantics.is_integer(right_node) and not self.semantics.is_integer(
                left_node
            )
        except Unanalyzable:
            return False

    def _comparison(self, spelling: str, left: Any, right: Any, rl: str, rr: str) -> str:
        if rl in self.handles and ((RELATIONAL_OPS[spelling], rr) in ((">", "0"), (">=", "1"))):
            # The Fortran asks whether the registration exists by comparing
            # the index it was given; the translation holds something that is
            # not an index, and the question is whether it is set.
            return f"bool({rl})"
        if not (self.semantics.is_character(left) or self.semantics.is_character(right)):
            return f"({rl} {RELATIONAL_OPS[spelling]} {rr})"
        if RELATIONAL_OPS[spelling] not in ("==", "!="):
            # Fortran orders character strings by the collating sequence after
            # blank padding; ordering them any other way is a different answer.
            raise NoRule(f"character comparison {spelling}")
        negated = "not " if RELATIONAL_OPS[spelling] == "!=" else ""
        return f"({negated}_fstr_eq({rl}, {rr}))"

    def _not(self, node: Any) -> str:
        operator, operand = node.children
        if str(operator).upper() != ".NOT.":
            raise NoRule(f"unary logical operator {operator}")
        if self.vector_boolean or self._array_valued(operand):
            return f"(~({self.render(operand)}))"
        return f"(not {self.render(operand)})"

    def _array_valued(self, node: Any) -> bool:
        """Whether a logical operand is an array, so ``.NOT.`` / ``.AND.`` /
        ``.OR.`` have to be elementwise -- ``any( .not. l_valid )`` over
        ``logical, dimension(nz) :: l_valid`` (CLUBB's new_pdf), outside any
        WHERE. Python's ``not`` on an array raises; ``~`` is the operator.

        A rank the semantics cannot settle -- a name no declaration in this
        file covers, a use-import of a stubbed module -- takes the scalar
        spelling, and the name goes on ``assumed_scalar`` so the block
        report carries the assumption: ``not`` of a length-1 array is a
        wrong answer that raises nothing (ledger #32 row 5)."""
        if isinstance(node, f03.Name) and not self.semantics.rank_declared(node):
            self.assumed_scalar.add(str(node).lower())
            return False
        try:
            return self.semantics.rank(node) > 0
        except Exception:  # rank refuses what it cannot settle
            for name in walk(node, f03.Name):
                self.assumed_scalar.add(str(name).lower())
            return False

    # -- references -----------------------------------------------------------

    def reference(self, node: Any) -> str:
        """``name(...)``: a subscript, an intrinsic, or a call."""
        name = str(node.children[0]).lower()
        if self.semantics.is_array(name):
            return self.subscript(name, node.children[1])
        # F77's specific spellings are the same intrinsic: canonicalise here,
        # once, so the constant folder, the scalar/array split and the
        # elemental dispatch below all see a name they know. A name that
        # canonicalised is the intrinsic, whatever else is declared under it.
        canonical = self.semantics.canonical_intrinsic(name)
        if canonical == name and self._is_procedure_dummy(name, node.children[1]):
            # The caller passed a callable, so this is a call whatever the
            # rest of the name resolution would make of it. It precedes the
            # intrinsic tables too: a dummy may be named after one, and the
            # argument is the caller's, not ours.
            arguments = self._arguments(_items(node.children[1]))
            return f"{self.names.symbol(name)}({', '.join(arguments)})"
        name = canonical

        items = _items(node.children[1])
        if name in self.semantics.generics:
            name = self.semantics.dispatch(name, items)

        folded = self._constant_fold(name, items)
        if folded is not None:
            return folded

        arguments = self._arguments(items)
        call = self._call(name, items, arguments)
        if call is not None:
            return call
        try:
            return self._intrinsic(name, items, arguments)
        except UnknownReference:
            bound = self.names.use_bindings.get(name)
            if bound is not None:
                # A USE-imported function, called through its module's alias.
                # This has to come before the subscript below, and did not:
                # ``parallelmin(min_area, hybrid)`` was emitted as
                # ``_reduction_mod.parallelmin[min_area - 1, hybrid - 1]``,
                # which is not a refusal but runnable, wrong code -- with the
                # zero-based shift applied to what are arguments.
                return f"{bound}({', '.join(arguments)})"
            if self._unspelled_intrinsic(name, items):
                # The standard's function, not an array: nothing in scope
                # declares or imports the name, or its arguments are ones no
                # subscript takes. ``digits(x)`` came out ``digits[x - 1]``,
                # a NameError at best (FNP-D0027).
                raise NoRule(f"intrinsic {name!r} has no spelling in this translation") from None
            # Neither a procedure this file declares, nor an intrinsic, nor
            # a name a USE statement bound: what is left is a subscript of
            # something it use-imports without the dimensions. Reading it as
            # a call would emit a call to a name nothing defines; a subscript
            # is what the source spelling says, and what the pipeline settled
            # on.
            declared = self.semantics.declaration(name)
            if (
                declared is not None
                and not declared.get("dims")
                and declared.get("dtype") not in (None, "UNDECLARED", "PROCEDURE", "str")
                and not declared.get("procedure")
            ):
                # A DECLARED scalar cannot be subscripted (a CHARACTER one
                # can: ``s(i:j)``), so this is a call to something no table
                # knows -- an unmapped intrinsic -- and never ``name[args -
                # 1]``. Undeclared names keep the fallback: the extractor has
                # blind spots (host-associated arrays, duplicate subprogram
                # names) where the name is a real array.
                raise NoRule(f"unknown function or array {name!r}") from None
            return self.subscript(name, node.children[1])

    def _unspelled_intrinsic(self, name: str, items: list[Any]) -> bool:
        """Whether an unresolved ``name(...)`` is a standard intrinsic this
        translation has no rule for, rather than an array it cannot see.

        The subscript fallback exists for arrays a module use-imports
        without their dimensions, so a name only a USE statement could have
        brought in keeps it: an ONLY list that names it binds it, and a bare
        USE of a module that is not intrinsic may. Even then, a keyword or a
        non-integer argument is no subscript.
        """
        if name not in STANDARD_FUNCTIONS:
            return False
        if (
            name in self.names.use_bindings
            or name in self.names.use_parameters
            or name in self.names.companion_globals
        ):
            return False
        bare_use = False
        for statement in self.semantics.module.get("use_statements", ()):
            match = USE_STATEMENT.match(statement)
            if match and not match.group(2) and match.group(1).lower() not in INTRINSIC_MODULES:
                bare_use = True
        if not bare_use:
            return True

        def no_subscript(item: Any) -> bool:
            if isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)):
                return True
            if isinstance(item, f03.Subscript_Triplet):
                return False
            # Definitely not integer, not merely untyped here: an index
            # the scope cannot type is still an index.
            return self.semantics._integral_or_unknown(item) is False

        return any(no_subscript(item) for item in items)

    def _is_procedure_dummy(self, name: str, arglist: Any) -> bool:
        """Whether ``name(...)`` calls a callable this subprogram was passed.

        Two ways in. ``EXTERNAL``, ``PROCEDURE`` and an explicit INTERFACE
        body say so in the declaration; F77 had none of those spellings, so a
        scalar non-character DUMMY referenced with an argument list is one by
        use -- there is nothing else it could be, because a dummy that were
        an array would have been declared with a shape.
        """
        if arglist is None:
            return False
        declared = self.semantics.declaration(name)
        if declared is not None and declared.get("procedure"):
            return True
        argument = next((a for a in self.semantics.subprogram["args"] if a["name"] == name), None)
        return argument is not None and not argument.get("dims") and argument.get("dtype") != "str"

    def bound(self, text: str, substitutions: dict[str, str] | None = None) -> str:
        """Declared bound text -> Python. Bound texts are simple -- names,
        integers, ``+ - * /``, parentheses, ``size(a, n)`` -- by construction;
        anything else refuses the statement that needed the bound.

        ``substitutions`` is for a *callee's* bound read in its caller
        (``extent``): a name that is one of the callee's dummies is the
        actual the call binds to it, parenthesised, and one the call leaves
        unbound refuses, as it does when it is the whole bound."""

        # An automatic array sized off another argument. UBOUND is the same
        # question on an axis based at one, and the dimension may be written
        # with or without ``dim=``. On an axis declared from another lower
        # bound, UBOUND is that bound plus the extent, less one (FNP-D0006).
        def extent(match: re.Match[str]) -> str:
            name = self.names.symbol(match.group(1).lower())
            dimension = match.group(2)
            if dimension is None:
                return self.extent_of(name)
            axis = int(DIM_KEYWORD.sub("", dimension)) - 1
            along = self.extent_along(name, axis)
            if match.group(0).lower().startswith("ubound"):
                origins = self._bound_origins(match.group(1).lower())
                if origins is not None and axis < len(origins):
                    if origins[axis] != indexing.UNIT_ORIGIN:
                        return f"(({self._origin(origins[axis])}) + {along} - 1)"
            return along

        if EXTENT.fullmatch(text):
            return EXTENT.sub(extent, text)
        rendered, position = [], 0
        opens_intrinsic = False  # the next "(" opens a max/min call
        calls: list[bool] = []  # per open parenthesis: a max/min call?
        for match in BOUND_TOKENS.finditer(text):
            if match.start() != position:
                raise NoRule(f"dim expr {text!r}")
            position = match.end()
            piece = match.group(0)
            inquiry = EXTENT.fullmatch(piece)
            if inquiry is not None:
                # ``size(x)-1``, ``2*size(c,2)``: an extent is a *term* of a
                # bound, not only a whole one. Substituting it into the text
                # before this loop spelled ``np.size(x) - 1``, which the loop
                # then refused at the ``.`` it has no token for -- an array
                # the source sizes off its argument, deferred over the
                # spelling of the answer rather than over the question.
                rendered.append(extent(inquiry))
            elif "%" in piece:
                # ``bounds%begp`` sizing a local: the component of a dummy,
                # which is an attribute of the same name on this side.
                root, component = (t.strip() for t in piece.split("%", 1))
                rendered.append(f"{self.names.symbol(root)}.{pysafe(component.lower())}")
            elif piece.lower() in ("max", "min") and text[match.end() :].lstrip().startswith("("):
                # ``max(2, edsclr_dim)`` sizing a local (CLUBB's windm
                # solver): Python spells the two intrinsics the same way,
                # and a bound's operands are integers. The only calls a
                # bound may carry; a comma is legal inside one of them alone.
                rendered.append(piece.lower())
                opens_intrinsic = True
            elif re.match(r"[A-Za-z_]", piece):
                if substitutions is not None and piece.lower() in substitutions:
                    actual = substitutions[piece.lower()]
                    if not actual:
                        raise NoRule(f"dummy dimension {text!r} is not bound by this call")
                    rendered.append(f"({actual})")
                else:
                    rendered.append(self.names.symbol(piece))
            elif piece.isdigit() and piece not in ("0", "1", "2"):
                hoisted = self.names.literals.get(piece)
                if hoisted is None:
                    raise NoRule(f"declared dim literal {piece}")
                rendered.append(hoisted)
            elif piece == "(":
                calls.append(opens_intrinsic)
                opens_intrinsic = False
                rendered.append(piece)
            elif piece == ")":
                if calls:
                    calls.pop()
                rendered.append(piece)
            elif piece == ",":
                if not (calls and calls[-1]):
                    raise NoRule(f"dim expr {text!r}")
                rendered.append(piece)
            else:
                rendered.append(piece)
        if position != len(text):
            raise NoRule(f"dim expr {text!r}")
        return _integer_divisions("".join(rendered))

    def extent_of(self, name: str) -> str:
        """How many elements an array has, as this target spells it."""
        return f"np.size({name})"

    def extent_along(self, name: str, axis: int) -> str:
        """The same along one zero-based axis.

        Named rather than written inline because it is a *spelling*, and the
        Numba backend's differs: ``np.size`` compiles under ``@njit`` but its
        axis argument does not, so a kernel asks the shape tuple instead. Two
        methods and not one because an unqualified extent has no axis to pass
        and the two targets agree on it.
        """
        return f"np.size({name}, {axis})"

    def _triplet(self, node: Any) -> str:
        """A range in a value position, as a Python ``slice`` object.

        Not the subscript path: that one knows which array it is indexing and
        can shift by the declared lower bound. A triplet reaching here has no
        array behind it -- it is an actual argument, and all that is known is
        Fortran's inclusive upper edge and one-based start. The pipeline's
        spelling, parentheses included, because a translation that differs
        here differs in text a differential compares.
        """
        lower, upper, step = node.children
        lower_text = self.render(lower) if lower is not None else ""
        upper_text = self.render(upper) if upper is not None else ""
        step_text = self.render(step) if step is not None else ""
        if lower_text:
            lower_text = f"({lower_text}) - 1"
        tail = f", {step_text}" if step_text else ""
        return f"slice({lower_text or 'None'}, {upper_text or 'None'}{tail})"

    def subscript(self, name: str, arglist: Any) -> str:
        """An array element or slice, shifted to zero-based."""
        declaration = self.semantics.declaration(name)
        dims = self.allocated_bounds.get(name, (declaration or {}).get("dims"))
        if dims == CONFLICTING_BOUNDS:
            raise NoRule(
                f"module allocatable {name!r} is allocated with lower bounds that do not "
                "agree, or with one this subprogram cannot evaluate"
            )
        positions = indexing.describe(arglist, dims, rank_of=self.semantics.rank)
        symbol = self.names.symbol(name)
        parts = [self._position(p, symbol, axis, dims) for axis, p in enumerate(positions)]
        return f"{symbol}[{', '.join(parts)}]"

    def _position(
        self,
        position: indexing.Position,
        array: str | None = None,
        axis: int | None = None,
        dims: list[dict[str, Any]] | None = None,
    ) -> str:
        """One subscript position of ``array`` along zero-based ``axis``.

        The array, axis and declared ``dims`` are only read by a section: one
        whose edges the runtime works out, where an implied edge has to be
        spelled as the bound it stands for, and one whose stop edge may have
        to be kept from going negative. ``None`` for a caller with no array
        to name.
        """
        if position.kind is Kind.VECTOR:
            return f"(({self.render(position.index)}) - 1)"
        if position.kind is Kind.INDEX:
            folded = indexing.fold(position, WHITELIST_INT)
            if folded is not None:
                return str(folded)
            return self._shift(self.render(position.index), position.origin)
        dim = dims[axis] if dims and axis is not None and axis < len(dims) else None
        return self._range(position, array, axis, dim)

    def _range(
        self,
        position: indexing.Position,
        array: str | None = None,
        axis: int | None = None,
        dim: dict[str, Any] | None = None,
    ) -> str:
        """``lo:hi`` inclusive becomes ``lo':hi'+1`` exclusive."""
        step = self.render(position.step) if position.step is not None else None
        if step is not None and step.lstrip("(").startswith("-"):
            # A descending section: the runtime works the edges out, because
            # the stop edge underflows at the first element. The declared
            # lower bound goes with them.
            lower, upper = self._stepped_edges(position, array, axis)
            return f"_f_rstep_lb({lower}, {upper}, {step}, {self._origin(position.origin)})"
        if step is not None and not step.strip("()").isdigit():
            # A step whose sign this cannot read -- a dummy, a variable
            # (CLUBB's ``grid_dir_indx``, 1 or -1 with the grid's direction):
            # the ascending spelling ``lo:hi+1:step`` stops one short when the
            # step turns out negative, and is empty at the first element. The
            # runtime reads the sign and builds the slice (#75).
            lower, upper = self._stepped_edges(position, array, axis)
            return f"_f_rstep_any({lower}, {upper}, {step}, {self._origin(position.origin)})"
        start = ""
        if position.lower is not None:
            folded = indexing.fold_index(position.lower, position.origin, WHITELIST_INT)
            start = (
                str(folded)
                if folded is not None
                else self._shift(self.render(position.lower), position.origin)
            )
        stop = self._stop(position, dim) if position.upper is not None else ""
        return f"{start}:{stop}" + (f":{step}" if step is not None else "")

    def _stop(self, position: indexing.Position, dim: dict[str, Any] | None) -> str:
        """The exclusive stop edge of an ascending section, kept off zero's
        far side.

        A section whose upper subscript is below its lower one is empty, and
        its subscripts need not be within the bounds (F2018 9.5.3.3.2):
        ``a(1:-1)`` selects nothing, as does ``s(1:-1)`` of a string. Rendered
        as it stood, the stop edge was ``-1`` and CPython counted it from the
        end -- ``a[0:-1]`` is every element but the last, ``s[0:(-1)]`` the
        string less its last character (FNP-D0037, FNP-D0043). A stop edge at
        zero or above means what Fortran means, so the one that could go
        below it is clamped there. Not where it cannot: a non-negative
        literal (a negative one is folded to the ``0`` it clamps to), the
        axis's own declared upper bound (below the lower one only on a
        zero-size axis, where every slice is empty) while the variables it
        names still hold the values the bounds took on entry, the section's
        own lower edge (``a(i:i)`` is never empty), and an inquiry that
        cannot answer below zero.
        """
        rendered = self.render(position.upper)
        literal = self.semantics.integer_literal(position.upper)
        low = _integer_text(position.origin)
        if literal is not None and low is not None:
            if literal - low + 1 < 0:
                return "0"
            if position.shifts_by_one:
                return rendered
        stop = (
            rendered
            if position.shifts_by_one
            else f"({rendered}) - ({self._origin(position.origin)}) + 1"
        )
        if literal is not None and low is not None:
            return stop
        upper = _bound_text(position.upper)
        if (
            upper is not None
            and upper == _bound_text((dim or {}).get("ub"))
            and self._kept_from_entry(upper)
        ) or (position.lower is not None and upper == _bound_text(position.lower)):
            return stop
        if (
            position.shifts_by_one
            and isinstance(position.upper, f03.Intrinsic_Function_Reference)
            and str(position.upper.children[0]).lower() in NON_NEGATIVE_INQUIRIES
        ):
            return stop
        return f"max(0, {stop})"

    def _kept_from_entry(self, text: str) -> bool:
        """Whether every variable a declared bound names keeps its value
        (``rwset.kept_from_entry``, which the read/write sets draw on too)."""
        return kept_from_entry(text, self.semantics)

    def capture_bounds(
        self, name: str, dims: list[dict[str, Any]], spelled: list[str]
    ) -> tuple[list[dict[str, Any]], list[str]]:
        """``dims`` with each lower bound that can move held in a local, and
        the assignments that fill those locals.

        An array's bounds are fixed when it comes into being -- on entry for
        an explicit-shape dummy or an automatic local (F2018 10.1.11), at
        its ALLOCATE for an allocatable (F2018 9.7.1.2) -- but a subscript
        was shifted by the bound *text*, re-read at every use. ``allocate(
        a(-n:n))``, then ``n = n + 1``, then ``a(0)``: the shift was
        ``(0) - (- n)`` with the new ``n``, and the element read was
        ``a(1)``; LBOUND and UBOUND answered -3 and 1 where gfortran says
        -2 and 2 (FNP-D0049). Where the subprogram can redefine a name the
        bound reads (``rwset.captured``), the bound's value is read once,
        into ``_lb_<array>_<axis>``, and the returned dims name that local,
        so the subscripts, LBOUND/UBOUND and section edges read it through
        ``subscript`` and ``_bound_origins`` as they read a bound text. The
        caller emits the assignments where the array comes into being.
        ``spelled`` is each axis's lower bound as Python. A bound that
        cannot move is left as it is, and so is its emitted text.
        """
        kept: list[dict[str, Any]] = []
        assignments: list[str] = []
        for axis, (dim, lower) in enumerate(zip(dims, spelled, strict=True)):
            if not captured(dim.get("lb"), self.semantics):
                kept.append(dim)
                continue
            hidden = f"_lb_{name.lower()}_{axis + 1}"
            assignments.append(f"{hidden} = {lower}")
            kept.append({**dim, "lb": hidden, "source_lb": dim["lb"]})
        return kept, assignments

    def _stepped_edges(
        self, position: indexing.Position, array: str | None, axis: int | None
    ) -> tuple[str, str]:
        """Both edges of a section the runtime slices, implied ones spelled.

        An omitted first subscript is the axis's lower bound and an omitted
        second one its upper bound, whatever the sign of the step (F2018
        9.5.3.3.2): ``a(::-1)`` is empty, and ``a(:1:-1)`` is ``a(1)`` alone.
        Handed to the runtime as ``None``, an implied edge took Python's
        meaning instead -- the end the step walks *from* -- so all three of
        ``a(:1:-1)``, ``a(n::-1)`` and ``a(::-1)`` summed the whole array
        reversed (FNP-D0001). The lower bound is the one the subscript
        already shifts by; the upper bound is the lower one plus the axis's
        extent less one, asked of the array the way every other extent is.
        """
        origin = self._origin(position.origin)
        lower = self.render(position.lower) if position.lower is not None else origin
        if position.upper is not None:
            return lower, self.render(position.upper)
        if array is None or axis is None:
            raise NoRule("an implied upper edge of a stepped section, with no array to size it")
        extent = self.extent_along(array, axis)
        upper = extent if position.shifts_by_one else f"({origin}) + {extent} - 1"
        return lower, upper

    def _origin(self, origin: str) -> str:
        """A declared lower bound, as Python.

        Through ``bound`` like every other declared bound. The origin is
        Fortran source text, and a name in it means whatever this module's USE
        statements bound it to -- emitting it raw put a bare ``nhe`` beside the
        ``_dimensions_mod.nhe`` that the same subscript already spelled
        correctly, which is a NameError rather than a matter of style.

        Constant folding upstream still works on the raw text: it is doing
        arithmetic on the Fortran, not naming anything.
        """
        return self.bound(origin)

    def _shift(self, rendered: str, origin: str) -> str:
        if origin == indexing.UNIT_ORIGIN:
            return f"{rendered} - 1"
        return f"({rendered}) - ({self._origin(origin)})"

    def _data_ref(self, node: Any) -> str:
        """``a % b % c(i, k)`` -> ``a.b.c[i - 1, k - 1]``."""
        parts = []
        for position, component in enumerate(node.children):
            if isinstance(component, f03.Name):
                name = str(component)
                parts.append(self.names.symbol(name) if position == 0 else pysafe(name.lower()))
            elif isinstance(component, f03.Part_Ref):
                name = str(component.children[0]).lower()
                head = self.names.symbol(name) if position == 0 else pysafe(name)
                if position > 0 and name in self.type_bound:
                    # `obj%method(args)` is a call; only the domain package
                    # knows which components are procedures rather than
                    # arrays, because the type is declared elsewhere.
                    called = ", ".join(self.render(item) for item in _items(component.children[1]))
                    parts.append(f"{head}({called})")
                    continue
                dims = self._component_dims(node, position, name)
                positions = indexing.describe(
                    component.children[1], dims, rank_of=self.semantics.rank
                )
                array = ".".join([*parts, head])
                subscripts = ", ".join(
                    self._position(p, array, axis, dims) for axis, p in enumerate(positions)
                )
                parts.append(f"{head}[{subscripts}]")
            elif isinstance(component, f03.Data_Ref):
                # fparser nests them when the chain is long enough.
                parts.append(self._data_ref(component))
            else:
                raise NoRule(f"data-ref component {type(component).__name__}")
        return ".".join(parts)

    def selector_dims(self, selector: Any) -> Any:
        """The dims an ``associate`` alias inherits from its selector.

        ``slatop => pftcon%slatop`` binds a name the body then subscripts as
        a plain array, so the shift the component carries -- its allocated
        lower bound -- has to travel with the alias, or the alias is shifted
        by one where the selector would have been shifted by zero.
        """
        if not isinstance(selector, f03.Data_Ref) or len(selector.children) != 2:
            return None
        component = selector.children[1]
        if isinstance(component, f03.Part_Ref):
            component = component.children[0]
        if not isinstance(component, f03.Name):
            return None
        return self._component_dims(selector, 1, str(component).lower())

    def _component_dims(self, node: Any, position: int, component: str) -> Any:
        """The declared -- or allocated -- dims of ``root%component``.

        Only the first component of a chain is resolved: the root's
        declaration names its type, the type record names the component,
        and an ``allocate (obj%c(0:n))`` seen by the frontend is on that
        record as ``allocated_dims``. Deeper chains and unknown types keep
        the unit origin, which is the shift every component had before.
        """
        if position != 1:
            return None
        root = node.children[0]
        root_name = str(root.children[0] if isinstance(root, f03.Part_Ref) else root).lower()
        declared = self.semantics.declaration(root_name)
        match = DERIVED_TYPE.match(str((declared or {}).get("dtype", "")))
        if match is None:
            return None
        record = self.semantics.types.get(match.group(1).lower(), {}).get(component)
        if not record:
            return None
        return record.get("allocated_dims") or record.get("dims")

    # -- calls ----------------------------------------------------------------

    def actual_argument(
        self, formal: dict[str, Any], actual: Any, substitutions: dict[str, str]
    ) -> str:
        """One actual argument, as the *callee's* dummy sees it.

        Fortran's sequence association lets an actual of lower rank -- a whole
        array passed to a two-dimensional dummy, an array element passed to an
        array dummy -- stand for the contiguous memory the dummy spans, and a
        target with real array objects has to say so. Applied wherever a call
        is bound against a record, which is both statements (``call qrfac(...,
        a, ...)``) and expressions: ``enorm(m, a(1, j))`` passes the whole of
        column ``j``, and rendering the element alone hands ``enorm`` a scalar
        to subscript.
        """
        rendered = self.render(actual)
        formal_dims = formal.get("dims") or []
        if not formal_dims:
            return rendered
        try:
            rank = self.semantics.rank(actual)
        except REFUSED:
            return rendered
        element = (
            rank == 0
            and isinstance(actual, f03.Part_Ref)
            and self.semantics.is_array(str(actual.children[0]).lower())
        )
        if self._assumed_size(formal_dims):
            # ``x(*)``: the dummy spans the caller's storage from the element
            # to the end of the array, and only the caller knows how far
            # that is. Rendering the element alone -- what an unbounded
            # dummy used to get -- hands the callee one number to subscript.
            if element:
                return self._association_tail(actual, formal_dims, substitutions)
            if rank is not None and 0 < rank < len(formal_dims):
                # ``vl(ldvl, *)`` handed a rank-1 ``vl``: the leading axes have
                # the extents the call binds, and the assumed-size last axis
                # takes whatever the actual's storage has left, in column-major
                # order. ``-1`` asks NumPy for exactly that only when the
                # storage fills whole columns, which a leading extent of 1
                # always does; any other raised on the partial last column
                # Fortran allows (``v(2, *)`` over five elements, FNP-D0045),
                # and the runtime keeps it.
                leading = [self.extent(d, substitutions) for d in formal_dims[:-1]]
                if all(axis == "1" for axis in leading):
                    return f"np.reshape({rendered}, ({', '.join(leading)}, -1), order='F')"
                return f"_f_seq_tail({', '.join([rendered, '0', *leading])})"
            if rank is not None and rank > len(formal_dims):
                # A whole matrix handed to ``c(*)`` -- ``h12(..., a, mda, 1,
                # i-1)``: the dummy spans all of its storage in column-major
                # order, from the first element. Rendering the array alone
                # handed the callee two axes to subscript with one index.
                leading = [self.extent(d, substitutions) for d in formal_dims[:-1]]
                return f"_f_seq_tail({', '.join([rendered, '0', *leading])})"
            return rendered
        if not all(d.get("ub") for d in formal_dims):
            return rendered
        if rank is not None and 0 < rank < len(formal_dims):
            # Fortran sequence association: a lower-rank actual fills the
            # dummy in column-major order, and the dummy takes only as much
            # of it as its extents span -- ``nnls(w, n1, n1, m, ...)`` hands
            # ``a(mda, n)`` the first ``n1*m`` cells of a longer workspace,
            # and reshaping the whole of ``w`` raised on the size.
            extents = [self.extent(d, substitutions) for d in formal_dims]
            span = " * ".join(f"({axis})" for axis in extents)
            flat = rendered if rank == 1 else f"np.ravel({rendered}, order='F')"
            return f"np.reshape({flat}[:{span}], ({', '.join(extents)},), order='F')"
        if element:
            return self.sequence_association(actual, formal_dims, substitutions)
        if rank == len(formal_dims) and rank > 1 and isinstance(actual, f03.Name):
            # A whole array of the dummy's rank. Where the dummy declares
            # other extents -- ``y(3, 2)`` for ``x(2, 2)`` -- Fortran
            # associates the leading part of the storage, and the array
            # handed over in its own shape had the callee index the caller's
            # axes: ``x(1, 2)`` read ``y(1, 2)``, not ``y(3, 1)`` (FNP-D0016).
            # The run time folds the storage onto the dummy's extents, and an
            # OUT or INOUT result goes back through ``_f_copy_out`` in the
            # same order. Where the declarations show the extents agree --
            # ``fjac(ldfjac, n)`` for ``a(lda, n)`` bound ``lda = ldfjac`` --
            # the array goes over as it is, as it always did. One axis has
            # no fold to get wrong: ``x(i)`` is the actual's ``i``-th element
            # either way, and a rank-1 actual goes over as it is.
            # A dummy extent the call does not bind is refused by ``extent``:
            # neither side can then say where the dummy ends.
            if not self._same_extents(str(actual).lower(), formal_dims, substitutions):
                extents = [self._dummy_extent(d, substitutions) for d in formal_dims]
                return f"_f_seq_shape({', '.join([rendered, *extents])})"
        return rendered

    def _dummy_extent(self, dim: dict[str, Any], substitutions: dict[str, str]) -> str:
        """One axis's extent of a callee's explicit-shape dummy, in the caller's names.

        An integer literal is written as it is: the extent is a count the
        runtime reads, not a value in the arithmetic, and the caller's module
        need not have hoisted the callee's ``x(4)``.
        """

        def side(text: str) -> str:
            return text if text.isdigit() else self.extent({"ub": text}, substitutions)

        upper = side(str(dim["ub"]))
        lower = str(dim.get("lb") or "1")
        if lower == "1":
            return upper
        return f"({upper}) - ({side(lower)}) + 1"

    def _same_extents(
        self, name: str, formal_dims: list[dict[str, Any]], substitutions: dict[str, str]
    ) -> bool:
        """Whether the actual ``name`` is declared with the dummy's extents.

        Read off the text, both sides in the caller's names: the same lower
        and upper bound spelled the same way. Anything the text cannot show
        -- an allocatable's or an assumed-shape array's axes, bounds spelled
        differently -- is left to the run time, where an equal shape costs a
        comparison and nothing else. So are bounds over a variable the body
        can change: the actual's extent is the one it had on entry, and the
        same text now may name another (FNP-D0049).
        """
        declared = (self.semantics.declaration(name) or {}).get("dims") or []
        if name in self.allocated_bounds or len(declared) != len(formal_dims):
            return False
        try:
            for mine, theirs in zip(declared, formal_dims, strict=True):
                if not mine.get("ub") or str(mine["ub"]) in ("*", ":"):
                    return False
                if not self._kept_from_entry(f"{mine.get('lb') or 1} {mine['ub']}".lower()):
                    return False
                lower = self.bound(str(mine.get("lb") or "1"))
                wanted = self.extent({"ub": str(theirs.get("lb") or "1")}, substitutions)
                if lower != wanted or self.bound(str(mine["ub"])) != self.extent(
                    theirs, substitutions
                ):
                    return False
        except REFUSED:
            return False
        return True

    @staticmethod
    def _assumed_size(formal_dims: list[dict[str, Any]]) -> bool:
        """``x(*)`` or ``x(n, *)``: the last axis has no extent of its own."""
        return bool(formal_dims and formal_dims[-1].get("assumed_size"))

    def _association_tail(
        self, actual: Any, formal_dims: list[dict[str, Any]], substitutions: dict[str, str]
    ) -> str:
        """An element actual for an assumed-size dummy: the actual's memory
        from the element on, as the array the callee reads and writes.

        A rank-1 actual to a rank-1 dummy is a plain slice, a view. Anything
        else -- ``a(i, 1)`` to ``dx(*)``, ``c(i, 1)`` to ``u(iue, *)``, a
        vector to ``x(2, *)`` -- goes through the runtime's ``_f_seq_tail``:
        the storage from the element on in column-major order, a view when
        the actual is Fortran-contiguous (the gate's inputs and every
        reshaped window are) and the tail fills whole columns, with a rank-2
        dummy's leading extents folded onto it the way Fortran lays it out,
        partial last column included. SLSQP's ``dcopy(n, a(i, 1),
        la, ...)`` and ``h12(..., c(i, 1), lc, ..., c(j, 1), ...)`` were
        refused here, which deferred every block that recovers a matrix row.
        """
        name = str(actual.children[0]).lower()
        declaration = self.semantics.declaration(name) or {}
        symbol = self.names.symbol(name)
        start = self._association_start(actual)
        if len(declaration.get("dims") or []) == 1 and len(formal_dims) == 1:
            return f"{symbol}[{start}:]"
        leading = [self.extent(d, substitutions) for d in formal_dims[:-1]]
        return f"_f_seq_tail({', '.join([symbol, start, *leading])})"

    def substitutions(self, record: dict[str, Any], actuals: list[Any]) -> dict[str, str]:
        """Formal name -> the actual bound to it, rendered in the caller's scope.

        Keyed in one case, because ``a(Lda, n)`` and the dummy ``lda`` are one
        name; and every formal gets an entry, bound or not, because a
        dimension no actual answered is a different thing from a name both
        sides can see. ``extent`` reads both facts.
        """
        table = {formal["name"].lower(): "" for formal in record["args"]}
        for formal, actual in zip(record["args"], actuals, strict=False):
            if actual is None:
                continue
            try:
                table[formal["name"].lower()] = self.render(actual)
            except REFUSED:
                pass
        return table

    # -- sequence association -------------------------------------------------

    def sequence_association_target(
        self, actual: Any, formal_dims: list[dict[str, Any]], substitutions: dict[str, str]
    ) -> tuple[str, bool]:
        """The same association, as somewhere a callee's OUT array can land.

        Returns the target text and whether the value has to be flattened in
        column-major order first. The input form is free to build a reshaped
        *copy*; a target cannot -- what is written has to reach the caller's
        own memory -- so only the two forms that are views are allowed here:
        the leading axes taken whole, and a slice of a rank-1 actual. Anything
        else is refused rather than written to a copy nobody reads, which is
        what ``wa(index + 1)`` was doing: assigned as if it were the single
        element the source spells, so the callee's whole array landed on one
        scalar and NumPy said so.
        """
        if self._assumed_size(formal_dims):
            # The tail view the callee was handed; the copy-out onto it is
            # the same memory, so the writes it made in place stand.
            tail = self._association_tail(actual, formal_dims, substitutions)
            if not tail.startswith("_f_seq_tail("):
                return tail, True
            # A higher-rank tail has no slice to assign through, so the
            # runtime writes the callee's array back into the caller's
            # column-major storage: ``{}`` is where the value goes, whole,
            # so the runtime can tell the view it handed out from a copy.
            name = str(actual.children[0]).lower()
            start = self._association_start(actual)
            return f"_f_seq_tail_out({self.names.symbol(name)}, {start}, {{}})", False
        whole = self._leading_axes_whole(actual, formal_dims)
        if whole is not None:
            # The view the callee was handed is where its result lands (#27);
            # ``[...]`` sends it through the runtime's copy-out like any other
            # whole-array target.
            return f"{whole}[...]", False
        name = str(actual.children[0]).lower()
        declaration = self.semantics.declaration(name) or {}
        if len(declaration.get("dims") or []) != 1:
            raise NoRule(f"seq-assoc target: {name} is not rank-1 and not at a lower bound")
        offset, span, _ = self._association_offset(actual, formal_dims, substitutions)
        return f"{self.names.symbol(name)}[{offset}:{offset} + {span}]", True

    def _leading_axes_whole(self, actual: Any, formal_dims: list[dict[str, Any]]) -> str | None:
        """``arr(1, k)`` to a rank-1 formal: the whole of column ``k``.

        ``None`` when the element is not at the lower bound of the leading
        axes, which is where the general offset form has to be used instead.
        """
        name = str(actual.children[0]).lower()
        subscripts = self._subscript_nodes(actual)
        declaration = self.semantics.declaration(name)
        if declaration is None:
            raise NoRule(f"seq-assoc: undeclared {name}")
        actual_dims = declaration.get("dims") or []
        if len(subscripts) != len(actual_dims):
            raise NoRule(f"seq-assoc: rank mismatch {name}")
        if len(formal_dims) > len(actual_dims):
            # A rank-1 actual filling a rank-2 dummy -- ``wa(index + 1)`` for
            # ``fjac(ldfjac, n)``. There are no leading axes to take whole,
            # but the association is ordinary: memory is memory, and the
            # offset form below spells it.
            return None
        first_scalar = None
        for at, subscript in enumerate(subscripts):
            if not isinstance(subscript, f03.Subscript_Triplet):
                if first_scalar is None:
                    first_scalar = at
            else:
                first_scalar = None
        if first_scalar is None:
            raise NoRule(f"seq-assoc: no scalar subscript in {name}")
        at_lower_bound = (
            first_scalar == 0
            and isinstance(subscripts[0], f03.Int_Literal_Constant)
            and str(subscripts[0]).split("_")[0] == str(actual_dims[0].get("lb", "1"))
        )
        if not (at_lower_bound and len(formal_dims) <= len(actual_dims) - first_scalar):
            return None
        parts = []
        for at in range(len(actual_dims)):
            if at < len(formal_dims):
                parts.append(":")
                continue
            # A trailing scalar subscript shifts by the axis's DECLARED
            # lower bound, not a blanket 1 (#39): an element ``a(1,1,ie)``
            # of a local ``a(np,np,nets:nete)`` is ``a[:, :, ie - nets]``.
            low = actual_dims[at].get("lb", "1")
            if low in (None, "1", ":"):
                parts.append(self._shifted(subscripts[at]))
            else:
                parts.append(f"({self.render(subscripts[at])}) - ({self.bound(low)})")
        return f"{self.names.symbol(name)}[{', '.join(parts)}]"

    @staticmethod
    def _subscript_nodes(actual: Any) -> list[Any]:
        arglist = actual.children[1]
        if arglist is None:
            return []
        return list(arglist.children) if hasattr(arglist, "children") else [arglist]

    def _association_offset(
        self, actual: Any, formal_dims: list[dict[str, Any]], substitutions: dict[str, str]
    ) -> tuple[str, str, str]:
        """``(offset, span, shape)`` for an element actual, in column-major order.

        The stride along an axis is the array's own extent there, not the text
        of its declared upper bound: the two agree, and asking the array does
        not read a name the block never mentions -- ``a(Lda, n)`` made every
        such call look like a read of ``lda``, which the static read/write
        gate reported as a disagreement -- nor does it get a non-unit lower
        bound wrong.
        """
        axes = [self.extent(d, substitutions) for d in formal_dims]
        offset = self._association_start(actual)
        return offset, " * ".join(f"({axis})" for axis in axes), ", ".join(axes)

    def _association_start(self, actual: Any) -> str:
        """The element's 0-based position in the actual's column-major storage."""
        name = str(actual.children[0]).lower()
        subscripts = self._subscript_nodes(actual)
        declaration = self.semantics.declaration(name) or {}
        actual_dims = declaration.get("dims") or []
        symbol = self.names.symbol(name)
        shifts = []
        for at, subscript in enumerate(subscripts):
            low = actual_dims[at].get("lb", "1")
            shifts.append(f"({self.render(subscript)} - {low})")
        offset = shifts[0]
        stride = "1"
        for at in range(1, len(shifts)):
            stride = f"{stride} * {self.extent_along(symbol, at - 1)}"
            offset = f"{offset} + {shifts[at]} * {stride}"
        return offset

    def sequence_association(
        self, actual: Any, formal_dims: list[dict[str, Any]], substitutions: dict[str, str]
    ) -> str:
        """A scalar element actual -- ``arr(i, k)`` -- passed to an array
        formal. Fortran passes contiguous memory starting at the element.

        The common pattern is the element sitting at the lower bound of the
        leading axes -- ``arr(1, k)`` -- which becomes taking those axes whole:
        ``arr[:, k - 1]``. The general form flattens in column-major order,
        offsets, and reshapes; correct everywhere, and worth avoiding where
        the cheap answer holds.
        """
        whole = self._leading_axes_whole(actual, formal_dims)
        if whole is not None:
            return whole
        offset, span, shape = self._association_offset(actual, formal_dims, substitutions)
        flat = f"{self.names.symbol(str(actual.children[0]).lower())}.ravel(order='F')"
        # The slice ends where the dummy does. Reshaping the whole tail is
        # what Fortran means only when the actual happens to end there too;
        # NumPy refuses any other size, so a dummy shorter than the memory
        # behind it -- ``enorm(m - j + 1, a(j, j))`` -- raised instead of
        # taking the first ``m - j + 1`` elements.
        return f"np.reshape({flat}[{offset}:{offset} + {span}], ({shape},), order='F')"

    def _shifted(self, node: Any) -> str:
        """A single 1-based index, 0-based. A literal folds only while the
        folded value stays inside the whitelist; otherwise it references the
        hoisted constant, minus one."""
        if isinstance(node, f03.Int_Literal_Constant):
            folded = int(str(node).split("_")[0]) - 1
            if folded in (0, 1, 2):
                return str(folded)
            return f"{self.names.literal(node)} - 1"
        return f"{self.render(node)} - 1"

    def extent(self, dim: dict[str, Any], substitutions: dict[str, str]) -> str:
        """One axis of a callee's dummy, as an expression the *caller* can read.

        A dummy's bound is written in the callee's names, so an axis named by
        one of the callee's own arguments is the actual the caller passed for
        it -- looked up without regard to case, because ``a(Lda, n)`` and the
        dummy ``lda`` are the same name. Missing that lookup renders the
        callee's parameter name in the caller's scope: ``r1mpyq(1, n, Qtf, 1,
        ...)`` reshaped ``qtf`` to ``(lda, n)``, and ``lda`` is a name the
        caller does not have.

        An axis the call did not bind is refused rather than guessed at, for
        the same reason. So is an arithmetic expression over the callee's
        arguments, name by name: ``dbint4``'s ``w(5, ndata + 2)`` bound
        ``ndata = nx`` is ``(nx) + 2`` in the caller, where it had been
        spelled ``ndata + 2`` -- a name the caller does not have, which the
        read/write gate reported and the call would have raised on. Anything
        that is not one of the callee's arguments -- a module parameter --
        means the same thing on both sides and goes through ``bound`` as it is.
        """
        text = str(dim["ub"])
        if text.lower() in substitutions:
            substituted = substitutions[text.lower()]
            if not substituted:
                raise NoRule(f"dummy dimension {text!r} is not bound by this call")
            return substituted
        return self.bound(text, substitutions)

    def _arguments(self, items: list[Any]) -> list[str]:
        rendered = []
        for item in items:
            if isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)):
                keyword, value = item.children
                rendered.append(f"{str(keyword).lower()}={self.render(value)}")
            else:
                rendered.append(self.render(item))
        return rendered

    def _constant_fold(self, name: str, items: list[Any]) -> str | None:
        """Emit a compile-time fold, when the reference compiler did one."""
        if not self.profile.cfold_mpfr or name not in CONSTANT_FOLDED or not items:
            return None
        if not all(
            not isinstance(a, (f03.Actual_Arg_Spec, f03.Component_Spec))
            and self.semantics.is_constant(a)
            for a in items
        ):
            return None
        return f"_f_cfold('{name}', {', '.join(self.render(a) for a in items)})"

    def _call(self, name: str, items: list[Any], arguments: list[str]) -> str | None:
        """A call to something with source, here or in a sibling module."""
        if name in self.statement_functions:
            return f"{pysafe(name)}({', '.join(arguments)})"

        transform = self.function_transforms.get(name)
        if transform is not None:
            return str(transform(list(arguments)))
        remote = self.remotes.get(name)
        record = self.semantics.procedures.get(name)
        if record is None and remote is None:
            if name not in self.semantics.companion_generics:
                return None
            # A generic reached through a sibling translated module: the
            # overload is picked here, from the companion's specifics.
            name = self.semantics.dispatch(name, items)
            remote = self.remotes[name]
            record = self.semantics.procedures.get(name)
        target = (
            f"{remote.alias}.{remote.name}"
            if remote
            else pysafe(emit_name(record or {"name": name}))
        )
        if record is not None:
            # Sequence association applies to a function reference as much as
            # to a CALL: ``enorm(m, a(1, j))`` hands the callee the whole of
            # column ``j``, and rendering the element alone hands it a scalar
            # to subscript. Only where every actual is positional -- a keyword
            # actual is not bound to a formal by position, and guessing which
            # dummy it answers is how the reshape lands on the wrong one. A
            # sibling's function is bound the same way: ``ddot(n, w(i4), 1,
            # w(iff), 1)`` into a translated BLAS handed ``ddot`` two scalars.
            positional = not any(
                isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)) for item in items
            )
            if positional and len(items) <= len(record["args"]):
                substitutions = self.substitutions(record, items)
                arguments = [
                    self.actual_argument(formal, item, substitutions)
                    for formal, item in zip(record["args"], items, strict=False)
                ]
        if record is not None and function_outputs(record):
            # The callee hands its OUT/INOUT dummies back beside its result,
            # and an expression has nowhere to put them. Only the statement
            # layer, for ``x = f(...)`` as a whole, unpacks that tuple.
            raise NoRule(
                f"function {name} has OUT/INOUT dummy argument(s) "
                f"{', '.join(a['name'] for a in function_outputs(record))}, which only a "
                "whole-statement reference `x = f(...)` can carry back"
            )
        if record is not None and not remote:
            if record.get("host_writes"):
                # A function reference is an expression: there is no place
                # in it for the host variables the body changes to land, and
                # a subroutine convention here would put a tuple where the
                # caller expects a value.
                raise NoRule(
                    f"internal function {name} writes host variable(s) "
                    f"{', '.join(record['host_writes'])}, which a function reference "
                    "cannot carry back"
                )
            arguments = [
                *arguments,
                *(self.names.symbol(hv) for hv in record.get("host_vars") or ()),
            ]
        if record is not None and self._broadcasts(record, items):
            # An ELEMENTAL procedure called with an array actual has to be
            # mapped over it; its body was written at scalar rank.
            return f"_f_ecall({target}, {', '.join(arguments)})"
        return f"{target}({', '.join(arguments)})"

    def _broadcasts(self, record: dict[str, Any], items: list[Any]) -> bool:
        if not any("ELEMENTAL" in str(p).upper() for p in (record.get("prefixes") or ())):
            return False
        for item in items:
            argument = item.children[1] if isinstance(item, f03.Actual_Arg_Spec) else item
            try:
                if self.semantics.rank(argument) > 0:
                    return True
            except Unanalyzable:
                pass
        return False

    # -- intrinsics -----------------------------------------------------------

    def _intrinsic(self, name: str, items: list[Any], arguments: list[str]) -> str:
        if name == "present":
            return self._present(arguments[0])
        if name in ("allocated", "associated"):
            return f"({arguments[0]} is not None)"
        external = self.externals.get(name)
        if external is not None and external.get("kind") == "function":
            return f"_ext.{name}({', '.join(arguments)})"
        # ``stubs`` is deliberately NOT consulted here. The pipeline answers
        # from that table only for references parsed as structure
        # constructors; a plainly-parsed ``hist_fld_active(name_out)`` is
        # refused and deferred to a human. Stubbing it here instead once
        # turned that whole IF construct into ``if False:`` -- emitted, dead,
        # and wrong in a way nothing downstream would notice.
        if name in ("lbound", "ubound"):
            return self._bound_inquiry(name, items)
        if name in KIND_INQUIRIES:
            return self._kind_inquiry(name, items)
        if name in ARRAY_TRANSFORM:
            return self._array_transform(name, items)
        if name in REDUCTIONS:
            return self._reduction(name, arguments)

        try:
            rank = max(
                (self.semantics.rank(a) for a in items if not isinstance(a, f03.Actual_Arg_Spec)),
                default=0,
            )
        except Unanalyzable:
            # Scalar, as the pipeline reads it. An argument this cannot rank
            # is usually not an intrinsic's at all -- the two paths below
            # both end in ``UnknownReference`` for a name the table does not
            # hold, which ``reference`` then resolves as a USE-bound call or
            # a subscript. Refusing here instead pre-empted both, and a name
            # with source one module over never got the chance to resolve.
            rank = 0

        if name in INTEGER_CONVERSIONS:
            return self._integer_conversion(name, items, arguments, rank)
        mapped = rank > 0 and not self._substring_rank(items)
        if name in BIT_INTRINSICS:
            arguments = self._bit_arguments(name, items, arguments)
            if mapped:
                return f"_f_ecall({self.scalar_table[name]}, {', '.join(arguments)})"
            return f"{self.scalar_table[name]}({', '.join(arguments)})"
        if mapped and name in MAPPED_OVER_ARRAYS and name not in self.array_table:
            # A scalar-only spelling handed an array: ``math.tan`` of one
            # raises, ``_f_mod`` of one calls ``int`` on it (FNP-D0036). The
            # scalar translation runs per element instead, which keeps the
            # libm call the reference makes, the way an ELEMENTAL
            # procedure's body is mapped over its actuals.
            values = _without_kind(name, arguments)
            return f"_f_ecall({self.scalar_table[name]}, {', '.join(values)})"
        if rank > 0:
            return self._over_arrays(name, arguments)
        return self._over_scalars(name, arguments)

    def _substring_rank(self, items: list[Any]) -> bool:
        """Whether the rank an argument has is only a substring's.

        ``s(i:i)`` of a CHARACTER scalar ranks as a section here, so
        ``index(letters, s(i:i))`` and ``ichar(s(i:i))`` reach the array
        path with scalar strings. Their scalar spelling is the right one,
        and mapping it would hand back a 0-d array where a number was.
        """
        for item in items:
            for reference in walk(item, f03.Part_Ref):
                declared = self.semantics.declaration(str(reference.children[0]))
                if declared and declared.get("dtype") == "str" and not declared.get("dims"):
                    return True
        return False

    def _integer_conversion(
        self, name: str, items: list[Any], arguments: list[str], rank: int
    ) -> str:
        """``INT``/``NINT`` into the INTEGER kind the source asked for.

        ``_without_kind`` drops a conversion's KIND, which for a REAL result
        is the double this translation computes in anyway. For an INTEGER
        one it is the *range*: ``k8 = int(x, kind=8)`` of 3.0d10 is
        30000000000 in Fortran, and ``_f_int(x)`` -- a default-kind
        conversion -- answered the int32 edge, -2147483648 (FNP-D0005). So
        the kind is read, and a 64-bit one handed on; the default kind
        keeps the spelling it always had. A kind this cannot resolve to 4
        or 8 is refused rather than guessed at, because a guess is exactly
        the defect.

        ``FLOOR`` and ``CEILING`` the same over an array, whose spelling
        converted into int32 whatever the KIND. A scalar one is
        ``math.floor``, an unbounded Python int that holds an 8-byte result
        already, so it keeps its spelling.
        """
        kind = self._conversion_kind(name, items)
        values = _without_kind(name, arguments)
        table = self.array_table if rank > 0 and name in self.array_table else self.scalar_table
        if kind == 8 and table[name].startswith("_f_"):
            values = [*values, "8"]
        return f"{table[name]}({', '.join(values)})"

    def _bit_arguments(self, name: str, items: list[Any], arguments: list[str]) -> list[str]:
        """``IAND``/``IOR``/``IEOR``/``ISHFT`` within the operands' KIND.

        The runtime reads the width off a NumPy dtype, and a literal or a
        Python int has none, so the operation went unbounded:
        ``ishft(1, 31)`` is -2147483648 in a default INTEGER and came out
        2147483648 (FNP-D0009). The width is the declared kind's, known
        here, so it is passed; an operand whose kind this cannot read
        leaves the runtime's dtype reading as it was.
        """
        values = [
            item.children[1]
            if isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec))
            else item
            for item in items
        ]
        operands = values[:1] if name == "ishft" else values
        widths = [self._integer_width(v) for v in operands]
        known = [w for w in widths if w is not None]
        if not known:
            return arguments
        keywords = any(isinstance(i, (f03.Actual_Arg_Spec, f03.Component_Spec)) for i in items)
        return [*arguments, f"bits={max(known)}" if keywords else str(max(known))]

    def _integer_width(self, node: Any) -> int | None:
        """An integer expression's KIND in bits, where its declaration or its
        literal spelling says -- a default one is 32 -- or ``None``.

        A BOZ constant answers ``None`` on purpose: in a bit intrinsic it
        takes the kind of the other operand.
        """
        while isinstance(node, f03.Parenthesis) or _signed(node):
            node = node.children[1]
        if isinstance(node, (f03.Hex_Constant, f03.Octal_Constant, f03.Binary_Constant)):
            return None
        if isinstance(node, (f03.Intrinsic_Function_Reference, f03.Part_Ref)) and not (
            self.semantics.is_array(str(node.children[0]))
        ):
            called = str(node.children[0]).lower()
            items = _items(node.children[1])
            if called in INTEGER_CONVERSIONS:
                return 64 if self._conversion_kind(called, items) == 8 else 32
            if called in BIT_INTRINSICS:
                inner = [self._integer_width(i) for i in items[: 1 if called == "ishft" else 2]]
                known = [w for w in inner if w is not None]
                return max(known) if known else None
            return None
        children = getattr(node, "children", None)
        if children and len(children) == 3 and _is_arithmetic(children[1]):
            left, right = self._integer_width(children[0]), self._integer_width(children[2])
            return max(left, right) if left is not None and right is not None else None
        return {"int32": 32, "int64": 64}.get(self.inquiry_dtype(node) or "")

    def _conversion_kind(self, name: str, items: list[Any]) -> int | None:
        """The KIND an ``INT``/``NINT`` asks for, as the number it is.

        ``None`` for no KIND, which is the default. 4 or 8 otherwise -- the
        two INTEGER kinds ``_f_int`` spells -- and a refusal for a kind that
        is neither, or that nothing here can evaluate.
        """
        node = None
        for at, item in enumerate(items):
            if isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)):
                if str(item.children[0]).lower() == "kind":
                    node = item.children[1]
            elif at == 1:
                node = item
        if node is None:
            return None
        value = self._kind_value(node)
        if value not in (4, 8):
            raise NoRule(
                f"{name} with KIND {node}: "
                + ("not a kind this translation can evaluate" if value is None else f"kind {value}")
                + "; only the 4- and 8-byte INTEGER kinds are spelled"
            )
        return value

    def _kind_value(self, node: Any) -> int | None:
        """A KIND expression's value, or ``None`` where this cannot tell.

        A literal is its value, a kind parameter what ``_kind_named`` reads
        it as, and ``selected_int_kind(r)`` with a literal range the kind
        gfortran gives -- 2 for ``r = 4``, which the frontend's "int32
        unless it needs int64" is not.
        """
        while isinstance(node, f03.Parenthesis):
            node = node.children[1]
        if isinstance(node, f03.Int_Literal_Constant):
            return int(str(node).split("_")[0])
        if isinstance(node, f03.Name):
            return self._kind_named(str(node).lower())
        if isinstance(node, (f03.Intrinsic_Function_Reference, f03.Part_Ref)):
            called = str(node.children[0]).lower()
            items = _items(node.children[1])
            if called == "selected_int_kind" and len(items) == 1:
                digits = self.semantics.integer_literal(items[0])
                if digits is not None:
                    return _selected_int_kind(digits)
        return None

    def _kind_named(self, name: str, depth: int = 4) -> int | None:
        """The kind a kind parameter names, or ``None`` where this cannot tell.

        Its initializer first, where this scope can see it: a bare number
        is itself, ``selected_int_kind(r)`` and ``kind(1_8)`` what gfortran
        makes of them, and another kind parameter what *that* names. The
        frontend's kind map is the fallback, as the byte width of the dtype
        it resolved the name to -- ``r8 = selected_real_kind(12)`` is 8 --
        because gfortran numbers kinds by bytes, which is why ``int(x, r8)``
        is a 64-bit integer. The map comes second because it reads an
        integer kind as "int32 unless it needs int64": ``selected_int_kind(4)``
        is ``int32`` there and kind 2 in gfortran.
        """
        declared = (
            self.semantics.declaration(name) or self.semantics.companion_parameters.get(name) or {}
        )
        initializer = str(declared.get("init_expr") or "").lower().replace(" ", "")
        if initializer.isdigit():
            return int(initializer)
        spelled = re.fullmatch(r"selected_int_kind\((?:r=)?(\d+)\)", initializer)
        if spelled:
            return _selected_int_kind(int(spelled.group(1)))
        spelled = re.fullmatch(r"kind\([-+]?\d+(?:_(\w+))?\)", initializer)
        if spelled:
            suffix = spelled.group(1)
            if suffix is None:
                return 4
            if suffix.isdigit():
                return int(suffix)
            return self._kind_named(suffix, depth - 1) if depth else None
        if depth and initializer != name and re.fullmatch(r"[a-z]\w*", initializer):
            aliased = self._kind_named(initializer, depth - 1)
            if aliased is not None:
                return aliased
        return KIND_BYTES.get(self.kind_map.get(name, ""))

    def _kind_inquiry(self, name: str, items: list[Any]) -> str:
        """``KIND``, ``HUGE``, ``TINY``, ``EPSILON``, ``PRECISION``: facts
        about the argument's *declared* kind, answered from the declaration.

        The runtime used to decide by the Python type of the value it was
        handed, and that is not the kind: ``kind(x4)`` of a ``real(4)``
        answered 8 and ``huge(k8)`` of an ``integer(8)`` the int32 maximum,
        because a float is a float and an int an int (FNP-D0007, FNP-D0008);
        and a ``real(8)`` assigned the integer ``0`` holds a Python ``0``,
        so ``huge(x)`` answered 2147483647 for it (FNP-D0039). The dtype is
        a translation-time fact, so it is passed -- ``_f_huge(k8,
        'int64')`` -- and the argument stays in the call because the source
        reads it. A kind this cannot resolve, and an argument the inquiry
        does not take, are refused.
        """
        values = [
            item.children[1]
            if isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec))
            else item
            for item in items
        ]
        if len(values) != 1:
            raise NoRule(f"{name} takes one argument")
        dtype = self.inquiry_dtype(values[0])
        if dtype is None:
            raise NoRule(f"{name}({values[0]}): the argument's kind is not one this resolves")
        if dtype not in KIND_INQUIRIES[name]:
            raise NoRule(f"{name} of a {dtype}")
        return f"{self.scalar_table[name]}({self.render(values[0])}, {dtype!r})"

    def inquiry_dtype(self, node: Any) -> str | None:
        """The dtype of an inquiry's argument, from its declaration or its
        literal spelling, or ``None`` where that is not settled.

        Only what a declaration or a literal says: a computed expression's
        kind is Fortran's promotion rules, and nothing here needs them.
        ``real(4)`` reaches here as ``UNKNOWN_REAL_KIND(4)``, a kind spelled
        by its number, which is the number; a kind *name* left unresolved
        where it was declared is read through this unit's kind map.
        """
        while isinstance(node, f03.Parenthesis) or _signed(node):
            node = node.children[1]
        if isinstance(node, f03.Int_Literal_Constant):
            kind = self._literal_kind(str(node))
            return {None: "int32", 4: "int32", 8: "int64"}.get(kind)
        if isinstance(node, f03.Real_Literal_Constant):
            kind = self._literal_kind(str(node))
            if kind is None:
                kind = 8 if "d" in str(node).lower() else 4
            return {4: "float32", 8: "float64"}.get(kind)
        if isinstance(node, f03.Logical_Literal_Constant):
            return "bool"
        if isinstance(node, f03.Char_Literal_Constant):
            return "str"
        dtype: Any = None
        record: dict[str, Any] | None = None
        if isinstance(node, f03.Name):
            dtype = self.semantics.scalar_target_dtype(node)
            record = self.semantics.declaration(str(node))
            subprogram = self.semantics.subprogram
            if record is None and subprogram.get("result") == str(node).lower():
                # A function's result is typed on the subprogram, not as a local.
                record = (
                    {"kind": subprogram.get("result_kind")} if "result_kind" in subprogram else None
                )
            companion = self.semantics.companion_parameters.get(str(node).lower())
            if dtype is None and companion is not None:
                # A parameter a sibling module declares: its kind is that
                # module's declaration of it.
                dtype, record = companion.get("dtype"), companion
        elif isinstance(node, f03.Part_Ref) and self.semantics.is_array(str(node.children[0])):
            dtype = self.semantics.declared_dtype(node)
            record = self.semantics.declaration(str(node.children[0]))
        elif isinstance(node, f03.Data_Ref) and len(node.children) == 2:
            root, last = node.children
            if isinstance(root, f03.Name) and isinstance(last, (f03.Name, f03.Part_Ref)):
                component = last if isinstance(last, f03.Name) else last.children[0]
                record = self.semantics.component(str(root), str(component))
                dtype = record.get("dtype") if record else None
        dtype = str(dtype or "")
        if dtype in ("int32", "int64"):
            return self._declared_integer_dtype(record)
        spelled = re.fullmatch(r"UNKNOWN_REAL_KIND\((\w+)\)", dtype)
        if spelled:
            # A kind the declaring scope left unresolved: a number is itself,
            # and a name is what it names here -- a companion analysed on
            # its own does not see the kinds module its reader does
            # (SLSQP's ``one`` is ``real(wp)`` in ``slsqp_support``).
            name = spelled.group(1).lower()
            resolved = {"4": "float32", "8": "float64"}.get(name) or self.kind_map.get(name)
            return resolved if resolved in _REAL else None
        return dtype if dtype in INQUIRY_DTYPES else None

    def _declared_integer_dtype(self, record: dict[str, Any] | None) -> str | None:
        """An INTEGER declaration's dtype, from the kind it spells.

        Not the dtype the frontend typed it with: that is ``int32`` for
        every INTEGER kind it could not resolve, with no marker, so
        ``integer(kind(1_8))``, ``integer(2)`` and ``integer(1)`` all arrive
        as ``int32``. Taken at its word, ``huge(a)`` of the first answered
        2147483647 and ``ishft(a, 40)`` of 1 shifted within 32 bits to 0,
        where Fortran says 9223372036854775807 and 1099511627776. So the
        kind is read again: none, or one that is 4, is the default
        ``int32``; one that is 8 is ``int64``; anything else, and a record
        that does not say, is ``None``.
        """
        if record is None or "kind" not in record:
            return None
        kind = str(record.get("kind") or "").strip().lower()
        if not kind:
            return "int32"
        number = int(kind) if kind.isdigit() else self._kind_named(kind)
        return None if number is None else _INTEGER_KIND_DTYPES.get(number)

    def _literal_kind(self, text: str) -> int | None:
        """A literal's kind suffix as its number: ``1_8`` is 8, ``1.0_r8`` is
        what ``r8`` names. ``None`` for no suffix; a suffix nothing here can
        evaluate is 0, which no table answers for."""
        if "_" not in text:
            return None
        suffix = text.rsplit("_", 1)[1].strip().lower()
        if suffix.isdigit():
            return int(suffix)
        return self._kind_named(suffix) or 0

    def _present(self, argument: str) -> str:
        """``present(x)``: a sentinel for an optional output, ``is not None``
        for everything else.

        An optional *output* cannot be spelled ``is None`` on the target side,
        because its value is always in the return tuple; the caller says
        whether it wanted it, and the callee reads that back.
        """
        name = argument.lower()
        for declared in self.semantics.subprogram["args"]:
            if declared["name"] == name and declared["optional"] and declared["intent"] == "OUT":
                return f"want_{name}"
        return f"({argument} is not None)"

    def _bound_inquiry(self, name: str, items: list[Any]) -> str:
        """``LBOUND``/``UBOUND``: the bounds the subscripts are shifted by.

        The runtime used to answer 1 for every LBOUND and the vocabulary
        spelled UBOUND as the extent, which is right only on an axis based
        at one: ``real(8) :: a(0:5)`` gave ``lbound(a, 1) = 1`` and
        ``ubound(a, 1) = 6`` where Fortran says 0 and 5 (FNP-D0006). The
        bounds reported are the ones ``subscript`` shifts by -- the
        declaration's, or an ``allocate``'s -- so a loop from ``lbound`` to
        ``ubound`` indexes exactly the elements it did in the source. An
        array expression, a section, or an array this scope has no shape
        for is based at one, as Fortran's own answer for them is.

        The array is still handed to the runtime rather than folded away:
        the inquiry reads it on both sides of the read/write check, and the
        runtime needs its shape anyway -- an axis of zero extent answers 1
        and 0 whatever its declaration says.
        """
        positional = [
            i for i in items if not isinstance(i, (f03.Actual_Arg_Spec, f03.Component_Spec))
        ]
        keyword = {
            str(i.children[0]).lower(): i.children[1]
            for i in items
            if isinstance(i, (f03.Actual_Arg_Spec, f03.Component_Spec))
        }
        array = keyword.get("array", positional[0] if positional else None)
        dim = keyword.get("dim", positional[1] if len(positional) > 1 else None)
        if array is None:
            raise NoRule(f"{name} without an array")
        rendered = self.render(array)
        dimension = self.render(dim) if dim is not None else None
        origins = self._bound_origins(array)
        if origins is not None and any(o != indexing.UNIT_ORIGIN for o in origins):
            lows = [self._origin(o) for o in origins]
            lower = f"({lows[0]},)" if len(lows) == 1 else f"({', '.join(lows)})"
            spelled = "_f_lbound" if name == "lbound" else "_f_ubound"
            return f"{spelled}({rendered}, {dimension or 'None'}, {lower})"
        if name == "lbound":
            called = self.scalar_table["lbound"]
            return f"{called}({rendered}, {dimension})" if dimension else f"{called}({rendered})"
        if dimension is not None:
            return self.axis_reduction(REDUCTIONS["ubound"], rendered, dimension)
        # Without DIM the answer is a vector, one upper bound per axis;
        # ``np.size`` of a rank-2 array is the element count.
        return f"_f_ubound({rendered})"

    def _bound_origins(self, array: Any) -> list[str] | None:
        """The lower bound of each axis ``subscript`` shifts ``array`` by,
        as source text, or ``None`` where it shifts by one throughout."""
        if isinstance(array, str):
            name = array
        elif isinstance(array, f03.Name):
            name = str(array).lower()
        elif (
            isinstance(array, f03.Data_Ref)
            and len(array.children) == 2
            and isinstance(array.children[1], f03.Name)
        ):
            dims = self._component_dims(array, 1, str(array.children[1]).lower())
            return [_axis_origin(dims, axis) for axis in range(len(dims))] if dims else None
        else:
            return None
        if not self.semantics.is_array(name):
            return None
        declaration = self.semantics.declaration(name) or {}
        dims = self.allocated_bounds.get(name, declaration.get("dims"))
        if dims == CONFLICTING_BOUNDS:
            raise NoRule(
                f"module allocatable {name!r} is allocated with lower bounds that do not "
                "agree, or with one this subprogram cannot evaluate"
            )
        return [_axis_origin(dims, axis) for axis in range(len(dims))] if dims else None

    def _reduction(self, name: str, arguments: list[str]) -> str:
        if len(arguments) == 2 and arguments[1].startswith("dim="):
            arguments = [arguments[0], arguments[1][len("dim=") :]]
        if any("=" in a.split("(")[0] for a in arguments):
            raise NoRule(f"{name} with a dim= or mask= keyword")
        collapses_an_axis = len(arguments) == 2 and name not in ("dot_product", "matmul")
        if collapses_an_axis:
            # Fortran's DIM is 1-based and names a dimension; an axis is 0-based.
            return self.axis_reduction(REDUCTIONS[name], arguments[0], arguments[1])
        if name == "sum" and len(arguments) == 1:
            # Whole-array SUM folds left to right; np.sum is pairwise and
            # rounds an ULP off. The runtime shim accumulates in order, the
            # way DOT_PRODUCT already does and for the same reason.
            return f"_f_vsum({arguments[0]})"
        return f"{REDUCTIONS[name]}({', '.join(arguments)})"

    def axis_reduction(self, spelling: str, array: str, dimension: str) -> str:
        """A reduction that collapses the axis Fortran's DIM names."""
        return f"{spelling}({array}, axis=({dimension}) - 1)"

    def _over_arrays(self, name: str, arguments: list[str]) -> str:
        if name == "merge":
            if len(arguments) != 3:
                raise NoRule("merge with keyword or missing arguments")
            return f"np.where({arguments[2]}, {arguments[0]}, {arguments[1]})"
        if name in ("max", "min") and len(arguments) > 2:
            # Fortran folds left: min(a, b, c) is min(min(a, b), c). Folding
            # the other way changes which NaN survives.
            folded = arguments[0]
            for argument in arguments[1:]:
                folded = f"{self.array_table[name]}({folded}, {argument})"
            return folded
        complex_spelled = self._complex_conversion(name, arguments)
        if complex_spelled is not None:
            return complex_spelled
        if name in ELEMENTAL_ARRAY:
            arguments = _without_kind(name, arguments)
            return f"{self.array_table[name]}({', '.join(arguments)})"
        if name in REDUCTIONS:
            return self._reduction(name, arguments)
        if name in ELEMENTAL_SCALAR:
            # An intrinsic with no vector spelling keeps its scalar one, as
            # the pipeline's rank>0 branch falls back to its scalar map.
            # Without this, ``index(letters, s(i:i))`` -- a substring actual
            # ranks as a section -- reached the subscript fallback and came
            # out ``index[letters - 1, ...]``: runnable, wrong, mechanical.
            arguments = _without_kind(name, arguments)
            return f"{self.scalar_table[name]}({', '.join(arguments)})"
        raise UnknownReference(name)

    def _complex_conversion(self, name: str, arguments: list[str]) -> str | None:
        """``cmplx`` and the real part of a complex, spelled at their kind.

        ``cmplx(x, kind = k)`` is ``np.complex128(x)`` (``np.complex64`` for
        a single kind): NumPy's scalar constructors take arrays too, where
        Python's ``complex`` -- the table's old spelling -- takes neither an
        array nor a kind. The kind is read through the unit's kind map; a
        ``cmplx`` without one is the default complex, single precision like
        the default real, which no source in the corpus means, so it is
        refused rather than guessed, as is a kind the map does not know and
        the two-part form ``cmplx(x, y)``.

        ``real(z)`` / ``dble(z)`` of a name declared complex is its real
        part, ``np.real``, and nothing more: the part already has the
        complex's real kind, and ``np.float64`` of a complex array discards
        the imaginary part behind a warning, of a complex scalar is a
        TypeError, and of a traced value cannot lower. A conversion of
        anything not declared complex is untouched here.
        """
        if name == "cmplx":
            values = [a for a in arguments if "kind=" not in a.lower()]
            kinds = [a for a in arguments if "kind=" in a.lower()]
            if not kinds and len(values) == 3:
                values, kinds = values[:2], values[2:]
            if len(values) != 1:
                raise NoRule("cmplx(x, y): the two-part form is not spelled; cmplx(x, kind=k) is")
            if not kinds:
                raise NoRule("cmplx without a kind is the default (single) complex; say the kind")
            dtype = self._kind_dtype(kinds[0])
            if dtype is None:
                raise NoRule(f"cmplx: kind {kinds[0]!r} names no real kind this unit knows")
            return f"np.{ {'float64': 'complex128', 'float32': 'complex64'}[dtype] }({values[0]})"
        if name in ("real", "dble", "float") and arguments:
            leading = _LEADING_NAME.match(arguments[0])
            declared = self.semantics.declaration(leading.group(1)) if leading else None
            if declared is None or not str(declared.get("dtype", "")).startswith("complex"):
                return None
            return f"np.real({arguments[0]})"
        return None

    def _kind_dtype(self, text: str) -> str | None:
        """``kind=core_rknd`` / ``core_rknd`` / ``8`` -> ``float64``, or None."""
        token = text.split("=", 1)[1] if "=" in text else text
        token = token.strip().strip("()").split(".")[-1].strip().lower()
        if token.startswith("i_") and token[2:].isdigit():
            # An integer literal as this backend spells it (``I_8``).
            token = token[2:]
        if token.isdigit():
            return {"8": "float64", "4": "float32"}.get(token)
        return self.kind_map.get(token)

    def _over_scalars(self, name: str, arguments: list[str]) -> str:
        if name == "merge":
            if len(arguments) != 3:
                raise NoRule("merge with keyword or missing arguments")
            # Fortran evaluates both branches. Safe as a conditional only
            # because expressions that reach here are pure.
            return f"(({arguments[0]}) if ({arguments[2]}) else ({arguments[1]}))"
        complex_spelled = self._complex_conversion(name, arguments)
        if complex_spelled is not None:
            return complex_spelled
        if name in ELEMENTAL_SCALAR:
            arguments = _without_kind(name, arguments)
            if self.elemental and name in ("exp", "log", "log10"):
                # An elemental body is written at scalar rank and runs over
                # arrays: math.* would reject one and np.* is an ULP off libm.
                return f"{self.array_table[name]}({', '.join(arguments)})"
            return f"{self.scalar_table[name]}({', '.join(arguments)})"
        raise UnknownReference(name)

    def _array_transform(self, name: str, items: list[Any]) -> str:
        positional, keyword = [], {}
        for item in items:
            if isinstance(item, f03.Actual_Arg_Spec):
                keyword[str(item.children[0]).lower()] = self.render(item.children[1])
            else:
                positional.append(self.render(item))

        def argument(index: int, key: str) -> str | None:
            if key in keyword:
                return keyword[key]
            return positional[index] if len(positional) > index else None

        def node(index: int, key: str) -> Any:
            """The source node ``argument`` rendered, for asking its rank."""
            for item in items:
                if isinstance(item, f03.Actual_Arg_Spec) and str(item.children[0]).lower() == key:
                    return item.children[1]
            values = [item for item in items if not isinstance(item, f03.Actual_Arg_Spec)]
            return values[index] if len(values) > index else None

        if name == "transpose":
            return f"np.asfortranarray({positional[0]}.T)"
        if name == "matmul":
            return f"np.matmul({positional[0]}, {positional[1]})"
        if name == "reshape":
            return self._reshape(items, argument)
        if name == "spread":
            source, dim, copies = argument(0, "source"), argument(1, "dim"), argument(2, "ncopies")
            if source is None or dim is None or copies is None:
                raise NoRule("spread with missing arguments")
            return f"np.repeat(np.expand_dims({source}, ({dim}) - 1), {copies}, axis=({dim}) - 1)"
        if name == "pack":
            array, mask, vector = argument(0, "array"), argument(1, "mask"), argument(2, "vector")
            if array is None or mask is None:
                raise NoRule("pack without an array or a mask")
            if vector is None and self._ranked(node(0, "array"), node(1, "mask")) == 1:
                # Boolean indexing is PACK only here: it walks the selected
                # elements in row-major order, which is array element order
                # at rank 1 alone.
                return f"({array})[({mask})]"
            # A rank-2 array packed by boolean indexing came out by rows --
            # 1 3 2 4 where Fortran packs 1 2 3 4 -- and VECTOR, which
            # sizes the result and supplies its tail, was dropped
            # (FNP-D0041, FNP-D0018). The runtime gathers column-major.
            tail = f", vector={vector}" if vector is not None else ""
            return f"_f_pack({array}, {mask}{tail})"
        if name == "unpack":
            mask, field_ = argument(1, "mask"), argument(2, "field")
            if mask is None or field_ is None:
                raise NoRule("unpack with missing arguments")
            return f"_f_unpack({positional[0]}, {mask}, {field_})"
        if name == "cshift":
            shift = argument(1, "shift")
            if shift is None:
                raise NoRule("cshift without a shift")
            dim = argument(2, "dim") or "1"
            return f"np.roll({positional[0]}, -({shift}), axis=({dim}) - 1)"
        if name == "eoshift":
            shift = argument(1, "shift")
            if shift is None:
                raise NoRule("eoshift without a shift")
            dim = argument(3, "dim") or "1"
            # EOSHIFT(ARRAY, SHIFT [, BOUNDARY, DIM]): the third argument is
            # what fills the vacated end. Dropped, ``eoshift(a, 1, -1d0)``
            # filled it with zero (FNP-D0019).
            boundary = argument(2, "boundary")
            fill = f", boundary={boundary}" if boundary is not None else ""
            return f"_f_eoshift({positional[0]}, {shift}, axis=({dim}) - 1{fill})"
        if name in ("maxloc", "minloc"):
            return self._locate(name, items, positional, keyword)
        raise NoRule(f"unhandled array transform {name!r}")

    def _ranked(self, *nodes: Any) -> int | None:
        """The one rank every node has, or ``None`` when they differ or one
        of them the semantics cannot rank."""
        try:
            ranks = {self.semantics.rank(n) for n in nodes if n is not None}
        except REFUSED:
            return None
        return ranks.pop() if len(ranks) == 1 and None not in nodes else None

    def _reshape(self, items: list[Any], argument: Any) -> str:
        """``RESHAPE(source, shape [, pad] [, order])``.

        ``np.reshape`` is RESHAPE only when the source has exactly as many
        elements as the shape asks for and neither PAD nor ORDER is given:
        a larger source is legal Fortran (the result takes its leading
        elements) and raised, and PAD and ORDER were dropped -- ``reshape(a,
        [2, 3], order=[2, 1])`` came out column by column, and a short
        source with a PAD raised (FNP-D0017, FNP-D0048). The runtime's
        ``_f_reshape`` is the standard's definition; the NumPy call stays
        where the two sizes are literals and agree, which is where it was
        right.
        """
        source, shape = argument(0, "source"), argument(1, "shape")
        if source is None or shape is None:
            raise NoRule("reshape without a source or a shape")
        pad, order = argument(2, "pad"), argument(3, "order")
        values = [a for a in items if not isinstance(a, f03.Actual_Arg_Spec)]
        wanted = _literal_product(values[1]) if len(values) > 1 else None
        if pad is None and order is None and wanted is not None:
            if wanted == self._literal_size(values[0]):
                return f"np.reshape({source}, {shape}, order='F')"
        keywords = [f"pad={pad}"] * (pad is not None) + [f"order={order}"] * (order is not None)
        return f"_f_reshape({', '.join([source, shape, *keywords])})"

    def _literal_size(self, node: Any) -> int | None:
        """How many elements a whole array has, when its declared bounds are
        integer literals, or an array constructor has, when every item is a
        scalar -- ``[1d0, 2d0, 3d0, 4d0]``, the commonest RESHAPE source,
        whose text ``np.reshape`` spelled before ``_f_reshape`` existed;
        ``None`` for anything else."""
        if isinstance(node, f03.Array_Constructor):
            listed = node.children[1]
            values = list(listed.children) if isinstance(listed, f03.Ac_Value_List) else [listed]
            try:
                scalar = all(
                    not isinstance(v, (f03.Ac_Implied_Do, f03.Ac_Spec))
                    and self.semantics.rank(v) == 0
                    for v in values
                )
            except REFUSED:
                return None
            return len(values) if scalar else None
        if not isinstance(node, f03.Name):
            return None
        name = str(node).lower()
        dims = self.allocated_bounds.get(name, (self.semantics.declaration(name) or {}).get("dims"))
        if not dims or dims == CONFLICTING_BOUNDS:
            return None
        size = 1
        for dim in dims:
            low = _integer_text(str(dim.get("lb") or "1"))
            high = _integer_text(str(dim.get("ub") or ""))
            if low is None or high is None:
                return None
            size *= max(high - low + 1, 0)
        return size

    def _locate(
        self, name: str, items: list[Any], positional: list[str], keyword: dict[str, str]
    ) -> str:
        """``MAXLOC``/``MINLOC`` return 1-based positions, not 0-based ones.

        Two argument lists share the name: ``(ARRAY, DIM [, MASK, KIND,
        BACK])`` and ``(ARRAY [, MASK, KIND, BACK])``, told apart by the
        second argument's type. MASK was read as a DIM when positional and
        dropped when a keyword -- ``maxloc(a, mask=m)`` answered the whole
        array's maximum (FNP-D0020) -- and the whole-array search took
        NumPy's row-major first maximum where Fortran's is the first in
        array element order, so a rank-2 tie came out at the other element
        (FNP-D0042). The runtime's ``_f_loc`` searches column-major under
        the mask. BACK, which asks for the last extremum, is refused;
        KIND sizes the result's integers and changes no value.
        """
        values = [item for item in items if not isinstance(item, f03.Actual_Arg_Spec)]
        array = positional[0] if values else keyword.get("array")
        if array is None:
            raise NoRule(f"{name} without an array")
        dim, mask = keyword.get("dim"), keyword.get("mask")
        back = keyword.get("back")
        if len(values) > 1:
            if self.semantics.is_logical_or_character(values[1]):
                mask, back = positional[1], (positional[3] if len(values) > 3 else back)
            elif self.semantics.is_integer(values[1]):
                dim = positional[1]
                mask = positional[2] if len(values) > 2 else mask
                back = positional[4] if len(values) > 4 else back
            else:
                raise NoRule(f"{name}: cannot tell whether its second argument is DIM or MASK")
        if back is not None and back != "False":
            raise NoRule(f"{name} with BACK=, the last extremum rather than the first")
        find = "np.argmax" if name == "maxloc" else "np.argmin"
        if dim is not None and mask is not None:
            raise NoRule(f"{name} with both DIM= and MASK=")
        extremum = "max" if name == "maxloc" else "min"
        # NumPy's search answers the first NaN, gfortran's skips them; only
        # an INTEGER array, which holds none, keeps NumPy's spelling.
        integer = bool(values) and self.semantics.is_integer(values[0])
        if dim is not None:
            if not integer:
                return f"_f_loc({array}, '{extremum}', dim={dim})"
            return f"({find}({array}, axis=({dim}) - 1) + 1)"
        if mask is None and integer and self._ranked(values[0]) == 1:
            # One axis: row-major and array element order are the same order.
            return f"(np.array(np.unravel_index({find}({array}), np.shape({array}))) + 1)"
        tail = f", mask={mask}" if mask is not None else ""
        return f"_f_loc({array}, '{extremum}'{tail})"

    # -- structure constructors -----------------------------------------------

    def _constructor_is_reference(self, name: str) -> bool:
        """Whether a constructor-parsed ``name(...)`` is really a reference.

        Nothing in scope claims the name as a procedure, a generic or an
        array, and either its canonical spelling is an intrinsic this
        translation maps, or the name is a procedure dummy -- declared one,
        or a scalar dummy of this subprogram used as one.
        """
        if (
            name in self.semantics.procedures
            or name in self.semantics.generics
            or name in self.semantics.companion_generics
            or self.semantics.is_array(name)
        ):
            return False
        canonical = F77_SPECIFIC_TO_GENERIC.get(name, name)
        if canonical in ELEMENTAL_SCALAR or canonical in REDUCTIONS or canonical in ARRAY_TRANSFORM:
            return True
        declared = self.semantics.declaration(name)
        return declared is not None and (
            bool(declared.get("procedure"))
            or (
                not declared.get("dims")
                and any(a["name"] == name for a in self.semantics.subprogram["args"])
            )
        )

    def _structure_constructor(self, node: Any) -> str:
        """fparser reads an ambiguous call as a constructor.

        A reference with character arguments, or one naming a generic, is
        parsed this way because it cannot be told from building a derived type
        without knowing what the name is. Resolving it here rather than in the
        parser keeps that knowledge in one place.
        """
        name = str(node.children[0]).lower()
        items = _items(node.children[1])
        if self._constructor_is_reference(name):
            # An intrinsic -- an F77 specific spelling included -- or a
            # procedure dummy parsed this way is a plain function reference,
            # and ``reference`` owns the mapping. Left here, ``datan2(0d0,
            # -one)`` was emitted verbatim: a call to a name nothing defines.
            return self.reference(node)
        arguments = self._arguments(items)

        if name in self.semantics.generics:
            name = self.semantics.dispatch(name, items)
        call = self._call(name, items, arguments)
        if call is not None:
            return call
        transform = self.function_transforms.get(name)
        if transform is not None:
            return str(transform(list(arguments)))
        external = self.externals.get(name)
        if external is not None and external.get("kind") == "function":
            return f"_ext.{name}({', '.join(arguments)})"
        if name in self.stubs:
            return self.stubs[name]
        # fparser reads an unknown `f(args)` as a constructor whenever the
        # arguments look like components -- a character actual, a keyword.
        # Nothing here defines a type of that name either, so it is a call,
        # which is what the source spelling says.
        return f"{self.names.symbol(name)}({', '.join(arguments)})"


def _is_arithmetic(operator: Any) -> bool:
    """``+ - * /`` between two operands, whose KIND is the wider one's."""
    return isinstance(operator, str) and operator in ("+", "-", "*", "/")


def _signed(node: Any) -> bool:
    """``-x`` or ``+x``: fparser's unary form, a sign and an operand."""
    children: Any = getattr(node, "children", None)
    return (
        bool(children)
        and len(children) == 2
        and isinstance(children[0], str)
        and children[0] in ("+", "-")
    )


def _axis_origin(dims: list[dict[str, Any]], axis: int) -> str:
    """One axis's declared lower bound as ``indexing.describe`` reads it."""
    lower = dims[axis].get("lb")
    return indexing.UNIT_ORIGIN if lower in (None, "", ":") else str(lower)


def _integer_text(text: str) -> int | None:
    """A declared bound that is an integer literal, as its value."""
    try:
        return int(text)
    except ValueError:
        return None


def _literal_product(node: Any) -> int | None:
    """The product of an array constructor of integer literals -- a shape
    written out, ``[2, 3]`` -- or ``None``."""
    if not isinstance(node, f03.Array_Constructor):
        return None
    values = _items(node.children[1])
    if not values or not all(isinstance(v, f03.Int_Literal_Constant) for v in values):
        return None
    product = 1
    for value in values:
        product *= int(value.children[0])
    return product


def _bound_text(node: Any) -> str | None:
    """Fortran text compared as text: case and blanks do not matter."""
    if node is None:
        return None
    return str(node).replace(" ", "").lower()


def _items(arglist: Any) -> list[Any]:
    if arglist is None:
        return []
    return list(arglist.children) if hasattr(arglist, "children") else [arglist]


def expand_power(base: str, exponent: int) -> str:
    """``x**5`` as ``x * x * x * x * x``, square-and-multiply, LSB first.

    Exactly the order gfortran's own expansion uses, because the point is to
    round the way the reference binary rounds. A ``pow`` call is one to two
    ULP away from this, which is enough to fail a bit-exact gate and not
    enough for anyone to notice by looking.
    """
    negative = exponent < 0
    remaining, result, square = abs(exponent), None, base
    while remaining:
        if remaining & 1:
            result = square if result is None else f"({result} * {square})"
        remaining >>= 1
        if remaining:
            square = f"({square} * {square})"
    return f"(1.0 / {result})" if negative else str(result)


def _complex_half(node: Any) -> str:
    """One component of a complex literal, kind suffix and D exponent gone."""
    return str(node).split("_")[0].replace("d", "e").replace("D", "E")
