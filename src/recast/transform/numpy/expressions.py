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

import re
from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03
from recast.fortran.interface import CONFLICTING_BOUNDS, emit_name
from recast.fortran.semantics import F77_SPECIFIC_TO_GENERIC, Semantics, Unanalyzable
from recast.transform.numpy.names import Names
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

BOUND_TOKENS = re.compile(r"[A-Za-z_]\w*\s*%\s*[A-Za-z_]\w*|[A-Za-z_]\w*|\d+|[()+\-*/ ]")
"""What a declared bound is allowed to be made of. Bound texts are simple by
construction; anything richer refuses the statement that needed the bound."""

__all__ = ["REFUSED", "Expressions", "Remote"]

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

MAX_EXPANDED_POWER = 16
"""Beyond this, expanding ``x**n`` to multiplications stops being worth reading
and starts being a place for a transcription error. Refused instead."""


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
            if self.vector_boolean and spelling in (".AND.", ".OR."):
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
        return f"{scalar}({left}, {right})" if scalar else None

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
        if self.vector_boolean:
            return f"(~({self.render(operand)}))"
        return f"(not {self.render(operand)})"

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

    def bound(self, text: str) -> str:
        """Declared bound text -> Python. Bound texts are simple -- names,
        integers, ``+ - * /``, parentheses, ``size(a, n)`` -- by construction;
        anything else refuses the statement that needed the bound."""

        # An automatic array sized off another argument. UBOUND is the same
        # question with unit lower bounds, which every translated array has,
        # and the dimension may be written with or without ``dim=``.
        def extent(match: re.Match[str]) -> str:
            name = self.names.symbol(match.group(1).lower())
            dimension = match.group(2)
            if dimension is None:
                return self.extent_of(name)
            axis = int(DIM_KEYWORD.sub("", dimension)) - 1
            return self.extent_along(name, axis)

        if EXTENT.fullmatch(text):
            return EXTENT.sub(extent, text)
        substituted = EXTENT.sub(extent, text)
        if substituted != text:
            text = substituted
        rendered, position = [], 0
        for match in BOUND_TOKENS.finditer(text):
            if match.start() != position:
                raise NoRule(f"dim expr {text!r}")
            position = match.end()
            token = match.group(0)
            if "%" in token:
                # ``bounds%begp`` sizing a local: the component of a dummy,
                # which is an attribute of the same name on this side.
                root, component = (t.strip() for t in token.split("%", 1))
                rendered.append(f"{self.names.symbol(root)}.{pysafe(component.lower())}")
            elif re.match(r"[A-Za-z_]", token):
                rendered.append(self.names.symbol(token))
            elif token.isdigit() and token not in ("0", "1", "2"):
                hoisted = self.names.literals.get(token)
                if hoisted is None:
                    raise NoRule(f"declared dim literal {token}")
                rendered.append(hoisted)
            else:
                rendered.append(token)
        if position != len(text):
            raise NoRule(f"dim expr {text!r}")
        return "".join(rendered)

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
        parts = [self._position(p) for p in positions]
        return f"{self.names.symbol(name)}[{', '.join(parts)}]"

    def _position(self, position: indexing.Position) -> str:
        if position.kind is Kind.VECTOR:
            return f"(({self.render(position.index)}) - 1)"
        if position.kind is Kind.INDEX:
            folded = indexing.fold(position, WHITELIST_INT)
            if folded is not None:
                return str(folded)
            return self._shift(self.render(position.index), position.origin)
        return self._range(position)

    def _range(self, position: indexing.Position) -> str:
        """``lo:hi`` inclusive becomes ``lo':hi'+1`` exclusive."""
        step = self.render(position.step) if position.step is not None else None
        if step is not None and step.lstrip("(").startswith("-"):
            # A descending section: the runtime works the edges out, because
            # either may be implied and the stop edge underflows at the first
            # element. The declared lower bound goes with them.
            lower = self.render(position.lower) if position.lower is not None else "None"
            upper = self.render(position.upper) if position.upper is not None else "None"
            return f"_f_rstep_lb({lower}, {upper}, {step}, {self._origin(position.origin)})"
        start = ""
        if position.lower is not None:
            folded = indexing.fold_index(position.lower, position.origin, WHITELIST_INT)
            start = (
                str(folded)
                if folded is not None
                else self._shift(self.render(position.lower), position.origin)
            )
        stop = ""
        if position.upper is not None:
            rendered = self.render(position.upper)
            stop = (
                rendered
                if position.shifts_by_one
                else f"({rendered}) - ({self._origin(position.origin)}) + 1"
            )
        return f"{start}:{stop}" + (f":{step}" if step is not None else "")

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
                parts.append(f"{head}[{', '.join(self._position(p) for p in positions)}]")
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
                return self._association_tail(actual, formal_dims)
            if rank is not None and 0 < rank < len(formal_dims):
                raise NoRule(
                    f"seq-assoc: rank-{rank} actual for the rank-{len(formal_dims)} "
                    f"assumed-size dummy {formal['name']}"
                )
            return rendered
        if not all(d.get("ub") for d in formal_dims):
            return rendered
        if rank is not None and 0 < rank < len(formal_dims):
            # Fortran sequence association: a lower-rank actual fills the
            # dummy in column-major order.
            shape = ", ".join(self.extent(d, substitutions) for d in formal_dims)
            return f"np.reshape({rendered}, ({shape},), order='F')"
        if element:
            return self.sequence_association(actual, formal_dims, substitutions)
        return rendered

    @staticmethod
    def _assumed_size(formal_dims: list[dict[str, Any]]) -> bool:
        """``x(*)`` or ``x(n, *)``: the last axis has no extent of its own."""
        return bool(formal_dims and formal_dims[-1].get("assumed_size"))

    def _association_tail(self, actual: Any, formal_dims: list[dict[str, Any]]) -> str:
        """An element actual for an assumed-size dummy: the actual's memory
        from the element on, as a view the callee reads and writes in place.

        Only a rank-1 actual has that as a view. A higher-rank actual's tail
        in column-major order is ``ravel(order='F')``, which is a copy unless
        the array happens to be Fortran-contiguous, and a copy is somewhere
        an OUT dummy's writes are lost; a rank-2 assumed-size dummy has no
        extent to reshape to at all. Both are refused.
        """
        name = str(actual.children[0]).lower()
        declaration = self.semantics.declaration(name) or {}
        if len(declaration.get("dims") or []) != 1 or len(formal_dims) != 1:
            raise NoRule(
                f"seq-assoc: element of {name} for an assumed-size dummy is only a view "
                "when both are rank-1"
            )
        return f"{self.names.symbol(name)}[{self._association_start(actual)}:]"

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
            return self._association_tail(actual, formal_dims), True
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
        the same reason. Anything that is not one of the callee's arguments --
        a module parameter, an arithmetic expression over them -- means the
        same thing on both sides and goes through ``bound``.
        """
        text = str(dim["ub"])
        if text.lower() in substitutions:
            substituted = substitutions[text.lower()]
            if not substituted:
                raise NoRule(f"dummy dimension {text!r} is not bound by this call")
            return substituted
        return self.bound(text)

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
        if record is not None and not remote:
            # Sequence association applies to a function reference as much as
            # to a CALL: ``enorm(m, a(1, j))`` hands the callee the whole of
            # column ``j``, and rendering the element alone hands it a scalar
            # to subscript. Only where every actual is positional -- a keyword
            # actual is not bound to a formal by position, and guessing which
            # dummy it answers is how the reshape lands on the wrong one.
            positional = not any(
                isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)) for item in items
            )
            if positional and len(items) <= len(record["args"]):
                substitutions = self.substitutions(record, items)
                arguments = [
                    self.actual_argument(formal, item, substitutions)
                    for formal, item in zip(record["args"], items, strict=False)
                ]
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

        if rank > 0:
            return self._over_arrays(name, arguments)
        return self._over_scalars(name, arguments)

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

    def _reduction(self, name: str, arguments: list[str]) -> str:
        if len(arguments) == 2 and arguments[1].startswith("dim="):
            arguments = [arguments[0], arguments[1][len("dim=") :]]
        if any("=" in a.split("(")[0] for a in arguments):
            raise NoRule(f"{name} with a dim= or mask= keyword")
        collapses_an_axis = len(arguments) == 2 and name not in ("dot_product", "matmul")
        if collapses_an_axis:
            # Fortran's DIM is 1-based and names a dimension; an axis is 0-based.
            return self.axis_reduction(REDUCTIONS[name], arguments[0], arguments[1])
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

    def _over_scalars(self, name: str, arguments: list[str]) -> str:
        if name == "merge":
            if len(arguments) != 3:
                raise NoRule("merge with keyword or missing arguments")
            # Fortran evaluates both branches. Safe as a conditional only
            # because expressions that reach here are pure.
            return f"(({arguments[0]}) if ({arguments[2]}) else ({arguments[1]}))"
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

        if name == "transpose":
            return f"np.asfortranarray({positional[0]}.T)"
        if name == "matmul":
            return f"np.matmul({positional[0]}, {positional[1]})"
        if name == "reshape":
            shape = argument(1, "shape")
            if shape is None:
                raise NoRule("reshape without a shape")
            return f"np.reshape({positional[0]}, {shape}, order='F')"
        if name == "spread":
            source, dim, copies = argument(0, "source"), argument(1, "dim"), argument(2, "ncopies")
            if source is None or dim is None or copies is None:
                raise NoRule("spread with missing arguments")
            return f"np.repeat(np.expand_dims({source}, ({dim}) - 1), {copies}, axis=({dim}) - 1)"
        if name == "pack":
            mask = argument(1, "mask")
            if mask is None:
                raise NoRule("pack without a mask")
            return f"({positional[0]})[({mask})]"
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
            return f"_f_eoshift({positional[0]}, {shift}, axis=({dim}) - 1)"
        if name in ("maxloc", "minloc"):
            return self._locate(name, positional, keyword)
        raise NoRule(f"unhandled array transform {name!r}")

    @staticmethod
    def _locate(name: str, positional: list[str], keyword: dict[str, str]) -> str:
        """``MAXLOC``/``MINLOC`` return 1-based positions, not 0-based ones."""
        find = "np.argmax" if name == "maxloc" else "np.argmin"
        array = positional[0]
        dim = keyword.get("dim", positional[1] if len(positional) > 1 else None)
        if dim is not None:
            return f"({find}({array}, axis=({dim}) - 1) + 1)"
        return f"(np.array(np.unravel_index({find}({array}), np.shape({array}))) + 1)"

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
