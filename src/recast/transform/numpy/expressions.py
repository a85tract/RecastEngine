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

from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03
from recast.fortran.semantics import Semantics, Unanalyzable
from recast.transform.numpy.names import Names
from recast.transform.numpy.vocabulary import (
    ARITH_OPS,
    ARRAY_TRANSFORM,
    ELEMENTAL_ARRAY,
    ELEMENTAL_SCALAR,
    INTEL_ARRAY,
    INTEL_SCALAR,
    LOGICAL_OPS,
    REDUCTIONS,
    RELATIONAL_OPS,
    WHITELIST_INT,
    pysafe,
)
from recast.transform.profiles import Profile
from recast.transform.rules import NoRule, indexing
from recast.transform.rules.indexing import Kind

__all__ = ["Expressions", "Remote"]

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

KIND_SECOND_ARGUMENT = frozenset(
    {"aint", "anint", "ceiling", "dble", "float", "floor", "int", "nint", "real"}
)
"""Conversions whose optional second argument names a kind, not a value."""

MAX_EXPANDED_POWER = 16
"""Beyond this, expanding ``x**n`` to multiplications stops being worth reading
and starts being a place for a transcription error. Refused instead."""


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

    stubs: dict[str, str] = field(default_factory=dict)
    """Framework function -> the text that stands in for it.

    A call into a framework the translation does not carry -- CAM's history
    buffer answering whether a field is active, its unit manager handing out a
    file unit -- has an answer that is a property of the framework, not of the
    language. So it is supplied, like ``intent_overrides`` and ``externals``,
    and the domain package that knows CAM ships the table. Without one the
    call is refused and becomes a deferred site, which is the honest outcome:
    the engine genuinely does not know what ``hist_fld_active`` returns.

    Consulted only for references fparser read as structure constructors,
    which is where the pipeline consults its copy. A plainly-parsed reference
    to a stubbed name refuses even when the table has an answer -- the wider
    placement this module briefly had turned a refusal the pipeline hands to
    a human into a fabricated constant.
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
        if isinstance(node, f03.Parenthesis):
            return f"({self.render(node.children[1])})"
        if isinstance(node, f03.Data_Ref):
            return self._data_ref(node)
        if isinstance(node, f03.Array_Constructor):
            items = node.children[1]
            values = items.children if hasattr(items, "children") else [items]
            return f"np.array([{', '.join(self.render(v) for v in values)}])"
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
        """Intrinsic -> spelling on a scalar, under this profile."""
        if self.profile.intel_math:
            return {**ELEMENTAL_SCALAR, **INTEL_SCALAR}
        return ELEMENTAL_SCALAR

    @property
    def array_table(self) -> dict[str, str]:
        """Intrinsic -> spelling on an array, under this profile."""
        if self.profile.intel_math:
            return {**ELEMENTAL_ARRAY, **INTEL_ARRAY}
        return ELEMENTAL_ARRAY

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
            if self.profile.intel_math:
                return f"intel_math.vpow({left}, {right})"
            return f"_f_vpow({left}, {right})"
        if self.profile.intel_math:
            return f"intel_math.pow({left}, {right})"
        return None

    def _comparison(self, spelling: str, left: Any, right: Any, rl: str, rr: str) -> str:
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
        return self._intrinsic(name, items, arguments)

    def subscript(self, name: str, arglist: Any) -> str:
        """An array element or slice, shifted to zero-based."""
        declaration = self.semantics.declaration(name)
        dims = self.allocated_bounds.get(name, (declaration or {}).get("dims"))
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
            return f"_f_rstep({self.render(position.lower)}, {self.render(position.upper)}, {step})"
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
            stop = rendered if position.shifts_by_one else f"({rendered}) - ({position.origin}) + 1"
        return f"{start}:{stop}" + (f":{step}" if step is not None else "")

    @staticmethod
    def _shift(rendered: str, origin: str) -> str:
        if origin == indexing.UNIT_ORIGIN:
            return f"{rendered} - 1"
        return f"({rendered}) - ({origin})"

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
                positions = indexing.describe(
                    component.children[1], None, rank_of=self.semantics.rank
                )
                parts.append(f"{head}[{', '.join(self._position(p) for p in positions)}]")
            else:
                raise NoRule(f"data-ref component {type(component).__name__}")
        return ".".join(parts)

    # -- calls ----------------------------------------------------------------

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
        target = f"{remote.alias}.{remote.name}" if remote else pysafe(name)
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
            raise NoRule(f"cannot rank the arguments of {name!r}") from None

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
            return f"{REDUCTIONS[name]}({arguments[0]}, axis=({arguments[1]}) - 1)"
        return f"{REDUCTIONS[name]}({', '.join(arguments)})"

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
            if name in KIND_SECOND_ARGUMENT and len(arguments) == 2:
                arguments = arguments[:1]
            return f"{self.array_table[name]}({', '.join(arguments)})"
        raise NoRule(f"no elementwise rule for {name!r}")

    def _over_scalars(self, name: str, arguments: list[str]) -> str:
        if name == "merge":
            if len(arguments) != 3:
                raise NoRule("merge with keyword or missing arguments")
            # Fortran evaluates both branches. Safe as a conditional only
            # because expressions that reach here are pure.
            return f"(({arguments[0]}) if ({arguments[2]}) else ({arguments[1]}))"
        if name in ELEMENTAL_SCALAR:
            if name in KIND_SECOND_ARGUMENT and len(arguments) == 2:
                arguments = arguments[:1]
            if self.elemental and name in ("exp", "log", "log10"):
                # An elemental body is written at scalar rank and runs over
                # arrays: math.* would reject one and np.* is an ULP off libm.
                return f"{self.array_table[name]}({', '.join(arguments)})"
            return f"{self.scalar_table[name]}({', '.join(arguments)})"
        raise NoRule(f"unknown function or array reference {name!r}")

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

    def _structure_constructor(self, node: Any) -> str:
        """fparser reads an ambiguous call as a constructor.

        A reference with character arguments, or one naming a generic, is
        parsed this way because it cannot be told from building a derived type
        without knowing what the name is. Resolving it here rather than in the
        parser keeps that knowledge in one place.
        """
        name = str(node.children[0]).lower()
        items = _items(node.children[1])
        arguments = self._arguments(items)

        if name in self.semantics.generics:
            name = self.semantics.dispatch(name, items)
        call = self._call(name, items, arguments)
        if call is not None:
            return call
        external = self.externals.get(name)
        if external is not None and external.get("kind") == "function":
            return f"_ext.{name}({', '.join(arguments)})"
        if name in self.stubs:
            return self.stubs[name]
        raise NoRule(f"structure constructor {name!r}")


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
