"""What an expression means, without deciding how to write it down.

Migrated from the type and shape reasoning inside CESM-language-translator's
``Translator``, where it sat among the emitters and could only be reached by
constructing one. It answers questions about Fortran -- what rank is this, is
it integer-valued, is it a compile-time constant, which specific procedure does
this generic call dispatch to -- and every answer would be the same for a
Julia or C++ backend.

That is the test for whether something belongs here: it belongs if it cannot
produce a line of target-language source. ``dim_expr``, which looks like a
sibling, stayed with the emitter because its answer is Python text.

Every method refuses rather than guesses. ``rank`` on an unrecognised
reference raises ``Unanalyzable`` instead of assuming scalar, because scalar is
the answer that silently produces working code with the wrong broadcast. A
Transform turns the refusal into a deferred site; a Verifier turns it into a
block it will not vouch for. Neither is a failure -- an honest gap is what the
agent layer consumes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03
from recast.fortran.intrinsics import ELEMENTAL, STATE_QUERY, TRANSFORMATIONAL

INTEGER_DTYPES = frozenset({"int32", "int64"})

ARITHMETIC = frozenset({"+", "-", "*", "/", "**"})

INTEGER_RESULT_INTRINSICS = frozenset({"int", "nint", "mod"})
"""Intrinsics whose result is integer whatever the argument is.

Short because it only has to be right, not complete: a name missing from it is
reported as non-integer, and the rule it feeds -- Fortran's ``/`` truncates
between two integers -- then keeps the safer real division.
"""

DERIVED_TYPE_MARKER = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)")
"""How ``interface.dtype_of`` spells a type it will not reduce to a dtype.

It refuses on purpose -- a derived type has no single numeric type -- and the
name is recoverable from the refusal, which is what makes component lookup
possible without a second pass.
"""


def _is_operator(node: Any, among: frozenset[str] | set[str]) -> bool:
    """Whether a child is one of these operator spellings.

    The ``isinstance`` is what makes this a function rather than an ``in``:
    fparser puts operators in the same child list as nodes, and several node
    types are unhashable, so testing membership directly raises a TypeError on
    an array constructor. Cheap to write, and it took a run over real source
    to notice.
    """
    return isinstance(node, str) and node in among


class Unanalyzable(RecastError):
    """This expression's meaning could not be settled without guessing."""


class AmbiguousDispatch(Unanalyzable):
    """A generic call matched none of its specifics, or more than one."""


@dataclass
class Semantics:
    """Type and shape questions, answered for one subprogram.

    Built from an ``interface.extract`` record rather than from a parse tree,
    so it costs nothing beyond the analysis the frontend already did.
    """

    module: dict[str, Any]
    """The whole-file record: module state, parameters, derived types."""

    subprogram: dict[str, Any]
    """The one being reasoned about: its arguments, locals and result."""

    procedures: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Name -> record for every callable in scope, including companions.

    Companion modules are merged in rather than kept apart: a call does not
    care which file its callee was declared in, and keeping two tables was how
    the original ended up with three lookups per question.
    """

    generics: dict[str, list[str]] = field(default_factory=dict)
    """Generic interfaces declared in this module."""

    companion_generics: dict[str, list[str]] = field(default_factory=dict)
    """Generic interfaces reached through a sibling translated module.

    Kept apart from the module's own only because ``rank`` treats them
    differently, and it treats them differently because the pipeline this came
    from did. See ``_reference_rank``.
    """
    types: dict[str, dict[str, Any]] = field(default_factory=dict)
    parameters: frozenset[str] = frozenset()
    """Names that are compile-time constants, including companions'."""

    statement_functions: frozenset[str] = frozenset()
    """Names defined by a statement function, which is scalar by definition.

    Supplied by the caller because recognising one needs the execution part,
    which this class deliberately does not read.
    """

    def __post_init__(self) -> None:
        self._declared: dict[str, dict[str, Any]] = {}
        for entry in (
            *self.subprogram["args"],
            *self.subprogram["locals"],
            *self.subprogram["local_parameters"],
            *self.module["module_state"],
            *self.module["module_parameters"],
        ):
            self._declared.setdefault(entry["name"], entry)
        # `rank` looks in a narrower table than everything else, and only
        # because the pipeline this came from did: its shape query skipped
        # local parameters where its type and array queries did not. A
        # 16-element lookup table therefore answers "array" to `is_array` and
        # "scalar" to `rank`. Reproduced rather than resolved -- the
        # pipeline's answers are the ones a bit-exact gate has been run
        # against, and every use of such a table in CAM is subscripted, so
        # nothing in the corpus distinguishes the two.
        local_parameters = {p["name"] for p in self.subprogram["local_parameters"]}
        self._declared_for_rank = {
            name: entry
            for name, entry in self._declared.items()
            if name not in local_parameters
            or any(a["name"] == name for a in self.subprogram["args"])
            or any(loc["name"] == name for loc in self.subprogram["locals"])
        }

    # -- declarations ---------------------------------------------------------

    def declaration(self, name: str) -> dict[str, Any] | None:
        """The nearest declaration of ``name``, innermost scope first.

        A local shadows module state, which is how Fortran scoping works and
        the reason this is a single ordered lookup rather than a search.
        """
        found = self._declared.get(name.lower())
        if found is not None:
            return found
        if self.subprogram.get("result") == name.lower() and self.subprogram.get("result_dims"):
            return {
                "name": name.lower(),
                "dtype": self.subprogram.get("result_dtype"),
                "dims": self.subprogram["result_dims"],
                "intent": "OUT",
                "optional": False,
            }
        return None

    def is_array(self, name: str) -> bool:
        declared = self.declaration(name)
        return bool(declared and declared.get("dims"))

    def derived_type_of(self, name: str) -> str | None:
        """The derived type ``name`` was declared with, if it has one."""
        lowered = name.lower()
        if self.subprogram.get("kind") == "function" and lowered == self.subprogram.get("result"):
            dtype = self.subprogram.get("result_dtype")
        else:
            declared = self.declaration(lowered)
            dtype = declared.get("dtype") if declared else None
        match = DERIVED_TYPE_MARKER.match(str(dtype or ""))
        return match.group(1).lower() if match else None

    def component(self, root: str, name: str) -> dict[str, Any] | None:
        """``b % q`` -> the record for ``q``, if ``b``'s type is known here."""
        type_name = self.derived_type_of(root)
        if type_name is None:
            return None
        return self.types.get(type_name, {}).get(name.lower())

    # -- shape ----------------------------------------------------------------

    def rank(self, node: Any) -> int:
        """Rank of an expression; 0 is scalar.

        Refuses on anything it cannot settle. Guessing scalar here is how a
        translation compiles, runs, and broadcasts one element where the
        Fortran operated on a whole array.
        """
        if isinstance(
            node,
            (
                f03.Real_Literal_Constant,
                f03.Int_Literal_Constant,
                f03.Char_Literal_Constant,
                f03.Logical_Literal_Constant,
            ),
        ):
            return 0
        if isinstance(node, f03.Name):
            declared = self._declared_for_rank.get(str(node).lower())
            return len(declared["dims"]) if declared and declared.get("dims") else 0
        if isinstance(node, f03.Parenthesis):
            return self.rank(node.children[1])
        if isinstance(node, (f03.Part_Ref, f03.Intrinsic_Function_Reference)):
            return self._reference_rank(node)
        if isinstance(node, f03.Array_Constructor):
            return 1
        if isinstance(node, f03.Structure_Constructor):
            return 0  # looks like a call, constructs one value
        if isinstance(node, (f03.Actual_Arg_Spec, f03.Component_Spec)):
            return self.rank(node.children[1])
        if isinstance(node, f03.Data_Ref):
            return self._dataref_rank(node)
        children = getattr(node, "children", None)
        if children and len(children) == 2 and isinstance(children[0], str):
            return self.rank(children[1])
        if children and len(children) == 3 and isinstance(children[1], str):
            return max(self.rank(children[0]), self.rank(children[2]))
        if isinstance(node, f03.And_Operand):
            return self.rank(node.children[1])
        raise Unanalyzable(f"rank of {type(node).__name__}")

    def _arguments(self, node: Any) -> list[Any]:
        if node.children[1] is None:
            return []
        args = node.children[1]
        return list(args.children) if hasattr(args, "children") else [args]

    def _reference_rank(self, node: Any) -> int:
        """A ``name(...)`` is a slice, an intrinsic, or a call, in that order."""
        name = str(node.children[0]).lower()
        items = self._arguments(node)

        if self.is_array(name):
            # Subscripting drops a rank per scalar index; a triplet keeps one.
            return sum(1 for s in items if isinstance(s, f03.Subscript_Triplet))
        if name in self.statement_functions:
            return 0
        if name in self.procedures:
            return self._call_rank(self.procedures[name], items)
        if name in self.companion_generics:
            return 0  # the overload decides, and dispatch is a separate question
        if name in TRANSFORMATIONAL or name in STATE_QUERY - {"merge"}:
            return 0
        if name in ELEMENTAL or name == "merge":
            return self._broadcast_rank(items)
        raise Unanalyzable(f"rank of unknown reference {name!r}")

    def _call_rank(self, record: dict[str, Any], items: list[Any]) -> int:
        """An ELEMENTAL function broadcasts; anything else returns its result."""
        if any("ELEMENTAL" in str(p).upper() for p in (record.get("prefixes") or [])):
            return self._broadcast_rank(items)
        return 0

    def _broadcast_rank(self, items: list[Any]) -> int:
        return max(
            (self.rank(a) for a in items if not isinstance(a, f03.Actual_Arg_Spec)), default=0
        )

    def _dataref_rank(self, node: Any) -> int:
        last = node.children[-1]
        if isinstance(last, f03.Part_Ref):
            items = last.children[1].children if last.children[1] is not None else []
            return sum(1 for s in items if isinstance(s, f03.Subscript_Triplet))
        # A bare component: its declared shape is in the type, which this
        # answer does not consult. Scalar is the assumption, and an array
        # component used in a scalar context fails loudly rather than quietly.
        return 0

    # -- type -----------------------------------------------------------------

    def is_integer(self, node: Any) -> bool:
        """Whether an expression is integer-valued.

        Drives the one rule where getting the type wrong changes arithmetic
        rather than just spelling: Fortran's ``/`` between two integers
        truncates. Answering False when unsure keeps real division, which is
        the direction that stays visible.
        """
        if isinstance(node, f03.Int_Literal_Constant):
            return True
        if isinstance(
            node,
            (
                f03.Real_Literal_Constant,
                f03.Char_Literal_Constant,
                f03.Logical_Literal_Constant,
            ),
        ):
            return False
        if isinstance(node, f03.Name):
            declared = self.declaration(str(node))
            return declared is not None and declared.get("dtype") in INTEGER_DTYPES
        if isinstance(node, f03.Parenthesis):
            return self.is_integer(node.children[1])
        if isinstance(node, (f03.Part_Ref, f03.Intrinsic_Function_Reference)):
            name = str(node.children[0]).lower()
            if name in INTEGER_RESULT_INTRINSICS:
                return True
            record = self.procedures.get(name)
            return record is not None and record.get("result_dtype") in INTEGER_DTYPES
        children = getattr(node, "children", None)
        if children and len(children) == 2 and isinstance(children[0], str):
            return self.is_integer(children[1])
        if children and len(children) == 3 and _is_operator(children[1], ARITHMETIC):
            return self.is_integer(children[0]) and self.is_integer(children[2])
        return False

    def is_character(self, node: Any) -> bool:
        if isinstance(node, f03.Char_Literal_Constant):
            return True
        if isinstance(node, f03.Name):
            declared = self.declaration(str(node))
            return declared is not None and declared.get("dtype") == "str"
        return False

    def is_constant(self, node: Any) -> bool:
        """Whether an expression is fixed at compile time.

        Literals and named parameters over ``+ - * / **``. It matters because a
        compiler evaluates an intrinsic over such an expression while
        compiling, at a precision no run-time library reproduces.
        """
        if isinstance(node, (f03.Real_Literal_Constant, f03.Int_Literal_Constant)):
            return True
        if isinstance(node, f03.Name):
            return str(node).lower() in self.parameters
        if isinstance(node, f03.Parenthesis):
            return self.is_constant(node.children[1])
        children = getattr(node, "children", None)
        if children and len(children) == 2 and _is_operator(children[0], {"+", "-"}):
            return self.is_constant(children[1])
        if children and len(children) == 3 and _is_operator(children[1], ARITHMETIC):
            return self.is_constant(children[0]) and self.is_constant(children[2])
        return False

    def integer_literal(self, node: Any) -> int | None:
        """The value of an integer literal, through parens and a sign.

        ``None`` for anything else. Used where a literal exponent changes how
        the expression is lowered, so a wrong answer would change arithmetic
        and a missing one only forgoes a rewrite.
        """
        if isinstance(node, f03.Int_Literal_Constant):
            return int(node.children[0])
        if isinstance(node, f03.Parenthesis):
            return self.integer_literal(node.children[1])
        children = getattr(node, "children", None)
        if children and len(children) == 2 and str(children[0]) in ("+", "-"):
            value = self.integer_literal(children[1])
            if value is not None:
                return -value if str(children[0]) == "-" else value
        return None

    # -- dispatch -------------------------------------------------------------

    def dispatch(self, name: str, actuals: list[Any]) -> str:
        """Which specific procedure a generic call resolves to.

        Matched on the two axes CAM's generics overload along: the rank of each
        actual, and whether a scalar one is integer. Exactly one match is an
        answer; none or several raises.

        There were two implementations of this, and they disagreed -- this one
        refuses, and the read/write analysis scored the candidates and took the
        best. They agreed on the thirty translated CAM modules only because all
        three of their generic call sites match cleanly. Refusing is the right
        half of that disagreement to keep: an overload picked wrongly changes
        which arguments are written, and nothing downstream re-checks it.
        """
        candidates = self.generics.get(name) or self.companion_generics.get(name)
        if not candidates:
            raise AmbiguousDispatch(f"{name!r} is not a generic interface here")

        positional = [
            a for a in actuals if not isinstance(a, (f03.Actual_Arg_Spec, f03.Component_Spec))
        ]
        keyword = {
            str(a.children[0]).lower(): a.children[1]
            for a in actuals
            if isinstance(a, (f03.Actual_Arg_Spec, f03.Component_Spec))
        }
        matches = [
            specific
            for specific in candidates
            if (record := self.procedures.get(specific)) is not None
            and self._matches(record, positional, keyword)
        ]
        if len(matches) == 1:
            return matches[0]
        raise AmbiguousDispatch(
            f"generic {name!r}: " + (f"ambiguous between {matches}" if matches else "no match")
        )

    def _signature(self, node: Any) -> tuple[int | None, bool | None]:
        """``(rank, integer-ness)``, with ``None`` where it cannot be told.

        An unanalyzable actual makes that argument a wildcard rather than
        making the whole call unanalyzable -- the other arguments usually
        discriminate, and refusing on the first hard one would refuse almost
        every call.
        """
        try:
            return self.rank(node), self.is_integer(node)
        except Unanalyzable:
            return None, None

    def _matches(
        self, record: dict[str, Any], positional: list[Any], keyword: dict[str, Any]
    ) -> bool:
        formals = record["args"]
        names = [f["name"] for f in formals]
        if any(k not in names for k in keyword):
            return False
        required = sum(1 for f in formals if not f["optional"])
        supplied = len(positional) + len(keyword)
        if not required <= supplied <= len(formals):
            return False

        elemental = "ELEMENTAL" in (record.get("prefixes") or [])
        pairs = [(formals[names.index(k)], v) for k, v in keyword.items()]
        pairs += list(zip(formals, positional, strict=False))
        for formal, actual in pairs:
            rank, integral = self._signature(actual)
            formal_rank = len(formal["dims"]) if formal.get("dims") else 0
            # An elemental procedure's scalar formal legally takes any rank.
            if not elemental and rank is not None and rank != formal_rank:
                return False
            formal_integral = formal["dtype"] in INTEGER_DTYPES
            if (
                integral is not None
                and rank == 0
                and formal_integral != integral
                and formal["dtype"] != "str"
            ):
                return False
        return True


def for_subprogram(
    record: dict[str, Any],
    name: str,
    *,
    companions: tuple[dict[str, Any], ...] = (),
    statement_functions: frozenset[str] = frozenset(),
) -> Semantics:
    """Build ``Semantics`` for one subprogram of an ``interface.extract`` record.

    ``companions`` are other modules' records, merged in so that a call does
    not have to know which file declared its callee.
    """
    procedures = {s["name"]: s for s in record["subprograms"]}
    generics = dict(record["generics"])
    companion_generics: dict[str, list[str]] = {}
    types = dict(record["types"])
    parameters = {p["name"] for p in record["module_parameters"]}
    for other in companions:
        procedures.update({s["name"]: s for s in other["subprograms"]})
        companion_generics.update(other["generics"])
        types.update(other["types"])
        parameters |= {p["name"] for p in other["module_parameters"]}

    subprogram = procedures[name] if name in procedures else None
    if subprogram is None or subprogram not in record["subprograms"]:
        subprogram = next((s for s in record["subprograms"] if s["name"] == name), None)
    if subprogram is None:
        raise Unanalyzable(f"{name!r} is not a subprogram of {record['module']!r}")
    parameters |= {p["name"] for p in subprogram["local_parameters"]}

    return Semantics(
        module=record,
        subprogram=subprogram,
        procedures=procedures,
        generics=generics,
        companion_generics=companion_generics,
        types=types,
        parameters=frozenset(parameters),
        statement_functions=statement_functions,
    )
