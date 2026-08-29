"""Fortran statements, written as NumPy.

The floor above ``expressions``: everything that emits a *line* rather than a
value. An assignment, a construct, a call. The expression layer answers what
a value is spelled like; this one decides what happens to it -- and the
decisions worth reading are the ones where Fortran's statement means more
than its Python lookalike:

* Assigning to a whole array is a copy into its storage, not a rebinding, so
  the target grows ``[...]``. Assigning a whole derived type is a deep copy.
* A WHERE is not an ``if``: its mask is an array, its assignments gather, and
  nesting ANDs the masks together.
* A ``do v = hi, lo, -1`` counts down through its last element; a slice's stop
  edge cannot say "one before the start", so the bounds shift differently by
  the sign of the step -- and a variable step defers the sign to run time.
* A ``goto`` is refused unless it is one of the two shapes that structure
  cleanly: the loop-exit pattern (label immediately after ``end do``), and the
  forward region (jump to a later label at the same level), which becomes a
  labelled exception because it escapes any nesting depth.
* An ``intent(out)`` argument does not exist on the target side: the callee
  returns it, and the call site assigns it back -- into the buffer, for an
  array, because the caller may be aliasing it.

Anything else -- a computed goto, a formatted internal write, an ELSEWHERE
with its own mask -- raises ``NoRule`` and becomes a deferred site.

Refusal here has two spellings, ``NoRule`` from this layer and its rules, and
``Unanalyzable`` out of ``semantics``; ``REFUSED`` is both, and is what the
Transform catches to defer a block.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from recast.fortran import intrinsics
from recast.fortran._parse import f03, f08, walk
from recast.fortran.interface import emit_name
from recast.fortran.semantics import Semantics, Unanalyzable
from recast.transform.numpy.calls import CallSite
from recast.transform.numpy.expressions import Expressions
from recast.transform.numpy.names import Names
from recast.transform.numpy.vocabulary import pysafe
from recast.transform.rules import NoRule

DATA_DTYPES = {"float64": "np.float64", "int32": "np.int32", "bool": "np.bool_"}
"""What a DATA fill is built as. Narrower than the allocate map, as the
pipeline has it here."""

DERIVED_TYPE = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)")

__all__ = ["REFUSED", "Statements"]

REFUSED = (NoRule, Unanalyzable)
"""What a refusal looks like from either floor below this one."""

INT_SENTINEL = -2147483647
"""``INT32_MIN + 1``: the fill for an undefined integer under ``poison_integers``.

An "impossible" index, so a read of an unwritten cell crashes on the subscript
or moves the answer, rather than passing as a plausible one. The value is the
one the tool this arm came from uses; a different sentinel would be a different
experiment.
"""


def undefined_array(owner: Any, shape: str, dtype: str) -> str:
    """An array Fortran leaves undefined, as this backend spells it.

    ``shape`` is the ``(a, b,)`` or ``np.shape(x)`` argument text, already
    formed. ``owner`` is whichever layer carries the two poison flags.

    Every site that would emit ``np.empty`` for undefined memory goes through
    here, because the tool this was taken from poisons by patching ``np.empty``
    itself and so reaches all of them at once -- the automatic locals, the
    intent(out) buffers, *and* the arrays an ``ALLOCATE`` statement brings into
    existence. Covering three of the four would report a clean run for a defect
    in the fourth.

    ``np.empty_like``, which an ``ALLOCATE`` with ``SOURCE=``/``MOLD=`` emits,
    is deliberately not covered: that patch reaches ``np.empty`` and not
    ``np.empty_like``, so poisoning it here would be this backend answering a
    question the experiment does not ask.
    """
    if getattr(owner, "poison_undefined", False):
        if dtype in ("np.float64", "np.float32"):
            return f"np.full({shape}, np.nan, dtype={dtype})"
        if getattr(owner, "poison_integers", False) and dtype in ("np.int32", "np.int64"):
            return f"np.full({shape}, {INT_SENTINEL}, dtype={dtype})"
    return f"np.empty({shape}, dtype={dtype})"


ALLOCATED_DTYPES = {
    "float64": "np.float64",
    "float32": "np.float32",
    "int32": "np.int32",
    "int64": "np.int64",
    "bool": "np.bool_",
    "str": "object",
}
"""Declared dtype -> the dtype an ``allocate`` requests. Anything else gets
``np.float64``, which is what the declaration meant if it said nothing --
except a derived type, which is not a dtype at all; see ``derived_array``."""


def derived_array(type_name: str, extents: list[str], known: dict[str, Any]) -> str | None:
    """An array of a derived type, as an object array already filled.

    ``np.empty(dtype=object)`` would hold ``None``s, and the first
    ``x(i)%c = ...`` against one is an AttributeError -- Fortran has the
    elements existing the moment the array does. Returns ``None`` for a type
    nothing here defines, which leaves the caller to say so its own way.
    """
    if type_name not in known:
        return None
    count = extents[0] if len(extents) == 1 else f"np.prod(({', '.join(extents)}))"
    return f"np.array([_make_{type_name}() for _ in range({count})], dtype=object)"


UNIT_NAME = re.compile(r"[a-z_]\w*")

FORMAT_SUPPORTED = re.compile(
    r"\(\s*(?:(?:\d*\s*(?:I\d+(?:\.\d+)?|F\d+\.\d+"
    r"|E[SN]?\d+\.\d+(?:E\d+)?|G\d+\.\d+|A(?:\d+)?|L\d+|\d*X|/"
    r"|'[^']*'|\"[^\"]*\"))\s*(?:,\s*)?)*\)",
    re.I,
)
"""The edit descriptors ``_f_fmt_write`` implements. A formatted internal
write using anything else is refused rather than silently list-directed."""


@dataclass
class Statements:
    """Render Fortran statements for one subprogram.

    Holds the state a statement can leave behind for a later one: the mask a
    WHERE puts over its body, the labels a ``do`` consumes as its exit, the
    bounds an ``allocate`` gives an array, the statement functions defined so
    far. All of it is per-subprogram, like the three layers this composes.
    """

    semantics: Semantics
    names: Names
    expressions: Expressions

    externals: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Procedures with an audited shim in the externals module."""

    call_transforms: dict[str, Any] = field(default_factory=dict)
    """Callee -> a domain package's answer for it; see ``calls.CallSite``."""

    stubs: dict[str, str] = field(default_factory=dict)
    """Framework subroutine -> the statement that stands in for it.

    The statement-shaped half of the table whose function-shaped half lives on
    ``Expressions``: calls into a framework the translation does not carry.
    Supplied by the domain package that knows the framework.
    """

    poison_undefined: bool = False
    """NaN-fill a float array Fortran leaves undefined; see ``Subprograms``."""

    poison_integers: bool = False
    """The integer arm of the above, and off even when it is on."""

    buffer_out_arrays: bool = False
    """Apply the caller-buffer convention at call sites; see ``Subprograms``."""

    masks: list[str] = field(default_factory=list)
    """Enclosing WHERE masks. Fortran ANDs a nested WHERE into its outer one."""

    assigned_names: set[str] = field(default_factory=set)
    """Names this subprogram's body assigns to. Filled by ``scan``."""

    called_names: set[str] = field(default_factory=set)
    """Names this subprogram's body calls or subscripts. Filled by ``scan``."""

    exit_labels: dict[int, str] = field(default_factory=dict)
    """``id(do-construct)`` -> the label that means ``exit`` inside it."""

    consumed_labels: set[str] = field(default_factory=set)
    """Labels a ``do`` or a region wrapper accounts for; their ``continue``
    statements emit as markers rather than as targets still owed a jump."""

    region_depth: int = 0
    """How many goto regions are open around the sequence being emitted."""

    active_labels: list[Any] = field(default_factory=list)
    """The loop-exit label (or ``None``) of each enclosing ``do``, innermost
    last, with ``("region", label)`` entries for open forward-goto regions.

    Every ``do`` pushes something: a ``goto`` deeper than the labeled do's own
    body would break the wrong loop, and the ``None`` on top blocks it.
    """

    construct_stack: list[tuple[str, str | None]] = field(default_factory=list)
    """``("do" | "block", construct-name-or-None)`` for each enclosing
    construct, outermost first.

    What an ``EXIT``/``CYCLE`` naming a construct has to consult. Kept apart
    from ``active_labels``, which answers a different question -- where a goto
    may land -- and whose entries are labels rather than construct names.
    """

    # -- entry points ---------------------------------------------------------

    def scan(self, subprogram: Any) -> None:
        """One pass over the body for the facts a later statement needs.

        Pairs every do-construct with an immediately-following labeled
        CONTINUE anywhere in the nesting tree -- inside such a loop, ``goto``
        to that label is exactly ``exit`` -- and records which names the body
        assigns and which it calls, which is what tells an F77 intrinsic
        declaration apart from a real local.
        """
        self.exit_labels = {}
        self.consumed_labels = set()
        self.assigned_names = {
            str(a.children[0]).lower()
            for a in walk(subprogram, f03.Assignment_Stmt)
            if isinstance(a.children[0], f03.Name)
        }
        self.called_names = {
            str(c.children[0]).lower()
            for c in walk(
                subprogram,
                (
                    f03.Part_Ref,
                    f03.Function_Reference,
                    f03.Intrinsic_Function_Reference,
                    f03.Structure_Constructor,
                ),
            )
        }

        def targeted_from_outside(do_node: Any, label: str) -> bool:
            inside = {id(g) for g in walk(do_node, f03.Goto_Stmt)}
            return any(
                str(g.children[0]) == label and id(g) not in inside
                for g in walk(subprogram, f03.Goto_Stmt)
            )

        def descend(children: Any) -> None:
            nodes = [child for child in children if child is not None]
            for at, child in enumerate(nodes):
                if (
                    isinstance(child, f03.Block_Nonlabel_Do_Construct)
                    and at + 1 < len(nodes)
                    and isinstance(nodes[at + 1], f03.Continue_Stmt)
                ):
                    label = self.label(nodes[at + 1])
                    # A label ALSO targeted by gotos outside this do is a
                    # forward-goto-region label, not a loop-exit label.
                    if label and not targeted_from_outside(child, label):
                        self.exit_labels[id(child)] = label
                        self.consumed_labels.add(label)
                if hasattr(child, "children") and not isinstance(
                    child, (f03.Name, f03.Int_Literal_Constant, f03.Real_Literal_Constant)
                ):
                    descend(child.children)

        descend(getattr(subprogram, "children", []))

    def render(self, node: Any, indent: int) -> list[str]:
        """One statement, as Python source lines."""
        pad = "    " * indent
        if isinstance(node, f03.Assignment_Stmt):
            return self._assignment(node, indent)
        if isinstance(node, f03.If_Stmt):
            # `if (c) action`: the action may be one that needs several lines
            # of its own -- a masked assignment, a stubbed call -- and they
            # all belong under the branch.
            condition, action = node.children
            inner = self.render(action, 0)
            return [
                f"{pad}if {self.expressions.render(condition)}:",
                *(f"{pad}    {line}" for line in inner),
            ]
        if isinstance(node, f03.If_Construct):
            return self._if_construct(node, indent)
        if isinstance(node, f03.Case_Construct):
            return self._case_construct(node, indent)
        if isinstance(node, f03.Return_Stmt):
            return [f"{pad}return {self.returned_value()}"]
        if isinstance(node, (f03.Cycle_Stmt, f03.Exit_Stmt)):
            return self.exit_stmt(node, pad)
        if isinstance(node, f03.Goto_Stmt):
            return self._goto(node, pad)
        if isinstance(node, f03.Continue_Stmt):
            label = self.label(node)
            if label in self.consumed_labels:
                return [f"{pad}pass  # {label} continue (break target)"]
            return [f"{pad}pass  # continue"]
        if isinstance(node, f03.Call_Stmt):
            return self._call(node, indent)
        if isinstance(
            node,
            (
                f03.Block_Nonlabel_Do_Construct,
                f03.Block_Label_Do_Construct,
                f03.Action_Term_Do_Construct,
            ),
        ):
            # The third is a labelled DO whose terminator is an action
            # statement rather than a CONTINUE -- F77 wrote them that way.
            return self._do_construct(node, indent)
        if isinstance(node, f03.Where_Stmt):
            return self._where_statement(node, indent)
        if isinstance(node, f03.Where_Construct):
            return self._where_construct(node, indent)
        if isinstance(node, f03.Allocate_Stmt):
            return self._allocate(node, indent)
        if isinstance(node, f03.Deallocate_Stmt):
            return self._deallocate(node, pad)
        if isinstance(node, f03.Pointer_Assignment_Stmt):
            return self._pointer_assignment(node, pad)
        if isinstance(node, f03.Nullify_Stmt):
            return self._nullify(node, pad)
        if isinstance(node, f03.Write_Stmt):
            return self._write(node, indent)
        if isinstance(node, f03.Format_Stmt):
            return [f"{pad}pass  # FORMAT statement (declarative)"]
        if isinstance(node, f03.Print_Stmt):
            return [f"{pad}pass  # PRINT (diagnostic only, no dataflow)"]
        if isinstance(node, (f03.Stop_Stmt, f08.Error_Stop_Stmt)):
            # ERROR STOP differs from STOP only in the exit status a compiler
            # is asked to produce; both end the program with the same message,
            # and nothing downstream of a SystemExit compares anything.
            message = ""
            if node.children and node.children[1]:
                message = str(node.children[1]).strip()
            keyword = "ERROR STOP" if isinstance(node, f08.Error_Stop_Stmt) else "STOP"
            return [f"{pad}raise SystemExit({message!r})  # {keyword}"]
        if isinstance(node, f03.Entry_Stmt):
            # A second entry point into a subprogram: F77's, deleted in F2018,
            # and the callers this translates reach the primary entry. The
            # statement itself does nothing where it stands.
            return [f"{pad}pass  # ENTRY (legacy)"]
        if isinstance(node, f03.Read_Stmt):
            return [f"{pad}pass  # READ (I/O stub)"]
        if isinstance(node, (f03.Open_Stmt, f03.Close_Stmt)):
            return [f"{pad}pass  # OPEN/CLOSE (I/O stub)"]
        if isinstance(
            node,
            (f03.Rewind_Stmt, f03.Backspace_Stmt, f03.Endfile_Stmt, f03.Flush_Stmt),
        ):
            # File positioning. Unlike INQUIRE below, none of these writes a
            # variable, so there is nothing for a read/write gate to compare
            # and nothing to lose by dropping them.
            return [f"{pad}pass  # {type(node).__name__[:-5].upper()} (I/O stub)"]
        if isinstance(node, f03.Inquire_Stmt):
            # Not a stub, on purpose, and this is where the pipeline this was
            # migrated from differs: it stubs INQUIRE to ``pass``. Every
            # output specifier -- ``opened=``, ``pos=``, ``iostat=`` -- is a
            # write, and a ``pass`` drops it silently, leaving the variable at
            # whatever it held.
            written = sorted(
                str(spec.children[0]).upper()
                for spec in walk(node, f03.Connect_Spec | f03.Inquire_Spec)
                if spec.children[0] is not None
                and str(spec.children[0]).upper() not in ("UNIT", "FILE")
            )
            raise NoRule(
                "inquire writes " + ", ".join(f"{k}=" for k in written)
                if written
                else "inquire with no output specifier"
            )
        if isinstance(node, (f03.Forall_Construct, f03.Forall_Stmt)):
            return self._forall(node, indent)
        if isinstance(node, f03.Data_Stmt):
            # A DATA among the executable statements: F77 allowed it there,
            # and fparser leaves it where it found it.
            return self.data_statement(node, indent)
        if isinstance(node, f03.Associate_Construct):
            return self._associate(node, indent)
        if isinstance(node, f08.Block_Construct):
            return self._block(node, indent)
        kind = type(node).__name__
        if kind.startswith("Cpp_"):
            # A preprocessor directive that survived into the tree: it said
            # something to the compiler, nothing to the program.
            return [f"{pad}pass  # {kind} (preprocessor directive)"]
        raise NoRule(f"no statement rule for {kind}")

    MAX_REGION_DEPTH = 20
    """How deep goto regions may nest before they stop being formed.

    A subprogram whose labels interleave can nest a region per label, and the
    pipeline hit a case where widening them did not terminate. Past this the
    statements are emitted flat: any goto among them then refuses, which is a
    visible refusal rather than a translation that took minutes to produce.
    """

    def sequence(self, nodes: list[Any], indent: int) -> list[str]:
        """A statement sequence, structuring forward goto-regions.

        ``goto L`` jumping to a later top-level ``L continue`` becomes a
        labelled exception raised out of any nesting depth; regions nest, and
        the outermost one belongs to the *last* matching label.
        """
        pad = "    " * indent
        if self.region_depth > self.MAX_REGION_DEPTH:
            return [line for node in nodes for line in self.render(node, indent)]
        labelled = {}
        targets = {}
        for at, node in enumerate(nodes):
            found = self.label(node)
            if found and found not in self.consumed_labels:
                labelled[at] = found
            reached = {str(goto.children[0]) for goto in walk(node, f03.Goto_Stmt)}
            if reached:
                targets[at] = reached

        # A backward goto -- label at i, `goto` to it at j > i -- is a loop:
        # everything from the label to the last such goto runs again. Taken
        # before the forward case, as the pipeline takes it, because a region
        # that is both is a loop with an early exit rather than the reverse.
        for at in sorted(labelled):
            label = labelled[at]
            back = [j for j in range(at + 1, len(nodes)) if label in targets.get(j, ())]
            if not back:
                continue
            lines = self.sequence(nodes[:at], indent) if at else []
            lines.append(f"{pad}while True:  # backward-goto region (label {label})")
            lines.append(f"{pad}    try:")
            self.active_labels.append(("region", label))
            self.consumed_labels.add(label)
            self.region_depth += 1
            try:
                body = self.sequence(nodes[at : back[-1] + 1], indent + 2)
            finally:
                self.region_depth -= 1
                self.active_labels.pop()
            lines.extend(body or [f"{pad}        pass"])
            lines.append(f"{pad}        break  # natural exit")
            lines.append(f"{pad}    except _FGoto as _g:")
            lines.append(f"{pad}        if _g.args[0] != '{label}':")
            lines.append(f"{pad}            raise")
            lines.append(f"{pad}        pass  # {label} (loop restart)")
            lines.extend(self.sequence(nodes[back[-1] + 1 :], indent))
            return lines

        # A forward goto -- `goto` at k, label at j > k -- is an early exit
        # from everything between them. The outermost region belongs to the
        # last matching label, so the scan runs from the end.
        for at in sorted(labelled, reverse=True):
            label = labelled[at]
            if not any(label in targets.get(earlier, ()) for earlier in range(at)):
                continue
            lines = [f"{pad}try:  # forward-goto region (label {label})"]
            self.active_labels.append(("region", label))
            self.region_depth += 1
            try:
                lines.extend(self.sequence(nodes[:at], indent + 1) or [f"{pad}    pass"])
            finally:
                self.region_depth -= 1
                self.active_labels.pop()
            lines.append(f"{pad}except _FGoto as _g:")
            lines.append(f"{pad}    if _g.args[0] != '{label}':")
            lines.append(f"{pad}        raise")
            lines.append(f"{pad}    pass  # {label} (region exit)")
            lines.extend(self.sequence(nodes[at + 1 :], indent))
            return lines
        lines = []
        for node in nodes:
            if isinstance(node, (f03.Cycle_Stmt, f03.Exit_Stmt)):
                lines.extend(self.exit_stmt(node, pad))
            else:
                lines.extend(self.render(node, indent))
        return lines

    # -- assignment -----------------------------------------------------------

    def _assignment(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        target, _, value = node.children
        if (
            isinstance(target, f03.Name)
            and isinstance(value, (f03.Part_Ref, f03.Structure_Constructor))
            and str(value.children[0]).lower() in self.expressions.handle_producers
        ):
            # The right-hand side is a lookup that answers with a handle, so
            # the name it lands in holds one too.
            self.expressions.handles.add(self.names.symbol(str(target)))
        if isinstance(target, f03.Name) and self.semantics.derived_type_of(str(target)):
            # Assigning a whole derived type is a DEEP COPY in Fortran.
            return [
                f"{pad}{self.names.symbol(str(target))} = "
                f"_copy_derived({self.expressions.render(value)})"
            ]
        if isinstance(target, f03.Part_Ref):
            root = str(target.children[0]).lower()
            declaration = self.semantics.declaration(root)
            arguments = target.children[1].children if target.children[1] is not None else []
            if (
                declaration is not None
                and not declaration.get("dims")
                and arguments
                and all(isinstance(a, f03.Name) for a in arguments)
            ):
                # An old-style STATEMENT FUNCTION: a scalar declaration with
                # dummy arguments becomes a nested def, whose closure mirrors
                # host association.
                formals = ", ".join(pysafe(str(a).lower()) for a in arguments)
                self._define_statement_function(root)
                return [
                    f"{pad}def {pysafe(root)}({formals}):  # statement function",
                    f"{pad}    return {self.expressions.render(value)}",
                ]
        rendered = self.expressions.render(value)
        if (
            self.semantics.is_scalar_integer_target(target)
            and not self.semantics.is_integer(value)
            and not self._rhs_is_opaque(value)
        ):
            # Fortran converts on assignment: a REAL expression stored into an
            # INTEGER scalar truncates toward zero, the mapping INT already
            # has. Without this the Python name silently holds a float, and
            # every later use of it as an index or a count either drifts or
            # raises "cannot be interpreted as an integer" somewhere else.
            try:
                rank = self.semantics.rank(value)
            except Unanalyzable:
                rank = 0
            if rank == 0 and not self.semantics.is_logical_or_character(value):
                rendered = f"int({rendered})"
        return [f"{pad}{self.target(target)} = {rendered}"]

    def _rhs_is_opaque(self, node: Any) -> bool:
        """Whether a stub, a domain transform or a foreign module decides the
        right-hand side's Python type, rather than Fortran typing.

        Those never take the integer conversion above. Wrapping one would be
        claiming Fortran typed it REAL, and nothing here typed it at all --
        the answer comes from a table this translation does not own.
        """
        while isinstance(node, f03.Parenthesis):
            node = node.children[1]
        if isinstance(node, f03.Name):
            name = str(node).lower()
            return name in self.names.use_bindings or name in self.names.use_parameters
        if isinstance(
            node,
            (
                f03.Part_Ref,
                f03.Intrinsic_Function_Reference,
                f03.Function_Reference,
                f03.Structure_Constructor,
            ),
        ):
            name = str(node.children[0]).lower()
            if self.semantics.is_array(name):
                return False
            return (
                name in self.expressions.function_transforms
                or name in self.expressions.stubs
                or name in self.externals
                or name in self.names.use_bindings
                or name in self.semantics.companion_generics
                # The engine's member of the same category: a handle producer
                # is supplied by the domain package beside the stubs and the
                # transforms above, and what it returns is not a number at all.
                or name in self.expressions.handle_producers
            )
        return False

    def target(self, node: Any) -> str:
        """An assignment target, honouring Fortran's COPY semantics: a whole
        array (or whole array component) gets ``[...]`` so the assignment
        fills its storage rather than rebinding its name."""
        if isinstance(node, f03.Name):
            name = str(node).lower()
            if self.semantics.is_array(name):
                return f"{self.names.symbol(name)}[...]"
            return self.names.symbol(name)
        if isinstance(node, f03.Part_Ref):
            # A subscripted name on the left is an array whatever this file
            # knows about it: nothing else can be assigned through. The
            # declaration is missing for a use-imported one, and refusing
            # there refused the assignment rather than the import.
            name = str(node.children[0]).lower()
            return self.expressions.subscript(name, node.children[1])
        if isinstance(node, f03.Data_Ref):
            rendered = self.expressions.render(node)
            last = node.children[-1]
            if (
                isinstance(last, f03.Name)
                and len(node.children) == 2
                and isinstance(node.children[0], f03.Name)
            ):
                component = self.semantics.component(str(node.children[0]), str(last))
                if component and component.get("dims"):
                    return f"{rendered}[...]"
            return rendered
        raise NoRule(f"assignment target {type(node).__name__}")

    def _define_statement_function(self, name: str) -> None:
        """From here on, ``name(...)`` is a local function, not an array."""
        self.semantics.statement_functions |= {name}
        self.expressions.statement_functions |= {name}

    # -- WHERE and FORALL -----------------------------------------------------

    def _mask_expression(self, node: Any) -> str:
        """WHERE masks are array-valued: ``.AND.``/``.OR.``/``.NOT.`` become
        ``&``/``|``/``~``."""
        saved = self.expressions.vector_boolean
        self.expressions.vector_boolean = True
        try:
            return self.expressions.render(node)
        finally:
            self.expressions.vector_boolean = saved

    def _effective_mask(self, local: str) -> str:
        return " & ".join([*self.masks, local]) if self.masks else local

    def _masked_assignment(self, mask: str, node: Any, indent: int) -> str:
        pad = "    " * indent
        target, _, value = node.children
        rendered = self.expressions.render(value)
        if self.semantics.rank(value) > 0:
            rendered = f"({rendered})[{mask}]"
        return f"{pad}{self.target(target)}[{mask}] = {rendered}"

    def _where_statement(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        mask, assignment = node.children
        depth = len(self.masks)
        variable = "_wm" if depth == 0 else f"_wm{depth + 1}"
        return [
            f"{pad}{variable} = {self._mask_expression(mask)}",
            self._masked_assignment(self._effective_mask(variable), assignment, indent),
        ]

    def _where_construct(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        depth = len(self.masks)
        variable = "_wm" if depth == 0 else f"_wm{depth + 1}"
        # What no branch so far has claimed. Carried explicitly because a
        # masked ELSEWHERE narrows it: each one takes the elements its own
        # condition selects and leaves the rest to the branches after it.
        remaining = "_wn" if depth == 0 else f"_wn{depth + 1}"
        masked_elsewhere = any(
            isinstance(child, f03.Masked_Elsewhere_Stmt) for child in node.children if child
        )
        seen = 0
        lines: list[str] = []
        local = variable
        for child in node.children:
            if isinstance(child, f03.Where_Construct_Stmt):
                lines.append(f"{pad}{variable} = {self._mask_expression(child.children[0])}")
                if masked_elsewhere:
                    lines.append(f"{pad}{remaining} = (~{variable})")
            elif isinstance(child, f03.Masked_Elsewhere_Stmt):
                seen += 1
                condition = self._mask_expression(child.children[0])
                local = f"_we{depth}_{seen}"
                lines.append(f"{pad}{local} = ({remaining} & {condition})")
                lines.append(f"{pad}{remaining} = ({remaining} & (~{condition}))")
            elif isinstance(child, f03.Elsewhere_Stmt):
                local = remaining if masked_elsewhere else f"(~{variable})"
            elif isinstance(child, f03.End_Where_Stmt):
                pass
            elif isinstance(child, f03.Assignment_Stmt):
                lines.append(self._masked_assignment(self._effective_mask(local), child, indent))
            elif isinstance(child, (f03.Where_Construct, f03.Where_Stmt)):
                self.masks.append(local)  # Fortran nested WHERE:
                try:  # the inner mask ANDs the outer one
                    lines.extend(self.render(child, indent))
                finally:
                    self.masks.pop()
            else:
                raise NoRule(f"WHERE body {type(child).__name__}")
        return lines

    def _forall(self, node: Any, indent: int) -> list[str]:
        """FORALL, as the nested loops it abbreviates."""
        if isinstance(node, f03.Forall_Stmt):
            header = walk(node, f03.Forall_Header)[0]
            body = [c for c in node.children if isinstance(c, f03.Assignment_Stmt)]
        else:
            header = walk(node, f03.Forall_Header)[0]
            body = [
                c
                for c in node.children
                if not isinstance(c, (f03.Forall_Construct_Stmt, f03.End_Forall_Stmt))
            ]
        triplets = walk(header, f03.Forall_Triplet_Spec)
        lines = []
        for depth, triplet in enumerate(triplets):
            parts = [str(c).strip() for c in triplet.children if c is not None]
            variable = parts[0].lower()
            low = self.expressions.render(triplet.children[1]) if triplet.children[1] else "1"
            high = self.expressions.render(triplet.children[2])
            step = ""
            if triplet.children[3] is not None:
                step = f", {self.expressions.render(triplet.children[3])}"
            pad = "    " * (indent + depth)
            lines.append(f"{pad}for {pysafe(variable)} in range({low}, ({high}) + 1{step}):")
        for statement in body:
            lines.extend(self.render(statement, indent + len(triplets)))
        return lines

    # -- allocation and pointers ----------------------------------------------

    def _allocate(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        lines = []
        # `allocate(x, source=e)` takes e's shape *and* its values;
        # `mold=e` takes the shape alone.
        template, copies = None, False
        options = node.children[2] if len(node.children) > 2 else None
        for option in self._items(options) if options is not None else []:
            children = list(getattr(option, "children", ()) or ())
            if len(children) == 2 and str(children[0]).upper() in ("SOURCE", "MOLD"):
                template = self.expressions.render(children[1])
                copies = str(children[0]).upper() == "SOURCE"
        if template is not None:
            for allocation in walk(node.children[1], (f03.Allocation, f03.Name)):
                target = (
                    allocation.children[0] if isinstance(allocation, f03.Allocation) else allocation
                )
                if not isinstance(target, f03.Name):
                    continue
                spelled = self.names.symbol(str(target).lower())
                value = (
                    f"np.array({template}, copy=True)" if copies else f"np.empty_like({template})"
                )
                lines.append(f"{pad}{spelled} = {value}")
            if lines:
                return lines
        for allocation in walk(node, f03.Allocation):
            target, shape = allocation.children[0], allocation.children[1]
            extents = []
            bounds = []
            for spec in walk(shape, f03.Allocate_Shape_Spec):
                low, high = spec.children
                if low is not None and str(low) != "1":
                    bounds.append({"lb": str(low), "ub": str(high)})
                    extents.append(
                        f"({self.expressions.render(high)}) - ({self.expressions.render(low)}) + 1"
                    )
                else:
                    bounds.append({"lb": "1", "ub": str(high)})
                    extents.append(self.expressions.render(high))
            if isinstance(target, f03.Name) and any(d["lb"] != "1" for d in bounds):
                key = str(target).lower()
                previous = self.expressions.allocated_bounds.get(key)
                if previous is not None and [d["lb"] for d in previous] != [
                    d["lb"] for d in bounds
                ]:
                    raise NoRule(f"conflicting allocate lower bounds for {key}")
                self.expressions.allocated_bounds[key] = bounds
            if isinstance(target, f03.Name):
                rendered = self.names.symbol(str(target))
            elif isinstance(target, f03.Data_Ref):
                rendered = ".".join(
                    (self.names.symbol(str(c)) if at == 0 else str(c).lower())
                    for at, c in enumerate(target.children)
                )
            else:
                raise NoRule(f"allocate target {type(target).__name__}")
            shape_text = (
                f"({', '.join(extents)},)" if len(extents) == 1 else f"({', '.join(extents)})"
            )
            name = (
                str(target).lower().split("%")[-1]
                if isinstance(target, (f03.Name, f03.Data_Ref))
                else None
            )
            declaration = self.semantics.declaration(name) if name else None
            derived = DERIVED_TYPE.match(str((declaration or {}).get("dtype", "")))
            filled = (
                derived_array(derived.group(1).lower(), extents, self.semantics.types)
                if derived
                else None
            )
            if filled is not None:
                lines.append(f"{pad}{rendered} = {filled}")
                continue
            dtype = (
                ALLOCATED_DTYPES.get(declaration["dtype"], "np.float64")
                if declaration
                else "np.float64"
            )
            lines.append(f"{pad}{rendered} = {undefined_array(self, shape_text, dtype)}")
        if not lines:
            # `allocate(x)` with no shape: a scalar allocatable, which for a
            # derived type is the object coming into existence.
            for item in self._items(node.children[1]):
                if isinstance(item, f03.Name):
                    lines.append(f"{pad}{self.names.symbol(str(item))} = _new_derived()")
                elif isinstance(item, f03.Data_Ref):
                    lines.append(f"{pad}{self.expressions.render(item)} = _new_derived()")
        if not lines:
            raise NoRule("allocate without shape specs")
        return lines

    def _deallocate(self, node: Any, pad: str) -> list[str]:
        # Allocatable tracking: only DIRECT variables go back to None (the
        # ``allocated()`` mirror); deallocating a derived-type COMPONENT is a
        # no-op for numpy buffers.
        lines = []
        objects = node.children[0]
        for item in objects.children if hasattr(objects, "children") else [objects]:
            if isinstance(item, f03.Name):
                lines.append(f"{pad}{self.names.symbol(str(item))} = None")
        if not lines:
            lines.append(f"{pad}pass  # deallocate of derived components")
        return lines

    def _pointer_assignment(self, node: Any, pad: str) -> list[str]:
        target, _, value = node.children
        rendered = (
            self.names.symbol(str(target))
            if isinstance(target, f03.Name)
            else self.expressions.render(target)
        )
        if (
            isinstance(value, (f03.Intrinsic_Function_Reference, f03.Part_Ref))
            and str(value.children[0]).upper() == "NULL"
        ):
            return [f"{pad}{rendered} = None  # ptr => null()"]
        return [f"{pad}{rendered} = {self.expressions.render(value)}  # ptr alias (view)"]

    def _nullify(self, node: Any, pad: str) -> list[str]:
        objects = node.children[1]
        lines = []
        for item in objects.children if hasattr(objects, "children") else [objects]:
            if isinstance(item, f03.Name):
                lines.append(f"{pad}{self.names.symbol(str(item))} = None")
            else:
                lines.append(f"{pad}{self.expressions.render(item)} = None  # nullify")
        return lines

    # -- I/O ------------------------------------------------------------------

    def _write(self, node: Any, indent: int) -> list[str]:
        """Log writes (``*``, unit numbers, use-imported units) carry no
        comparable dataflow and become ``pass``; a list-directed INTERNAL
        write -- the unit is a local character variable -- carries real
        dataflow and becomes ``_f_list_write``."""
        pad = "    " * indent
        control, items = node.children
        specifiers = list(control.children) if hasattr(control, "children") else []
        unit = format_ = None
        position = 0
        for specifier in specifiers:
            keyword, value = specifier.children
            key = str(keyword).upper() if keyword is not None else None
            if key in ("IOSTAT", "ERR", "ADVANCE", "REC"):
                raise NoRule(f"write with {key}= control")
            if key == "UNIT" or (key is None and position == 0):
                unit = value
            elif key == "FMT" or (key is None and position == 1):
                format_ = value
            position += 1
        name = str(unit).strip().lower() if unit is not None else "*"
        declaration = self.semantics.declaration(name) if UNIT_NAME.fullmatch(name) else None
        if declaration is not None and declaration.get("dtype") == "str":
            arguments = ", ".join(
                self.expressions.render(item)
                for item in (items.children if hasattr(items, "children") else [items])
            )
            spelled = str(format_).strip() if format_ is not None else "*"
            if spelled == "*":
                return [f"{pad}{pysafe(name)} = _f_list_write({arguments})"]
            # A formatted internal write: the FMT decides the layout, so the
            # list-directed shim would be a silently wrong string (#16).
            if isinstance(format_, f03.Char_Literal_Constant):
                text = str(format_)[1:-1]
                if not FORMAT_SUPPORTED.fullmatch(text.strip()):
                    raise NoRule(
                        f"formatted internal write: unsupported edit descriptor in {text!r}"
                    )
                return [f"{pad}{pysafe(name)} = _f_fmt_write({text!r}, {arguments})"]
            raise NoRule("formatted internal write with a non-literal format")
        return [f"{pad}pass  # write({name},...) log — no dataflow"]

    # -- control flow ---------------------------------------------------------

    def _goto(self, node: Any, pad: str) -> list[str]:
        label = str(node.children[0])
        if self.active_labels and self.active_labels[-1] == label:
            return [f"{pad}break  # goto {label} == exit (label follows end do)"]
        if any(
            isinstance(entry, tuple) and entry[0] == "cycle" and entry[1] == label
            for entry in self.active_labels
        ):
            return [f"{pad}continue  # goto {label} == cycle (labeled-DO terminator)"]
        if any(
            isinstance(entry, tuple) and entry[0] == "region" and entry[1] == label
            for entry in self.active_labels
        ):
            # The exception escapes any nesting depth inside the region.
            return [f"{pad}raise _FGoto('{label}')  # goto {label}"]
        raise NoRule(f"goto {label} is not a loop-exit pattern")

    def _if_construct(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        lines: list[str] = []
        body: list[str] = []
        opened = False
        for child in node.children:
            if isinstance(child, f03.If_Then_Stmt):
                lines.append(f"{pad}if {self._condition(child)}:")
                opened = True
            elif isinstance(child, f03.Else_If_Stmt):
                self._flush(lines, body, pad)
                lines.append(f"{pad}elif {self._condition(child)}:")
            elif isinstance(child, f03.Else_Stmt):
                self._flush(lines, body, pad)
                lines.append(f"{pad}else:")
            elif isinstance(child, f03.End_If_Stmt):
                self._flush(lines, body, pad)
            else:
                body.extend(self.render(child, indent + 1))
        if not opened:
            raise NoRule("if construct without If_Then_Stmt")
        return lines

    @staticmethod
    def _flush(lines: list[str], body: list[str], pad: str) -> None:
        if body:
            lines.extend(body)
            body.clear()
        elif lines and lines[-1].endswith(":"):
            lines.append(f"{pad}    pass")

    # -- DATA ------------------------------------------------------------------

    def data_statement(self, node: Any, indent: int) -> list[str]:
        """``data a, b /1.0, 2.0/``: the initial values, as assignments.

        DATA sits in the specification part and is a *static* initialisation,
        so the assignments belong at the top of the body, before anything can
        read the names. Only literal values and literal subscripts are taken:
        a parameter name here would have to be resolved in an order this
        stage does not track, and a computed subscript is not static.
        """
        pad = "    " * indent
        lines: list[str] = []
        for group in walk(node, f03.Data_Stmt_Set):
            objects = [o for o in group.children[0].children if o is not None]
            # Plain names and integer-literal subscripts pair one-for-one with
            # the values and are emitted below. An implied-do or a section
            # target does not: it stands for a run of elements, so the list has
            # to be flattened before anything can be paired with it.
            direct = True
            for target in objects:
                if isinstance(target, f03.Name):
                    continue
                if isinstance(target, f03.Part_Ref):
                    if any(
                        not isinstance(s, f03.Int_Literal_Constant)
                        for s in self._items(target.children[1])
                    ):
                        direct = False
                    continue
                direct = False
            if not direct:
                lines.extend(
                    self._data_expanded(pad, objects, self._data_values(group.children[1]))
                )
                continue
            values = self._data_values(group.children[1])
            at = 0
            for target in objects:
                indexed = isinstance(target, f03.Part_Ref)
                raw = str(target.children[0] if indexed else target).lower()
                name = self.names.symbol(raw)
                if indexed:
                    if at >= len(values):
                        raise NoRule("DATA has fewer values than objects")
                    where = ", ".join(
                        f"{int(str(s).split('_')[0])} - 1" for s in self._items(target.children[1])
                    )
                    lines.append(f"{pad}{name}[{where}] = {values[at]}")
                    at += 1
                elif self.semantics.is_array(raw):
                    if len(objects) > 1:
                        raise NoRule("DATA list mixes an array with other objects")
                    spelled = ", ".join(values[at:])
                    at = len(values)
                    lines.append(f"{pad}{name}[:] = np.array([{spelled}], dtype=np.float64)")
                else:
                    if at >= len(values):
                        raise NoRule("DATA has fewer values than objects")
                    lines.append(f"{pad}{name} = {values[at]}")
                    at += 1
        return lines or [f"{pad}pass  # DATA (empty)"]

    def _data_literal_int(self, node: Any, env: dict[str, int], default: int | None = None) -> int:
        """A DATA bound or subscript, as an integer. Literals and the enclosing
        implied-do's variable only -- anything else is not a static bound."""
        if node is None:
            if default is None:
                raise NoRule("DATA bound missing")
            return default
        if isinstance(node, f03.Int_Literal_Constant):
            return int(str(node).split("_")[0])
        if isinstance(node, f03.Name) and str(node).lower() in env:
            return env[str(node).lower()]
        raise NoRule(f"DATA non-literal bound {node}")

    def _data_targets(
        self, objects: list[Any], env: dict[str, int] | None = None
    ) -> list[tuple[str, tuple[int, ...] | None]]:
        """A DATA object list flattened to ``(name, one-based index or None)``
        in definition order, expanding implied-dos and literal-bound sections.

        The order is the whole point: DATA pairs its objects with its values
        positionally, and an implied-do stands for as many objects as it has
        iterations.
        """
        env = env or {}
        out: list[tuple[str, tuple[int, ...] | None]] = []
        for target in objects:
            if target is None:
                continue
            if isinstance(target, f03.Name):
                out.append((str(target).lower(), None))
            elif isinstance(target, f03.Part_Ref):
                name = str(target.children[0]).lower()
                expanded: list[list[int]] = [[]]
                for subscript in self._items(target.children[1]):
                    if isinstance(subscript, f03.Subscript_Triplet):
                        lower, upper, step = subscript.children
                        low = self._data_literal_int(lower, env)
                        high = self._data_literal_int(upper, env)
                        by = self._data_literal_int(step, env, 1)
                        values = list(range(low, high + (1 if by > 0 else -1), by))
                    else:
                        values = [self._data_literal_int(subscript, env)]
                    expanded = [[*e, v] for e in expanded for v in values]
                out.extend((name, tuple(e)) for e in expanded)
            elif isinstance(target, f03.Data_Implied_Do):
                inner, variable, lower, upper, step = target.children
                low = self._data_literal_int(lower, env)
                high = self._data_literal_int(upper, env)
                by = self._data_literal_int(step, env, 1)
                name = str(variable).lower()
                for iteration in range(low, high + (1 if by > 0 else -1), by):
                    out.extend(self._data_targets(self._items(inner), {**env, name: iteration}))
            else:
                raise NoRule(f"DATA object {type(target).__name__}: {target}")
        return out

    def _data_expanded(self, pad: str, objects: list[Any], values: list[str]) -> list[str]:
        """Flattened DATA targets, emitted as fills.

        A run that is contiguous in the last dimension becomes one slice
        assignment rather than one statement per element -- which is what a
        400-element lookup table needs to stay readable.
        """
        targets = self._data_targets(objects)
        if len(targets) != len(values):
            raise NoRule(
                f"DATA value count mismatch ({len(targets)} targets, {len(values)} values)"
            )

        def origins(name: str, rank: int) -> list[int]:
            dims = (self.semantics.declaration(name) or {}).get("dims")
            if dims is None or len(dims) != rank:
                raise NoRule(f"DATA dims unknown for {name!r}")
            found = []
            for dim in dims:
                text = str(dim.get("lb") or "1")
                if not re.fullmatch(r"-?\d+", text):
                    raise NoRule(f"DATA non-literal lower bound for {name!r}")
                found.append(int(text))
            return found

        lines: list[str] = []
        at = 0
        while at < len(targets):
            name, index = targets[at]
            if index is None:
                raise NoRule("DATA whole-array mixed with indexed")
            end = at + 1
            while end < len(targets):
                following, next_index = targets[end]
                previous = targets[end - 1][1]
                if following != name or next_index is None or previous is None:
                    break
                if next_index[:-1] != index[:-1] or next_index[-1] != previous[-1] + 1:
                    break
                end += 1
            lower = origins(name, len(index))
            leading = ", ".join(str(v - lb) for v, lb in zip(index[:-1], lower[:-1], strict=True))
            start = index[-1] - lower[-1]
            spelled = self.names.symbol(name)
            if end - at == 1:
                where = (leading + ", " if leading else "") + str(start)
                lines.append(f"{pad}{spelled}[{where}] = {values[at]}")
            else:
                where = (leading + ", " if leading else "") + f"{start}:{start + (end - at)}"
                dtype = DATA_DTYPES.get(
                    (self.semantics.declaration(name) or {}).get("dtype", ""), "np.float64"
                )
                lines.append(
                    f"{pad}{spelled}[{where}] = "
                    f"np.array([{', '.join(values[at:end])}], dtype={dtype})"
                )
            at = end
        return lines

    @staticmethod
    def _items(node: Any) -> list[Any]:
        children = getattr(node, "children", None)
        return [c for c in (children if children is not None else [node]) if c is not None]

    def _data_values(self, node: Any) -> list[str]:
        """The value list, flat, with each ``N*value`` repeat written out."""
        values: list[str] = []
        for item in self._items(node):
            if isinstance(item, f03.Data_Stmt_Value):
                repeat_node, value = item.children
                try:
                    repeat = int(str(repeat_node).split("_")[0])
                except (TypeError, ValueError):
                    # A named constant as the count: known to the compiler,
                    # not to this stage, which does not evaluate parameters.
                    # Spelled as the pipeline spells it: the caller already
                    # writes "DATA " in front of the reason, so a reason that
                    # begins with the word again reads "DATA DATA".
                    raise NoRule(f"data_repeat({repeat_node})") from None
                values.extend([self._data_value(value)] * repeat)
            else:
                values.append(self._data_value(item))
        return values

    def _data_value(self, node: Any) -> str:
        """One DATA value. Literals only, signed or not."""
        if isinstance(node, (f03.Real_Literal_Constant, f03.Int_Literal_Constant)):
            return self.names.literal(node)
        if isinstance(node, (f03.Signed_Real_Literal_Constant, f03.Signed_Int_Literal_Constant)):
            text = str(node)
            try:
                return self.names.literal(node)
            except NoRule:
                # The hoisting saw the magnitude; fparser reads the sign as
                # part of the literal only in some positions.
                unsigned = self.names.literals.get(text.lstrip("-+"))
                if unsigned is None:
                    raise
                return f"-{unsigned}" if text.startswith("-") else unsigned
        raise NoRule(f"DATA value {type(node).__name__}: {node}")

    def _condition(self, node: Any) -> str:
        """An IF's condition. fparser reports it as ``None`` for a construct
        whose test it could not attach; the branch still has to open."""
        rendered = self.expressions.render(node.children[0]) if node.children[0] else ""
        return rendered or "True"

    def _case_construct(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        select = node.children[0]
        selector = self.expressions.render(select.children[0])
        character = self.semantics.is_character(select.children[0])
        lines: list[str] = []
        body: list[str] = []
        first = True
        for child in node.children[1:]:
            if isinstance(child, f03.Case_Stmt):
                if not first:
                    self._flush(lines, body, pad)
                chosen = child.children[0]
                if chosen is None or getattr(chosen, "children", (None,))[0] is None:
                    # `case default` with nothing before it: there is no `if`
                    # for an `else` to attach to.
                    if first:
                        lines.append(f"{pad}if True:  # SELECT CASE default-only")
                        first = False
                    else:
                        lines.append(f"{pad}else:")
                else:
                    values = walk(
                        chosen,
                        (
                            f03.Name,
                            f03.Int_Literal_Constant,
                            f03.Char_Literal_Constant,
                            f03.Real_Literal_Constant,
                            f03.Logical_Literal_Constant,
                        ),
                    )
                    if any(
                        isinstance(v, f03.Case_Value_Range) for v in getattr(chosen, "children", [])
                    ):
                        raise NoRule("case value range")
                    conditions = []
                    for value in values:
                        if character or isinstance(value, f03.Char_Literal_Constant):
                            conditions.append(
                                f"_fstr_eq({selector}, {self.expressions.render(value)})"
                            )
                        else:
                            conditions.append(f"({selector} == {self.expressions.render(value)})")
                    keyword = "if" if first else "elif"
                    lines.append(f"{pad}{keyword} {' or '.join(conditions) or 'True'}:")
                    first = False
            elif isinstance(child, f03.End_Select_Stmt):
                self._flush(lines, body, pad)
            else:
                body.extend(self.render(child, indent + 1))
        return lines

    # -- named EXIT and CYCLE -------------------------------------------------

    LOOP_CONSTRUCTS = (
        f03.Block_Nonlabel_Do_Construct,
        f03.Block_Label_Do_Construct,
        f03.Action_Term_Do_Construct,
    )

    @staticmethod
    def construct_name(node: Any) -> str | None:
        """The label a construct was opened with -- ``col:`` of ``col: do ...``.

        fparser hangs it off the *reader line* rather than the statement's
        children, which is why it is easy to translate a whole corpus without
        noticing it is there.
        """
        item = getattr(node, "item", None)
        name = getattr(item, "name", None)
        return str(name).lower() if name else None

    def opening_name(self, node: Any) -> str | None:
        """The construct name of a DO or BLOCK construct node."""
        opener = walk(node, (f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt, f08.Block_Stmt))
        return self.construct_name(opener[0]) if opener else None

    def exit_stmt(self, node: Any, pad: str) -> list[str]:
        """``EXIT``/``CYCLE``, named or not.

        Unnamed, or naming the loop it already sits in, this is Python's own
        ``break``/``continue``. Naming an *outer* loop it is not: ``break``
        leaves one loop, and the statement means to leave two. The engine
        emitted the bare keyword for every form, so ``cycle col`` from inside
        an inner loop continued the inner one -- a program that still runs,
        still terminates, and is wrong in a way no structural check sees.
        """
        is_exit = isinstance(node, f03.Exit_Stmt)
        keyword = "break" if is_exit else "continue"
        target = node.children[1] if len(node.children) > 1 else None
        if target is None:
            return [f"{pad}{keyword}"]
        name = str(target).lower()
        crossed_a_loop = False
        for kind, construct in reversed(self.construct_stack):
            if kind == "block":
                if construct != name:
                    continue
                if not is_exit:
                    raise NoRule(f"CYCLE names a BLOCK construct ({name})")
                return [f"{pad}raise _FBlockExit({name!r})"]
            if construct != name:
                crossed_a_loop = True
                continue
            if not crossed_a_loop:
                # The loop it names is the one it is in: plain, and this is
                # the overwhelmingly common case -- a named loop with an
                # unnamed body reads better and costs nothing.
                return [f"{pad}{keyword}"]
            raiser = "_FLoopExit" if is_exit else "_FLoopCycle"
            return [f"{pad}raise {raiser}({name!r})"]
        raise NoRule(f"{keyword.upper()} names {name!r}, which encloses nothing here")

    def _cross_loop_targets(self, node: Any, name: str | None) -> tuple[bool, bool]:
        """Whether anything under a *nested* loop leaves or cycles ``name``.

        Only those need a catcher. A named loop whose EXITs all sit in its own
        body emits exactly what it emitted before, so a corpus that never
        crosses a loop boundary is byte-identical either way.
        """
        if not name:
            return False, False
        inner = [c for c in walk(node, self.LOOP_CONSTRUCTS) if c is not node]
        if not inner:
            return False, False
        exits = cycles = False
        for stmt in walk(node, (f03.Exit_Stmt, f03.Cycle_Stmt)):
            target = stmt.children[1] if len(stmt.children) > 1 else None
            if target is None or str(target).lower() != name:
                continue
            if not any(any(s is stmt for s in walk(loop, type(stmt))) for loop in inner):
                continue
            if isinstance(stmt, f03.Exit_Stmt):
                exits = True
            else:
                cycles = True
        return exits, cycles

    def _do_construct(self, node: Any, indent: int) -> list[str]:
        """A DO construct, with catchers for whatever crosses it.

        An EXIT that names this loop from inside a nested one wraps the whole
        loop, so the loop is what is left; a CYCLE wraps the *body*, so the
        header runs again. Both re-raise a name that is not this loop's,
        which is what lets them nest. Neither is emitted unless something
        actually crosses a loop boundary to reach this one.
        """
        name = self.opening_name(node)
        self.construct_stack.append(("do", name))
        try:
            exits, cycles = self._cross_loop_targets(node, name)
            cycle_name = name if cycles else None
            if not exits:
                return self._do_construct_inner(node, indent, cycle_name)
            pad = "    " * indent
            return [
                f"{pad}try:  # exit {name}",
                *self._do_construct_inner(node, indent + 1, cycle_name),
                f"{pad}except _FLoopExit as _le:",
                f"{pad}    if _le.args[0] != {name!r}:",
                f"{pad}        raise",
            ]
        finally:
            self.construct_stack.pop()

    def _do_construct_inner(
        self, node: Any, indent: int, cycle_name: str | None = None
    ) -> list[str]:
        pad = "    " * indent
        do_statement = walk(node, (f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt))[0]
        if not walk(do_statement, f03.Loop_Control):
            # `do` with no control at all: an unbounded loop something inside
            # leaves, which in Fortran is an EXIT or a goto past the END DO.
            return [f"{pad}while True:", *self._loop_body(node, indent, cycle_name)]
        control = walk(do_statement, f03.Loop_Control)
        if not control:
            raise NoRule("do while / infinite do")
        # Loop_Control children: (while-condition, (variable, [lo, hi, step]), ...)
        if control[0].children[0] is not None:
            lines = [f"{pad}while {self.expressions.render(control[0].children[0])}:"]
            body = []
            for child in node.children:
                if isinstance(child, (f03.Nonlabel_Do_Stmt, f03.End_Do_Stmt)):
                    continue
                body.extend(self.render(child, indent + 1))
            lines.extend(self._caught_cycle(body or [f"{pad}    pass"], indent, cycle_name))
            return lines
        if control[0].children[1] is None:
            raise NoRule("do with a loop control that is neither a count nor a condition")
        variable, bounds = control[0].children[1]
        bounds = list(bounds)
        low = self.expressions.render(bounds[0])
        high = self.expressions.render(bounds[1])
        step = self.expressions.render(bounds[2]) if len(bounds) > 2 else None
        name = pysafe(str(variable).lower())
        if step is None:
            head = f"{pad}for {name} in range({low}, {high} + 1):"
        else:
            text = step.lstrip("(").lstrip()
            if text.startswith("-") and not any(c.isalpha() or c == "_" for c in text[1:2]):
                # A literal negative step: the stop edge is known at compile time.
                head = f"{pad}for {name} in range({low}, {high} - 1, {step}):"
            elif re.fullmatch(r"-?\d+", text):
                sign = "-" if text.startswith("-") else "+"
                head = f"{pad}for {name} in range({low}, {high} {sign} 1, {step}):"
            else:
                # A variable step: the direction, and so the stop edge, is
                # only known at run time.
                head = (
                    f"{pad}for {name} in range({low}, "
                    f"({high}) + (1 if ({step}) > 0 else -1), {step}):"
                )
        return [head, *self._loop_body(node, indent, cycle_name)]

    def _caught_cycle(self, body: list[str], indent: int, cycle_name: str | None) -> list[str]:
        """A loop body, wrapped so a CYCLE naming *this* loop reaches its header."""
        if not cycle_name:
            return body
        pad = "    " * (indent + 1)
        return [
            f"{pad}try:",
            *(f"    {line}" for line in body),
            f"{pad}except _FLoopCycle as _lc:",
            f"{pad}    if _lc.args[0] != {cycle_name!r}:",
            f"{pad}        raise",
        ]

    def _loop_body(self, node: Any, indent: int, cycle_name: str | None = None) -> list[str]:
        """The statements between the DO and its END DO, at one more indent."""
        pad = "    " * indent
        # Every do pushes something, label or None: a goto from deeper than
        # this loop's own body must not break the wrong loop, and a None on
        # top is what stops it.
        self.active_labels.append(self.exit_labels.get(id(node)))
        # `do 100 i = ...` / `100 continue`: a goto to that terminator from
        # inside the body is a cycle, not an exit.
        terminator = node.children[-1] if node.children else None
        cycle_label = (
            self.label(terminator)
            if walk(node, f03.Label_Do_Stmt) and isinstance(terminator, f03.Continue_Stmt)
            else None
        )
        if cycle_label:
            self.active_labels.append(("cycle", cycle_label))
        inner = []
        started = False
        for child in node.children:
            if isinstance(child, (f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt)):
                started = True
                continue
            if isinstance(child, f03.End_Do_Stmt):
                break
            if (
                isinstance(child, f03.Continue_Stmt)
                and node.children
                and child is node.children[-1]
            ):
                continue  # a labeled do's own `NNN continue` terminator
            if started:
                inner.append(child)
        try:
            body = self.sequence(inner, indent + 1)
        finally:
            if cycle_label:
                self.active_labels.pop()
            self.active_labels.pop()
        return self._caught_cycle(body or [f"{pad}    pass"], indent, cycle_name)

    def _associate(self, node: Any, indent: int) -> list[str]:
        """``associate (a => expr)``: the name is bound once, then read.

        A plain assignment is right for the reads and for writes through a
        whole-array or component target, which is what the physics corpora use it
        for -- the body sees the same object. It is *not* right for a scalar
        target written through the association, where Fortran writes back to
        the selector and Python would rebind the local; nothing in the corpus
        does that, and it is a refusal worth having when something does.
        """
        pad = "    " * indent
        lines = []
        inherited: dict[str, Any] = {}
        for association in walk(node, f03.Association):
            alias, _, selector = association.children
            name = pysafe(str(alias).lower())
            lines.append(f"{pad}{name} = {self.expressions.render(selector)}")
            dims = self.expressions.selector_dims(selector)
            if dims:
                inherited[str(alias).lower()] = dims
        # The alias subscripts like its selector for the body's duration: a
        # component allocated from zero keeps its zero through the alias.
        bounds = self.expressions.allocated_bounds
        previous = {name: bounds.get(name) for name in inherited}
        bounds.update(inherited)
        try:
            body = [
                line
                for child in node.children
                if not isinstance(child, (f03.Associate_Stmt, f03.End_Associate_Stmt))
                for line in self.render(child, indent)
            ]
        finally:
            for name, before in previous.items():
                if before is None:
                    bounds.pop(name, None)
                else:
                    bounds[name] = before
        return lines + (body or [f"{pad}pass"])

    def _block(self, node: Any, indent: int) -> list[str]:
        """``block ... end block``: its declarations, then its statements.

        The declarations become zero-initialised locals at the enclosing
        indent rather than a scope of their own. Python has no block scope, so
        a name declared here and also declared outside would collide -- which
        is a refusal the frontend's own local table would have to raise, and
        no source in the corpus does it.
        """
        pad = "    " * indent
        name = self.opening_name(node)
        # An ``EXIT`` naming this block has no Python construct to leave --
        # the body is inlined at the enclosing indent, so ``break`` would
        # bind to whatever loop happens to be outside it. The catcher is
        # emitted only when something actually names this block.
        exited = bool(name) and any(
            len(stmt.children) > 1
            and stmt.children[1] is not None
            and str(stmt.children[1]).lower() == name
            for stmt in walk(node, f03.Exit_Stmt)
        )
        body_indent = indent + 1 if exited else indent
        body_pad = "    " * body_indent
        self.construct_stack.append(("block", name))
        try:
            lines: list[str] = []
            for child in node.children:
                if isinstance(child, (f08.Block_Stmt, f08.End_Block_Stmt)):
                    continue
                if isinstance(child, f03.Specification_Part):
                    for declaration in walk(child, f03.Type_Declaration_Stmt):
                        lines.extend(self._block_declaration(declaration, body_pad))
                    continue
                lines.extend(self.render(child, body_indent))
        finally:
            self.construct_stack.pop()
        body = lines or [f"{body_pad}pass"]
        if not exited:
            return body
        return [
            f"{pad}try:  # exit {name}",
            *body,
            f"{pad}except _FBlockExit as _be:",
            f"{pad}    if _be.args[0] != {name!r}:",
            f"{pad}        raise",
        ]

    def _block_declaration(self, declaration: Any, pad: str) -> list[str]:
        """One declaration inside a BLOCK, as a zero-initialised local."""
        spelled = str(declaration.children[0]).upper()
        initial = "0"
        if "LOGICAL" in spelled:
            initial = "False"
        elif "CHARACTER" in spelled:
            initial = "''"
        elif "INTEGER" not in spelled:
            initial = "0.0"
        lines = []
        for entity in walk(declaration, f03.Entity_Decl):
            name = pysafe(str(entity.children[0]).lower())
            value = initial
            for child in entity.children:
                if isinstance(child, f03.Initialization):
                    value = self.expressions.render(child.children[1])
            if walk(entity, f03.Explicit_Shape_Spec_List):
                raise NoRule(f"array {name!r} declared inside a BLOCK")
            lines.append(f"{pad}{name} = {value}")
        return lines

    # -- calls ----------------------------------------------------------------

    def _call(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        items = list(node.children[1].children) if node.children[1] is not None else []
        designator = node.children[0]
        if isinstance(designator, f03.Procedure_Designator):
            # ``call obj%method(...)``. The receiver's methods are not
            # modelled -- ``_new_derived`` has no methods -- so a call would
            # crash. A framework stub answers first; a call carrying
            # arguments is value-bearing (``obj%pack(state, buf)`` writes
            # through ``buf``) and is refused rather than dropped (#6); an
            # argument-less one passes no value and stays the no-op the
            # pipeline emits.
            obj = str(designator.children[0]).lower()
            method = str(designator.children[2]).lower()
            stub = self.stubs.get(f"{obj}.{method}")
            if stub is not None:
                return [f"{pad}{stub}  # {obj}%{method} (infra stub)"]
            if items:
                raise NoRule(f"type-bound call {obj}%{method} with arguments")
            return [f"{pad}pass  # {obj}%{method} (OOP method stub)"]
        name = str(designator).lower()
        transform = self.call_transforms.get(name)
        if transform is not None:
            # A call whose meaning is a framework's: answered by the package
            # that knows the framework, before anything here is consulted.
            return list(
                transform(
                    CallSite(
                        name=name,
                        indent=indent,
                        actuals=tuple(items),
                        keywords={
                            str(a.children[0]).lower(): a.children[1]
                            for a in items
                            if isinstance(a, f03.Actual_Arg_Spec)
                        },
                        render=self.expressions.render,
                        holds_handle=self.expressions.handles.add,
                    )
                )
            )
        if name == "move_alloc":
            # `move_alloc(from, to)`: the array changes hands and the source
            # becomes unallocated, which is what `allocated()` mirrors.
            moved = [self.expressions.render(CallSite.bare(item)) for item in items]
            if len(moved) >= 2:
                return [f"{pad}{moved[1]} = {moved[0]}  # move_alloc", f"{pad}{moved[0]} = None"]
        # A framework call with a stub is the stub, before anything else is
        # asked -- the pipeline consults its INFRA_STUBS ahead of generics,
        # companions and externals alike.
        stub = self.stubs.get(name)
        if stub is not None:
            return [f"{pad}{stub}  # {name} (infra stub)"]
        if name in self.semantics.generics:
            name = self.semantics.dispatch(name, items)
        record = self.semantics.procedures.get(name)
        if record is not None and record not in self.semantics.module["subprograms"]:
            record = None  # a companion's; resolved below through remotes
        prefix = ""
        if record is None and name in self.semantics.companion_generics:
            name = self.semantics.dispatch(name, items)
            record = self.semantics.procedures.get(name)
            prefix = self.expressions.remotes[name].alias + "."
        elif record is None and name in self.expressions.remotes:
            remote = self.expressions.remotes[name]
            record = self.semantics.procedures.get(remote.name)
            name, prefix = remote.name, remote.alias + "."
        if record is None:
            external = self.externals.get(name)
            if external and external.get("kind") == "subroutine":
                return self._external_call(name, external, node, pad)
            if name in intrinsics.SUBROUTINE:
                # Not an external -- the standard defines it, and every one of
                # them writes an argument, so the ``pass`` the pipeline emits
                # would drop a write. Refused under its own name so the work
                # list says "the engine does not know this intrinsic" rather
                # than "this tree is missing a library".
                raise NoRule(f"intrinsic subroutine {name!r} has no rule")
            raise NoRule(f"call to external subroutine {name!r}")

        # Bind actuals to formals BY NAME for keyword arguments: Fortran
        # allows any trailing mix once a keyword appears.
        formal_names = [a["name"] for a in record["args"]]
        actuals: list[Any] = [None] * len(formal_names)
        position = 0
        for item in items:
            if isinstance(item, f03.Actual_Arg_Spec):
                keyword = str(item.children[0]).lower()
                if keyword not in formal_names:
                    raise NoRule(f"unknown keyword {keyword} in call {name}")
                actuals[formal_names.index(keyword)] = item.children[1]
            else:
                if position >= len(actuals):
                    raise NoRule(
                        f"more actuals ({len(items)}) than "
                        f"formals ({len(formal_names)}) in call {name}"
                    )
                actuals[position] = item
                position += 1

        # Formal name -> rendered actual: sequence-association reshapes need
        # the callee's dimension names resolved to caller expressions.
        substitutions = {}
        for formal, actual in zip(record["args"], actuals, strict=True):
            if actual is not None:
                try:
                    substitutions[formal["name"]] = self.expressions.render(actual)
                except REFUSED:
                    pass

        inputs: list[str] = []
        outputs: list[str] = []
        for formal, actual in zip(record["args"], actuals, strict=True):
            if actual is None:  # an unsupplied optional
                if not formal["optional"]:
                    raise NoRule(f"missing required actual {formal['name']}")
                if self.is_optional_output(formal):
                    outputs.append("_")  # the return tuple has fixed length
                continue
            if self.is_optional_output(formal):
                inputs.append(f"want_{formal['name']}=True")
            passes = formal["intent"] in ("IN", "INOUT", "UNKNOWN") or bool(
                formal.get("buffer") and self.buffer_out_arrays
            )
            view = None
            if passes and not self.is_optional_output(formal):
                # A buffer formal is a parameter of the callee as well as one
                # of its returns: the callee writes into the storage this
                # caller owns, so the actual has to be passed in too.
                argument, view = self._input_argument(formal, actual, substitutions, inputs)
                inputs.append(argument)
            if formal["intent"] in ("OUT", "INOUT"):
                # A sequence-associated element actual: the callee's array IS
                # the view into this caller's buffer, so the result comes
                # back through that view (#27).
                outputs.append(f"{view}[...]" if view else self._output_target(actual))

        elemental = any("ELEMENTAL" in str(p).upper() for p in (record.get("prefixes") or []))
        broadcasts = False
        if elemental:
            for actual in actuals:
                if actual is None:
                    continue
                try:
                    if self.semantics.rank(actual) > 0:
                        broadcasts = True
                        break
                except REFUSED:
                    pass
        # Host association: an internal callee takes the host variables it
        # touches as extra trailing actuals.
        for host_var in record.get("host_vars") or ():
            inputs.append(self.names.symbol(host_var))
        # Python takes no positional argument after a keyword one, and
        # Fortran's optionals can leave a gap anywhere in the list.
        inputs = [a for a in inputs if "=" not in a] + [a for a in inputs if "=" in a]
        target = f"{prefix}{pysafe(emit_name(record))}"
        if broadcasts:
            call = f"_f_ecall({target}, {', '.join(inputs)})"
        else:
            call = f"{target}({', '.join(inputs)})"
        if outputs:
            # A whole-array OUT actual is copied into the caller's buffer by
            # the runtime rather than assigned through ``[...]``: the callee
            # may return a narrower array than the buffer it was handed.
            has_array = any("[...]" in target for target in outputs)
            if has_array and len(outputs) == 1:
                base = outputs[0].replace("[...]", "")
                return [f"{pad}_f_copy_out({base}, {call})"]
            if has_array:
                lines = [f"{pad}_out = {call}"]
                for i, target in enumerate(outputs):
                    if "[...]" in target:
                        lines.append(f"{pad}_f_copy_out({target.replace('[...]', '')}, _out[{i}])")
                    else:
                        lines.append(f"{pad}{target} = _out[{i}]")
                return lines
            return [f"{pad}{', '.join(outputs)} = {call}"]
        return [f"{pad}{call}"]

    def _input_argument(
        self, formal: dict[str, Any], actual: Any, substitutions: dict[str, str], inputs: list[str]
    ) -> tuple[str, str | None]:
        """The rendered actual, and -- when it is a sequence-associated
        element -- the view of the caller's storage it stands for."""
        rendered = self.expressions.render(actual)
        view = None
        formal_dims = formal.get("dims") or []
        if len(formal_dims) >= 1 and all(d.get("ub") for d in formal_dims):
            try:
                rank = self.semantics.rank(actual)
            except REFUSED:
                rank = None
            if rank is not None and 0 < rank < len(formal_dims):
                # Fortran sequence association: a lower-rank actual fills the
                # dummy in column-major order.
                shape = ", ".join(
                    substitutions.get(d["ub"], "") or self.bound(d["ub"]) for d in formal_dims
                )
                rendered = f"np.reshape({rendered}, ({shape},), order='F')"
            elif (
                rank == 0
                and len(formal_dims) >= 1
                and isinstance(actual, f03.Part_Ref)
                and self.semantics.is_array(str(actual.children[0]).lower())
            ):
                rendered = view = self._sequence_association(actual, formal_dims, substitutions)
        keyword = formal["optional"] or any("=" in a for a in inputs)
        return (f"{pysafe(formal['name'])}={rendered}" if keyword else rendered), view

    def _output_target(self, actual: Any) -> str:
        if isinstance(actual, f03.Name):
            name = str(actual).lower()
            # A whole-array out actual: assign INTO the buffer, preserving
            # Fortran's aliasing semantics.
            if self.semantics.is_array(name):
                return f"{self.names.symbol(name)}[...]"
            return self.names.symbol(name)
        if isinstance(actual, f03.Part_Ref):
            # Subscripted, so an array whatever this file was told about it.
            return self.expressions.subscript(str(actual.children[0]).lower(), actual.children[1])
        if isinstance(actual, f03.Data_Ref):
            return self.expressions.render(actual)
        raise NoRule("out actual arg is not a variable/section")

    def _external_call(self, name: str, external: dict[str, Any], node: Any, pad: str) -> list[str]:
        """A call to a registered external: out positions come from the
        registry, whose shim signature mirrors the Fortran intent layout."""
        positional = []
        keywords = []
        if node.children[1] is not None:
            for item in node.children[1].children:
                if isinstance(item, f03.Actual_Arg_Spec):
                    # Registry shims take **kwargs: keywords pass through.
                    # Only safe for pure-IN keywords; an OUT keyword would
                    # need registry support, so those stay refused.
                    keyword, value = item.children
                    keywords.append(f"{str(keyword).lower()}={self.expressions.render(value)}")
                    continue
                positional.append(item)
        out_positions = set(external.get("out_positions", []))
        inputs, outputs = [], []
        for at, actual in enumerate(positional):
            if at in out_positions:
                if isinstance(actual, f03.Name):
                    name_ = str(actual).lower()
                    outputs.append(
                        f"{self.names.symbol(name_)}[...]"
                        if self.semantics.is_array(name_)
                        else self.names.symbol(name_)
                    )
                elif isinstance(actual, f03.Part_Ref) and self.semantics.is_array(
                    str(actual.children[0]).lower()
                ):
                    outputs.append(
                        self.expressions.subscript(
                            str(actual.children[0]).lower(), actual.children[1]
                        )
                    )
                elif isinstance(actual, f03.Data_Ref):
                    outputs.append(self.expressions.render(actual))
                else:
                    raise NoRule(f"external out actual {name}[{at}]")
            else:
                inputs.append(self.expressions.render(actual))
        call = f"_ext.{name}({', '.join(inputs + keywords)})"
        if outputs:
            return [f"{pad}{', '.join(outputs)} = {call}"]
        return [f"{pad}{call}"]

    @staticmethod
    def is_optional_output(formal: dict[str, Any]) -> bool:
        """Optional OUT, the ``want_<name>`` sentinel convention: a trailing
        keyword ``want_<name>=False`` carries ``present()``, and the value is
        ALWAYS in the fixed-length return tuple -- callers that did not ask
        ignore it via ``_``."""
        return bool(formal["optional"]) and formal["intent"] == "OUT"

    def returned_value(self) -> str:
        subprogram = self.semantics.subprogram
        if subprogram["kind"] == "function":
            return pysafe(subprogram["result"])
        outputs = [pysafe(a["name"]) for a in subprogram["args"] if a["intent"] in ("OUT", "INOUT")]
        if not outputs:
            return ""
        return ", ".join(outputs) if len(outputs) > 1 else outputs[0]

    # -- sequence association -------------------------------------------------

    def _sequence_association(
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
        name = str(actual.children[0]).lower()
        arglist = actual.children[1]
        subscripts = (
            (arglist.children if hasattr(arglist, "children") else [arglist])
            if arglist is not None
            else []
        )
        declaration = self.semantics.declaration(name)
        if declaration is None:
            raise NoRule(f"seq-assoc: undeclared {name}")
        actual_dims = declaration.get("dims") or []
        if len(subscripts) != len(actual_dims):
            raise NoRule(f"seq-assoc: rank mismatch {name}")
        if len(formal_dims) > len(actual_dims):
            raise NoRule(f"seq-assoc: formal rank > actual rank {name}")

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
        if at_lower_bound and len(formal_dims) <= len(actual_dims) - first_scalar:
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
                    rendered = self.expressions.render(subscripts[at])
                    parts.append(f"({rendered}) - ({self.bound(low)})")
            return f"{self.names.symbol(name)}[{', '.join(parts)}]"

        shape = ", ".join(
            substitutions.get(d["ub"], "") or self.bound(d["ub"]) for d in formal_dims
        )
        flat = f"{self.names.symbol(name)}.ravel(order='F')"
        shifts = []
        for at, subscript in enumerate(subscripts):
            low = actual_dims[at].get("lb", "1")
            shifts.append(f"({self.expressions.render(subscript)} - {low})")
        offset = shifts[0]
        stride = "1"
        for at in range(1, len(shifts)):
            high = actual_dims[at - 1].get("ub", "1")
            high_py = self.bound(high) if not high.isdigit() else high
            stride = f"{stride} * {high_py}"
            offset = f"{offset} + {shifts[at]} * {stride}"
        return f"np.reshape({flat}[{offset}:], ({shape},), order='F')"

    def _shifted(self, node: Any) -> str:
        """A single 1-based index, 0-based. A literal folds only while the
        folded value stays inside the whitelist; otherwise it references the
        hoisted constant, minus one."""
        if isinstance(node, f03.Int_Literal_Constant):
            folded = int(str(node).split("_")[0]) - 1
            if folded in (0, 1, 2):
                return str(folded)
            return f"{self.names.literal(node)} - 1"
        return f"{self.expressions.render(node)} - 1"

    def bound(self, text: str) -> str:
        """Declared bound text -> Python. Lives on the expression layer, which
        is where a bound's *origin* also has to go through it."""
        return self.expressions.bound(text)

    @staticmethod
    def label(node: Any) -> str | None:
        item = getattr(node, "item", None)
        found = getattr(item, "label", None) if item is not None else None
        if found:
            return str(found)
        # A construct carries its label on its first statement, not on itself:
        # `100 if (x) then` is an If_Construct whose If_Then_Stmt has the 100.
        first = node.children[0] if getattr(node, "children", None) else None
        item = getattr(first, "item", None) if first is not None else None
        found = getattr(item, "label", None) if item is not None else None
        return str(found) if found else None
