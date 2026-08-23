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

from recast.fortran._parse import f03, f08, walk
from recast.fortran.interface import emit_name
from recast.fortran.semantics import Semantics, Unanalyzable
from recast.transform.numpy.expressions import Expressions
from recast.transform.numpy.names import Names
from recast.transform.numpy.vocabulary import pysafe
from recast.transform.rules import NoRule

__all__ = ["REFUSED", "Statements"]

REFUSED = (NoRule, Unanalyzable)
"""What a refusal looks like from either floor below this one."""

ALLOCATED_DTYPES = {
    "float64": "np.float64",
    "float32": "np.float32",
    "int32": "np.int32",
    "int64": "np.int64",
    "bool": "np.bool_",
}
"""Declared dtype -> the dtype an ``allocate`` requests. Anything else gets
``np.float64``, which is what the declaration meant if it said nothing."""

EXTENT = re.compile(r"(?:SIZE|UBOUND)\(\s*(\w+)\s*(?:,\s*((?:dim\s*=\s*)?\d+)\s*)?\)", re.I)
"""``size(a)``, ``size(a, 2)``, ``ubound(a, dim=2)`` inside a declared bound."""

DIM_KEYWORD = re.compile(r"dim\s*=\s*", re.I)

BOUND_TOKENS = re.compile(r"[A-Za-z_]\w*|\d+|[()+\-*/ ]")
"""What a declared bound is allowed to be made of. Bound texts are simple by
construction; anything richer refuses the statement that needed the bound."""

UNIT_NAME = re.compile(r"[a-z_]\w*")


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

    stubs: dict[str, str] = field(default_factory=dict)
    """Framework subroutine -> the statement that stands in for it.

    The statement-shaped half of the table whose function-shaped half lives on
    ``Expressions``: calls into a framework the translation does not carry.
    Supplied by the domain package that knows the framework.
    """

    masks: list[str] = field(default_factory=list)
    """Enclosing WHERE masks. Fortran ANDs a nested WHERE into its outer one."""

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

    # -- entry points ---------------------------------------------------------

    def scan(self, subprogram: Any) -> None:
        """Pair every do-construct with an immediately-following labeled
        CONTINUE anywhere in the nesting tree -- inside such a loop, ``goto``
        to that label is exactly ``exit``."""
        self.exit_labels = {}
        self.consumed_labels = set()

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
            condition, action = node.children
            inner = self.render(action, 0)
            if len(inner) != 1:
                raise NoRule("multi-line action in single-line if")
            return [f"{pad}if {self.expressions.render(condition)}:", f"{pad}    {inner[0]}"]
        if isinstance(node, f03.If_Construct):
            return self._if_construct(node, indent)
        if isinstance(node, f03.Case_Construct):
            return self._case_construct(node, indent)
        if isinstance(node, f03.Return_Stmt):
            return [f"{pad}return {self.returned_value()}"]
        if isinstance(node, f03.Cycle_Stmt):
            return [f"{pad}continue"]
        if isinstance(node, f03.Exit_Stmt):
            return [f"{pad}break"]
        if isinstance(node, f03.Goto_Stmt):
            return self._goto(node, pad)
        if isinstance(node, f03.Continue_Stmt):
            label = self.label(node)
            if label in self.consumed_labels:
                return [f"{pad}pass  # {label} continue (break target)"]
            return [f"{pad}pass  # continue"]
        if isinstance(node, f03.Call_Stmt):
            return self._call(node, indent)
        if isinstance(node, (f03.Block_Nonlabel_Do_Construct, f03.Block_Label_Do_Construct)):
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
        if isinstance(node, f03.Stop_Stmt):
            message = ""
            if node.children and node.children[1]:
                message = str(node.children[1]).strip()
            return [f"{pad}raise SystemExit({message!r})  # STOP"]
        if isinstance(node, f03.Read_Stmt):
            return [f"{pad}pass  # READ (I/O stub)"]
        if isinstance(node, (f03.Open_Stmt, f03.Close_Stmt)):
            return [f"{pad}pass  # OPEN/CLOSE (I/O stub)"]
        if isinstance(node, (f03.Forall_Construct, f03.Forall_Stmt)):
            return self._forall(node, indent)
        if isinstance(node, f03.Associate_Construct):
            return self._associate(node, indent)
        if isinstance(node, f08.Block_Construct):
            return self._block(node, indent)
        raise NoRule(f"no statement rule for {type(node).__name__}")

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
            if isinstance(node, f03.Cycle_Stmt):
                lines.append(f"{pad}continue")
            elif isinstance(node, f03.Exit_Stmt):
                lines.append(f"{pad}break")
            else:
                lines.extend(self.render(node, indent))
        return lines

    # -- assignment -----------------------------------------------------------

    def _assignment(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        target, _, value = node.children
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
        return [f"{pad}{self.target(target)} = {self.expressions.render(value)}"]

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
            name = str(node.children[0]).lower()
            if self.semantics.is_array(name):
                return self.expressions.subscript(name, node.children[1])
            raise NoRule(f"assignment to non-array ref {name}")
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
            dtype = (
                ALLOCATED_DTYPES.get(declaration["dtype"], "np.float64")
                if declaration
                else "np.float64"
            )
            lines.append(f"{pad}{rendered} = np.empty({shape_text}, dtype={dtype})")
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
            if str(format_).strip() != "*":
                raise NoRule("formatted internal write")
            arguments = ", ".join(
                self.expressions.render(item)
                for item in (items.children if hasattr(items, "children") else [items])
            )
            return [f"{pad}{pysafe(name)} = _f_list_write({arguments})"]
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

    def _do_construct(self, node: Any, indent: int) -> list[str]:
        pad = "    " * indent
        do_statement = walk(node, (f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt))[0]
        if not walk(do_statement, f03.Loop_Control):
            # `do` with no control at all: an unbounded loop something inside
            # leaves, which in Fortran is an EXIT or a goto past the END DO.
            return [f"{pad}while True:", *self._loop_body(node, indent)]
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
            lines.extend(body or [f"{pad}    pass"])
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
        return [head, *self._loop_body(node, indent)]

    def _loop_body(self, node: Any, indent: int) -> list[str]:
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
        return body or [f"{pad}    pass"]

    def _associate(self, node: Any, indent: int) -> list[str]:
        """``associate (a => expr)``: the name is bound once, then read.

        A plain assignment is right for the reads and for writes through a
        whole-array or component target, which is what CAM and CLOUDSC use it
        for -- the body sees the same object. It is *not* right for a scalar
        target written through the association, where Fortran writes back to
        the selector and Python would rebind the local; nothing in the corpus
        does that, and it is a refusal worth having when something does.
        """
        pad = "    " * indent
        lines = []
        for association in walk(node, f03.Association):
            name = pysafe(str(association.children[0]).lower())
            lines.append(f"{pad}{name} = {self.expressions.render(association.children[2])}")
        body = [
            line
            for child in node.children
            if not isinstance(child, (f03.Associate_Stmt, f03.End_Associate_Stmt))
            for line in self.render(child, indent)
        ]
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
        lines: list[str] = []
        for child in node.children:
            if isinstance(child, (f08.Block_Stmt, f08.End_Block_Stmt)):
                continue
            if isinstance(child, f03.Specification_Part):
                for declaration in walk(child, f03.Type_Declaration_Stmt):
                    lines.extend(self._block_declaration(declaration, pad))
                continue
            lines.extend(self.render(child, indent))
        return lines or [f"{pad}pass"]

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
        name = str(node.children[0]).lower()
        items = list(node.children[1].children) if node.children[1] is not None else []
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
            if formal["intent"] in ("IN", "INOUT", "UNKNOWN") and not self.is_optional_output(
                formal
            ):
                inputs.append(self._input_argument(formal, actual, substitutions, inputs))
            if formal["intent"] in ("OUT", "INOUT"):
                outputs.append(self._output_target(actual))

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
    ) -> str:
        rendered = self.expressions.render(actual)
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
                rendered = self._sequence_association(actual, formal_dims, substitutions)
        keyword = formal["optional"] or any("=" in a for a in inputs)
        return f"{pysafe(formal['name'])}={rendered}" if keyword else rendered

    def _output_target(self, actual: Any) -> str:
        if isinstance(actual, f03.Name):
            name = str(actual).lower()
            # A whole-array out actual: assign INTO the buffer, preserving
            # Fortran's aliasing semantics.
            if self.semantics.is_array(name):
                return f"{self.names.symbol(name)}[...]"
            return self.names.symbol(name)
        if isinstance(actual, f03.Part_Ref) and self.semantics.is_array(
            str(actual.children[0]).lower()
        ):
            return self.expressions.subscript(str(actual.children[0]).lower(), actual.children[1])
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
                else:
                    parts.append(self._shifted(subscripts[at]))
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
                return f"np.size({name})"
            axis = int(DIM_KEYWORD.sub("", dimension)) - 1
            return f"np.size({name}, {axis})"

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
            if re.match(r"[A-Za-z_]", token):
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
