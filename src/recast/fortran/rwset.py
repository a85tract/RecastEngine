"""Per-block read and write sets.

Migrated from the source pipeline's ``pipeline/rwset.py``, Fortran half.
The original held both halves -- this analysis and a walk of the generated
Python -- in one file, together with the comparison between them. Only this
half is analysis of the source, so only this half is a Frontend's job; the
Python walk and the comparison are a Verifier's, and live there.

What the split buys: the sets this produces land in ``Facts`` and can be
compared against *any* target language by a Verifier that never learns
Fortran. What it costs is nothing, because the two halves only ever met
through a set equality.

Block ids come from ``chunk``, so a mismatch names the same piece of code
every other stage names.

The analysis is deliberately syntactic and slightly over-approximate in one
direction: an unrecognised statement falls back to "every name in it is a
read". A spurious read makes a block fail the cross-check and go to the agent
queue, which is recoverable; a missed write does not, so nothing here is
allowed to under-report one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03, walk
from recast.fortran.chunk import chunk_subprogram
from recast.fortran.intrinsics import ALL as INTRINSICS
from recast.fortran.intrinsics import STATE_QUERY, TRANSFORMATIONAL
from recast.fortran.semantics import Semantics, Unanalyzable, for_subprogram

KIND_ARG_FNS = frozenset({"real", "dble", "int", "nint", "aint", "anint", "floor", "ceiling"})
"""Conversions whose optional second argument is a kind name, not a value.

``real(x, r8)`` reads ``x`` and not ``r8``. Counting the kind as a read is the
one over-approximation that would fire on nearly every line of model physics.
"""


@dataclass(frozen=True)
class Scope:
    """Everything the analysis needs that is not in the statement itself.

    The pipeline this came from kept all five as module globals that the
    command line filled in, which meant the read/write sets of a file depended
    on which file had been processed before it.
    """

    subprograms: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Name -> interface record, for procedures defined alongside this one."""

    generics: dict[str, list[str]] = field(default_factory=dict)
    """Generic name -> specific names, for call dispatch."""

    ranks: dict[str, int] = field(default_factory=dict)
    """Symbol -> declared rank. A local named ``sum`` shadows the intrinsic."""

    chars: frozenset[str] = frozenset()
    """Character-typed symbols. ``write(buf, ...)`` to one of these is a write."""

    semantics: Semantics | None = None
    """Type and shape answers, for the questions dispatch needs.

    Optional so a caller can still build a ``Scope`` by hand; without it a
    generic call is treated as an unresolved external, which is the
    conservative reading rather than a guess.
    """

    externals: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Procedures with no source here, and which arguments they write.

    ``{"endrun": {"out_positions": []}}``. Operator-supplied, like intent
    overrides: without it a call to an unanalyzed procedure has to be treated
    as reading its arguments and writing none, which is wrong in whichever
    direction the procedure actually behaves.
    """


def scope_for(
    record: dict[str, Any],
    sub_name: str,
    *,
    externals: dict[str, dict[str, Any]] | None = None,
) -> Scope:
    """Build a ``Scope`` for one subprogram out of an ``interface.extract`` record."""
    from recast.fortran.interface import subprogram_key

    subs = {subprogram_key(s): s for s in record["subprograms"]}
    sub = (
        subs[sub_name]
        if sub_name in subs
        else next(s for s in record["subprograms"] if s["name"] == sub_name)
    )

    ranks: dict[str, int] = {}
    chars: set[str] = set()
    declared: list[dict[str, Any]] = [
        *sub["args"],
        *sub["locals"],
        *sub["local_parameters"],
        *record["module_state"],
        *record["module_parameters"],
    ]
    for entry in declared:
        name = entry["name"]
        ranks[name] = len(entry.get("dims") or [])
        if entry.get("dtype") == "str":
            chars.add(name)
    if sub["result"] is not None:
        ranks.setdefault(sub["result"], len(sub["result_dims"] or []))

    return Scope(
        subprograms=subs,
        generics=dict(record["generics"]),
        ranks=ranks,
        chars=frozenset(chars),
        semantics=for_subprogram(record, sub_name),
        externals=dict(externals or {}),
    )


def expr_reads(node: Any, scope: Scope) -> set[str]:
    """Every symbol an expression reads.

    A name that is a declared local or dummy is a read even when it collides
    with an intrinsic -- Fortran lets a variable named ``sum`` shadow the
    function, and treating that as a call loses a real dataflow edge.
    """
    reads: set[str] = set()
    if node is None or isinstance(node, str):
        return reads

    if isinstance(node, f03.Name):
        name = str(node).lower()
        if name in scope.ranks:
            return {name}
        if name not in STATE_QUERY and name not in TRANSFORMATIONAL:
            if (
                name not in scope.subprograms
                and name not in scope.generics
                and name not in scope.externals
            ):
                reads.add(name)  # an arg-less function reference parses as a bare Name
        return reads

    if isinstance(
        node, (f03.Part_Ref, f03.Intrinsic_Function_Reference, f03.Structure_Constructor)
    ):
        fname = str(node.children[0]).lower()
        if node.children[1] is not None:
            args = node.children[1]
            items = list(args.children) if hasattr(args, "children") else [args]
            if fname in KIND_ARG_FNS and len(items) == 2:
                items = items[:1]
            for item in items:
                reads |= expr_reads(item, scope)
        if scope.ranks.get(fname, 0) > 0:
            # A declared array shadows an intrinsic name -- the same rule the
            # bare-Name branch applies. zm_conv declares `gamma(pcols,pver)`,
            # and reading `gamma(i,k)` is dataflow, not a call to GAMMA.
            reads.add(fname)
            return reads
        known = (
            fname in scope.subprograms
            or fname in scope.generics
            or fname in scope.externals
            or fname in INTRINSICS
        )
        if not known:
            reads.add(fname)  # not a call: an array element read
        return reads

    if isinstance(node, (f03.Actual_Arg_Spec, f03.Component_Spec)):
        return expr_reads(node.children[1], scope)  # the keyword is not a read

    if isinstance(node, f03.Data_Ref):
        # The root object is the read; component names are attributes of it,
        # which the target side spells the same way and also does not count.
        reads |= expr_reads(node.children[0], scope)
        for comp in node.children[1:]:
            if isinstance(comp, f03.Part_Ref) and comp.children[1] is not None:
                reads |= expr_reads(comp.children[1], scope)
        return reads

    for child in getattr(node, "children", []) or []:
        reads |= expr_reads(child, scope)
    return reads


def _resolve_generic(name: str, actuals: list[Any], scope: Scope) -> str | None:
    """The specific procedure a generic call dispatches to, or ``None``.

    Delegates to ``semantics.dispatch``, which refuses when the call matches
    none of the specifics or more than one. This module used to score the
    candidates and take the best instead -- a second implementation of one
    language question, disagreeing with the emitter's in exactly the cases
    that matter. Picking an overload wrongly changes which arguments are
    written, and this is the analysis a gate compares those writes against, so
    a wrong pick would be checked against itself.

    ``None`` means the call is read as an unresolved external: its arguments
    are read, none is written. Conservative, and visible as a mismatch rather
    than as a silently different answer.
    """
    if scope.semantics is None:
        return None
    try:
        return scope.semantics.dispatch(name, actuals)
    except Unanalyzable:
        return None


def _bind_actuals(callee: dict[str, Any], items: list[Any]) -> list[Any]:
    """Line actual arguments up with formals, honouring keyword arguments."""
    formals = [a["name"] for a in callee["args"]]
    bound: list[Any] = [None] * len(formals)
    position = 0
    for item in items:
        if isinstance(item, (f03.Actual_Arg_Spec, f03.Component_Spec)):
            keyword = str(item.children[0]).lower()
            if keyword in formals:
                bound[formals.index(keyword)] = item.children[1]
        elif position < len(bound):
            bound[position] = item
            position += 1
    return bound


def rwset(node: Any, scope: Scope) -> tuple[set[str], set[str]]:
    """``(reads, writes)`` for one statement or construct."""
    reads: set[str] = set()
    writes: set[str] = set()

    def _write_actual(actual: Any) -> None:
        """Record an out-argument, the way the pipeline this came from did.

        Differs from ``write_target`` on a derived-type actual: this counts the
        *component* name as a read as well, where the assignment path does not.
        The two disagree, and this repository keeps the disagreement rather
        than resolving it, because the pipeline's answers are the ones a
        bit-exact gate has been run against and this one has not.
        """
        if isinstance(actual, f03.Name):
            writes.add(str(actual).lower())
        elif isinstance(actual, (f03.Part_Ref, f03.Data_Ref)):
            writes.add(str(actual.children[0]).lower())
            for child in actual.children[1:]:
                reads.update(expr_reads(child, scope))

    def write_target(target: Any) -> None:
        """Record an assignment target: the root is written, subscripts are read."""
        if isinstance(target, f03.Name):
            writes.add(str(target).lower())
        elif isinstance(target, f03.Data_Ref):
            writes.add(str(target.children[0]).lower())
            for comp in target.children[1:]:
                if isinstance(comp, f03.Part_Ref) and comp.children[1] is not None:
                    reads.update(expr_reads(comp.children[1], scope))
        else:
            writes.add(str(target.children[0]).lower())
            for child in target.children[1:]:
                reads.update(expr_reads(child, scope))

    def call(stmt: Any) -> None:
        name = str(stmt.children[0]).lower()
        items = list(stmt.children[1].children) if stmt.children[1] is not None else []
        if name in scope.generics:
            name = _resolve_generic(name, items, scope) or name

        callee = scope.subprograms.get(name)
        actuals = _bind_actuals(callee, items) if callee is not None else items

        if callee is None:
            external = scope.externals.get(name)
            out_positions = set(external.get("out_positions", [])) if external else set()
            for j, actual in enumerate(actuals):
                if j in out_positions:
                    write_target(actual)
                else:
                    reads.update(expr_reads(actual, scope))
            return

        for formal, actual in zip(callee["args"], actuals, strict=False):
            if actual is None:
                continue
            # UNKNOWN counts as a read: an undeclared intent might be one, and
            # a missed read only ever costs a spurious mismatch.
            if formal["intent"] in ("IN", "INOUT", "UNKNOWN"):
                reads.update(expr_reads(actual, scope))
            if formal["intent"] in ("OUT", "INOUT"):
                _write_actual(actual)

    def visit(stmt: Any) -> None:
        if isinstance(stmt, f03.Assignment_Stmt):
            lhs, _, rhs = stmt.children
            if isinstance(lhs, f03.Part_Ref):
                root = str(lhs.children[0]).lower()
                args = lhs.children[1].children if lhs.children[1] is not None else []
                if (
                    scope.ranks.get(root) == 0
                    and args
                    and all(isinstance(a, f03.Name) for a in args)
                ):
                    return  # a statement-function definition, not dataflow
            write_target(lhs)
            reads.update(expr_reads(rhs, scope))

        elif isinstance(stmt, f03.If_Stmt):
            reads.update(expr_reads(stmt.children[0], scope))
            visit(stmt.children[1])

        elif isinstance(stmt, f03.If_Construct):
            for child in stmt.children:
                if isinstance(child, (f03.If_Then_Stmt, f03.Else_If_Stmt)):
                    reads.update(expr_reads(child.children[0], scope))
                elif not isinstance(child, (f03.Else_Stmt, f03.End_If_Stmt)):
                    visit(child)

        elif isinstance(stmt, f03.Case_Construct):
            for child in stmt.children:
                if isinstance(child, (f03.Select_Case_Stmt, f03.Case_Stmt)):
                    reads.update(expr_reads(child.children[0], scope))
                elif not isinstance(child, f03.End_Select_Stmt):
                    visit(child)

        elif isinstance(stmt, f03.Call_Stmt):
            call(stmt)

        elif isinstance(stmt, (f03.Return_Stmt, f03.Cycle_Stmt, f03.Exit_Stmt)):
            pass  # a construct name is control flow, not a symbol read

        elif isinstance(stmt, f03.Where_Stmt):
            mask, assignment = stmt.children
            reads.update(expr_reads(mask, scope))
            visit(assignment)

        elif isinstance(stmt, f03.Where_Construct):
            for child in stmt.children:
                if isinstance(child, f03.Where_Construct_Stmt):
                    reads.update(expr_reads(child.children[0], scope))
                elif not isinstance(child, (f03.Elsewhere_Stmt, f03.End_Where_Stmt)):
                    visit(child)

        elif isinstance(stmt, f03.Allocate_Stmt):
            for allocation in walk(stmt, f03.Allocation):
                target = allocation.children[0]
                writes.add(
                    str(target).lower()
                    if isinstance(target, f03.Name)
                    else str(target.children[0]).lower()
                )
                for shape in walk(allocation, f03.Allocate_Shape_Spec):
                    reads.update(expr_reads(shape, scope))

        elif isinstance(stmt, f03.Deallocate_Stmt):
            # A plain allocatable becomes ``x = None``, which is a write. A
            # derived-type *component* dealloc becomes ``pass`` -- the object
            # is collected -- so it carries no dataflow either way.
            for obj in stmt.children[0].children:
                if isinstance(obj, f03.Name):
                    writes.add(str(obj).lower())

        elif isinstance(stmt, f03.Write_Stmt):
            # A write to a log unit has no dataflow. An *internal* write, whose
            # unit is a character variable, writes that variable.
            control, items = stmt.children
            units = [str(n).lower() for n in walk(control, f03.Name)]
            if units and units[0] in scope.chars:
                writes.add(units[0])
                if items is not None:
                    reads.update(expr_reads(items, scope))

        elif isinstance(stmt, f03.Pointer_Assignment_Stmt):
            target, _, rhs = stmt.children
            if isinstance(target, f03.Name):
                writes.add(str(target).lower())
            if not str(rhs).strip().lower().startswith("null("):
                reads.update(expr_reads(rhs, scope))

        elif isinstance(stmt, f03.Nullify_Stmt):
            for obj in walk(stmt.children[1], f03.Name):
                writes.add(str(obj).lower())

        elif isinstance(stmt, (f03.Block_Nonlabel_Do_Construct, f03.Block_Label_Do_Construct)):
            do_stmt = walk(stmt, (f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt))[0]
            control = walk(do_stmt, f03.Loop_Control)
            if control:
                # Three forms, and the pipeline this came from handled only the
                # first: `do i = lo, hi` writes the counter and reads its
                # bounds, `do while (c)` reads c and writes nothing, and a bare
                # `do` does neither. Assuming the counted form crashed the
                # analysis outright on four of the thirty translated modules.
                condition, counter = control[0].children[0], control[0].children[1]
                if counter is not None:
                    var, bounds = counter
                    writes.add(str(var).lower())
                    for bound in bounds:
                        reads.update(expr_reads(bound, scope))
                elif condition is not None:
                    reads.update(expr_reads(condition, scope))
            body = False
            for child in stmt.children:
                if isinstance(child, (f03.Nonlabel_Do_Stmt, f03.Label_Do_Stmt)):
                    body = True
                elif isinstance(child, f03.End_Do_Stmt):
                    break
                elif body and not isinstance(child, (f03.Cycle_Stmt, f03.Exit_Stmt)):
                    visit(child)

        else:
            # Conservative fallback. Over-reporting a read costs a block a
            # review; under-reporting a write costs a wrong answer nobody sees.
            reads.update(expr_reads(stmt, scope))

    visit(node)
    return reads, writes


def block_rwsets(sub: Any, scope: Scope) -> list[dict[str, Any]]:
    """``[{id, reads, writes}, ...]`` for every block of one subprogram."""
    out = []
    for block_id, node, _span in chunk_subprogram(sub):
        reads, writes = rwset(node, scope)
        out.append({"id": block_id, "reads": sorted(reads), "writes": sorted(writes)})
    return out
