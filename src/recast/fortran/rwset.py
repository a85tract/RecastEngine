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

import re
from dataclasses import dataclass, field
from typing import Any

from recast.fortran._parse import f03, f08, walk
from recast.fortran.chunk import chunk_subprogram
from recast.fortran.intrinsics import ALL as INTRINSICS
from recast.fortran.intrinsics import STATE_QUERY, TRANSFORMATIONAL
from recast.fortran.semantics import Semantics, Unanalyzable, for_subprogram

KIND_ARG_FNS = frozenset(
    {"real", "dble", "int", "nint", "aint", "anint", "floor", "ceiling", "cmplx", "float"}
)
"""Conversions whose KIND argument is a kind name, not a value.

``real(x, r8)`` reads ``x`` and not ``r8``. Counting the kind as a read is the
one over-approximation that would fire on nearly every line of model physics.
The rule has to match the emitter's exactly, which is the pipeline's #37: a
``kind=`` keyword anywhere, the second positional of a two-argument
conversion -- except ``cmplx``, whose second is the imaginary part -- and
``cmplx``'s third positional.
"""


def _without_kind_argument(fname: str, items: list[Any]) -> list[Any]:
    """The actuals of a conversion that are values, the KIND name dropped."""
    kept = []
    for at, item in enumerate(items):
        if isinstance(item, f03.Actual_Arg_Spec) and str(item.children[0]).lower() == "kind":
            continue
        if len(items) == 2 and at == 1 and fname != "cmplx":
            continue
        if fname == "cmplx" and len(items) == 3 and at == 2:
            continue
        kept.append(item)
    return kept


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

    dummy_procedures: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Dummy procedure argument -> the interface record it was declared with.

    ``procedure(func) :: fcn`` makes ``call fcn(...)`` a call whose argument
    intents are stated by the abstract interface ``func``. Without them the
    call falls to the unresolved-external reading -- every actual read, none
    written -- which loses the write the callback exists to make.
    """

    externals: dict[str, dict[str, Any]] = field(default_factory=dict)
    alias_dims: dict[str, Any] = field(default_factory=dict)
    """An ``associate`` alias -> the dims of the component it selects, so a
    subscript of the alias reads the component's lower-bound names the way
    the translation spells them. Filled while the construct is visited."""
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
    companions: tuple[dict[str, Any], ...] = (),
) -> Scope:
    """Build a ``Scope`` for one subprogram out of an ``interface.extract`` record."""
    from recast.fortran.interface import subprogram_key

    by_key = {subprogram_key(s): s for s in record["subprograms"]}
    sub = (
        by_key[sub_name]
        if sub_name in by_key
        else next(s for s in record["subprograms"] if s["name"] == sub_name)
    )
    # Keyed by the *bare* name, because that is the spelling a call site uses
    # and every question this scope answers about ``subprograms`` is "is this
    # name a call rather than an array". ``subprogram_key`` qualifies an
    # internal procedure with its host -- right for selecting one to analyse,
    # and wrong here: ``frobenius_norm_companion`` is filed under
    # ``qr_algeq_solver/frobenius_norm_companion``, so the bare lookup missed
    # it and the call was scored as an array-element *read*. The translation
    # counts it as a call on the other side, so the two sides disagreed on
    # every block that calls a host-associated procedure.
    subs = {s["name"]: s for s in record["subprograms"]}
    # An explicit interface is the shape a procedure dummy calls through:
    # ``call func(x, val)`` binds its actuals the way a call to a known
    # procedure does, or ``val`` is counted read where the callee wrote it.
    for name, body in (record.get("interfaces") or {}).items():
        subs.setdefault(name, body)

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

    interfaces = record.get("interfaces") or {}
    dummy_procedures = {
        argument["name"]: interfaces[argument["interface"]]
        for argument in sub["args"]
        if argument.get("procedure") and argument.get("interface") in interfaces
    }

    return Scope(
        subprograms=subs,
        dummy_procedures=dummy_procedures,
        generics=dict(record["generics"]),
        ranks=ranks,
        chars=frozenset(chars),
        semantics=for_subprogram(record, sub_name, companions=companions),
        externals=dict(externals or {}),
    )


def _selector_dims(selector: Any, scope: Scope) -> Any:
    """The dims of ``root%component`` -- allocated where the frontend saw an
    ALLOCATE, declared otherwise -- for an associate alias to inherit."""
    semantics = scope.semantics
    if semantics is None or not isinstance(selector, f03.Data_Ref):
        return None
    if len(selector.children) != 2:
        return None
    root, component = selector.children
    if isinstance(component, f03.Part_Ref):
        component = component.children[0]
    declared = semantics.declaration(str(root).lower()) or {}
    match = re.match(r"UNKNOWN\(TYPE\((\w+)\)\)", str(declared.get("dtype", "")), re.I)
    if match is None:
        return None
    record = semantics.types.get(match.group(1).lower(), {}).get(str(component).lower())
    if not record:
        return None
    return record.get("allocated_dims") or record.get("dims")


def lower_bound_reads(name: str, scope: Scope) -> set[str]:
    """The names a subscript of ``name`` reads through its declared lower
    bounds. ``rho(bounds%begp:bounds%endp, ...)`` then ``rho(p, ic)``: the
    element's address is ``p - bounds%begp``, which the translation spells
    out and the source computes silently. Both sides read ``bounds``."""
    semantics = scope.semantics
    if semantics is None:
        return set()
    declared = semantics.declaration(name) or {}
    dims = scope.alias_dims.get(name) or declared.get("dims") or ()
    found: set[str] = set()
    for dim in dims:
        lower = str(dim.get("lb") or "").strip()
        if not lower or re.fullmatch(r"-?\d+", lower):
            continue
        for token in re.findall(r"[A-Za-z_]\w*", lower.split("%", 1)[0]):
            name_ = token.lower()
            if name_ in INTRINSICS or name_ in STATE_QUERY or name_ in TRANSFORMATIONAL:
                continue
            # A local, a dummy, or a use-imported parameter (``nlevsno``):
            # the translation spells each of them in the shift.
            found.add(name_)
    return found


def expr_reads(node: Any, scope: Scope) -> set[str]:
    """Every symbol an expression reads.

    A name that is a declared local or dummy is a read even when it collides
    with an intrinsic -- Fortran lets a variable named ``sum`` shadow the
    function, and treating that as a call loses a real dataflow edge.
    """
    if isinstance(node, type):
        # fparser hangs the *class* ``Int_Literal_Constant`` under a
        # ``Data_Edit_Desc`` (``I4.4`` in a FORMAT statement); it is not a
        # node, and iterating its ``children`` property raised a TypeError
        # that stopped the read/write analysis of every module with such a
        # format (ELM's histFileMod).
        return set()

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
            if fname in KIND_ARG_FNS:
                items = _without_kind_argument(fname, items)
            for item in items:
                reads |= expr_reads(item, scope)
        if scope.ranks.get(fname, 0) > 0 or fname in scope.alias_dims:
            # A declared array shadows an intrinsic name -- the same rule the
            # bare-Name branch applies. zm_conv declares `gamma(pcols,pver)`,
            # and reading `gamma(i,k)` is dataflow, not a call to GAMMA. An
            # associate alias of an array component is an array here too.
            reads.add(fname)
            reads |= lower_bound_reads(fname, scope)
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
            root = str(target.children[0]).lower()
            writes.add(root)
            reads.update(lower_bound_reads(root, scope))
            for child in target.children[1:]:
                reads.update(expr_reads(child, scope))

    def call(stmt: Any) -> None:
        name = str(stmt.children[0]).lower()
        items = list(stmt.children[1].children) if stmt.children[1] is not None else []
        if name in scope.generics:
            name = _resolve_generic(name, items, scope) or name

        callee = scope.subprograms.get(name)
        if callee is None and name in scope.dummy_procedures:
            # Calling through a dummy procedure reads it: which code runs is
            # decided by the value the caller passed, and the translation
            # spells that read the same way -- the argument's own name at
            # callee position.
            callee = scope.dummy_procedures[name]
            reads.add(name)
        actuals = _bind_actuals(callee, items) if callee is not None else items
        # A call through a dummy reads the dummy: the callable is data this
        # subprogram was passed, and the translation spells it as a name.
        dummies = (
            {a["name"].lower() for a in scope.semantics.subprogram["args"]}
            if (scope.semantics is not None)
            else set()
        )
        if name in dummies:
            reads.add(name)

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
            # a missed read only ever costs a spurious mismatch. A buffer OUT
            # (#36) is passed by the caller *and* returned -- the emitter
            # renders the actual as a call argument and as an unpack target
            # -- so this side records the read as well as the write (#38).
            if formal["intent"] in ("IN", "INOUT", "UNKNOWN") or formal.get("buffer"):
                reads.update(expr_reads(actual, scope))
            if formal["intent"] in ("OUT", "INOUT"):
                _write_actual(actual)

    def visit(stmt: Any) -> None:
        if isinstance(stmt, f08.Block_Construct):
            # A named block is a scope wrapper around ordinary statements.
            # Without this case it fell into the conservative fallback, which
            # counted every name (construct labels included) as a read and
            # dropped every write -- silently under-reporting the exact thing
            # this module's invariant forbids.
            # fparser hangs the block's statements directly off the
            # construct (an Execution_Part wrapper, if one ever appears, is
            # unwrapped). Each statement is visited AS a statement: descending
            # one level further visited its Name children instead, which the
            # fallback counted as reads and never as writes.
            for child in stmt.children:
                if isinstance(child, (f08.Block_Stmt, f08.End_Block_Stmt)):
                    continue
                if isinstance(child, f03.Specification_Part):
                    continue
                if isinstance(child, f03.Execution_Part):
                    for inner in child.children:
                        visit(inner)
                    continue
                visit(child)
            return
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

        elif isinstance(stmt, f03.Associate_Construct):
            # ``associate (a => x%comp, ...)``: each association binds a name
            # to a selector, which is how the emitter spells it too -- an
            # assignment of the alias from the selector's root -- so the alias
            # is a write and the selector's roots and subscripts are reads.
            # The body then uses the aliases as ordinary variables. Falling
            # through to the fallback below instead reported every name in
            # the whole construct as a read and none as a write, which failed
            # every block of a model whose physics is written this way.
            for child in stmt.children:
                if isinstance(child, f03.Associate_Stmt):
                    for association in walk(child, f03.Association):
                        alias, _, selector = association.children
                        writes.add(str(alias).lower())
                        reads.update(expr_reads(selector, scope))
                        dims = _selector_dims(selector, scope)
                        if dims:
                            scope.alias_dims[str(alias).lower()] = dims
                elif not isinstance(child, f03.End_Associate_Stmt):
                    visit(child)

        else:
            # Conservative fallback. Over-reporting a read costs a block a
            # review; under-reporting a write costs a wrong answer nobody sees.
            reads.update(expr_reads(stmt, scope))

    visit(node)
    return reads, writes


def _data_rwset(statement: Any, scope: Scope) -> tuple[set[str], set[str]]:
    """``(reads, writes)`` for one DATA statement.

    Not the general statement rule: a DATA object list is written, not read,
    and the general rule -- which sees names in expression positions --
    reports the objects as reads and nothing as written, the exact inverse of
    what the emitted assignments do. Subscripts on an object are reads, as
    they are on any assignment target, and so is everything in the value list.
    """
    reads: set[str] = set()
    writes: set[str] = set()
    for group in statement.children or ():
        children = getattr(group, "children", None)
        if not children or len(children) < 2:
            continue
        objects, values = children[0], children[1]
        for target in walk(objects, (f03.Name, f03.Part_Ref, f03.Data_Ref)):
            if isinstance(target, f03.Name):
                writes.add(str(target).lower())
                continue
            writes.add(str(target.children[0]).lower())
            for child in target.children[1:]:
                reads.update(expr_reads(child, scope))
            break
        reads.update(expr_reads(values, scope))
    return reads - writes, writes


def block_rwsets(sub: Any, scope: Scope) -> list[dict[str, Any]]:
    """``[{id, reads, writes}, ...]`` for every block of one subprogram.

    DATA statements first, under their own ``D``-numbered ids. They sit in
    the specification part, so ``chunk_subprogram`` -- which walks the
    execution part -- never sees them, and a translation that emits them as
    assignments had no read/write set to be checked against. That was not
    visible while every DATA statement in reach was being refused for want of
    a hoisted literal.
    """
    out = []
    specification = next((c for c in sub.children if isinstance(c, f03.Specification_Part)), None)
    if specification is not None:
        for at, statement in enumerate(walk(specification, f03.Data_Stmt), start=1):
            reads, writes = _data_rwset(statement, scope)
            out.append({"id": f"D{at:03d}", "reads": sorted(reads), "writes": sorted(writes)})
    for block_id, node, _span in chunk_subprogram(sub):
        reads, writes = rwset(node, scope)
        out.append({"id": block_id, "reads": sorted(reads), "writes": sorted(writes)})
    return out
