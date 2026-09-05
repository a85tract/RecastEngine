"""Flattening a derived-type interface so both sides of a gate can call it.

A physics routine that takes ``type(model_type), intent(inout) :: inst``
-- one object with a few hundred pointer components -- and reads module
state of the same shape cannot be handed to f2py's flat wrapper, and the
differential gate cannot sample an object. Every such unit would stop at
the oracle for that reason alone.

The answer is an *adapter* on each side, generated from one analysis:

* which components of which object the subprogram touches, and whether it
  writes them -- from the source's own read/write analysis, with the
  ``associate`` aliases resolved back to ``object%component``, transitively
  through the calls the object is passed down;
* each component's type, rank and allocation bounds -- from the type's
  definition and the ``allocate (this%…)`` statements in its module;
* the plain module variables the body reads that nothing in the tree
  initializes, which a run sets and a recording carries.

This module produces the *plan*: a flat signature in the engine's own
interface vocabulary, serialisable into ``Facts.extra`` so the oracle, the
transform and the recorder read one analysis rather than each redoing it.
The Fortran adapter is the oracle's (``recast.oracle.flat``), the Python
adapter the transform's (``recast.transform.numpy.flat``).

What is *not* claimed: the adapter allocates every touched component over
the one extent the driver chooses (``np_``, the patch count) and the
model's fixed layer counts, with values the gate generated. That is a
kernel-in-isolation test, not a column run.

Which names mean "begin/end of the driver's range" and which modules hold
constants are conventions of the tree; they arrive in ``FlatConventions``
from whoever knows the tree, and the defaults are only the common CESM-family
spelling.
"""

from __future__ import annotations

import dataclasses
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from recast.fortran.tree import (
    MODULE_DEFINITION,
    integer_parameters,
    module_sources,
    named_extents,
    sources,
)

__all__ = [
    "Component",
    "FlatConventions",
    "FlatObject",
    "FlatPlan",
    "StateVar",
    "plans_for",
    "plans_from_facts",
    "signature",
]

DERIVED = re.compile(r"UNKNOWN\(TYPE\((\w+)\)\)", re.IGNORECASE)
ALLOCATE_THIS = re.compile(r"allocate\s*\(\s*this\s*%\s*(\w+)\s*\(([^()]*)\)", re.I)
ALLOCATE_STMT = re.compile(r"^\s*allocate\s*\((.*)\)\s*(?:!.*)?$", re.I | re.M)
ALLOCATION = re.compile(r"(?<![%\w])(\w+)\s*\(([^()]*)\)")
"""One ``name(bounds)`` inside an ALLOCATE: a module variable, not a
``this%`` component (the ``%`` look-behind) -- those are ``ALLOCATE_THIS``."""
FORTRAN_TYPES = {"float64": "real(8)", "float32": "real(4)", "int32": "integer", "bool": "logical"}


@dataclass(frozen=True)
class FlatConventions:
    """What the tree does not say about itself and the plan needs."""

    kind_assumptions: dict[str, str] = field(default_factory=dict)
    constant_modules: frozenset[str] = frozenset()
    """Modules whose parameters size things: looked up for integer extents."""
    stub_modules: frozenset[str] = frozenset()
    """Modules the frontend does not resolve as companions; their state is
    still module state the adapter may have to carry."""
    patch_count: str = "np_"
    """The one extent the tree leaves to the driver. Every
    ``begin:end`` bound over the driver's range becomes ``1:<patch_count>``."""
    bounds_pattern: str = r"^(beg|end)[pcgl]$"
    """Names that spell the driver's range: ``begp``/``endp`` and the
    column/gridcell/landunit forms, in the CESM family."""
    counter_prefix: str = "num_"
    """An assumed-shape dummy ``x(:)`` paired with a scalar ``num_x`` takes
    that as its extent; anything else is sized over the patch count."""


@dataclass
class Component:
    name: str
    dtype: str
    bounds: list[tuple[str, str]]
    """Per axis ``(lb, ub)`` as rendered for the adapter's ``allocate``."""
    extents: list[str]
    """Per axis extent as the flat argument declares it (``np_``, ``100``)."""
    written: bool = False
    pointer: bool = False
    owner: str = ""

    @property
    def flat(self) -> str:
        return f"{self.owner}__{self.name}"


@dataclass
class FlatObject:
    name: str
    """The dummy's name, or the module variable's."""
    type_name: str
    kind: str
    """``dummy`` or ``state``."""
    module: str | None = None
    """For state: the module that declares it."""
    type_module: str | None = None
    """The module that defines the type, for the adapters' ``use`` lines."""
    components: list[Component] = field(default_factory=list)


@dataclass
class StateVar:
    """A plain module variable the subprogram reads that nothing initializes
    -- a table read from a file, a namelist value. The run's value is what
    the reference computed with, so it is recorded and passed like a
    component, under ``<module>__<name>``."""

    module: str
    name: str
    dtype: str
    extents: list[str]
    written: bool = False
    bounds: list[tuple[str, str]] = field(default_factory=list)
    """Per axis ``(lb, ub)`` for an *allocatable* module array, from the
    module's own ALLOCATE: the adapter allocates it so before assigning,
    because ``x = flat`` alone would give it a lower bound of one. Empty for
    an array whose declaration carries its shape."""

    @property
    def flat(self) -> str:
        return f"{self.module}__{self.name}"


@dataclass
class FlatPlan:
    subprogram: dict[str, Any]
    objects: list[FlatObject]
    unsupported: list[str] = field(default_factory=list)
    states: list[StateVar] = field(default_factory=list)
    dim_constants: dict[str, int] = field(default_factory=dict)
    """Named extents of the original dummies that are tree constants
    (``a(nrk,nrk)``), so the flat signature can spell them as numbers."""
    patch_count: str = "np_"
    counter_prefix: str = "num_"

    @property
    def name(self) -> str:
        return f"{self.subprogram['name']}_flat"

    @property
    def usable(self) -> bool:
        return not self.unsupported and bool(self.objects)

    @property
    def gated(self) -> bool:
        """A public subroutine: what the oracle can wrap and the gate compare."""
        return bool(self.subprogram.get("public", True)) and self.subprogram["kind"] == "subroutine"

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FlatPlan:
        objects = [
            FlatObject(
                **{k: v for k, v in o.items() if k != "components"},
                components=[
                    Component(**{**c, "bounds": [tuple(b) for b in c["bounds"]]})
                    for c in o["components"]
                ],
            )
            for o in data["objects"]
        ]
        states = [
            StateVar(**{**s, "bounds": [tuple(b) for b in s.get("bounds", [])]})
            for s in data.get("states", [])
        ]
        return cls(
            subprogram=data["subprogram"],
            objects=objects,
            unsupported=list(data.get("unsupported", [])),
            states=states,
            dim_constants=dict(data.get("dim_constants", {})),
            patch_count=data.get("patch_count", "np_"),
            counter_prefix=data.get("counter_prefix", "num_"),
        )

    @property
    def flat_args(self) -> list[dict[str, Any]]:
        """The adapter's argument list, in the engine's interface vocabulary."""
        args: list[dict[str, Any]] = []
        names = {a["name"].lower() for a in self.subprogram["args"]}
        for argument in self.subprogram["args"]:
            if str(argument["dtype"]) == "PROCEDURE":
                continue  # never flat: a callback is specialized
            if str(argument["dtype"]) == "str" and argument.get("dims"):
                continue  # a character array is not spelled; a scalar is a value

            if DERIVED.match(str(argument["dtype"])) or argument.get("optional"):
                # The object is what the adapter exists to replace; an optional
                # is left absent on both sides, the way the engine's own
                # wrapper leaves it.
                continue
            entry = dict(argument)
            if entry.get("dims") and entry.get("intent") == "OUT":
                # Caller-buffer convention: the storage comes in and goes
                # back out, so the flat signature says INOUT -- and for an
                # assumed-shape dummy that is the only way the shape is known.
                entry["intent"] = "INOUT"
            if entry.get("dims"):
                sized = []
                for dim in entry["dims"]:
                    if dim.get("ub"):
                        ub = str(dim["ub"]).strip().lower()
                        sized.append({**dim, "ub": str(self.dim_constants.get(ub, dim["ub"]))})
                    else:
                        counter = f"{self.counter_prefix}{entry['name'].lower()}"
                        extent = counter if counter in names else self.patch_count
                        sized.append({"lb": "1", "ub": extent})
                entry["dims"] = sized
            args.append(entry)
        args.append(
            {
                "name": self.patch_count,
                "dtype": "int32",
                "intent": "IN",
                "optional": False,
                "dims": None,
            }
        )
        for obj in self.objects:
            for comp in obj.components:
                args.append(
                    {
                        "name": comp.flat,
                        "dtype": comp.dtype,
                        "intent": "INOUT" if comp.written else "IN",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": extent} for extent in comp.extents],
                    }
                )
        for state in self.states:
            args.append(
                {
                    "name": state.flat,
                    "dtype": state.dtype,
                    "intent": "INOUT" if state.written else "IN",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": extent} for extent in state.extents] or None,
                }
            )
        return args


def signature(plan: FlatPlan) -> dict[str, Any]:
    """The adapter's entry for a ``_SIGNATURES`` table or an interface record."""
    return {
        "kind": "subroutine",
        "args": [
            {
                "name": a["name"],
                "dtype": a["dtype"],
                "intent": a["intent"],
                "optional": bool(a.get("optional")),
                "dims": a.get("dims"),
            }
            for a in plan.flat_args
        ],
        "result": None,
        "result_dtype": None,
    }


def plans_from_facts(facts: Any, *, gated: bool = True) -> list[FlatPlan]:
    """The plans the frontend stored on ``Facts.extra``, usable ones only.

    ``gated`` (the default) keeps the public subroutines: the ones an
    adapter can be compiled and compared for. A lowering that follows calls
    inward wants every plan, private subprograms and functions included."""
    stored = (facts.extra or {}).get("flat_plans") or []
    plans = [p for p in (FlatPlan.from_dict(d) for d in stored) if p.usable]
    if gated:
        plans = [p for p in plans if p.gated]
    return plans


# --- the tree ----------------------------------------------------------------


def type_file(type_name: str, root: Path) -> Path | None:
    pattern = re.compile(
        rf"^\s*type\s*(?:,\s*[\w()=]+\s*)*(?:::)?\s*{re.escape(type_name)}\s*$", re.I | re.M
    )
    for path in sources(root):
        if pattern.search(path.read_text(errors="replace")):
            return path
    return None


def _module_of(path: Path) -> str | None:
    match = MODULE_DEFINITION.search(path.read_text(errors="replace"))
    return match.group(1) if match else None


def _state_declaration(name: str, root: Path) -> tuple[str, str] | None:
    """``(type_name, module)`` for a module variable of derived type."""
    pattern = re.compile(
        rf"^\s*type\s*\(\s*(\w+)\s*\)[^:!]*::\s*{re.escape(name)}\s*(?:!.*)?$", re.I | re.M
    )
    for path in sources(root):
        text = path.read_text(errors="replace")
        for match in pattern.finditer(text):
            if "intent" in match.group(0).lower():
                continue  # a dummy of that type, not the module's variable
            module = MODULE_DEFINITION.search(text)
            return match.group(1).lower(), (module.group(1).lower() if module else "")
    return None


def _allocation_bounds(path: Path) -> dict[str, list[str]]:
    """``component -> [axis bound text, ...]`` from ``allocate (this%c (…))``."""
    out: dict[str, list[str]] = {}
    for match in ALLOCATE_THIS.finditer(path.read_text(errors="replace")):
        out.setdefault(match.group(1).lower(), [b.strip() for b in match.group(2).split(",")])
    return out


_MODULE_ALLOCATIONS: dict[Path, dict[str, list[str]]] = {}


def _module_allocation_bounds(path: Path) -> dict[str, list[str]]:
    """``variable -> [axis bound text, ...]`` from every ``allocate (x (…))``
    of a plain name in the module: a module allocatable's shape is what its
    own init routine gave it (``allocate (vcmax_np1 (0:mxpft))`` in
    ``pftconrd``), which its declaration ``(:)`` does not say."""
    if path not in _MODULE_ALLOCATIONS:
        out: dict[str, list[str]] = {}
        for statement in ALLOCATE_STMT.finditer(path.read_text(errors="replace")):
            for match in ALLOCATION.finditer(statement.group(1)):
                out.setdefault(
                    match.group(1).lower(), [b.strip() for b in match.group(2).split(",")]
                )
        _MODULE_ALLOCATIONS[path] = out
    return _MODULE_ALLOCATIONS[path]


def _axis(
    bound: str, constants: dict[str, int], conventions: FlatConventions
) -> tuple[tuple[str, str], str] | None:
    """One allocation axis -> ``((lb, ub), extent)`` in adapter spelling."""
    lb, ub = bound.split(":", 1) if ":" in bound else ("1", bound)
    lb, ub = lb.strip().lower(), ub.strip().lower()
    bounds = re.compile(conventions.bounds_pattern)
    patch = conventions.patch_count

    def spell(text: str) -> str | None:
        if bounds.match(text):
            return "1" if text.startswith("beg") else patch
        if re.fullmatch(r"-?\d+", text):
            return text
        if text in constants:
            return str(constants[text])
        # An arithmetic bound over constants: ``-nlevsno+1``.
        expression = re.sub(
            r"[A-Za-z_]\w*", lambda m: str(constants.get(m.group(0).lower(), m.group(0))), text
        )
        if re.fullmatch(r"[\d\s()+\-*/]+", expression):
            try:
                return str(int(eval(expression.replace("/", "//"), {"__builtins__": {}})))  # noqa: S307
            except Exception:
                return None
        # A bound over a module *variable* (set at run time): kept symbolic,
        # spelled by the variable's flat name once the plan's states are
        # known (see ``_bind_symbolic_extents``).
        if re.fullmatch(r"[A-Za-z_\d\s()+\-*/]+", text):
            return f"({text})"
        return None

    low, high = spell(lb), spell(ub)
    if low is None or high is None:
        return None
    if high == patch:
        extent = patch if low == "1" else None
    elif re.fullmatch(r"-?\d+", low) and re.fullmatch(r"-?\d+", high):
        extent = str(int(high) - int(low) + 1)
    else:
        extent = f"({high}) - ({low}) + 1"
    if extent is None:
        return None
    return (low, high), extent


# --- the analysis ------------------------------------------------------------


def _subprogram_node(source: Path, name: str) -> Any:
    from recast.fortran._parse import f03, parse, walk

    for node in walk(parse(source), (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        stmt = walk(node, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
        if str(stmt.children[1]).lower() == name.lower():
            return node
    return None


def _accesses(
    node: Any,
    record: dict[str, Any],
    name: str,
    procedures: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] | None = None,
    objects: frozenset[str] = frozenset(),
    depth: int = 0,
    visited: set[tuple[str, str]] | None = None,
    globals_out: set[str] | None = None,
    externals: dict[str, dict[str, Any]] | None = None,
    companions: tuple[dict[str, Any], ...] = (),
) -> tuple[dict[str, str], set[str], set[str]]:
    """``(alias -> "root%comp", reads, writes)`` over the subprogram's body,
    with the association statements themselves left out of the sets.

    Transitive through calls: an object passed to a procedure of this
    module or a companion is analysed as that procedure's dummy, and what
    it touches comes back spelled on the caller's name (``procedures`` maps
    a callee name to its file, its module record and its own record). A
    model passes its object down three or four levels, and an adapter that
    allocated only the caller's own components would hand the callee an
    object with attributes missing.
    """
    from recast.fortran import rwset as rw
    from recast.fortran._parse import f03, walk

    aliases: dict[str, str] = {}
    for construct in walk(node, f03.Associate_Construct):
        for association in walk(construct, f03.Association):
            alias, _, selector = association.children
            if isinstance(selector, f03.Data_Ref) and len(selector.children) == 2:
                root, comp = selector.children
                if isinstance(comp, f03.Part_Ref):
                    comp = comp.children[0]
                aliases[str(alias).lower()] = f"{str(root).lower()}%{str(comp).lower()}"
    # A pointer assignment is an alias too -- ``psn_z => photosyns_vars%
    # psnsun_z_patch`` under one branch and the shade array under the
    # other -- and a write through it is a write of every target it may
    # have. Left out, the routine's own outputs (ELM's leaf photosynthesis,
    # stomatal resistance, ci) were planned read-only and the gate never
    # compared them.
    pointed: dict[str, set[str]] = {}
    for statement in walk(node, f03.Pointer_Assignment_Stmt):
        target, _, selector = statement.children
        if not isinstance(target, f03.Name):
            continue
        if isinstance(selector, f03.Data_Ref) and len(selector.children) == 2:
            root, comp = selector.children
            if isinstance(comp, f03.Part_Ref):
                comp = comp.children[0]
            if isinstance(root, f03.Name) and isinstance(comp, f03.Name):
                pointed.setdefault(str(target).lower(), set()).add(
                    f"{str(root).lower()}%{str(comp).lower()}"
                )
    # With the companions' procedures in scope: a call into a sibling
    # module writes its intent(out) actuals, and ``call tridiag(..., t(p,:))``
    # is how a component gets written here. Without them every call scores
    # as reads of its actuals and the component is planned read-only --
    # which the NumPy adapter hides (the record's arrays are written in
    # place) and a functional lowering does not.
    scope = rw.scope_for(record, name, externals=externals or {}, companions=companions)
    reads: set[str] = set()
    writes: set[str] = set()
    execution = next((c for c in node.children if isinstance(c, f03.Execution_Part)), None)
    if execution is None:
        return aliases, reads, writes
    own: dict[str, Any] = next(
        (s for s in record.get("subprograms", ()) if s["name"].lower() == name.lower()), {}
    )
    declared_here = {
        e["name"].lower()
        for group in ("args", "locals", "local_parameters")
        for e in own.get(group, ())
    } | {str(own.get("result") or "").lower()}

    def take(stmt: Any) -> None:
        if isinstance(stmt, f03.Associate_Construct):
            for child in stmt.children:
                if not isinstance(child, (f03.Associate_Stmt, f03.End_Associate_Stmt)):
                    take(child)
            return
        r, w = rw.rwset(stmt, scope)
        reads.update(r)
        writes.update(w)

    for stmt in execution.children:
        take(stmt)
    # Plain names this body touches that it does not declare are module
    # state -- its own module's or a use-imported module's -- and the caller
    # collects them to see which the run sets outside the tree's initializers.
    if globals_out is not None:
        for plain in reads | writes:
            if "%" not in plain and plain not in declared_here and plain not in aliases:
                globals_out.add(("w:" if plain in writes else "r:") + plain)
        # Names in the local declarations' bounds (``tk(bounds%begc:bounds%endc,
        # -nlevsno+1:nlevgrnd)``): read at entry, module variables included.
        for entry in own.get("locals", ()):
            for dim in entry.get("dims") or ():
                for bound in (dim.get("lb"), dim.get("ub")):
                    for token in re.findall(r"[A-Za-z_]\w*", str(bound or "").split("%", 1)[0]):
                        if token.lower() not in declared_here:
                            globals_out.add("r:" + token.lower())
    # An OUT actual of a call: ``call fill(inst%ncan, v)`` writes the
    # component, whether the callee is this module's or a companion's. The
    # sets record the actual by its root only, so the component is named
    # here from the callee's out positions.
    for call in walk(node, f03.Call_Stmt):
        callee = str(call.children[0]).lower()
        actuals = call.children[1].children if call.children[1] is not None else []
        positions: list[int] = []
        dummies: list[str] = []
        if procedures and callee in procedures:
            args = procedures[callee][2]["args"]
            dummies = [a["name"].lower() for a in args]
            positions = [i for i, a in enumerate(args) if a.get("intent") in ("OUT", "INOUT")]
        elif externals and callee in externals:
            positions = [int(i) for i in externals[callee].get("out_positions", [])]
        for at, actual in enumerate(actuals):
            value = actual
            where = at
            if isinstance(actual, f03.Actual_Arg_Spec):
                keyword, value = actual.children
                if str(keyword).lower() in dummies:
                    where = dummies.index(str(keyword).lower())
            if where not in positions:
                continue
            if isinstance(value, f03.Data_Ref) and len(value.children) == 2:
                root, comp = value.children
                if isinstance(comp, f03.Part_Ref):
                    comp = comp.children[0]
                writes.add(f"{str(root).lower()}%{str(comp).lower()}")
            elif isinstance(value, f03.Name):
                writes.add(str(value).lower())
            elif isinstance(value, f03.Part_Ref):
                # ``tair(p,:)``: a section of an alias, or of a plain array.
                writes.add(str(value.children[0]).lower())
    # An alias written in this body is its selector's component written --
    # spelled that way here, so a caller following a call into this
    # subprogram sees ``inst%flux`` and not the alias.
    for name_ in list(reads):
        if name_ in aliases:
            reads.add(aliases[name_])
    for name_ in list(writes):
        if name_ in aliases:
            writes.add(aliases[name_])
    # Direct ``obj%comp`` references, which the sets record by root only.
    for assignment in walk(node, f03.Assignment_Stmt):
        lhs = assignment.children[0]
        if isinstance(lhs, f03.Data_Ref) and len(lhs.children) == 2:
            root, comp = lhs.children
            if isinstance(comp, f03.Part_Ref):
                comp = comp.children[0]
            writes.add(f"{str(root).lower()}%{str(comp).lower()}")
    for ref in walk(node, f03.Data_Ref):
        if len(ref.children) == 2:
            root, comp = ref.children
            if isinstance(comp, f03.Part_Ref):
                comp = comp.children[0]
            reads.add(f"{str(root).lower()}%{str(comp).lower()}")
    # Through calls. ``call sub(p, ic, inst)`` and ``f(x, inst)`` alike:
    # every actual that names one of our objects is followed into the
    # callee under the callee's dummy name, and the result is renamed back.
    if procedures and depth < 5:
        visited = visited if visited is not None else set()
        interesting = objects | set(aliases)
        # ``dummy = hybrid(...)`` parses as any of three node kinds depending
        # on what fparser could tell about the name.
        for call in [
            *walk(node, f03.Call_Stmt),
            *walk(node, f03.Part_Ref),
            *walk(node, f03.Function_Reference),
            *walk(node, f03.Structure_Constructor),
        ]:
            callee = str(call.children[0]).lower()
            if callee not in procedures:
                continue
            path, module_record, callee_record = procedures[callee]
            actuals = call.children[1].children if call.children[1] is not None else []
            mapping: dict[str, str] = {}
            for at, actual in enumerate(actuals):
                dummy: str | None
                if isinstance(actual, f03.Actual_Arg_Spec):
                    keyword, value = actual.children
                    dummy = str(keyword).lower()
                else:
                    value = actual
                    dummy = (
                        callee_record["args"][at]["name"].lower()
                        if at < len(callee_record["args"])
                        else None
                    )
                if dummy is None or not isinstance(value, f03.Name):
                    continue
                spelled = str(value).lower()
                if spelled in aliases:
                    spelled = aliases[spelled].split("%", 1)[0]
                if spelled in objects or spelled in interesting:
                    mapping[dummy] = spelled
            # A procedure passed as an actual (``hybrid(..., func, ...)``) is
            # called back with the object; its own derived-type dummies are
            # mapped by name, which is how the model spells them.
            for actual in actuals:
                value = actual.children[1] if isinstance(actual, f03.Actual_Arg_Spec) else actual
                if not isinstance(value, f03.Name):
                    continue
                passed = str(value).lower()
                if passed in procedures and passed != callee:
                    _, p_module, p_record = procedures[passed]
                    p_map = {
                        a["name"].lower(): a["name"].lower()
                        for a in p_record["args"]
                        if a["name"].lower() in objects or a["name"].lower() in interesting
                    }
                    p_key = (passed, ",".join(sorted(p_map.values())))
                    if p_map and p_key not in visited:
                        visited.add(p_key)
                        p_node = _subprogram_node(procedures[passed][0], passed)
                        if p_node is not None:
                            _, p_reads, p_writes = _accesses(
                                p_node,
                                p_module,
                                passed,
                                procedures,
                                frozenset(p_map) | objects,
                                depth + 1,
                                visited,
                                globals_out,
                                externals,
                                companions,
                            )
                            p_dummies = {a["name"].lower() for a in p_record["args"]}
                            for inner, target in ((p_reads, reads), (p_writes, writes)):
                                for item in inner:
                                    if "%" not in item:
                                        continue
                                    root, comp = item.split("%", 1)
                                    if root in p_map:
                                        target.add(f"{p_map[root]}%{comp}")
                                    elif root not in p_dummies:
                                        target.add(item)
            key = (callee, ",".join(sorted(mapping.values())))
            # A callee handed no object is still followed for the module
            # state it reads (a lookup table and its grids); once.
            if key in visited or (not mapping and globals_out is None):
                continue
            visited.add(key)
            callee_node = _subprogram_node(path, callee)
            if callee_node is None:
                continue
            _, inner_reads, inner_writes = _accesses(
                callee_node,
                module_record,
                callee,
                procedures,
                frozenset(mapping) | objects,
                depth + 1,
                visited,
                globals_out,
                externals,
                companions,
            )
            callee_dummies = {a["name"].lower() for a in callee_record["args"]}
            for inner, target in ((inner_reads, reads), (inner_writes, writes)):
                for item in inner:
                    if "%" not in item:
                        continue
                    root, comp = item.split("%", 1)
                    if root in mapping:
                        target.add(f"{mapping[root]}%{comp}")
                    elif root not in callee_dummies:
                        # Module state (``patch%itype``): spelled the same
                        # everywhere, and the callee's reads are ours.
                        target.add(item)
    for alias, targets in pointed.items():
        if alias in reads:
            reads |= targets
        if alias in writes:
            writes |= targets
    return aliases, reads, writes


def _companion_scope(facts: Any) -> tuple[dict[str, dict[str, Any]], tuple[dict[str, Any], ...]]:
    """The companions' procedures the way the frontend puts them in scope:
    which names are calls and which positions they write, use-renames
    looked up under the local spelling."""
    from recast.fortran.interface import companion_externals

    externals: dict[str, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for companion in facts.provenance.get("companions") or []:
        record = companion.get("record")
        if not record:
            continue
        records.append(record)
        table = companion_externals(record)
        for local, remote in (companion.get("renames") or {}).items():
            if remote in table:
                table[local] = table[remote]
        for name, entry in table.items():
            externals.setdefault(name, entry)
    return externals, tuple(records)


def _procedure_index(
    facts: Any, root: Path, source: Path, kinds: dict[str, str] | None = None
) -> dict[str, tuple[Path, dict[str, Any], dict[str, Any]]]:
    """Callee name -> (file, module record, subprogram record) over this
    unit, its companions, and the modules those reach through ``use`` --
    the closure a call can actually reach. The companions themselves stay
    one level deep (a companion's own companions are its translation's
    business); this index is for following *calls* inward, and a companion
    procedure calling into a module the unit never uses (``GetObu`` into
    ``hybrid``) is still this unit's dataflow."""
    index: dict[str, tuple[Path, dict[str, Any], dict[str, Any]]] = {}
    for sub in facts.interface.get("subprograms", ()):
        index[sub["name"].lower()] = (source, facts.interface, sub)
    seen_modules = {str(facts.interface.get("module", "")).lower()}
    pending: list[tuple[Path, dict[str, Any]]] = []
    for companion in facts.provenance.get("companions") or []:
        record = companion.get("record") or {}
        path = (root / str(companion.get("source", ""))).resolve()
        seen_modules.add(str(record.get("module", "")).lower())
        pending.append((path, record))
    while pending:
        path, record = pending.pop(0)
        for sub in record.get("subprograms", ()):
            index.setdefault(sub["name"].lower(), (path, record, sub))
        used = set()
        for statement in record.get("use_statements") or ():
            match = re.match(r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", statement.strip(), re.I)
            if match and match.group(1).lower() not in seen_modules:
                used.add(match.group(1).lower())
        seen_modules |= used
        for further in module_sources(root, frozenset(used)):
            further_record = _module_record(further, kinds or {})
            if further_record:
                pending.append((further.resolve(), further_record))
    return index


def plans_for(facts: Any, root: Path, conventions: FlatConventions | None = None) -> list[FlatPlan]:
    """One plan per subprogram that takes a derived-type dummy -- private
    ones and functions included, because a lowering that follows calls
    inward needs their plans; ``FlatPlan.gated`` says which the oracle can
    wrap."""
    from recast.fortran import interface

    conventions = conventions or FlatConventions()
    kinds = conventions.kind_assumptions
    root = root.resolve()
    source = (root / facts.provenance["source"]).resolve()
    record = facts.interface
    plans: list[FlatPlan] = []
    type_records: dict[str, tuple[dict[str, Any], dict[str, list[str]]]] = {}
    type_files: dict[str, Path] = {}
    externals, companions = _companion_scope(facts)

    def type_info(type_name: str) -> tuple[dict[str, Any], dict[str, list[str]]] | None:
        if type_name not in type_records:
            path = type_file(type_name, root)
            if path is None:
                return None
            rec = interface.extract(path, kind_assumptions=kinds)
            types = rec.get("types", {})
            comps = types.get(type_name) or types.get(type_name.lower())
            if comps is None:
                return None
            type_records[type_name] = (comps, _allocation_bounds(path))
            type_files[type_name] = path
        return type_records[type_name]

    for sub in record["subprograms"]:
        if sub["kind"] not in ("subroutine", "function"):
            continue
        dummies = {
            a["name"].lower(): m.group(1).lower()
            for a in sub["args"]
            if (m := DERIVED.match(str(a["dtype"])))
        }
        if not dummies:
            continue
        if any(str(a["dtype"]) == "PROCEDURE" for a in sub["args"]):
            continue
        plan = FlatPlan(
            subprogram=sub,
            objects=[],
            patch_count=conventions.patch_count,
            counter_prefix=conventions.counter_prefix,
        )
        node = _subprogram_node(source, sub["name"])
        if node is None:
            plan.unsupported.append("subprogram not found in source")
            plans.append(plan)
            continue
        procedures = _procedure_index(facts, root, source, kinds)
        globals_touched: set[str] = set()
        aliases, reads, writes = _accesses(
            node,
            record,
            sub["name"],
            procedures,
            frozenset(dummies),
            0,
            set(),
            globals_touched,
            externals,
            companions,
        )
        touched: dict[str, dict[str, bool]] = {}  # root -> comp -> written
        for name in reads | writes:
            ref = aliases.get(name, name if "%" in name else None)
            if ref is None:
                continue
            obj, member = ref.split("%", 1)
            touched.setdefault(obj, {}).setdefault(member, False)
            if name in writes:
                touched[obj][member] = True
        for obj in sorted(touched):
            if obj in dummies:
                flat = FlatObject(name=obj, type_name=dummies[obj], kind="dummy")
            else:
                declared = _state_declaration(obj, root)
                if declared is None:
                    plan.unsupported.append(f"{obj}: not a dummy and not module state")
                    continue
                flat = FlatObject(name=obj, type_name=declared[0], kind="state", module=declared[1])
            info = type_info(flat.type_name)
            if info is None:
                plan.unsupported.append(f"{obj}: type {flat.type_name} not found in tree")
                continue
            flat.type_module = _module_of(type_files[flat.type_name])
            comps, bounds = info
            names_needed: list[str] = []
            for member, member_axes in bounds.items():
                if member in touched[obj]:
                    names_needed += re.findall(r"[A-Za-z_]\w*", " ".join(member_axes))
            constants = integer_parameters(
                sorted({n.lower() for n in names_needed}),
                root,
                conventions.constant_modules,
                (type_files[flat.type_name],),
                kinds,
            )
            for member, written in sorted(touched[obj].items()):
                spec = comps.get(member)
                if spec is None:
                    plan.unsupported.append(f"{obj}%{member}: no such component")
                    continue
                found_axes = bounds.get(member)
                if found_axes is None and spec.get("dims"):
                    plan.unsupported.append(f"{obj}%{member}: no allocate statement found")
                    continue
                axes = found_axes or []
                resolved = [_axis(a, constants, conventions) for a in axes]
                if any(r is None for r in resolved):
                    plan.unsupported.append(f"{obj}%{member}: bounds {axes} not resolvable")
                    continue
                flat.components.append(
                    Component(
                        name=member,
                        dtype=spec["dtype"],
                        bounds=[r[0] for r in resolved if r],
                        extents=[r[1] for r in resolved if r],
                        written=written,
                        pointer=bool(spec.get("pointer")),
                        owner=obj,
                    )
                )
            plan.objects.append(flat)
        for obj in dummies:
            if obj not in touched:
                path = type_file(dummies[obj], root)
                plan.objects.append(
                    FlatObject(
                        name=obj,
                        type_name=dummies[obj],
                        kind="dummy",
                        type_module=_module_of(path) if path else None,
                    )
                )
        # Names a symbolic extent uses are module variables the run set:
        # they are inputs of the adapter like any other state.
        for flat_obj in plan.objects:
            for comp in flat_obj.components:
                for text in [*comp.extents, *(b for pair in comp.bounds for b in pair)]:
                    for token in re.findall(r"[A-Za-z_]\w*", text):
                        if token.lower() != conventions.patch_count:
                            globals_touched.add("r:" + token.lower())
        plan.states = _state_vars(globals_touched, facts, root, plan, conventions)
        _bind_symbolic_extents(plan)
        plan.dim_constants = integer_parameters(
            named_extents([sub]), root, conventions.constant_modules, (source,), kinds
        )
        for argument in plan.flat_args:
            if str(argument["dtype"]) not in FORTRAN_TYPES and str(argument["dtype"]) != "str":
                plan.unsupported.append(f"{argument['name']}: {argument['dtype']} is not flat")
        plans.append(plan)
    return plans


_STATE_RECORDS: dict[Path, dict[str, Any]] = {}


def _module_record(path: Path, kinds: dict[str, str]) -> dict[str, Any] | None:
    from recast.fortran import interface

    if path not in _STATE_RECORDS:
        try:
            _STATE_RECORDS[path] = interface.extract(path, kind_assumptions=kinds)
        except Exception:  # an unparsable sibling has no state to offer
            _STATE_RECORDS[path] = {}
    return _STATE_RECORDS[path] or None


def _state_vars(
    touched: set[str], facts: Any, root: Path, plan: FlatPlan, conventions: FlatConventions
) -> list[StateVar]:
    """The module variables among the plain names the subprogram (and its
    callees) touch, resolved against the modules the unit and its
    companions use."""
    wanted = {t.split(":", 1)[1] for t in touched}
    written = {t.split(":", 1)[1] for t in touched if t.startswith("w:")}
    if not wanted:
        return []
    modules: set[str] = set(conventions.constant_modules | conventions.stub_modules)
    statements = list(facts.interface.get("use_statements") or [])
    for companion in facts.provenance.get("companions") or []:
        modules.add(str(companion.get("module", "")).lower())
        statements.extend((companion.get("record") or {}).get("use_statements") or [])
    for statement in statements:
        match = re.match(r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", statement.strip(), re.I)
        if match:
            modules.add(match.group(1).lower())
    # Transitively: a companion's callee module carries its own ``use``d
    # state (``ops_mod`` uses ``state2_mod``); the closure a call can reach
    # is the closure whose state the run may have set.
    frontier = set(modules)
    while frontier:
        grown: set[str] = set()
        for path in module_sources(root, frozenset(frontier)):
            record = _module_record(path, conventions.kind_assumptions)
            for statement in (record or {}).get("use_statements") or ():
                match = re.match(r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(\w+)", statement.strip(), re.I)
                if match and match.group(1).lower() not in modules:
                    grown.add(match.group(1).lower())
        modules |= grown
        frontier = grown
    own_module = str(facts.interface.get("module", "")).lower()
    modules.discard(own_module)
    found: list[StateVar] = []
    seen: set[str] = set()
    symbolic: set[str] = set()
    for path in module_sources(root, frozenset(modules)):
        record = _module_record(path, conventions.kind_assumptions)
        if not record:
            continue
        module = str(record.get("module", "")).lower()
        for entry in record.get("module_state", ()):
            name = str(entry["name"]).lower()
            # Every module *variable* the run may have set -- initialized in
            # the tree or not -- is the run's to say; parameters are not here.
            if name not in wanted or name in seen:
                continue
            if (
                DERIVED.match(str(entry.get("dtype")))
                or str(entry.get("dtype")) not in FORTRAN_TYPES
            ):
                continue
            dims = entry.get("dims") or []
            allocated: list[tuple[str, str]] | None = None
            if any(d.get("ub") is None for d in dims):
                # Deferred shape: the ALLOCATE in the module is the shape.
                axes = _module_allocation_bounds(path).get(name)
                if axes and len(axes) == len(dims):
                    allocated = [
                        (a.split(":", 1)[0].strip(), a.split(":", 1)[1].strip())
                        if ":" in a
                        else ("1", a.strip())
                        for a in axes
                    ]
                    dims = [{"lb": lb, "ub": ub} for lb, ub in allocated]
            names = [
                t.lower()
                for d in dims
                for t in re.findall(r"[A-Za-z_]\w*", f"{d.get('lb') or ''} {d.get('ub') or ''}")
            ]
            constants = integer_parameters(
                sorted(set(names)),
                root,
                conventions.constant_modules,
                (path,),
                conventions.kind_assumptions,
            )
            extents: list[str] = []
            bounds: list[tuple[str, str]] = []
            ok = True
            for d in dims:
                lb = str(d.get("lb") or "1").strip().lower()
                ub = str(d.get("ub") or "").strip().lower()
                low = constants.get(lb, int(lb) if re.fullmatch(r"-?\d+", lb) else None)
                high = constants.get(ub, int(ub) if re.fullmatch(r"-?\d+", ub) else None)
                if low is not None and high is not None:
                    extents.append(str(high - low + 1))
                    bounds.append((str(low), str(high)))
                elif allocated is not None and ub:
                    # A run-time extent: spelled by the module variables that
                    # size it, which are the run's state like any other and
                    # are bound to their flat names after the scan.
                    spelled_lb = str(low) if low is not None else lb
                    spelled_ub = str(high) if high is not None else ub
                    extents.append(f"(({spelled_ub}) - ({spelled_lb}) + 1)")
                    bounds.append((spelled_lb, spelled_ub))
                    symbolic.update(
                        t.lower() for t in re.findall(r"[A-Za-z_]\w*", f"{spelled_lb} {spelled_ub}")
                    )
                else:
                    ok = False
                    break
            if not ok:
                plan.unsupported.append(f"{module}%{name}: extent {dims} not resolvable")
                continue
            seen.add(name)
            found.append(
                StateVar(
                    module=module,
                    name=name,
                    dtype=str(entry["dtype"]),
                    extents=extents,
                    written=name in written,
                    bounds=bounds if allocated is not None else [],
                )
            )
    # The names a run-time extent uses are module variables the run set:
    # inputs of the adapter like any other state, found by one more scan.
    needed = symbolic - {v.name for v in found} - wanted
    if needed:
        return _state_vars(
            touched | {f"r:{n}" for n in needed}, facts, root, plan, conventions
        )
    return sorted(found, key=lambda v: (v.module, v.name))


def _bind_symbolic_extents(plan: FlatPlan) -> None:
    """Spell every module variable in a component's bounds by its flat
    state name, which is a scalar argument of the adapter; a name no state
    answers for makes the component unsupported."""
    by_name = {state.name: state.flat for state in plan.states if not state.extents}

    def bind(text: str) -> str | None:
        missing: list[str] = []

        def swap(match: re.Match[str]) -> str:
            token = match.group(0)
            lowered = token.lower()
            if lowered == plan.patch_count or re.fullmatch(r"\d+", token):
                return token
            if lowered in by_name:
                return by_name[lowered]
            missing.append(token)
            return token

        out = re.sub(r"[A-Za-z_]\w*", swap, text)
        return None if missing else out

    kept_states = []
    for state in plan.states:
        if not any(re.search(r"[A-Za-z_]", e) for e in state.extents):
            kept_states.append(state)
            continue
        extents = [bind(e) for e in state.extents]
        bounds = [(bind(lo), bind(hi)) for lo, hi in state.bounds]
        if any(e is None for e in extents) or any(
            lo is None or hi is None for lo, hi in bounds
        ):
            plan.unsupported.append(f"{state.module}%{state.name}: extent names no run state")
            continue
        state.extents = [e for e in extents if e is not None]
        state.bounds = [(lo, hi) for lo, hi in bounds if lo is not None and hi is not None]
        kept_states.append(state)
    plan.states = kept_states

    for obj in plan.objects:
        kept = []
        for comp in obj.components:
            extents = [bind(e) for e in comp.extents]
            bounds = [(bind(lo), bind(hi)) for lo, hi in comp.bounds]
            if any(e is None for e in extents) or any(
                lo is None or hi is None for lo, hi in bounds
            ):
                plan.unsupported.append(f"{obj.name}%{comp.name}: extent names no run state")
                continue
            comp.extents = [e for e in extents if e is not None]
            comp.bounds = [(lo, hi) for lo, hi in bounds if lo is not None and hi is not None]
            kept.append(comp)
        obj.components = kept
