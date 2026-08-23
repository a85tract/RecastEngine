"""Interface extraction: signatures, dtypes, dims, module state, call graph.

Migrated from CESM-language-translator ``pipeline/extract_interface.py``. The
analysis is unchanged; what changed is that the kind table is an argument
rather than a module global the command line reached in and mutated. A frontend
that answers differently depending on which CLI ran last is not cacheable, and
the ``Frontend`` contract requires ``analyze`` to be deterministic.

Nothing here writes. It reports what the source says, including where the
source is silent -- an argument whose intent is undeclared comes back as
``UNKNOWN`` rather than as a guess, because roughly a third of CAM's dummy
arguments have no usable declared intent and pretending otherwise is how a
translation gets a plausible wrong answer.

``intent_overrides`` is the one way a fact the source does not state may enter,
and it is deliberately narrow: it may fill an ``UNKNOWN`` in, never contradict a
declared intent, and the record keeps both what the source said and the fact
that a human supplied the rest. Someone reading the Facts can always tell the
two apart, which is the whole difference between an override and a guess.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03, parse, walk

INTENTS = frozenset({"IN", "OUT", "INOUT"})


class IntentConflict(RecastError):
    """An override contradicts an intent the source declares.

    Raised rather than resolved. If the two disagree, one of them is wrong, and
    which one is not a question analysis can answer -- silently preferring
    either would produce Facts nobody can check against the file.
    """


class UnknownOverride(RecastError):
    """An override names an argument its subprogram does not have.

    Almost always a typo, and a typo that is ignored is an override the
    operator believes is in force and is not.
    """


SHR_KIND_DTYPES: dict[str, str] = {
    "shr_kind_r8": "float64",
    "shr_kind_r4": "float32",
    "shr_kind_i8": "int64",
    "shr_kind_i4": "int32",
}
"""CESM's shared kind module, the one cross-domain fact this frontend carries.

It is here rather than in the domain extension because ``shr_kind_mod`` is how
essentially every Fortran climate code spells its precision, and a frontend
that cannot resolve ``r8`` is not useful on any of them. Anything narrower than
that belongs in the domain extension.
"""

KIND_FN_RE = re.compile(r"selected_real_kind\s*\(\s*(\d+)", re.I)


def node_span(node: Any) -> tuple[int | None, int | None]:
    """``(first_line, last_line)`` of a node, via its statement items."""
    lines: list[int] = []
    for n in walk(node):
        item = getattr(n, "item", None)
        if item is not None and getattr(item, "span", None):
            lines.extend(item.span)
    return (min(lines), max(lines)) if lines else (None, None)


def stmt_line(node: Any) -> int | None:
    return node_span(node)[0]


def names_in(node: Any) -> list[str]:
    """All identifier strings appearing under ``node`` (lowercased)."""
    return [str(n).lower() for n in walk(node, f03.Name)]


def dims_of(spec_node: Any) -> list[dict[str, str | None]] | None:
    """Array-spec node -> ``[{"lb", "ub"}, ...]``.

    ``ub`` of ``None`` means deferred or assumed -- an allocatable, or a dummy
    declared ``(:)``. The distinction matters downstream: a deferred bound has
    to be recovered at call time, an explicit one can be emitted.
    """
    if spec_node is None:
        return None
    dims: list[dict[str, str | None]] = []
    children = getattr(spec_node, "children", None)
    if children is None:
        children = [spec_node]
    for d in children:
        if isinstance(d, f03.Explicit_Shape_Spec):
            lb, ub = d.children
            dims.append({"lb": str(lb) if lb is not None else "1", "ub": str(ub)})
        elif isinstance(d, f03.Assumed_Shape_Spec | f03.Deferred_Shape_Spec):
            dims.append({"lb": "1", "ub": None})
        elif isinstance(d, f03.Assumed_Size_Spec):
            dims.append({"lb": "1", "ub": None})
        else:
            dims.append({"lb": "1", "ub": str(d)})
    return dims


def kind_aliases_from_use(ast: Any, kind_dtypes: dict[str, str]) -> dict[str, str]:
    """Resolve use-renamed kind params (``use shr_kind_mod, only: r8 => shr_kind_r8``)."""
    aliases: dict[str, str] = {}
    for use in walk(ast, f03.Use_Stmt):
        for rename in walk(use, f03.Rename):
            local = str(rename.children[1]).lower()
            remote = str(rename.children[2]).lower()
            if remote in kind_dtypes:
                aliases[local] = kind_dtypes[remote]
        for nm in walk(use, f03.Name):
            n = str(nm).lower()
            if n in kind_dtypes:
                aliases[n] = kind_dtypes[n]
    return aliases


def resolve_kind_map(module_params: list[dict[str, Any]]) -> dict[str, str]:
    """Map kind-parameter names (``r8``) to numpy dtype strings."""
    kind_map: dict[str, str] = {}
    for p in module_params:
        init = (p.get("init_expr") or "").lower().replace(" ", "")
        m = KIND_FN_RE.search(init)
        if m:
            kind_map[p["name"]] = "float64" if int(m.group(1)) >= 10 else "float32"
        elif init in ("8", "kind(1.d0)", "kind(1.0d0)"):
            kind_map[p["name"]] = "float64"
        elif init == "4":
            kind_map[p["name"]] = "float32"
    return kind_map


def dtype_of(base_type: str | None, kind: str | None, kind_map: dict[str, str]) -> str:
    """Fortran type + kind -> dtype name, or an ``UNKNOWN`` marker.

    Unresolved kinds come back as ``UNKNOWN_REAL_KIND(k)`` rather than
    defaulting to float64. A silently wrong precision is the one failure this
    stage can cause that no downstream gate would catch as a *type* error --
    it shows up much later as a tolerance failure nobody can explain.
    """
    bt = (base_type or "").upper()
    k = (kind or "").lower()
    if bt == "REAL":
        if k in kind_map:
            return kind_map[k]
        if k in ("8", ""):
            return "float64" if k == "8" else "float32"
        return f"UNKNOWN_REAL_KIND({k})"
    if bt == "INTEGER":
        return "int32"  # gfortran default integer
    if bt == "LOGICAL":
        return "bool"
    if bt == "CHARACTER":
        return "str"
    if bt.startswith("DOUBLE"):
        return "float64"
    return f"UNKNOWN({bt})"


def parse_decl_stmt(decl: Any) -> dict[str, Any]:
    """Decompose a ``Type_Declaration_Stmt``.

    Returns base type, kind, collapsed ``INTENT``, attributes, and one entity
    record per declared name.
    """
    type_spec, attr_list, _entity_list = decl.children

    base_type: str | None = None
    kind: str | None = None
    if isinstance(type_spec, f03.Intrinsic_Type_Spec):
        base_type = str(type_spec.children[0])
        sel = type_spec.children[1]
        if sel is not None:
            for nm in walk(sel, (f03.Name, f03.Int_Literal_Constant)):
                kind = str(nm)
                break
            # character(len=*) -> kind stays None; the length is recorded apart
            if base_type.upper() == "CHARACTER":
                kind = None
    else:  # derived types etc.
        base_type = str(type_spec)

    intent: str | None = None
    attrs: list[str] = []
    attr_dims: list[dict[str, str | None]] | None = None
    if attr_list is not None:
        for a in attr_list.children:
            if isinstance(a, f03.Intent_Attr_Spec):
                intent = str(a.children[1]).upper()
                attrs.append("INTENT")
            elif isinstance(a, f03.Dimension_Attr_Spec):
                attrs.append("DIMENSION:" + str(a.children[1]))
                # a DIMENSION attribute applies to every entity without its own
                attr_dims = dims_of(a.children[1])
            else:
                attrs.append(str(a).upper())

    entities: list[dict[str, Any]] = []
    for ent in walk(decl, f03.Entity_Decl):
        name_node, array_spec, char_len, init = ent.children
        entities.append(
            {
                "name": str(name_node).lower(),
                "array_spec": str(array_spec) if array_spec is not None else None,
                "dims": dims_of(array_spec) if array_spec is not None else attr_dims,
                "char_len": str(char_len) if char_len is not None else None,
                "init_expr": str(init.children[1]) if init is not None else None,
            }
        )

    char_len_spec: str | None = None
    if base_type and base_type.upper() == "CHARACTER":
        m = re.search(r"len\s*=\s*([^),]+)", str(type_spec), re.I)
        if m:
            char_len_spec = m.group(1).strip()

    return {
        "base_type": base_type,
        "kind": kind,
        "intent": intent,
        "attrs": attrs,
        "entities": entities,
        "char_len_spec": char_len_spec,
        "line": stmt_line(decl),
    }


def collect_decls(spec_part: Any) -> list[dict[str, Any]]:
    decls = [parse_decl_stmt(d) for d in walk(spec_part, f03.Type_Declaration_Stmt)]
    # A separate DIMENSION statement (F77 style: `complex a` then
    # `dimension a(0:*)`) gives an already-declared entity its shape. Without
    # this the entity reads as a scalar and every subscript of it refuses.
    by_name = {e["name"]: e for d in decls for e in d["entities"]}
    for stmt in walk(spec_part, f03.Dimension_Stmt):
        for name, spec in stmt.children[0]:
            entity = by_name.get(str(name).lower())
            if entity is not None and entity["dims"] is None:
                entity["dims"] = dims_of(spec)
                entity["array_spec"] = str(spec)
    return decls


def sub_name_of(sub: Any) -> str:
    """Lowercased name of a subprogram node."""
    stmt = walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
    return str(stmt.children[1]).lower()


def normalize_overrides(table: dict[str, Any] | None) -> dict[str, dict[str, str]]:
    """``{subprogram: {arg: INTENT}}``, lowercased, validated, metadata dropped.

    Keys beginning with ``_`` are notes about the table rather than entries in
    it -- the hand-maintained file this feature replaces carried its reasoning
    under ``_provenance``, and that is worth keeping next to the values it
    explains rather than in a sibling document nobody reads.
    """
    out: dict[str, dict[str, str]] = {}
    for sub_name, args in (table or {}).items():
        if sub_name.startswith("_"):
            continue
        if not isinstance(args, dict):
            raise UnknownOverride(
                f"intent overrides for {sub_name!r} must be a mapping of arg -> intent"
            )
        entries = {}
        for arg, intent in args.items():
            if str(intent).upper() not in INTENTS:
                raise UnknownOverride(
                    f"intent override {sub_name}.{arg} = {intent!r}; "
                    f"expected one of {sorted(INTENTS)}"
                )
            entries[arg.lower()] = str(intent).upper()
        out[sub_name.lower()] = entries
    return out


def apply_intent_override(arg: dict[str, Any], sub_name: str, override: str | None) -> None:
    """Fill an ``UNKNOWN`` intent in from an override, in place.

    An override may only supply what the source omitted. Contradicting a
    declared intent raises: the source and the operator disagree, and analysis
    is not the place that gets to decide which of them is right.

    When an override is applied the record grows two keys, so that a reader can
    still see what the file itself said:

    * ``intent_declared`` -- ``UNKNOWN``, what the source states
    * ``intent_override`` -- ``True``, meaning this value did not come from the
      source. Where it *did* come from is recorded once, in
      ``Facts.provenance``, rather than repeated on every argument.
    """
    if override is None:
        return
    declared = arg["intent"]
    if declared != "UNKNOWN":
        if declared != override:
            raise IntentConflict(
                f"{sub_name}.{arg['name']} is declared intent({declared.lower()}) "
                f"but an override says {override}; fix one of them"
            )
        return
    arg["intent"] = override
    arg["intent_declared"] = "UNKNOWN"
    arg["intent_override"] = True


def extract_subprogram(
    sub: Any,
    kind_map: dict[str, str],
    module_state_names: set[str],
    module_sub_names: set[str],
    intent_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    """One subroutine or function: args, locals, calls, module-state effects.

    ``intent_overrides`` maps this subprogram's argument names to ``IN``,
    ``OUT`` or ``INOUT``. See ``apply_intent_override``.
    """
    is_function = isinstance(sub, f03.Function_Subprogram)
    stmt_cls = f03.Function_Stmt if is_function else f03.Subroutine_Stmt
    stmt = walk(sub, stmt_cls)[0]

    prefix, name_node, dummy_args, suffix = (list(stmt.children) + [None] * 4)[:4]
    name = str(name_node).lower()

    prefixes: list[str] = []
    result_name = name
    if prefix is not None:
        prefixes = [str(p).upper() for p in walk(prefix, f03.Prefix_Spec)]
    if is_function and suffix is not None:
        res = walk(suffix, f03.Name)
        if res:
            result_name = str(res[0]).lower()

    arg_names = [str(a).lower() for a in walk(dummy_args, f03.Name)] if dummy_args else []

    spec = next((c for c in sub.children if isinstance(c, f03.Specification_Part)), None)
    decls = collect_decls(spec) if spec is not None else []

    ent_info: dict[str, dict[str, Any]] = {}
    local_parameters: list[dict[str, Any]] = []
    for d in decls:
        for e in d["entities"]:
            info = {
                "fortran_type": d["base_type"],
                "kind": d["kind"],
                "dtype": dtype_of(d["base_type"], d["kind"], kind_map),
                "intent": d["intent"],
                "optional": "OPTIONAL" in d["attrs"],
                "parameter": "PARAMETER" in d["attrs"],
                "array_spec": e["array_spec"]
                or next(
                    (a.split(":", 1)[1] for a in d["attrs"] if a.startswith("DIMENSION:")), None
                ),
                "dims": e["dims"],
                "char_len": e["char_len"] or d["char_len_spec"],
                "init_expr": e["init_expr"],
                "line": d["line"],
            }
            ent_info[e["name"]] = info
            if info["parameter"]:
                local_parameters.append(
                    {
                        "name": e["name"],
                        "dtype": info["dtype"],
                        "dims": info.get("dims"),
                        "init_expr": e["init_expr"],
                        "line": d["line"],
                    }
                )
    # separate-form `PARAMETER (NAME=expr, ...)` statements (F77 style)
    if spec is not None:
        for pstmt in walk(spec, f03.Parameter_Stmt):
            for pdef in walk(pstmt, f03.Named_Constant_Def):
                pname = str(pdef.children[0]).lower()
                declared = ent_info.get(pname)
                if declared is not None:
                    declared["parameter"] = True
                    declared["init_expr"] = str(pdef.children[1])
                    local_parameters.append(
                        {
                            "name": pname,
                            "dtype": declared["dtype"],
                            "dims": declared.get("dims"),
                            "init_expr": declared["init_expr"],
                            "line": declared["line"],
                        }
                    )

    args: list[dict[str, Any]] = []
    for pos, an in enumerate(arg_names):
        info = ent_info.get(an, {})
        arg = {
            "name": an,
            "position": pos,
            "fortran_type": info.get("fortran_type"),
            "kind": info.get("kind"),
            "dtype": info.get("dtype", "UNDECLARED"),
            "intent": info.get("intent") or "UNKNOWN",
            "optional": info.get("optional", False),
            "array_spec": info.get("array_spec"),
            "dims": info.get("dims"),
            "char_len": info.get("char_len"),
            "line": info.get("line"),
        }
        apply_intent_override(arg, name, (intent_overrides or {}).get(an))
        args.append(arg)

    unknown = sorted(set(intent_overrides or {}) - set(arg_names))
    if unknown:
        raise UnknownOverride(
            f"intent overrides for {name!r} name arguments it does not have: {unknown}"
        )

    result_dtype: str | None = None
    result_dims: list[dict[str, str | None]] | None = None
    if is_function:
        if result_name in ent_info:
            result_dtype = ent_info[result_name]["dtype"]
            result_dims = ent_info[result_name].get("dims")
        elif prefix is not None:  # type in the prefix: `real(r8) function f(...)`
            for t in walk(prefix, f03.Intrinsic_Type_Spec):
                k = None
                for nm in walk(t.children[1] or [], (f03.Name, f03.Int_Literal_Constant)):
                    k = str(nm)
                    break
                result_dtype = dtype_of(str(t.children[0]), k, kind_map)

    locals_ = [
        {"name": n, "dtype": i["dtype"], "array_spec": i["array_spec"], "dims": i.get("dims")}
        for n, i in ent_info.items()
        if n not in arg_names and n != result_name and not i["parameter"]
    ]

    exec_part = next((c for c in sub.children if isinstance(c, f03.Execution_Part)), None)

    present_args: list[str] = []
    calls: list[str] = []
    state_read: set[str] = set()
    state_written: set[str] = set()
    if exec_part is not None:
        for ref in walk(exec_part, (f03.Part_Ref, f03.Intrinsic_Function_Reference)):
            fn = str(ref.children[0]).lower()
            if fn == "present":
                for nm in walk(ref.children[1], f03.Name):
                    present_args.append(str(nm).lower())
            elif fn in module_sub_names:
                calls.append(fn)
        for call in walk(exec_part, f03.Call_Stmt):
            cn = str(call.children[0]).lower()
            if cn in module_sub_names:
                calls.append(cn)
        # An arg-less function reference is a bare Name in the expression, not a
        # Part_Ref, so the two forms have to be caught separately.
        used = set(names_in(exec_part))
        calls.extend(sorted((used & module_sub_names) - {name} - set(calls)))

        for asgn in walk(exec_part, f03.Assignment_Stmt):
            lhs, _, rhs = asgn.children
            state_written |= set(names_in(lhs)) & module_state_names
            state_read |= set(names_in(rhs)) & module_state_names
        # deallocate(X) is a write: Fortran sets the allocation status to
        # unallocated, and the translator maps that to ``X = None``
        for dealloc in walk(exec_part, f03.Deallocate_Stmt):
            state_written |= set(names_in(dealloc)) & module_state_names
        # So is allocate(X): the name goes from unallocated to an array.
        for allocate in walk(exec_part, f03.Allocate_Stmt):
            for item in walk(allocate, f03.Allocation):
                target = str(item.children[0]).lower()
                if target in module_state_names:
                    state_written.add(target)
        # Module state passed as a call actual may be written through an
        # intent(out) or intent(inout) dummy, and the callee's intents are
        # not in view here. Counted as written, which costs a name on the
        # ``global`` list if it turns out to be read-only and loses the
        # write to a local if it is not.
        for call in walk(exec_part, f03.Call_Stmt):
            if call.children[1] is not None:
                state_written |= set(names_in(call.children[1])) & module_state_names
        # reads in non-assignment contexts (if conditions, call arguments)
        state_read |= (used & module_state_names) - state_written

    return {
        "name": name,
        "kind": "function" if is_function else "subroutine",
        "prefixes": prefixes,
        "result": result_name if is_function else None,
        "result_dtype": result_dtype,
        "result_dims": result_dims,
        "line_span": list(node_span(sub)),
        "args": args,
        "local_parameters": local_parameters,
        "locals": locals_,
        "present_calls": sorted(set(present_args)),
        "calls": sorted({c for c in calls if c != name}),
        "module_state_read": sorted(state_read),
        "module_state_written": sorted(state_written),
    }


def _scope_of(ast: Any, path: Path) -> tuple[str, Any, Any]:
    """``(module_name, specification_part, subprogram_scope)`` for a source file.

    Handles the three shapes this frontend meets: a module, a main program, and
    a file of bare subprograms. The last two borrow the file stem for a name so
    that every Unit still has a stable ``uid``.
    """
    modules = walk(ast, f03.Module)
    if modules:
        mod = modules[0]
        mod_name = str(walk(mod, f03.Module_Stmt)[0].children[1]).lower()
        spec = next((c for c in mod.children if isinstance(c, f03.Specification_Part)), None)
        return mod_name, spec, mod

    programs = walk(ast, f03.Main_Program)
    stem = path.stem.lower().replace("_cpp", "")
    mod_name = stem
    spec = None
    if programs:
        prog_stmts = walk(programs[0], f03.Program_Stmt)
        if prog_stmts and prog_stmts[0].children[1] is not None:
            mod_name = str(prog_stmts[0].children[1]).lower()
        spec = next(
            (c for c in programs[0].children if isinstance(c, f03.Specification_Part)), None
        )
    return mod_name, spec, ast


def _derived_types(mod_spec: Any, kind_map: dict[str, str]) -> dict[str, Any]:
    """``{type_name: {component: {dtype, dims, allocatable, pointer}}}``."""
    types: dict[str, Any] = {}
    if mod_spec is None:
        return types
    for td in walk(mod_spec, f03.Derived_Type_Def):
        tname = None
        for st in walk(td, f03.Derived_Type_Stmt):
            tname = str(st.children[1]).lower()
        comps: dict[str, Any] = {}
        for decl in walk(td, f03.Data_Component_Def_Stmt):
            tspec = decl.children[0]
            base = (
                str(tspec.children[0]) if isinstance(tspec, f03.Intrinsic_Type_Spec) else str(tspec)
            )
            kind = None
            if isinstance(tspec, f03.Intrinsic_Type_Spec) and tspec.children[1] is not None:
                for nm in walk(tspec.children[1], (f03.Name, f03.Int_Literal_Constant)):
                    kind = str(nm)
                    break
            attr_list = decl.children[1]
            attrs = [str(a).upper() for a in (attr_list.children if attr_list else [])]
            for ent in walk(decl, f03.Component_Decl):
                comps[str(ent.children[0]).lower()] = {
                    "dtype": dtype_of(base, kind, kind_map),
                    "dims": dims_of(ent.children[1]),
                    "allocatable": "ALLOCATABLE" in attrs,
                    "pointer": "POINTER" in attrs,
                }
        if tname:
            types[tname] = comps
    return types


def _generics(mod_spec: Any) -> dict[str, list[str]]:
    """``{generic_name: [specific names]}`` from interface blocks."""
    generics: dict[str, list[str]] = {}
    if mod_spec is None:
        return generics
    for ib in walk(mod_spec, f03.Interface_Block):
        gname = None
        for st in walk(ib, f03.Interface_Stmt):
            if st.children[0] is not None:
                gname = str(st.children[0]).lower()
        if gname is None:
            continue
        specs = [str(n).lower() for ps in walk(ib, f03.Procedure_Stmt) for n in walk(ps, f03.Name)]
        if specs:
            generics[gname] = specs
    return generics


def extract(
    path: Path,
    *,
    kind_assumptions: dict[str, str] | None = None,
    intent_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full interface record for one Fortran source file.

    ``kind_assumptions`` names kind parameters this file use-imports from a
    module that is not being parsed (``{"r8": "float64"}``). The command-line
    ancestor of this function mutated a module global to do the same thing,
    which made the answer depend on invocation order.

    ``intent_overrides`` is ``{subprogram: {arg: "IN"|"OUT"|"INOUT"}}`` for
    dummy arguments the source declares no intent for. Subprograms this file
    does not define are ignored -- one table may cover a whole source tree --
    but an argument a named subprogram does not have is an error.
    """
    ast = parse(path)
    overrides = normalize_overrides(intent_overrides)
    kind_dtypes = dict(SHR_KIND_DTYPES)
    if kind_assumptions:
        kind_dtypes.update({k.lower(): v for k, v in kind_assumptions.items()})

    mod_name, mod_spec, sub_scope = _scope_of(ast, path)

    module_parameters: list[dict[str, Any]] = []
    module_state: list[dict[str, Any]] = []
    if mod_spec is not None:
        for d in collect_decls(mod_spec):
            for e in d["entities"]:
                rec = {
                    "name": e["name"],
                    "fortran_type": d["base_type"],
                    "kind": d["kind"],
                    "dims": e["dims"],
                    "init_expr": e["init_expr"],
                    "line": d["line"],
                }
                if "PARAMETER" in d["attrs"]:
                    module_parameters.append(rec)
                else:
                    module_state.append(rec)

    kind_map = resolve_kind_map(module_parameters)
    for k, v in kind_aliases_from_use(ast, kind_dtypes).items():
        kind_map.setdefault(k, v)
    for rec in module_parameters + module_state:
        rec["dtype"] = dtype_of(rec["fortran_type"], rec["kind"], kind_map)

    state_names = {s["name"] for s in module_state}

    # Accessibility. CAM's convention is a bare `private` up top and an
    # explicit public list -- and a wrapper that `use`s a private symbol does
    # not compile, so who is public is a fact consumers genuinely need.
    public_names: list[str] = []
    private_names: list[str] = []
    default_private = False
    if mod_spec is not None:
        for acc in walk(mod_spec, f03.Access_Stmt):
            spec = str(acc.children[0]).upper()
            names = [str(n).lower() for n in walk(acc, f03.Name)]
            if spec == "PUBLIC":
                public_names.extend(names)
            elif names:
                private_names.extend(names)
            else:
                default_private = True

    def is_public(name: str) -> bool:
        if default_private:
            return name in public_names
        return name not in private_names

    subs = walk(sub_scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram))
    sub_names = {
        str(walk(s, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() for s in subs
    }
    subprograms = [
        extract_subprogram(s, kind_map, state_names, sub_names, overrides.get(sub_name_of(s)))
        for s in subs
    ]
    _infer_write_only_intents(subs, subprograms)
    _host_associate(subs, subprograms, sub_names, state_names)
    for record in subprograms:
        record["public"] = is_public(record["name"])

    return {
        "source_file": str(path),
        "module": mod_name,
        # A file of bare subprograms borrows its stem for a name; consumers
        # that emit `use <module>` need to know the name is borrowed.
        "is_module": bool(walk(ast, f03.Module)),
        "kind_map": kind_map,
        "use_statements": [str(u) for u in walk(sub_scope, f03.Use_Stmt)],
        "module_parameters": module_parameters,
        "module_state": module_state,
        "public": sorted(set(public_names)),
        "types": _derived_types(mod_spec, kind_map),
        "generics": _generics(mod_spec),
        "subprograms": subprograms,
    }


def companion_externals(record: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Externals-table entries for every procedure of a sibling module's record.

    A module that calls into an already-translated sibling resolves those
    calls outside its own file, and the read/write analysis needs the same
    fact the pipeline's ``--companions`` flag carried: which names are
    procedures, and which argument positions they write. Deriving the table
    from the sibling's own interface record keeps the intents from being
    transcribed by hand, which is how they would drift.
    """
    table: dict[str, dict[str, Any]] = {}
    for sub in record["subprograms"]:
        table[sub["name"]] = {
            "kind": sub["kind"],
            "out_positions": [
                at
                for at, argument in enumerate(sub["args"])
                if argument["intent"] in ("OUT", "INOUT")
            ],
        }
    return table


def subprogram_key(record: dict[str, Any]) -> str:
    """The name a subprogram is looked up by: ``host/name`` for an internal
    procedure, because two hosts may each contain a ``func``."""
    host = record.get("host")
    return f"{host}/{record['name']}" if host else record["name"]


def emit_name(record: dict[str, Any]) -> str:
    """The name the translation gives this subprogram.

    The Fortran name, as the pipeline emits it -- an internal procedure comes
    out flat, beside its host, with the host's variables it touches as
    trailing arguments. Only when two internal procedures of different hosts
    share a name is the second form used, ``host__name``, because one flat
    file cannot define ``func`` twice; the pipeline has no rule for that case
    and this is the smallest one that keeps every call resolvable.
    """
    return str(record.get("emit_name") or record["name"])


def _infer_write_only_intents(subs: list[Any], records: list[dict[str, Any]]) -> None:
    """Give a write-only F77 dummy the intent its use says it has.

    A dummy declared without INTENT that the body only ever assigns to --
    bare name on the left, never read anywhere -- is semantically
    ``intent(out)``. Without the attribute the return convention leaves it
    out of the signature and out of the return, so the value the routine
    computed is dropped on the floor. Scalars only: an array dummy mutates
    through the buffer it was passed and is not lost either way.
    """
    by_name = {sub_name_of(s): s for s in subs}
    for record in records:
        candidates = [
            argument
            for argument in record["args"]
            if argument["intent"] == "UNKNOWN"
            and not argument.get("dims")
            and not argument.get("optional")
            and argument.get("dtype") in ("float64", "float32", "int32", "int64", "bool")
        ]
        node = by_name.get(record["name"])
        if not candidates or node is None:
            continue
        exec_part = next((c for c in node.children if isinstance(c, f03.Execution_Part)), None)
        if exec_part is None:
            continue
        mentions: dict[str, int] = {}
        for name in walk(exec_part, f03.Name):
            key = str(name).lower()
            mentions[key] = mentions.get(key, 0) + 1
        assigned: dict[str, int] = {}
        for assignment in walk(exec_part, f03.Assignment_Stmt):
            target = assignment.children[0]
            if isinstance(target, f03.Name):
                key = str(target).lower()
                assigned[key] = assigned.get(key, 0) + 1
        for argument in candidates:
            written = assigned.get(argument["name"], 0)
            if written and mentions.get(argument["name"], 0) == written:
                argument["intent"] = "OUT"
                argument["intent_inferred"] = "write-only"


def _host_associate(
    subs: list[Any],
    records: list[dict[str, Any]],
    sub_names: set[str],
    state_names: set[str],
) -> None:
    """Mark internal procedures with their host and the host variables they use.

    An internal procedure -- one inside another subprogram's CONTAINS --
    reads and writes its host's variables without declaring them (host
    association). The translation passes those as extra trailing arguments,
    so the record has to say which they are: names used in the internal
    procedure's execution part that are not its own, but are declared by the
    host. As the pipeline's ``extract_interface`` does it.
    """
    parents: dict[int, Any] = {}
    for s in subs:
        for inner in walk(s, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
            if inner is not s:
                parents[id(inner)] = s
    for s, rec in zip(subs, records, strict=True):
        parent = parents.get(id(s))
        if parent is None:
            continue
        rec["host"] = sub_name_of(parent)
        exec_part = next((c for c in s.children if isinstance(c, f03.Execution_Part)), None)
        if exec_part is None:
            continue
        used = set(names_in(exec_part))
        own = {a["name"] for a in rec["args"]}
        own |= {loc["name"] for loc in rec["locals"]}
        own |= {p["name"] for p in rec.get("local_parameters", [])}
        host_names: set[str] = set()
        parent_spec = next(
            (c for c in parent.children if isinstance(c, f03.Specification_Part)), None
        )
        if parent_spec is not None:
            for d in collect_decls(parent_spec):
                host_names |= {e["name"] for e in d["entities"]}
        parent_stmt = walk(parent, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
        if len(parent_stmt.children) > 2 and parent_stmt.children[2] is not None:
            host_names |= {str(n).lower() for n in walk(parent_stmt.children[2], f03.Name)}
        host_vars = sorted((used & host_names) - own - sub_names - state_names)
        if host_vars:
            rec["host_vars"] = host_vars
    # Two internal procedures of different hosts with one name cannot both
    # be `def func` in one file.
    seen: dict[str, int] = {}
    for rec in records:
        seen[rec["name"]] = seen.get(rec["name"], 0) + 1
    for rec in records:
        if rec.get("host") and seen[rec["name"]] > 1:
            rec["emit_name"] = f"{rec['host']}__{rec['name']}"
