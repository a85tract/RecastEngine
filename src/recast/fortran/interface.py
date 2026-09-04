"""Interface extraction: signatures, dtypes, dims, module state, call graph.

Migrated from the source pipeline's ``pipeline/extract_interface.py``. The
analysis is unchanged; what changed is that the kind table is an argument
rather than a module global the command line reached in and mutated. A frontend
that answers differently depending on which CLI ran last is not cacheable, and
the ``Frontend`` contract requires ``analyze`` to be deterministic.

Nothing here writes. It reports what the source says, including where the
source is silent -- an argument whose intent is undeclared comes back as
``UNKNOWN`` rather than as a guess, because roughly a third of one corpus's dummy
arguments have no usable declared intent and pretending otherwise is how a
translation gets a plausible wrong answer.

The two exceptions are not guesses but readings of the body, and each says so
in ``intent_inferred``: a dummy the body only ever assigns to is ``OUT``, and
one it only ever reads -- never assigned, never a DO variable, never handed to
anything that could write it -- is ``IN``. Both are what the code does, not
what it probably meant, and a body that does anything else leaves the argument
``UNKNOWN``.

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
from recast.fortran._parse import f03, f08, parse, walk

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


KIND_FN_RE = re.compile(r"selected_real_kind\s*\(\s*(\d+)", re.I)
INT_KIND_FN_RE = re.compile(r"selected_int_kind\s*\(\s*(\d+)", re.I)
KIND_OF_RE = re.compile(r"^kind\(([^)]*)\)$", re.I)
INT_LITERAL_RE = re.compile(r"^[-+]?\d+$")
REAL_LITERAL_RE = re.compile(r"^[-+]?(\d+\.\d*|\.\d+|\d+)([de][-+]?\d+)?$")

ISO_FORTRAN_ENV_KINDS = {
    "real32": "float32",
    "real64": "float64",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
}
"""The named constants of the intrinsic ``iso_fortran_env`` module, which is
how most of the Fortran written since 2008 spells a kind. ``real128`` is
deliberately absent: numpy has no portable 128-bit float, so a translation
that took one for float64 would be the silent precision loss ``dtype_of``
exists to refuse.

A file that declares its own parameter of one of these names shadows the
intrinsic, so this table is consulted only after the file's own parameters
have been resolved."""


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


def dims_of(spec_node: Any) -> list[dict[str, Any]] | None:
    """Array-spec node -> ``[{"lb", "ub"}, ...]``.

    ``ub`` of ``None`` means deferred or assumed -- an allocatable, or a dummy
    declared ``(:)``. The distinction matters downstream: a deferred bound has
    to be recovered at call time, an explicit one can be emitted.
    """
    if spec_node is None:
        return None
    dims: list[dict[str, Any]] = []
    if isinstance(spec_node, f03.Assumed_Size_Spec):
        # ``x(*)`` and ``x(3,*)``. Taken here rather than in the loop below,
        # because fparser gives this node children -- the leading explicit
        # dimensions, and the last dimension's lower bound -- and descending
        # into them reads the assumed-size dimension as a pair of ``None``s
        # and reports a rank the declaration does not have. The ``*`` itself
        # is not a child: it is the node.
        leading, lower = spec_node.children
        if leading is not None:
            dims.extend(dims_of(leading) or [])
        # Marked, because an assumed-size ``ub`` of None is otherwise
        # indistinguishable from an assumed-shape one, and only the first
        # means the caller owns storage the callee cannot size.
        dims.append(
            {"lb": str(lower) if lower is not None else "1", "ub": None, "assumed_size": True}
        )
        return dims
    children = getattr(spec_node, "children", None)
    if children is None:
        children = [spec_node]
    for d in children:
        if isinstance(d, f03.Explicit_Shape_Spec):
            lb, ub = d.children
            dims.append({"lb": str(lb) if lb is not None else "1", "ub": str(ub)})
        elif isinstance(d, f03.Assumed_Shape_Spec):
            # ``tk(bounds%begc:, -nlevsno+1:)``: the shape is the caller's,
            # the lower bound is this declaration's, and a subscript shifts
            # by it.
            lower = d.children[0] if getattr(d, "children", None) else None
            dims.append({"lb": str(lower) if lower is not None else "1", "ub": None})
        elif isinstance(d, f03.Deferred_Shape_Spec):
            dims.append({"lb": "1", "ub": None})
        elif isinstance(d, f03.Assumed_Size_Spec):
            dims.extend(dims_of(d) or [])
        else:
            dims.append({"lb": "1", "ub": str(d)})
    return dims


INTRINSIC_KIND_MODULES = frozenset({"iso_fortran_env", "iso_c_binding"})
"""The intrinsic modules whose named constants are kinds."""

INTRINSIC_KINDS: dict[str, str] = {
    **ISO_FORTRAN_ENV_KINDS,
    "c_float": "float32",
    "c_double": "float64",
    "c_long_double": "float64",
    "c_int": "int32",
    "c_short": "int16",
    "c_long": "int64",
    "c_long_long": "int64",
    "c_size_t": "int64",
    "c_int32_t": "int32",
    "c_int64_t": "int64",
}
"""What a USE of an intrinsic kind module can bring in: ``iso_fortran_env``'s
names above, and ``iso_c_binding``'s C kinds. ``real128`` stays out for the
reason ``ISO_FORTRAN_ENV_KINDS`` gives."""


def kind_aliases_from_use(ast: Any, kind_dtypes: dict[str, str]) -> dict[str, str]:
    """Kind names made available by USE: the operator's table (``use
    kinds_mod, only: r8 => wp_r8``, or the bare names) and the standard
    intrinsic modules -- ``wp => real64``, ``only: real64``, or a bare ``use
    iso_fortran_env``, which brings every name in (the pipeline's #26)."""
    aliases: dict[str, str] = {}
    for use in walk(ast, f03.Use_Stmt):
        module = str(use.children[2]).lower() if len(use.children) > 2 else ""
        intrinsic = module in INTRINSIC_KIND_MODULES
        table = dict(kind_dtypes)
        if intrinsic:
            table.update(INTRINSIC_KINDS)
        for rename in walk(use, f03.Rename):
            local = str(rename.children[1]).lower()
            remote = str(rename.children[2]).lower()
            if remote in table:
                aliases[local] = table[remote]
        only = use.children[4] if len(use.children) > 4 else None
        if only is not None:
            for nm in walk(only, f03.Name):
                n = str(nm).lower()
                if n in table:
                    aliases.setdefault(n, table[n])
        elif intrinsic:
            for n, dtype in INTRINSIC_KINDS.items():
                aliases.setdefault(n, dtype)
        else:
            for nm in walk(use, f03.Name):
                n = str(nm).lower()
                if n in kind_dtypes:
                    aliases[n] = kind_dtypes[n]
    return aliases


def _kind_dtype(init: str, resolved: dict[str, str]) -> str | None:
    """One kind parameter's initializer -> a dtype, or ``None`` if it is not
    one this reads. ``init`` is already lowercased and stripped of spaces.

    Four spellings of the same fact, and the aliasing between them. Fortran
    lets a kind be named (``selected_real_kind(12)``), imported from
    ``iso_fortran_env`` (``real64``), taken from a literal (``kind(0.d0)``),
    or written as the bare kind number -- and one kind parameter is often
    just a shorter name for another, which is why ``resolved`` is consulted
    first and why the caller iterates.
    """
    if init in resolved:  # wp = slsqp_rk
        return resolved[init]
    m = KIND_FN_RE.search(init)
    if m:
        return "float64" if int(m.group(1)) >= 10 else "float32"
    m = INT_KIND_FN_RE.search(init)
    if m:
        # selected_int_kind(r) is decimal *range*; 9 is the most a 32-bit
        # integer holds.
        return "int64" if int(m.group(1)) > 9 else "int32"
    if init in ISO_FORTRAN_ENV_KINDS:
        return ISO_FORTRAN_ENV_KINDS[init]
    m = KIND_OF_RE.match(init)
    if m:
        literal = m.group(1)
        if INT_LITERAL_RE.match(literal):
            return "int32"  # kind(1) -- default integer
        if REAL_LITERAL_RE.match(literal):
            # kind(0.d0) is double, kind(1.0) is default real.
            return "float64" if "d" in literal else "float32"
        return None
    if init == "8":
        return "float64"
    if init == "4":
        return "float32"
    return None


def resolve_kind_map(module_params: list[dict[str, Any]]) -> dict[str, str]:
    """Map kind-parameter names (``r8``) to numpy dtype strings.

    Passes over the parameters until one changes nothing, because a kind
    parameter may be spelled in terms of another declared after it
    (``slsqp_rk = real64`` then ``wp = slsqp_rk``) and a single pass would
    leave the second unresolved -- which reaches the f2py wrapper as a dummy
    argument whose type it cannot spell, and the whole subprogram drops out
    of the bit-exact gate over a name.
    """
    resolved: dict[str, str] = {}
    pending = {
        p["name"]: (p.get("init_expr") or "").lower().replace(" ", "") for p in module_params
    }
    while pending:
        progressed = False
        for name, init in list(pending.items()):
            dtype = _kind_dtype(init, resolved)
            if dtype is None:
                continue
            resolved[name] = dtype
            del pending[name]
            progressed = True
        if not progressed:
            break
    return resolved


def dtype_of(base_type: str | None, kind: str | None, kind_map: dict[str, str]) -> str:
    """Fortran type + kind -> dtype name, or an ``UNKNOWN`` marker.

    Unresolved kinds come back as ``UNKNOWN_REAL_KIND(k)`` rather than
    defaulting to float64. A silently wrong precision is the one failure this
    stage can cause that no downstream gate would catch as a *type* error --
    it shows up much later as a tolerance failure nobody can explain.

    The kind map is read by base type: a name in it that resolved to an
    integer width is not an answer for a ``real``, and vice versa. Mixing them
    would let one bad parameter turn a float64 argument into an int64 one,
    which is the same silent corruption in a different direction.
    """
    bt = (base_type or "").upper()
    k = (kind or "").lower()
    if bt == "REAL":
        if kind_map.get(k, "").startswith("float"):
            return kind_map[k]
        if k in ("8", ""):
            return "float64" if k == "8" else "float32"
        return f"UNKNOWN_REAL_KIND({k})"
    if bt == "INTEGER":
        if kind_map.get(k, "").startswith("int"):
            return kind_map[k]
        if k == "8":
            return "int64"
        # An unresolved integer kind still reports the default rather than an
        # UNKNOWN marker: this is where the pipeline stood, every integer in
        # the corpus so far is a 32-bit one, and turning the default into a
        # refusal is a change to make against evidence, not in passing.
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
    attr_dims: list[dict[str, Any]] | None = None
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
    """Type declarations that belong to THIS scope: the direct children of
    the specification part.

    A declaration inside an ``interface`` block declares a dummy argument of
    somebody *else's* procedure. Walking the subtree picked those up too, and
    a module of nothing but abstract interfaces reported two dozen variables
    named ``x`` and ``f`` as its state, several of them twice -- and then
    subtracted a host's ``me`` from an internal function's host variables
    because a module "state" of that name existed. The pipeline's fix for
    its #7, relayed.
    """
    parts = spec_part if isinstance(spec_part, (list, tuple)) else [spec_part]
    return [
        parse_decl_stmt(child)
        for part in parts
        if part is not None
        for child in (part.children or [])
        if isinstance(child, f03.Type_Declaration_Stmt)
    ]


def dimension_stmt_shapes(spec_part: Any) -> dict[str, dict[str, Any]]:
    """``dimension a(10), b(0:n,3)`` -> ``{name: {"dims", "array_spec"}}``.

    The F77 standalone form, which says the shape of a name declared -- or
    not declared -- somewhere else. Nothing here read it, so such a name
    stayed a scalar, and a scalar with subscripts on the left of an ``=`` is
    exactly the shape of an old-style statement function: ``pm(i,i) = ...``
    became ``def pm(i, i)``, a duplicate-argument ``SyntaxError`` that takes
    the whole translated module down at import.
    """
    shapes: dict[str, dict[str, Any]] = {}
    for stmt in walk(spec_part, f03.Dimension_Stmt):
        # ``Dimension_Stmt.children`` is a one-tuple holding a list of plain
        # ``(Name, Array_Spec)`` pairs -- ordinary tuples, not nodes, so there
        # is no ``.children`` to descend into on the pair itself.
        for group in stmt.children:
            for entry in group or ():
                if not isinstance(entry, tuple) or len(entry) != 2:
                    continue
                name_node, spec_node = entry
                if spec_node is None:
                    continue
                shapes[str(name_node).lower()] = {
                    "dims": dims_of(spec_node),
                    "array_spec": str(spec_node),
                }
    return shapes


def procedure_declarations(spec_part: Any) -> dict[str, str | None]:
    """``procedure(func) :: fcn`` -> ``{"fcn": "func"}``.

    A dummy procedure is declared by a PROCEDURE statement rather than a type
    declaration, so nothing in ``collect_decls`` sees it and the argument was
    left with no type, no intent and no hint that it is callable at all. The
    value is the interface the declaration names -- an abstract interface of
    this file, usually -- or ``None`` for the typed forms (``procedure(real)``,
    ``procedure()``), which say the result type but not the argument list and
    so cannot be called from a translation.
    """
    declared: dict[str, str | None] = {}
    if spec_part is None:
        return declared
    for stmt in walk(spec_part, f03.Procedure_Declaration_Stmt):
        spec, _attrs, decl_list = (list(stmt.children) + [None] * 3)[:3]
        interface = str(spec).lower() if isinstance(spec, f03.Name) else None
        entities = getattr(decl_list, "children", None) or ([decl_list] if decl_list else [])
        for entity in entities:
            # ``fcn`` is a bare Name; ``fcn => null()`` is a Proc_Decl whose
            # first child is the name. Walking every Name below the list would
            # also collect the initializer's.
            head = entity.children[0] if isinstance(entity, f03.Proc_Decl) else entity
            if isinstance(head, f03.Name):
                declared[str(head).lower()] = interface
    return declared


def apply_dimension_stmts(
    entities: dict[str, dict[str, Any]], shapes: dict[str, dict[str, Any]]
) -> None:
    """Give a DIMENSION statement's shape to the entity it names.

    A shape only ever *fills in*: an entity that carries its own array spec
    keeps it, because a type declaration and a DIMENSION statement for the
    same name is a conflict the compiler rejects and not something to
    silently resolve. A name that appears in no type declaration at all --
    legal under implicit typing, which is how the F77 sources write it --
    gets a minimal record, so the later passes see an array rather than
    nothing.
    """
    for name, shape in shapes.items():
        entity = entities.get(name)
        if entity is None:
            entities[name] = {
                "fortran_type": None,
                "kind": None,
                "dtype": "UNDECLARED",
                "intent": None,
                "optional": False,
                "parameter": False,
                "array_spec": shape["array_spec"],
                "dims": shape["dims"],
                "char_len": None,
                "init_expr": None,
                "line": None,
            }
            continue
        if not entity.get("dims"):
            entity["dims"] = shape["dims"]
            entity["array_spec"] = entity.get("array_spec") or shape["array_spec"]


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


_INT_LITERAL = re.compile(r"^[+-]?\d+(?:_\w+)?$")


def _fold_local_parameter_bounds(
    args: list[dict[str, Any]],
    result_dims: list[dict[str, Any]] | None,
    local_parameters: list[dict[str, Any]],
) -> dict[str, str]:
    """A dummy's bound that names a *local* integer parameter of the
    subprogram, folded to the parameter's literal value.

    CLUBB's ``w_term_ma_zt_lhs`` declares ``integer, parameter :: t_above =
    1, t_below = 2`` and then ``weights_zt2zm(ngrdcol, nzm, t_above:t_below)``.
    The name is the subprogram's own and reaches no consumer of the record:
    an f2py wrapper that spells it does not compile, and a sampler that
    reads it has no table for it. The value does reach every consumer, and
    it is the same array. Only a bare name with a literal integer
    initializer folds; an expression stays as written and is the wrapper's
    to refuse. Returns ``{"arg.bound": "name -> value"}`` for the record.
    """
    values: dict[str, str] = {}
    for parameter in local_parameters:
        init = str(parameter.get("init_expr") or "").strip()
        if parameter.get("dims") or not _INT_LITERAL.match(init):
            continue
        if str(parameter.get("dtype", "")).startswith("int"):
            values[str(parameter["name"]).lower()] = init.split("_")[0]
    folded: dict[str, str] = {}
    if not values:
        return folded
    targets: list[tuple[str, list[dict[str, Any]]]] = [
        (a["name"], a.get("dims") or []) for a in args
    ]
    if result_dims:
        targets.append(("<result>", result_dims))
    for owner, dims in targets:
        for axis, dim in enumerate(dims):
            for key in ("lb", "ub"):
                bound = dim.get(key)
                if bound is None:
                    continue
                name = str(bound).strip().lower()
                if name in values:
                    dim[key] = values[name]
                    folded[f"{owner}[{axis}].{key}"] = f"{name} -> {values[name]}"
    return folded


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
    # An interface body (``Function_Body``) has the same shape as a
    # subprogram and arrives here from ``_interfaces``; a function's body
    # under ``isinstance(sub, Function_Subprogram)`` alone was read as a
    # subroutine, and ``walk`` found no ``Subroutine_Stmt`` to index.
    is_function = isinstance(sub, (f03.Function_Subprogram, f03.Function_Body))
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
    kind = "function" if is_function else "subroutine"
    return _record_of(
        sub,
        name,
        kind,
        arg_names,
        result_name,
        prefixes,
        kind_map,
        module_state_names,
        module_sub_names,
        intent_overrides,
        prefix=prefix,
    )


def extract_program(
    prog: Any,
    name: str,
    kind_map: dict[str, str],
    module_state_names: set[str],
    module_sub_names: set[str],
) -> dict[str, Any]:
    """A main program's body as a subprogram record of kind ``program``: no
    dummies, no result, its declarations as locals, its calls as calls. The
    body has the shape of a subroutine's -- a specification part and an
    execution part -- and is read by the same code."""
    return _record_of(
        prog, name, "program", [], name, [], kind_map, module_state_names, module_sub_names, None
    )


def _record_of(
    sub: Any,
    name: str,
    kind: str,
    arg_names: list[str],
    result_name: str,
    prefixes: list[str],
    kind_map: dict[str, str],
    module_state_names: set[str],
    module_sub_names: set[str],
    intent_overrides: dict[str, str] | None,
    prefix: Any = None,
) -> dict[str, Any]:
    """The record of one scope with a subprogram's shape -- a subroutine, a
    function, an interface body, a program -- once its header is read."""
    is_function = kind == "function"

    # Every specification part, not the first: statement functions between
    # declaration blocks make fparser split one into several. ``walk`` takes
    # a list, so the readers below need no change.
    spec = [c for c in sub.children if isinstance(c, f03.Specification_Part)] or None
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
    # A standalone `dimension x(n)` says the shape of a name a type
    # declaration may not have mentioned at all. Applied after the loop
    # above, so an entity's own array spec still wins, and re-synced into
    # the parameter records the loop already appended.
    if spec is not None:
        apply_dimension_stmts(ent_info, dimension_stmt_shapes(spec))
        for parameter in local_parameters:
            parameter["dims"] = ent_info[parameter["name"]].get("dims")

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

    # A procedure dummy is a callable, never an array, and reads like a
    # scalar declaration. Four spellings say so, and the fourth says it only
    # through use: F77 had no PROCEDURE, so `real(8) :: rad` for a dummy that
    # the body then calls as `rad(v)` is one. Without this the reference
    # falls through to the subscript rule and `f(a)` is emitted `f[a - 1]`.
    procedure_names: set[str] = set()
    if spec is not None:
        for pstmt in walk(spec, f03.Procedure_Declaration_Stmt):
            procedure_names.update(str(n).lower() for n in walk(pstmt.children[-1], f03.Name))
        for estmt in walk(spec, f03.External_Stmt):
            procedure_names.update(str(n).lower() for n in walk(estmt.children[-1], f03.Name))
    for d in decls:
        if "EXTERNAL" in d["attrs"]:
            procedure_names.update(e["name"] for e in d["entities"])
    execution = [c for c in sub.children if isinstance(c, f03.Execution_Part)] or None
    if execution is not None:
        for ref in walk(
            execution, (f03.Part_Ref, f03.Function_Reference, f03.Structure_Constructor)
        ):
            referenced = str(ref.children[0]).lower()
            found = ent_info.get(referenced)
            if (
                referenced in arg_names
                and found is not None
                and not found.get("dims")
                and found.get("dtype") != "str"
                and ref.children[1] is not None
            ):
                procedure_names.add(referenced)
    for procedure_name in procedure_names:
        info = ent_info.setdefault(
            procedure_name,
            {
                "fortran_type": None,
                "kind": None,
                "dtype": "UNDECLARED",
                "intent": None,
                "optional": False,
                "parameter": False,
                "array_spec": None,
                "dims": None,
                "char_len": None,
                "init_expr": None,
                "line": None,
            },
        )
        info["procedure"] = True
        info["fortran_type"] = "PROCEDURE"
        info["dtype"] = "PROCEDURE"
    dummy_procedures = procedure_declarations(spec)

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
            **({"procedure": True} if info.get("procedure") else {}),
        }
        # A dummy procedure is not data: it carries the interface it was
        # declared with instead of a dtype, and every consumer that asks
        # "what does calling this mean" reads that name. ``procedure`` is its
        # dtype because every consumer of the record already switches on
        # that field, and a name declared PROCEDURE is passed in and never
        # assigned -- which is intent(in), stated by the declaration itself.
        if an in dummy_procedures:
            arg["procedure"] = True
            arg["interface"] = dummy_procedures[an]
            arg["dtype"] = "PROCEDURE"
            arg["intent"] = "IN"
        apply_intent_override(arg, name, (intent_overrides or {}).get(an))
        args.append(arg)

    unknown = sorted(set(intent_overrides or {}) - set(arg_names))
    if unknown:
        raise UnknownOverride(
            f"intent overrides for {name!r} name arguments it does not have: {unknown}"
        )

    result_dtype: str | None = None
    result_dims: list[dict[str, Any]] | None = None
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
        {
            "name": n,
            "dtype": i["dtype"],
            "array_spec": i["array_spec"],
            "dims": i.get("dims"),
            # ``real(r8) :: x = -2._r8``: the declaration's value, which the
            # prologue emits instead of its UB-guard zero.
            "init_expr": i.get("init_expr"),
        }
        for n, i in ent_info.items()
        if n not in arg_names and n != result_name and not i["parameter"]
    ]

    # Every execution part, for the same reason as ``spec`` above.
    exec_part = [c for c in sub.children if isinstance(c, f03.Execution_Part)] or None

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

    folded = _fold_local_parameter_bounds(args, result_dims, local_parameters)

    return {
        "name": name,
        "kind": kind,
        "prefixes": prefixes,
        "result": result_name if is_function else None,
        "result_dtype": result_dtype,
        "result_dims": result_dims,
        "line_span": list(node_span(sub)),
        "args": args,
        "folded_bounds": folded,
        "local_parameters": local_parameters,
        "locals": locals_,
        "present_calls": sorted(set(present_args)),
        "calls": sorted({c for c in calls if c != name}),
        "module_state_read": sorted(state_read),
        "module_state_written": sorted(state_written),
    }


def scopes_in(ast: Any, path: Path) -> list[tuple[str, str, Any]]:
    """``(name, kind, node)`` for every program unit a file defines, in source
    order: each ``module`` (kind ``module``), ``submodule`` (``submodule``)
    and ``program`` (``program``). A file of bare subprograms has none and
    borrows its stem, kind ``file``, with the whole tree as its scope.

    One file, several units: the synthetic slicing corpus puts a module and
    the program that uses it in one file, and a test tree puts two modules
    in one. Reading only the first left the rest unreachable -- a ``use``
    of the second module resolved to the file and came back with the
    first's record."""
    out: list[tuple[str, str, Any]] = []
    for mod in walk(ast, f03.Module):
        out.append((str(walk(mod, f03.Module_Stmt)[0].children[1]).lower(), "module", mod))
    for sub in walk(ast, f08.Submodule):
        out.append((str(walk(sub, f08.Submodule_Stmt)[0].children[1]).lower(), "submodule", sub))
    for prog in walk(ast, f03.Main_Program):
        stmts = walk(prog, f03.Program_Stmt)
        name = path.stem.lower().replace("_cpp", "")
        if stmts and stmts[0].children[1] is not None:
            name = str(stmts[0].children[1]).lower()
        out.append((name, "program", prog))
    if not out:
        out.append((path.stem.lower().replace("_cpp", ""), "file", ast))
    return out


def _scope_of(ast: Any, path: Path, name: str | None = None) -> tuple[str, Any, Any]:
    """``(module_name, specification_part, subprogram_scope)`` for one program
    unit of a source file: the one called ``name``, or the first when none
    is named (a file with one module, which is nearly every file).

    Handles the three shapes this frontend meets: a module, a main program, and
    a file of bare subprograms. The last two borrow the file stem for a name so
    that every Unit still has a stable ``uid``.
    """
    scopes = scopes_in(ast, path)
    chosen = scopes[0]
    if name is not None:
        chosen = next((sc for sc in scopes if sc[0] == name.lower()), None) or chosen
    scope_name, kind, node = chosen
    if kind == "file":
        return scope_name, None, ast
    spec = next((c for c in node.children if isinstance(c, f03.Specification_Part)), None)
    return scope_name, spec, node


def _only_names(statement: str) -> set[str]:
    """The local names a ``use ..., only:`` statement brings in."""
    match = re.search(r"only\s*:\s*(.*)$", statement, re.I | re.S)
    if not match:
        return set()
    return {
        item.split("=>", 1)[0].strip().lower() for item in match.group(1).split(",") if item.strip()
    }


def _component_allocate_bounds(scope: Any, visible: set[str]) -> dict[str, Any]:
    """Component -> the bounds an ``allocate (obj%comp(lb:ub, ...))`` in this
    module gave it, when a lower bound is not one.

    The same fact ``_module_allocate_bounds`` records for module state, for
    the components of a derived type: ``allocate (this%slatop(0:mxpft))`` in
    a type's init routine sets a lower bound the component's ``(:)`` does not
    carry, and every ``pftcon%slatop(itype)`` elsewhere then lands a slot off
    under the blanket one-based shift. A bound is usable only when a literal
    or a name every reader of the type can see; a lower bound that is an
    expression or a local of the allocating routine, or two allocations that
    disagree, leaves the component unrecorded and the declaration wins.
    Keyed by component name alone: which type ``this`` is would need the
    declaration table, and a module whose types share a component name with
    different bounds has not been seen.
    """
    found: dict[str, Any] = {}
    if scope is None:
        return found
    for allocation in walk(scope, f03.Allocation):
        target, shape = allocation.children[0], allocation.children[1]
        if not isinstance(target, f03.Data_Ref) or len(target.children) != 2:
            continue
        component = target.children[1]
        if isinstance(component, f03.Part_Ref):
            component = component.children[0]
        if not isinstance(component, f03.Name):
            continue
        name = str(component).lower()
        bounds: list[dict[str, str]] = []
        rebased = False
        unusable = False
        for spec in walk(shape, f03.Allocate_Shape_Spec):
            low, high = spec.children
            text = "1" if low is None else str(low).split("_")[0]
            # A literal, or a name every reader of the type can see, is a
            # bound every reference can shift by. A local of the allocating
            # routine (``begp``) is neither: that axis keeps the unit origin,
            # which is what every reference assumed before -- per axis,
            # because ``(begp:endp, 0:nlevmlcan)`` re-bases its second axis
            # whatever its first one is called.
            tokens = [t.lower() for t in re.findall(r"[A-Za-z_]\w*", text)]
            if text != "1" and any(t not in visible for t in tokens):
                text = "1"
            if text != "1":
                rebased = True
            bounds.append({"lb": text, "ub": str(high)})
        if not rebased:
            continue
        previous = found.get(name)
        if previous is not None and previous != bounds:
            unusable = True
        found[name] = CONFLICTING_BOUNDS if unusable else bounds
    return {k: v for k, v in found.items() if v != CONFLICTING_BOUNDS}


def _derived_types(
    mod_spec: Any, kind_map: dict[str, str], scope: Any = None, visible: set[str] | None = None
) -> dict[str, Any]:
    """``{type_name: {component: {dtype, dims, allocatable, pointer}}}``,
    plus ``allocated_dims`` on a component whose ALLOCATE re-bases it."""
    types: dict[str, Any] = {}
    if mod_spec is None:
        return types
    allocated = _component_allocate_bounds(scope, visible or set())
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
            # ``real, dimension(4) :: edge`` says the same as ``real :: edge(4)``,
            # and a component that carries its shape on the attribute rather
            # than on the entity was reported as a scalar -- not a refusal, a
            # wrong answer, and one every reader of the record inherits.
            attr_dims = None
            for spec in walk(attr_list, f03.Dimension_Component_Attr_Spec):
                attr_dims = dims_of(spec.children[1])
            for ent in walk(decl, f03.Component_Decl):
                cname = str(ent.children[0]).lower()
                comps[cname] = {
                    "dtype": dtype_of(base, kind, kind_map),
                    # The entity's own shape wins where it has one: Fortran
                    # lets ``dimension(4) :: a, b(7)`` give ``b`` a different
                    # one from the attribute's.
                    "dims": dims_of(ent.children[1]) or attr_dims,
                    "allocatable": "ALLOCATABLE" in attrs,
                    "pointer": "POINTER" in attrs,
                }
                if cname in allocated:
                    comps[cname]["allocated_dims"] = allocated[cname]
        if tname:
            types[tname] = comps
    return types


def _interfaces(
    mod_spec: Any, kind_map: dict[str, str], state_names: set[str], sub_names: set[str]
) -> dict[str, Any]:
    """``{name: subprogram record}`` for the bodies of interface blocks.

    An explicit interface for a procedure the module does not define -- the
    shape a procedure dummy has to have (``external :: func`` in a solver,
    ``interface / subroutine func(...)`` above it) -- and an ``abstract
    interface``, which says what calling a ``procedure(func) :: fcn`` dummy
    means: which arguments it takes, and which of them it writes. Without it
    a translation can only refuse the call, because it has no way to know
    that ``call fcn(n, x, fvec, iflag)`` writes ``fvec``. It is what a call
    through that dummy is bound against, and the record has the same shape
    as a subprogram's so the call renderer needs no second path.
    """
    interfaces: dict[str, Any] = {}
    if mod_spec is None:
        return interfaces
    for ib in walk(mod_spec, f03.Interface_Block):
        # ``interface gen`` names a generic: its specifics are procedures, see
        # _generics. ``abstract interface`` is spelled with the same slot.
        if any(
            st.children[0] is not None and str(st.children[0]).upper() != "ABSTRACT"
            for st in walk(ib, f03.Interface_Stmt)
        ):
            continue
        for body in walk(ib, (f03.Subroutine_Body, f03.Function_Body)):
            record = extract_subprogram(body, kind_map, state_names, sub_names)
            interfaces[record["name"]] = record
    return interfaces


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


def submodule_parent(ast: Any) -> str | None:
    """The parent module of a ``submodule (parent[:ancestor]) name`` file,
    or ``None`` for anything else. The ancestor, when spelled, is a
    submodule of the same parent and is not what a synthetic USE names."""
    submodules = walk(ast, f08.Submodule)
    if not submodules:
        return None
    statement = walk(submodules[0], f08.Submodule_Stmt)[0]
    return str(walk(statement.children[0], f03.Name)[0]).lower()


def extract(
    path: Path,
    *,
    kind_assumptions: dict[str, str] | None = None,
    intent_overrides: dict[str, Any] | None = None,
    buffer_out_arrays: str = "unsizable",
    scope: str | None = None,
) -> dict[str, Any]:
    """Full interface record for one program unit of a Fortran source file:
    the module, submodule or program called ``scope``, or the file's first
    when none is named. A ``program``'s body is recorded as a subprogram
    of kind ``program``, named after it, with no dummies.

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
    kind_dtypes = {k.lower(): v for k, v in (kind_assumptions or {}).items()}

    mod_name, mod_spec, sub_scope = _scope_of(ast, path, scope)

    module_parameters: list[dict[str, Any]] = []
    module_state: list[dict[str, Any]] = []
    if mod_spec is not None:
        # An interface body's dummies are declared inside the module's
        # specification part, and they are the interface's, not the module's:
        # ``subroutine func(p, x, val)`` in an ``interface`` block declares no
        # module variable ``x``, and a subprogram whose own dummy is ``x``
        # shadows nothing.
        within_interfaces = [node_span(ib) for ib in walk(mod_spec, f03.Interface_Block)]
        for d in collect_decls(mod_spec):
            if any(lo <= d["line"] <= hi for lo, hi in within_interfaces):
                continue
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

    if mod_spec is not None:
        by_name = {rec["name"]: rec for rec in module_parameters + module_state}
        shapes = dimension_stmt_shapes(mod_spec)
        apply_dimension_stmts(by_name, shapes)
        # A module-level DIMENSION for a name no declaration mentioned is
        # state, not a parameter: a PARAMETER has to be declared to have a
        # value, and this name has none.
        for added in shapes:
            if added not in {rec["name"] for rec in module_parameters + module_state}:
                record = by_name[added]
                record["name"] = added
                module_state.append(record)

    kind_map = resolve_kind_map(module_parameters)
    for k, v in kind_aliases_from_use(ast, kind_dtypes).items():
        kind_map.setdefault(k, v)
    # A bare ``use kinds_mod`` names nothing, so the alias pass above sees no
    # local name to bind: every public entity of that module is visible here
    # under its own name. Supplied kinds therefore also stand under their own
    # names, behind anything this file declares itself.
    for k, v in kind_dtypes.items():
        kind_map.setdefault(k, v)
    for rec in module_parameters + module_state:
        rec["dtype"] = dtype_of(rec["fortran_type"], rec["kind"], kind_map)

    state_names = {s["name"] for s in module_state}
    visible = state_names | {p["name"] for p in module_parameters}
    allocated_bounds = _module_allocate_bounds(sub_scope, state_names, visible)

    # Accessibility. A common convention is a bare `private` up top and an
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
    _infer_read_only_intents(subs, subprograms, sub_names)
    # After the inference, not before: an intent this pass just gave a
    # dummy is one this rule has to see.
    _mark_buffer_out_arrays(subprograms, every=buffer_out_arrays == "all")
    buffer_convention = buffer_out_arrays
    # A component's allocate bound may spell a use-imported name -- CLM's
    # ``(begc:endc, -nlevsno+1:nlevgrnd)`` -- which every module that reads
    # the component imports the same way; those count as visible for it.
    imported = {n for u in walk(sub_scope, f03.Use_Stmt) for n in _only_names(str(u))}
    _host_associate(subs, subprograms, sub_names, state_names)
    if isinstance(sub_scope, f03.Main_Program):
        # After the passes that walk ``subs`` and ``subprograms`` side by
        # side: the program body has no node in ``subs``.
        subprograms.insert(
            0, extract_program(sub_scope, mod_name, kind_map, state_names, sub_names)
        )
    generics = _generics(mod_spec)
    # A specific of a public generic is reachable through the generic even
    # when the module keeps the specific itself private (CLUBB's banded
    # solvers: ``public :: tridiag_lu_solve`` over three private specifics).
    # It is gate-visible, and ``public_via`` says what to call it through.
    via = {
        specific: generic
        for generic, specifics in generics.items()
        if is_public(generic)
        for specific in specifics
    }
    for record in subprograms:
        record["public"] = is_public(record["name"])
        if not record["public"] and record["name"] in via:
            record["public"] = True
            record["public_via"] = via[record["name"]]

    parent = submodule_parent(sub_scope) if isinstance(sub_scope, f08.Submodule) else None
    use_statements = [str(u) for u in walk(sub_scope, f03.Use_Stmt)]
    if parent:
        # The parent's entities reach a submodule by host association; the
        # translation reaches them as a companion, through this USE (#29).
        use_statements.insert(0, f"USE {parent}")

    return {
        "source_file": str(path),
        "module": mod_name,
        # A file of bare subprograms borrows its stem for a name; consumers
        # that emit `use <module>` need to know the name is borrowed. A
        # submodule is a module scope, reached through its parent's name.
        "is_module": isinstance(sub_scope, f03.Module) or parent is not None,
        **({"submodule_of": parent} if parent else {}),
        "kind_map": kind_map,
        "use_statements": use_statements,
        "module_parameters": module_parameters,
        "module_state": module_state,
        # Module state whose ALLOCATE gave it a lower bound its declaration
        # does not carry. Module-wide because the ALLOCATE is usually in the
        # init routine and the references are everywhere else.
        "module_allocate_bounds": allocated_bounds,
        "public": sorted(set(public_names)),
        "types": _derived_types(mod_spec, kind_map, scope=sub_scope, visible=visible | imported),
        "generics": generics,
        "interfaces": _interfaces(mod_spec, kind_map, state_names, sub_names),
        "buffer_convention": buffer_convention,
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
            # A buffer OUT is passed in and returned (the caller's storage),
            # so the caller reads the actual as well as writing it -- the
            # same rule the scope applies to its own subprograms (#38).
            "buffer_positions": [
                at for at, argument in enumerate(sub["args"]) if argument.get("buffer")
            ],
            # What the caller reads: IN and INOUT actuals, and a buffer OUT.
            # An INOUT actual is written *and* read; out_positions alone said
            # only the first.
            "read_positions": [
                at
                for at, argument in enumerate(sub["args"])
                if argument["intent"] in ("IN", "INOUT", "UNKNOWN") or argument.get("buffer")
            ],
            # So a keyword actual (``rcond = rcond``, CLUBB's band_solve) lands
            # on its own position and not on whichever comes next.
            "arg_names": [str(argument["name"]).lower() for argument in sub["args"]],
            # Handing the caller's own optional to one of these queries its
            # presence -- a read of it, as ``present()`` is.
            "optional_out_positions": [
                at
                for at, argument in enumerate(sub["args"])
                if argument.get("optional") and argument["intent"] == "OUT"
            ],
        }
    # The sibling's generics too: a call spells the generic (CLUBB's
    # ``zt2zm_api`` over grid_class's specifics), and a name the scope does
    # not know as a procedure it counts as a read of data. The entry's
    # writes are the union over the specifics -- which agree, in every
    # generic seen so far, on being functions with no OUT argument.
    for generic, specifics in (record.get("generics") or {}).items():
        known = [(s, table[s]) for s in specifics if s in table]
        if generic in table or not known:
            continue
        # Which specific a call reaches depends on its arity (and ranks the
        # scope does not resolve): the entry carries every specific with its
        # argument count, and the scope picks by the actuals it sees. A
        # union over specifics of different arity marked the wrong
        # positions -- CLUBB's tridiag_solve, zm2zt_api.
        signatures = {s: next(x for x in record["subprograms"] if x["name"] == s) for s, _ in known}
        arity = {s: len(sig["args"]) for s, sig in signatures.items()}
        required = {
            s: sum(1 for a in sig["args"] if not a.get("optional")) for s, sig in signatures.items()
        }
        table[generic] = {
            "kind": known[0][1]["kind"],
            "out_positions": sorted({at for _, entry in known for at in entry["out_positions"]}),
            "buffer_positions": sorted(
                {at for _, entry in known for at in entry.get("buffer_positions", [])}
            ),
            "specifics": [
                {
                    "name": s,
                    "args": arity[s],
                    "required": required[s],
                    "out_positions": entry["out_positions"],
                    "buffer_positions": entry.get("buffer_positions", []),
                    "read_positions": entry.get("read_positions", []),
                    "arg_names": entry.get("arg_names", []),
                    "optional_out_positions": entry.get("optional_out_positions", []),
                    # Two specifics of one arity (CLUBB's tridiag_solve
                    # with and without its optional rcond) are told apart
                    # by the ranks of their dummies.
                    "ranks": [len(a.get("dims") or []) for a in signatures[s]["args"]],
                }
                for s, entry in known
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


def _intent_inferable(dtype: Any) -> bool:
    """Numeric and logical scalars, the kinds an intent can be inferred for.

    A REAL whose kind name did not resolve is still a REAL scalar: excluding
    it left exactly the F77 sources this rule exists for -- the ones that
    spell their kind through a module parameter -- without an intent.
    """
    return dtype in ("float64", "float32", "int32", "int64", "bool") or str(dtype).startswith(
        "UNKNOWN_REAL_KIND("
    )


def _infer_write_only_intents(subs: list[Any], records: list[dict[str, Any]]) -> None:
    """Give an un-INTENTed F77 dummy the intent its use says it has.

    Two rules, both on scalar dummies the declaration left UNKNOWN:

    * assigned and never read -- bare name on the left, mentioned nowhere
      else -- is ``intent(out)``;
    * assigned AND read is ``intent(inout)``: Fortran passes by reference, so
      the update is the caller's to see.

    Either way, without the attribute the return convention leaves the dummy
    out of the signature and out of the return, and the value the routine
    computed is dropped on the floor. Scalars only: an array dummy mutates
    through the buffer it was passed and is not lost either way. A dummy that
    is only *passed on* to another procedure stays UNKNOWN -- its fate is the
    callee's, and is not decidable here.
    """
    by_name = {sub_name_of(s): s for s in subs}
    for record in records:
        candidates = [
            argument
            for argument in record["args"]
            if argument["intent"] == "UNKNOWN"
            and not argument.get("dims")
            and not argument.get("optional")
            and _intent_inferable(argument.get("dtype"))
        ]
        node = by_name.get(record["name"])
        if not candidates or node is None:
            continue
        # Every execution part, not the first: a subprogram with an internal
        # CONTAINS has more than one, and reading only the first scored the
        # rest as if the body never mentioned the dummy at all.
        exec_parts = [c for c in node.children if isinstance(c, f03.Execution_Part)]
        if not exec_parts:
            continue
        mentions: dict[str, int] = {}
        assigned: dict[str, int] = {}
        for exec_part in exec_parts:
            for name in walk(exec_part, f03.Name):
                key = str(name).lower()
                mentions[key] = mentions.get(key, 0) + 1
            for assignment in walk(exec_part, f03.Assignment_Stmt):
                target = assignment.children[0]
                if isinstance(target, f03.Name):
                    key = str(target).lower()
                    assigned[key] = assigned.get(key, 0) + 1
        for argument in candidates:
            written = assigned.get(argument["name"], 0)
            if not written:
                continue
            write_only = mentions.get(argument["name"], 0) == written
            argument["intent"] = "OUT" if write_only else "INOUT"
            argument["intent_inferred"] = True


def _mark_buffer_out_arrays(records: list[dict[str, Any]], every: bool = False) -> None:
    """Mark an intent(out) ARRAY dummy the callee cannot size as the caller's.

    ``every`` marks every intent(out) array so, sizable or not: Fortran
    passes the caller's storage in every case, and a callee that writes
    only ``t(1:n)`` of its ``t(nlev)`` leaves the rest as the caller had it
    -- which a fresh buffer returned whole cannot do (CLM-ml's
    ``tridiag_2eq`` into ``tair(p,:)``, the layers above the canopy). The
    default keeps the narrower convention the engine's emitted corpus was
    differenced under; a frontend that wants the faithful one asks.

    Fortran passes array storage, so an ``intent(out)`` array whose extent
    this subprogram cannot derive was allocated by the caller and has to stay
    a parameter -- returned like an INOUT rather than created here. It cannot
    derive one when the dummy is assumed-size (``a(*)``), or assumed-shape
    with neither a same-rank assumed-shape IN/INOUT donor to take the shape
    from nor explicit-bound IN arguments covering every dimension.

    Without the mark the return convention drops the argument from the
    signature and allocates a fresh array, so the caller's buffer is never
    written and every call is built with the wrong arity.
    """
    for record in records:
        args = record.get("args") or []
        for argument in args:
            if (
                argument.get("intent") != "OUT"
                or not argument.get("dims")
                or argument.get("optional")
                or argument.get("buffer")
            ):
                continue
            dims = argument["dims"]
            if every or any(d.get("assumed_size") for d in dims):
                argument["buffer"] = True
                continue
            if not all(d.get("ub") is None for d in dims):
                continue
            rank = len(dims)
            donor = any(
                other.get("intent") in ("IN", "INOUT")
                and len(other.get("dims") or []) == rank
                and any(d.get("ub") is None for d in other["dims"])
                for other in args
            )
            if donor:
                continue
            explicit = [
                other
                for other in args
                if other.get("intent") in ("IN", "INOUT")
                and other.get("dims")
                and all(d.get("ub") is not None for d in other["dims"])
            ]
            covered = all(
                any(axis < len(other["dims"]) for other in explicit) for axis in range(rank)
            )
            if not covered:
                argument["buffer"] = True


def _written_or_escaping(exec_part: Any, sub_names: set[str]) -> set[str]:
    """Names this execution part could change, read conservatively.

    A name is here if it is assigned to, if it controls a DO, if a READ fills
    it, if an ALLOCATE/DEALLOCATE/NULLIFY names it -- or if it is handed to
    something that might write it. An intrinsic never writes its argument and
    a subscript is not a call, so neither of those escapes; a reference to one
    of this file's own subprograms, and anything fparser could not resolve to
    either, does.
    """
    escaping: set[str] = set()

    def leftmost(node: Any) -> str | None:
        while node is not None and not isinstance(node, f03.Name):
            children = getattr(node, "children", None)
            if not children:
                return None
            node = children[0]
        return str(node).lower() if isinstance(node, f03.Name) else None

    for assignment in walk(exec_part, (f03.Assignment_Stmt, f03.Pointer_Assignment_Stmt)):
        name = leftmost(assignment.children[0])
        if name:
            escaping.add(name)
    for control in walk(exec_part, f03.Loop_Control):
        for name in walk(control, f03.Name):
            escaping.add(str(name).lower())
            break
    for statement in walk(
        exec_part,
        (f03.Read_Stmt, f03.Allocate_Stmt, f03.Deallocate_Stmt, f03.Nullify_Stmt),
    ):
        for name in walk(statement, f03.Name):
            escaping.add(str(name).lower())
    for call in walk(exec_part, f03.Call_Stmt):
        arguments = call.children[1]
        for name in walk(arguments, f03.Name) if arguments is not None else []:
            escaping.add(str(name).lower())
    for reference in walk(exec_part, (f03.Part_Ref, f03.Structure_Constructor)):
        base = reference.children[0]
        if not isinstance(base, f03.Name):
            continue
        if isinstance(reference, f03.Part_Ref) and str(base).lower() not in sub_names:
            continue  # a subscript, or a reference to something with no body here
        for name in walk(reference.children[1], f03.Name):
            escaping.add(str(name).lower())
    return escaping


def _infer_read_only_intents(
    subs: list[Any], records: list[dict[str, Any]], sub_names: set[str]
) -> None:
    """Give a dummy the body never changes the intent its use says it has.

    The mirror of ``_infer_write_only_intents``, and the commoner half: a
    dummy declared without INTENT that is only ever read is ``intent(in)``.
    Left ``UNKNOWN`` it is not a wrong answer, but it is an unanswerable one
    for anything downstream that has to know whether the value after the call
    is an output -- the differential gate refuses the whole subprogram over
    it, so one undeclared argument costs the routine its evidence.
    """
    by_name = {sub_name_of(s): s for s in subs}
    for record in records:
        candidates = [
            argument
            for argument in record["args"]
            if argument["intent"] == "UNKNOWN"
            and not argument.get("optional")
            and not argument.get("procedure")
        ]
        node = by_name.get(record["name"])
        if not candidates or node is None:
            continue
        exec_part = next((c for c in node.children if isinstance(c, f03.Execution_Part)), None)
        if exec_part is None:
            continue
        escaping = _written_or_escaping(exec_part, sub_names)
        for argument in candidates:
            if argument["name"] not in escaping:
                argument["intent"] = "IN"
                argument["intent_inferred"] = "read-only"


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
        exec_part = [c for c in s.children if isinstance(c, f03.Execution_Part)] or None
        if exec_part is None:
            continue
        used = set(names_in(exec_part))
        own = {a["name"] for a in rec["args"]}
        own |= {loc["name"] for loc in rec["locals"]}
        own |= {p["name"] for p in rec.get("local_parameters", [])}
        # A function's result variable is its own, not a local: when the host
        # declares the same name, the result shadows it, and reporting it as
        # host-associated passed it as a trailing actual the Fortran does not
        # have (translator #43, shared by both sides until relayed).
        if rec.get("result"):
            own.add(str(rec["result"]).lower())
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


CONFLICTING_BOUNDS = "conflicting"
"""A module allocatable allocated with lower bounds that do not agree, or
with one no other subprogram can evaluate. A reference to it cannot be
shifted statically, and saying so is the only honest answer."""


def _module_allocate_bounds(scope: Any, state_names: set[str], visible: set[str]) -> dict[str, Any]:
    """Module state -> the bounds its ALLOCATE gave it, module-wide.

    ``allocate(x(ntot, 0:nspec))`` sets a lower bound the declaration does
    not carry, and the ALLOCATE is typically in an init routine while the
    references are in every other one. Without this each of those gets the
    blanket one-based shift and lands a slot off.

    A bound is usable only if every subprogram can evaluate it: a literal,
    or a name visible module-wide. A local of the allocating subprogram, an
    expression, or two ALLOCATEs that disagree leave the name marked
    conflicting, and a reference to it refuses rather than shifting wrongly.
    """
    found: dict[str, Any] = {}
    for allocation in walk(scope, f03.Allocation):
        target, shape = allocation.children[0], allocation.children[1]
        if not isinstance(target, f03.Name):
            continue
        name = str(target).lower()
        if name not in state_names:
            continue
        bounds: list[dict[str, str]] = []
        rebased = False
        unusable = False
        for spec in walk(shape, f03.Allocate_Shape_Spec):
            low, high = spec.children
            text = "1" if low is None else str(low).split("_")[0]
            if text != "1":
                rebased = True
                if not re.fullmatch(r"-?\d+", text) and text.lower() not in visible:
                    unusable = True
            bounds.append({"lb": text, "ub": str(high)})
        if not rebased:
            continue
        previous = found.get(name)
        disagrees = previous is not None and (
            previous == CONFLICTING_BOUNDS
            or [d["lb"] for d in previous] != [d["lb"] for d in bounds]
        )
        found[name] = CONFLICTING_BOUNDS if unusable or disagrees else bounds
    return found
