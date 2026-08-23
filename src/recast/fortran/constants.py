"""Parameters and inline literals, classified but not yet spelled.

Migrated from CESM-language-translator ``pipeline/extract_constants.py``, and
split where the ``Frontend`` contract says it has to be. The original did two
jobs: it worked out what every parameter and every inline numeric literal was,
and it wrote a NumPy constants module. Only the first job is analysis.

So ``classify_init`` returns a decision -- integer, real, logical, reference,
array, an expression over earlier constants, or an honest refusal -- and never
a line of target-language source. The rendering it used to do belongs to the
Transform that has a target language, and lives with it.

The zero-literal rule survives the split intact: every non-structural numeric
literal in an execution part is hoisted to a deterministic name, so that a
translated routine contains no bare magic numbers and every constant has one
definition that both sides of a differential check can be pointed at.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from recast.fortran._parse import f03, parse, walk
from recast.fortran.expr import _normalize_real
from recast.fortran.interface import _scope_of

WHITELIST_INT = frozenset({"0", "1", "2"})
WHITELIST_REAL = frozenset({0.0, 1.0, 0.5})
"""Literals that stay inline.

These are structural -- loop bounds, indices, halves -- not physical. Hoisting
them would bury the arithmetic in names without making anything checkable.
"""

_ARRAY_MODULE_RE = re.compile(r"[0-9eE.+\-*/\s]*\[[0-9eE.,+\-\s]+\][0-9eE.+\-*/\s]*")
_ARRAY_LOCAL_RE = re.compile(r"\[[0-9eE.,+\-\s]+\]")
_KIND_SUFFIX_RE = re.compile(r"_\w+")
_TOKEN_RE = re.compile(r"[A-Za-z_]\w*|\d+\.?\d*(?:[edED][+-]?\d+)?(?:_\w+)?|\*\*|[()+\-*/]")


def strip_kind(text: str) -> str:
    """``7.90298_r8`` -> ``7.90298``; ``10_i8`` -> ``10``."""
    return text.split("_")[0]


def canon_name(text: str, is_real: bool) -> str:
    """Deterministic constant name for a literal.

    Derived from the literal's own digits, so the same number hoisted from two
    different routines lands on one name and one definition. ``F_`` for reals,
    ``I_`` for integers, ``P`` for the point, ``M`` for a minus.
    """
    t = strip_kind(text).lower().replace("d", "e")
    t = t.replace(".", "P").replace("+", "").replace("-", "M").replace("e", "E")
    return f"{'F' if is_real else 'I'}_{t}"


def is_whitelisted(text: str, is_real: bool) -> bool:
    t = strip_kind(text)
    if is_real:
        try:
            return float(t.lower().replace("d", "e")) in WHITELIST_REAL
        except ValueError:
            return False
    return t in WHITELIST_INT


def literals_with_lines(
    node: Any, line: int | None = None, out: list[tuple[str, bool, int | None]] | None = None
) -> list[tuple[str, bool, int | None]]:
    """Depth-first ``(literal_text, is_real, nearest_line)`` under a node.

    Descends into plain tuples and lists as well as node children: some
    fparser2 nodes -- ``Loop_Control`` among them -- hold their children in bare
    Python containers, and a traversal that only follows ``.children`` silently
    misses every do-bound literal in the file.
    """
    if out is None:
        out = []
    if isinstance(node, list | tuple):
        for ch in node:
            if ch is not None and not isinstance(ch, str):
                literals_with_lines(ch, line, out)
        return out
    item = getattr(node, "item", None)
    if item is not None and getattr(item, "span", None):
        line = item.span[0]
    if isinstance(node, f03.Real_Literal_Constant):
        out.append((str(node), True, line))
        return out
    if isinstance(node, f03.Int_Literal_Constant):
        out.append((str(node), False, line))
        return out
    children = getattr(node, "children", None)
    if children is None or not isinstance(children, list | tuple):
        return out
    for ch in children:
        if ch is not None and not isinstance(ch, str):
            literals_with_lines(ch, line, out)
    return out


def classify_init(init_expr: str, known_names: set[str]) -> tuple[str, Any]:
    """Classify a parameter initializer. Returns ``(kind, payload)``.

    ``int`` / ``int64`` -> a Python ``int``.
    ``real``            -> normalized decimal text, digits untouched.
    ``logical``         -> a Python ``bool``.
    ``ref``             -> the lower-cased name of another constant.
    ``expr``            -> tokens over earlier constants, in source order.
    ``skip``            -> a reason, in words.

    ``skip`` is a result, not an error. A kind parameter has no runtime value to
    carry, and an initializer this frontend cannot evaluate must be reported as
    unresolved rather than approximated.
    """
    e = init_expr.strip()

    m = re.search(r"selected_real_kind\s*\(\s*(\d+)", e, re.I)
    if m:
        # The kind value itself is referenceable at runtime (`if (kind /= r8)`).
        # gfortran: selected_real_kind(p >= 10) -> 8, otherwise 4.
        return "int", 8 if int(m.group(1)) >= 10 else 4
    m = re.search(r"selected_int_kind\s*\(\s*(\d+)", e, re.I)
    if m:
        # The smallest kind holding 10**N. gfortran: N <= 4 -> 2, N <= 9 -> 4,
        # otherwise 8; a value, because the source can compare against it.
        n = int(m.group(1))
        return "int", 2 if n <= 4 else (4 if n <= 9 else 8)
    if re.search(r"kind\s*\(", e, re.I):
        return "skip", "kind parameter (compile-time only)"

    m = re.fullmatch(r"int\s*\(\s*z'([0-9a-f]+)'\s*,\s*\w+\s*\)", e, re.I)
    if m:  # BOZ literal: int(Z'...', i8)
        return "int64", int(m.group(1), 16)

    compact = e.replace(" ", "")
    if compact.lower() in (".true.", ".false."):
        return "logical", compact.lower() == ".true."
    if re.fullmatch(r"-?\d+", compact):
        return "int", int(compact)
    m = re.fullmatch(r"(-?)(\d+\.?\d*(?:[edED][+-]?\d+)?)(_\w+)?", compact)
    if m and ("." in compact or "e" in compact.lower() or "d" in compact.lower()):
        return "real", m.group(1) + _normalize_real(m.group(2))
    if compact.lower() in known_names:
        return "ref", compact.lower()

    # A constant expression over earlier parameters. Re-emitted token-wise so
    # the target language evaluates the same arithmetic the compiler folded,
    # rather than this stage folding it at a different precision.
    toks = _TOKEN_RE.findall(e)
    if toks and "".join(toks).replace(" ", "") == compact:
        out: list[dict[str, str]] = []
        for t in toks:
            if re.match(r"[A-Za-z_]", t):
                if t.lower() not in known_names:
                    return "skip", f"unknown name {t!r} in expression: {e}"
                out.append({"t": "ref", "v": t.lower()})
            elif re.match(r"\d", t):
                base = _normalize_real(t)
                is_real = "." in base or "e" in base
                out.append({"t": "real" if is_real else "int", "v": base})
            else:
                out.append({"t": "op", "v": t})
        return "expr", out

    return "skip", f"unevaluated expression: {e}"


def _array_literal(init: str, module_level: bool) -> str | None:
    """Kind-stripped text of a pure-literal array constructor, or ``None``.

    Lookup tables are declared as ``dnu(16) = [ ... ]``. Stripping the kind
    suffix textually is exact when the constructor holds nothing but literals,
    which is what the pattern check enforces before the strip is trusted.
    """
    txt = _KIND_SUFFIX_RE.sub("", init).replace("D", "E").replace("d", "e")
    pattern = _ARRAY_LOCAL_RE if module_level is False else _ARRAY_MODULE_RE
    return txt if pattern.fullmatch(txt) else None


def _decl_line(decl: Any) -> int | None:
    for n in walk(decl):
        item = getattr(n, "item", None)
        if item is not None and getattr(item, "span", None):
            return int(item.span[0])
    return None


def extract(path: Path, *, extern_names: set[str] | None = None) -> dict[str, Any]:
    """Every parameter and hoisted literal in one Fortran source file.

    ``extern_names`` are constants this module use-imports and that a sibling
    translation already defines; they count as known when classifying an
    expression. The command-line ancestor learned them by ``exec``-ing a
    generated Python file, which made the frontend's answer depend on a
    previously generated artifact being present and importable.
    """
    ast = parse(path)
    mod_name, mod_spec, sub_scope = _scope_of(ast, path)

    known: set[str] = set(extern_names or ())
    module_parameters: list[dict[str, Any]] = []

    for decl in walk(mod_spec, f03.Type_Declaration_Stmt) if mod_spec is not None else []:
        type_spec, attr_list, _ = decl.children
        attrs = [str(a).upper() for a in (attr_list.children if attr_list else [])]
        if "PARAMETER" not in attrs:
            continue
        base = (
            str(type_spec.children[0]).upper()
            if isinstance(type_spec, f03.Intrinsic_Type_Spec)
            else "?"
        )
        line = _decl_line(decl)
        for ent in walk(decl, f03.Entity_Decl):
            name = str(ent.children[0]).lower()
            init = str(ent.children[3].children[1]) if ent.children[3] is not None else None
            rec: dict[str, Any] = {
                "name": name,
                "base_type": base,
                "init_expr": init,
                "line": line,
            }
            array_text = (
                _array_literal(init, module_level=True)
                if init and "[" in init and ent.children[1] is not None
                else None
            )
            if array_text is not None:
                rec["kind"], rec["payload"] = "array", array_text
            else:
                rec["kind"], rec["payload"] = classify_init(init or "", known)
            module_parameters.append(rec)
            known.add(name)

    local_parameters: list[dict[str, Any]] = []
    literal_map: dict[str, dict[str, str]] = {}
    hoisted: dict[str, dict[str, Any]] = {}

    def hoist(text: str, is_real: bool, location: str, subprogram: str) -> None:
        cname = canon_name(text, is_real)
        entry = hoisted.setdefault(
            cname,
            {
                "value": _normalize_real(text) if is_real else strip_kind(text),
                "is_real": is_real,
                "locations": [],
            },
        )
        entry["locations"].append(location)
        literal_map[subprogram][text] = cname

    for sub in walk(sub_scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
        st = walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
        sname = str(st.children[1]).lower()
        literal_map[sname] = {}

        spec = next((c for c in sub.children if isinstance(c, f03.Specification_Part)), None)
        if spec is not None:
            # F77 separate form: PARAMETER (NAME = expr, ...)
            for pstmt in walk(spec, f03.Parameter_Stmt):
                for pdef in walk(pstmt, f03.Named_Constant_Def):
                    pname = str(pdef.children[0]).lower()
                    init = str(pdef.children[1])
                    kind, payload = classify_init(init, known)
                    local_parameters.append(
                        {
                            "subprogram": sname,
                            "name": pname,
                            "const": f"{sname.upper()}__{pname.upper()}",
                            "form": "parameter_stmt",
                            "init_expr": init,
                            "kind": kind,
                            "payload": payload,
                        }
                    )
            for decl in walk(spec, f03.Type_Declaration_Stmt):
                _, attr_list, _ = decl.children
                attrs = [str(a).upper() for a in (attr_list.children if attr_list else [])]
                if "PARAMETER" not in attrs:
                    continue
                for ent in walk(decl, f03.Entity_Decl):
                    pname = str(ent.children[0]).lower()
                    init = str(ent.children[3].children[1]) if ent.children[3] is not None else None
                    const = f"{sname.upper()}__{pname.upper()}"
                    array_text = (
                        _array_literal(init, module_level=False)
                        if init and init.strip().startswith("[") and ent.children[1] is not None
                        else None
                    )
                    if array_text is not None:
                        kind, payload = "array", array_text
                    else:
                        kind, payload = classify_init(init or "", known)
                    local_parameters.append(
                        {
                            "subprogram": sname,
                            "name": pname,
                            "const": const,
                            "form": "declaration",
                            "init_expr": init,
                            "kind": kind,
                            "payload": payload,
                        }
                    )
                    literal_map[sname][f"@param:{pname}"] = const

            # Declaration bounds take part in the zero-literal rule too: the
            # prologue that allocates capeten(pcols, 5) must name the 5.
            for shp in walk(spec, f03.Explicit_Shape_Spec):
                for text, is_real, _line in literals_with_lines(shp):
                    if is_real or is_whitelisted(text, is_real):
                        continue
                    hoist(text, False, f"{sname}:decl", sname)

        exec_part = next((c for c in sub.children if isinstance(c, f03.Execution_Part)), None)
        if exec_part is None:
            continue
        for text, is_real, line in literals_with_lines(exec_part):
            if is_whitelisted(text, is_real):
                continue
            hoist(text, is_real, f"{sname}:{line}", sname)

    return {
        "source_file": str(path),
        "module": mod_name,
        "module_parameters": module_parameters,
        "local_parameters": local_parameters,
        "hoisted_literals": hoisted,
        "literal_map": literal_map,
    }
