"""Resolving use-imported constants across source files.

Migrated from the source pipeline's ``pipeline/resolve_use.py``. A module
under translation imports named constants -- ``cpair``, ``epsilo`` -- from
modules that are not themselves being translated. This finds their
initializers in the real sources, follows them transitively, and returns them
in dependency order as ``Expr`` trees.

The original wrote two files here, a Fortran stand-in module and a Python
constants file. Both moved out: emitting source is not a ``Frontend``'s job,
and ``expr.render`` is what keeps them agreeing.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03, parse, walk
from recast.fortran.expr import (
    Expr,
    UnsupportedExpression,
    build,
    names_used,
    real_kind_of,
    substitute,
)


class UnresolvedConstant(RecastError):
    """A use-imported name whose initializer is in none of the given sources."""


def harvest(path: Path) -> dict[str, tuple[Any, int | None, str | None, str | None]]:
    """``name -> (initializer node, line, declared base type, kind spelling)``
    for module-level initialized entities; the base type is ``real``, ``int``
    ``str`` or ``None``, the kind spelling ``r8`` / ``8`` / ``None`` for the
    default.

    Covers parameters and initialized ``save``/``protected`` variables alike: a
    constant that a physics module reads is a constant whether or not the
    author spelled ``parameter``. A stub file may hold several modules, so all
    of them are harvested.
    """
    ast = parse(path)
    out: dict[str, tuple[Any, int | None, str | None, str | None]] = {}
    for mod in walk(ast, f03.Module):
        spec = next((c for c in mod.children if isinstance(c, f03.Specification_Part)), None)
        if spec is None:
            continue
        for decl in walk(spec, f03.Type_Declaration_Stmt):
            line = None
            for n in walk(decl):
                item = getattr(n, "item", None)
                if item is not None and getattr(item, "span", None):
                    line = item.span[0]
                    break
            # The declared base type, which is what says whether ``rd / rv``
            # is a real quotient: the fold cannot tell from two names.
            type_text = str(decl.children[0])
            base = type_text.split("(")[0].strip().upper()
            declared = {
                "REAL": "real",
                "DOUBLE PRECISION": "real",
                "INTEGER": "int",
                "COMPLEX": "complex",
                "CHARACTER": "str",
            }.get(base)
            kind_spelling_ = kind_spelling(type_text, base)
            for ent in walk(decl, f03.Entity_Decl):
                if ent.children[3] is not None:
                    initializer = ent.children[3].children[1]
                    out[str(ent.children[0]).lower()] = (
                        initializer,
                        line,
                        declared,
                        kind_spelling_,
                    )
    return out


def kind_spelling(type_text: str, base: str) -> str | None:
    """``real(kind=r8)`` -> ``r8``; ``real(8)`` -> ``8``; ``double precision``
    -> ``8``; a bare ``real`` -> ``None``, the default kind."""
    if base == "DOUBLE PRECISION":
        return "8"
    if "(" not in type_text:
        return None
    inside = type_text[type_text.index("(") + 1 : type_text.rindex(")")]
    return inside.split("=", 1)[-1].strip().lower() or None


USE_RENAME = re.compile(
    r"^\s*use\s*(?:,\s*\w+\s*)?(?:::)?\s*(?P<module>[A-Za-z_]\w*)\s*,\s*only\s*:(?P<only>.*)$",
    re.I | re.M,
)


def _kind_tables(
    table: dict[str, tuple[Any, int | None, str | None, str | None]],
    origin: dict[str, Path],
    sources: list[Path],
    kind_assumptions: dict[str, str] | None,
) -> dict[Path, dict[str, str]]:
    """Kind-parameter names to dtypes, per source file.

    What the sources define (``integer, parameter :: r8 =
    selected_real_kind(12)``) is read by the interface's kind rules, iterated
    because one kind is often spelled by another; the extension's
    ``kind_assumptions`` cover what no source defines. A file that renames a
    kind on import (``use precision_mod, only: r8 => wp_r8``) sees it under
    its own name, which is why the table is per file: ``1.0_r8`` in that
    file is ``wp_r8``'s width and nobody else's ``r8``.
    """
    from recast.fortran.interface import resolve_kind_map

    shared = {k.lower(): v for k, v in (kind_assumptions or {}).items()}
    params = [
        {"name": name, "init_expr": str(node)}
        for name, (node, _line, declared, _kind) in table.items()
        if declared == "int"
    ]
    shared.update(resolve_kind_map(params))
    tables: dict[Path, dict[str, str]] = {}
    for path in sources:
        own = dict(shared)
        try:
            text = path.read_text(errors="replace")
        except OSError:
            tables[path] = own
            continue
        for match in USE_RENAME.finditer(text):
            for item in match.group("only").split(","):
                if "=>" not in item:
                    continue
                local, remote = (x.strip().lower() for x in item.split("=>", 1))
                if remote in shared and local not in (kind_assumptions or {}):
                    own[local] = shared[remote]
        tables[path] = own
    return tables


def declared_dtype(
    declared: str | None, kind_spelling: str | None, kinds: dict[str, str]
) -> str | None:
    """The dtype a declaration names, ``None`` when its kind is not known.

    A complex is two reals of one kind and reads as ``complex128`` or
    ``complex64`` by that kind; the default complex is single, like the
    default real."""
    if declared in ("int", "str"):
        # No real width to place: an integer's storage is exact, and a
        # character constant is its text.
        return declared
    if declared not in ("real", "complex"):
        return None
    if kind_spelling is None:
        parts: str | None = "float32"  # default real
    else:
        try:
            parts = real_kind_of(kind_spelling, kinds)
        except UnsupportedExpression:
            return None
    if declared == "complex":
        return {"float64": "complex128", "float32": "complex64"}.get(parts or "")
    return parts


def resolve(
    symbols: list[str], sources: list[Path], kind_assumptions: dict[str, str] | None = None
) -> list[dict[str, Any]]:
    """Resolve ``symbols`` and everything they depend on, dependency-first.

    Returns one record per constant: ``name``, its ``expr`` tree (every node
    typed by kind), the ``source`` it was found in, its ``line``, its
    declared base type (``dtype``: ``real`` / ``int`` / ``str``), the dtype
    of its declared kind (``kind_dtype``: ``float64`` / ``float32`` / ``int``
    / ``str`` / ``None``), and whether it was ``requested`` or pulled in transitively.
    Order is safe to emit or evaluate top to bottom. ``kind_assumptions``
    names the kinds the sources use but do not define.

    Raises ``UnresolvedConstant`` rather than skipping. A missing physical
    constant that silently becomes undefined downstream is far more expensive to
    diagnose than a failure here that names it.
    """
    table: dict[str, tuple[Any, int | None, str | None, str | None]] = {}
    origin: dict[str, Path] = {}
    for path in sources:
        for name, rec in harvest(path).items():
            table[name] = rec
            origin[name] = path
    kinds_of = _kind_tables(table, origin, sources, kind_assumptions)
    declared_kinds = {
        name: declared_dtype(declared, kind_spelling, kinds_of[origin[name]])
        for name, (_node, _line, declared, kind_spelling) in table.items()
    }

    requested = [s.strip().lower() for s in symbols if s.strip()]
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def need(name: str) -> None:
        if name in seen:
            return
        if name not in table:
            raise UnresolvedConstant(f"no initializer for {name!r} in {[str(s) for s in sources]}")
        seen.add(name)
        node, line, declared, _kind_spelling = table[name]
        kind_dtype = declared_kinds[name]
        # The one legal self-reference, a kind inquiry on the constant being
        # declared, stands for its kind alone; see ``expr.substitute``.
        expr: Expr = substitute(
            build(node, kinds_of[origin[name]], declared_kinds),
            name,
            Expr("real", "1.0", dtype=kind_dtype),
        )
        for dep in names_used(expr):
            need(dep)
        ordered.append(
            {
                "name": name,
                "expr": expr,
                "dtype": declared,
                "kind_dtype": kind_dtype,
                "source": str(origin[name]),
                "line": line,
                "requested": name in requested,
            }
        )

    for symbol in requested:
        need(symbol)
    return ordered
