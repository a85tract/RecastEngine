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

from pathlib import Path
from typing import Any

from recast.errors import RecastError
from recast.fortran._parse import f03, parse, walk
from recast.fortran.expr import Expr, build, names_used, substitute


class UnresolvedConstant(RecastError):
    """A use-imported name whose initializer is in none of the given sources."""


def harvest(path: Path) -> dict[str, tuple[Any, int | None, str | None]]:
    """``name -> (initializer node, line, declared base type)`` for module-level
    initialized entities; the base type is ``real``, ``int`` or ``None``.

    Covers parameters and initialized ``save``/``protected`` variables alike: a
    constant that a physics module reads is a constant whether or not the
    author spelled ``parameter``. A stub file may hold several modules, so all
    of them are harvested.
    """
    ast = parse(path)
    out: dict[str, tuple[Any, int | None, str | None]] = {}
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
            base = str(decl.children[0]).split("(")[0].strip().upper()
            declared = {"REAL": "real", "DOUBLE PRECISION": "real", "INTEGER": "int"}.get(base)
            for ent in walk(decl, f03.Entity_Decl):
                if ent.children[3] is not None:
                    initializer = ent.children[3].children[1]
                    out[str(ent.children[0]).lower()] = (initializer, line, declared)
    return out


def resolve(symbols: list[str], sources: list[Path]) -> list[dict[str, Any]]:
    """Resolve ``symbols`` and everything they depend on, dependency-first.

    Returns one record per constant: ``name``, its ``expr`` tree, the ``source``
    it was found in, its ``line``, and whether it was ``requested`` or pulled in
    transitively. Order is safe to emit or evaluate top to bottom.

    Raises ``UnresolvedConstant`` rather than skipping. A missing physical
    constant that silently becomes undefined downstream is far more expensive to
    diagnose than a failure here that names it.
    """
    table: dict[str, tuple[Any, int | None, str | None]] = {}
    origin: dict[str, Path] = {}
    for path in sources:
        for name, rec in harvest(path).items():
            table[name] = rec
            origin[name] = path

    requested = [s.strip().lower() for s in symbols if s.strip()]
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    def need(name: str) -> None:
        if name in seen:
            return
        if name not in table:
            raise UnresolvedConstant(f"no initializer for {name!r} in {[str(s) for s in sources]}")
        seen.add(name)
        node, line, declared = table[name]
        # The one legal self-reference, a kind inquiry on the constant being
        # declared, stands for its kind alone; see ``expr.substitute``.
        expr: Expr = substitute(build(node), name, Expr("real", "1.0"))
        for dep in names_used(expr):
            need(dep)
        ordered.append(
            {
                "name": name,
                "expr": expr,
                "dtype": declared,
                "source": str(origin[name]),
                "line": line,
                "requested": name in requested,
            }
        )

    for symbol in requested:
        need(symbol)
    return ordered
