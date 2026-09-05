"""Questions a stage asks of the source *tree* rather than of one unit.

The frontend analyses one file at a time. Some facts a later stage needs
are spread over the tree: which file defines module ``X``; what integer a
parameter another module declares evaluates to; which names a unit
use-imports from a given set of modules. This module answers those, over
the tree the operator pointed the run at, and nothing else -- no domain
table lives here. Which modules count as *constants modules* is the
caller's to say; a domain extension says it from its conventions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

__all__ = [
    "MODULE_DEFINITION",
    "SUFFIXES",
    "integer_parameters",
    "module_sources",
    "named_extents",
    "parameter_names",
    "use_imports",
]

SUFFIXES = frozenset({".f90", ".f95", ".f03", ".f08", ".f"})
MODULE_DEFINITION = re.compile(r"^\s*module\s+(?!procedure\b)(\w+)", re.IGNORECASE | re.MULTILINE)
USE = re.compile(
    r"USE\b\s*(?:,\s*\w+\s*)?(?:::)?\s*(?P<module>\w+)(?:\s*,\s*ONLY\s*:\s*(?P<only>.*))?$",
    re.IGNORECASE | re.DOTALL,
)


def sources(root: Path) -> list[Path]:
    """Every Fortran file under ``root``, in a fixed order."""
    return [p for p in sorted(root.rglob("*")) if p.suffix.lower() in SUFFIXES and p.is_file()]


def module_sources(root: Path, modules: frozenset[str]) -> list[Path]:
    """The files under ``root`` that define one of ``modules`` (lower-case names)."""
    found: list[Path] = []
    for path in sources(root):
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        if any(m.group(1).lower() in modules for m in MODULE_DEFINITION.finditer(text)):
            found.append(path)
    return found


def use_imports(
    record: dict[str, Any], modules: frozenset[str], skip: frozenset[str] = frozenset()
) -> list[str]:
    """Remote names this unit use-imports, with an ONLY list, from ``modules``.

    Module scope and every subprogram scope alike. A bare ``use`` of one of
    the modules names nothing and imports everything; that is not resolved
    here (there is no list to resolve) and the rules refuse the references,
    which is the right answer for a request nobody spelled out. ``skip``
    names to leave out -- kind parameters, typically, which are assumptions
    rather than constants.
    """
    statements: list[str] = list(record.get("use_statements") or [])
    for sub in record.get("subprograms", ()):
        statements.extend(sub.get("use_statements") or [])
    names: list[str] = []
    for statement in statements:
        match = USE.match(statement.strip())
        if not match or match.group("module").lower() not in modules or not match.group("only"):
            continue
        for item in match.group("only").split(","):
            remote = item.split("=>", 1)[-1].strip().lower()
            if remote and remote not in skip and remote not in names:
                names.append(remote)
    return names


def named_extents(subprograms: list[dict[str, Any]]) -> list[str]:
    """Names that spell a declared extent of a dummy array and are not
    themselves dummies -- ``a1(nlev)`` -- so a parameter the file
    use-imports. The oracle's wrapper and the gate's sampler both need its
    value."""
    names: list[str] = []
    for sub in subprograms:
        dummies = {a["name"].lower() for a in sub["args"]}
        for argument in sub["args"]:
            for dim in argument.get("dims") or ():
                for bound in (dim.get("lb"), dim.get("ub")):
                    for token in re.findall(r"[A-Za-z_]\w*", str(bound or "")):
                        lowered = token.lower()
                        if lowered not in dummies and lowered not in names:
                            names.append(lowered)
    return names


_PARAMETERS: dict[Path, set[str]] = {}


def parameter_names(files: list[Path], kind_assumptions: dict[str, str] | None = None) -> set[str]:
    """The names declared ``parameter`` in these modules, by the engine's
    own interface record; cached per file."""
    from recast.fortran import interface

    names: set[str] = set()
    for path in files:
        if path not in _PARAMETERS:
            try:
                record = interface.extract(path, kind_assumptions=kind_assumptions or {})
                _PARAMETERS[path] = {
                    str(p["name"]).lower() for p in record.get("module_parameters", ())
                }
            except Exception:  # an unparsable file declares nothing we can read
                _PARAMETERS[path] = set()
        names |= _PARAMETERS[path]
    return names


def integer_parameters(
    names: list[str],
    root: Path,
    modules: frozenset[str],
    extra: tuple[Path, ...] = (),
    kind_assumptions: dict[str, str] | None = None,
) -> dict[str, int]:
    """``{name: value}`` for the names that resolve to an integer constant
    in the tree's ``modules`` -- or in ``extra`` files, for a module that
    sizes its own type with its own parameter; the rest are left out, and
    the stage that needed them says so.

    Parameters only: a module *variable*'s initializer (``nlev = -1``) is
    not its value in the run, and an extent folded from it is wrong.
    """
    from recast.fortran.expr import render
    from recast.fortran.use import UnresolvedConstant, resolve

    files = [*extra, *module_sources(root, modules)]
    values: dict[str, int] = {}
    if not names or not files:
        return values
    parameters = parameter_names(files, kind_assumptions)
    for name in names:
        if name not in parameters:
            continue
        value = _evaluate(name, files, resolve, render, UnresolvedConstant)
        if isinstance(value, int) and not isinstance(value, bool):
            values[name] = value
    return values


def _evaluate(
    name: str, files: list[Path], resolve: Any, render: Any, unresolved: type[BaseException]
) -> Any:
    """The value of one named constant, or ``None`` where the tree does not
    initialize it with something a parameter can be folded from."""
    try:
        records = resolve([name], files)
    except unresolved:
        return None
    env: dict[str, Any] = {}
    try:
        for entry in records:
            text = render(
                entry["expr"],
                real=lambda t: f"float('{t}')",
                integer=lambda t: t,
                name=lambda t: t.upper(),
            )
            if "float(" not in text and entry["expr"].kind != "str":
                # Integer arithmetic throughout: Fortran's ``/`` truncates.
                text = text.replace("/", "//")
            env[entry["name"].upper()] = eval(text, {"__builtins__": {}}, dict(env))  # noqa: S307
    except Exception:  # an initializer shape the renderer has no rule for
        return None
    return env.get(name.upper())
