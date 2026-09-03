"""Discover Python modules that implement numerical work with NumPy.

The frontend deliberately uses only :mod:`ast`.  Importing a project during
discovery would execute untrusted module top-level code, make discovery depend
on optional dependencies, and turn a missing NumPy install into "no units".
Execution belongs to the independent source oracle later in the recipe.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recast import WORKSPACE_DIRNAME
from recast.errors import ConfigError
from recast.model import Facts, Unit
from recast.plugins.frontend import Frontend

__all__ = ["PythonNumpyFrontend", "factory"]

_SKIPPED_DIRS = frozenset(
    {
        WORKSPACE_DIRNAME,
        ".git",
        "__pycache__",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
    }
)


class PythonNumpyFrontend(Frontend):
    """One deterministic unit per importable Python/NumPy module."""

    name = "python-numpy"
    languages = ("python",)

    def discover(self, root: Path) -> Iterable[Unit]:
        root = root.resolve()
        for source in sorted(
            root.rglob("*.py"), key=lambda path: path.relative_to(root).as_posix()
        ):
            relative = source.relative_to(root)
            if any(part in _SKIPPED_DIRS for part in relative.parts):
                continue
            text = source.read_text(encoding="utf-8")
            try:
                tree = ast.parse(text, filename=relative.as_posix())
            except SyntaxError as error:
                # A file that appears to target NumPy must not disappear merely
                # because it is malformed.  Analysis will report the exact error.
                if "numpy" not in text:
                    continue
                attrs: dict[str, Any] = {"parse_error": _syntax_detail(error)}
            else:
                if not _is_numpy_module(tree) or not _module_functions(tree):
                    continue
                attrs = {}
            yield Unit(
                uid=f"python:{relative.as_posix()}",
                kind="python-module",
                sources=(relative,),
                attrs=attrs,
            )

    def analyze(self, unit: Unit, root: Path) -> Facts:
        if unit.kind != "python-module" or len(unit.sources) != 1:
            raise ConfigError(f"python-numpy frontend cannot analyze {unit.uid!r}")
        relative = unit.sources[0]
        source = (root / relative).resolve()
        try:
            source.relative_to(root.resolve())
        except ValueError as error:
            raise ConfigError(f"python source escapes the project root: {relative}") from error
        try:
            text = source.read_text(encoding="utf-8")
            tree = ast.parse(text, filename=relative.as_posix())
        except (OSError, UnicodeError, SyntaxError) as error:
            detail = _syntax_detail(error) if isinstance(error, SyntaxError) else str(error)
            raise ConfigError(f"cannot analyze {relative.as_posix()}: {detail}") from error
        if not _is_numpy_module(tree):
            raise ConfigError(f"{relative.as_posix()} no longer imports NumPy")

        functions = _module_functions(tree)
        exports = _exports(tree, functions)
        module = _module_name(relative)
        calls = {name: _local_calls(node, functions) for name, node in functions.items()}
        return Facts(
            unit=unit.uid,
            interface={
                "module": module,
                "source": relative.as_posix(),
                "functions": [
                    _function_record(functions[name], name in exports) for name in functions
                ],
                # The runner understands this key when deciding whether a unit
                # legitimately has an empty analysis result.
                "subprograms": [{"name": name, "public": name in exports} for name in functions],
                "exports": sorted(exports),
            },
            callgraph={f"{unit.uid}/{name}": calls[name] for name in functions},
            effects={
                "numpy_aliases": sorted(_numpy_aliases(tree)),
                "source_imports": sorted(_imports(tree)),
                "import_aliases": dict(sorted(_import_aliases(tree).items())),
            },
            provenance={
                "digest": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "source": relative.as_posix(),
                "frontend": self.name,
            },
        )


def _is_numpy_module(tree: ast.Module) -> bool:
    return bool(_numpy_aliases(tree))


def _numpy_aliases(tree: ast.Module) -> set[str]:
    aliases: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                if item.name == "numpy":
                    aliases.add(item.asname or "numpy")
        elif isinstance(node, ast.ImportFrom) and node.module == "numpy":
            aliases.update(item.asname or item.name for item in node.names)
    return aliases


def _imports(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(item.name for item in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    return imported


def _import_aliases(tree: ast.Module) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for item in node.names:
                aliases[item.asname or item.name.split(".", 1)[0]] = item.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                aliases[item.asname or item.name] = f"{node.module}.{item.name}"
    return aliases


def _module_functions(tree: ast.Module) -> dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def _exports(
    tree: ast.Module, functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef]
) -> set[str]:
    for node in tree.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        target = node.target if isinstance(node, ast.AnnAssign) else node.targets[0]
        if not isinstance(target, ast.Name) or target.id not in {"__all__", "_RECAST_EXPORTS"}:
            continue
        if node.value is None:
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, TypeError):
            continue
        if isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
            return {item for item in value if item in functions}
    return {name for name in functions if not name.startswith("_")}


def _local_calls(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef],
) -> list[str]:
    called = {
        child.func.id
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id in functions
    }
    return sorted(called)


def _function_record(node: ast.FunctionDef | ast.AsyncFunctionDef, public: bool) -> dict[str, Any]:
    positional = [*node.args.posonlyargs, *node.args.args]
    return {
        "name": node.name,
        "public": public,
        "async": isinstance(node, ast.AsyncFunctionDef),
        "arguments": [
            {"name": arg.arg, "annotation": ast.unparse(arg.annotation) if arg.annotation else None}
            for arg in [*positional, *node.args.kwonlyargs]
        ],
        "returns": ast.unparse(node.returns) if node.returns else None,
    }


def _module_name(relative: Path) -> str:
    parts = list(relative.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts) or "__init__"


def _syntax_detail(error: SyntaxError) -> str:
    return f"line {error.lineno or '?'}: {error.msg}"


def factory(**_config: Any) -> PythonNumpyFrontend:
    return PythonNumpyFrontend()
