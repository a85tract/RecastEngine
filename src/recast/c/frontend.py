"""``c-kernel``: every program directory under a tree, as one Unit each.

A kernel directory is one with a ``main.c`` / ``main.cpp`` in it, or a
Makefile that names its ``program``. The Unit's ``attrs["build"]`` is a
build spec (see ``recast.c.build``) that a plain kernel gets as ``make`` in
its directory; a frontend that knows a suite's conventions -- where the
serial program is, what the Makefile variables are -- subclasses this one
and writes the specs it knows, the way the CESM extension subclasses the
Fortran frontend.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recast import WORKSPACE_DIRNAME
from recast.c.build import SOURCE_SUFFIXES, Spec
from recast.c.scan import scan
from recast.model import Facts, Unit
from recast.plugins.frontend import Frontend

__all__ = ["CKernelFrontend", "factory", "git_revision", "makefile_vars", "source_files"]

_ASSIGN = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*([:?+]?=)\s*(.*?)\s*$")


def makefile_vars(makefile: Path) -> dict[str, str]:
    """Top-level ``NAME = value`` assignments of a Makefile, last one wins."""
    out: dict[str, str] = {}
    for line in makefile.read_text(errors="replace").splitlines():
        m = _ASSIGN.match(line)
        if not m or line.lstrip().startswith("#"):
            continue
        name, op, value = m.groups()
        out[name] = (out.get(name, "") + " " + value).strip() if op == "+=" else value
    return out


def source_files(directory: Path, *, recursive: bool = False) -> list[Path]:
    it = directory.rglob("*") if recursive else directory.iterdir()
    return sorted(p for p in it if p.is_file() and p.suffix in SOURCE_SUFFIXES)


def git_revision(root: Path) -> str | None:
    try:
        out = subprocess.run(  # noqa: S603
            ["git", "-C", str(root), "rev-parse", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return out.stdout.strip() or None


def _read_all(paths: Iterable[Path]) -> str:
    return "\n".join(p.read_text(errors="replace") for p in paths if p.exists())


class CKernelFrontend(Frontend):
    name = "c-kernel"
    languages = ("c", "c++")

    def discover(self, root: Path) -> Iterable[Unit]:
        for directory in sorted(p for p in root.rglob("*") if p.is_dir()):
            rel = directory.relative_to(root)
            if WORKSPACE_DIRNAME in rel.parts or any(part.startswith(".") for part in rel.parts):
                continue
            unit = self.unit_for(root, directory)
            if unit is not None:
                yield unit

    def unit_for(self, root: Path, directory: Path) -> Unit | None:
        """A Unit for one kernel directory, or None if it is not one."""
        makefile = next(
            (m for m in ("Makefile", "makefile", "GNUmakefile") if (directory / m).exists()), None
        )
        variables = makefile_vars(directory / makefile) if makefile else {}
        main = next(
            (m for m in ("main.cpp", "main.c", "main.cc") if (directory / m).exists()), None
        )
        program = variables.get("program") or ("main" if main else None)
        if program is None:
            return None
        sources = tuple(variables.get("source", main or "").split()) or ((main,) if main else ())
        rel = directory.relative_to(root)
        steps: list[list[str]] = []
        if makefile:
            steps = [["make", "-f", makefile, "clean"], ["make", "-f", makefile, "CC={cc}"]]
        elif main:
            steps = [["{cc}", "-O2", *sources, "-o", program]]
        return Unit(
            uid=f"c:{rel.as_posix()}",
            kind="kernel",
            sources=tuple(rel / s for s in sources if (directory / s).exists()),
            attrs={
                "build": {
                    "dir": rel.as_posix(),
                    "steps": steps,
                    "program": program,
                    "args": variables.get("RUN_ARGS", "").split(),
                    "sources": list(sources),
                }
            },
        )

    def analyze(self, unit: Unit, root: Path) -> Facts:
        spec = Spec.from_attrs(unit.attrs["build"])
        directory = spec.resolve_dir(root)
        text = _read_all(directory / s for s in spec.sources)
        found = scan(text)
        headers = _read_all(
            p for p in source_files(directory, recursive=True) if p.suffix in {".h", ".hpp"}
        )
        return Facts(
            unit=unit.uid,
            interface={
                "program": spec.program,
                "sources": list(spec.sources),
                "run_args": list(spec.args),
                "subprograms": found.functions,
                "entry": "main",
            },
            callgraph={unit.uid: [f for f in found.functions if f != "main"]},
            effects={
                "loops": found.loops,
                "loop_depth": found.loop_depth,
                "allocations": found.allocations,
                "timed_region": found.timed_region,
                "io": ["stdout"],
                "input_files": [a for a in spec.args if "/" in a or "." in a],
                "omp_pragmas": found.omp_pragmas,
                "target_regions": found.target_regions,
            },
            provenance={
                "revision": git_revision(root),
                "dir": spec.dir.as_posix(),
                "scanner": "regex; counts are lexical, not semantic",
                "includes": sorted(set(found.includes) | set(scan(headers).includes)),
            },
            extra={"lines": found.lines},
        )


def factory(**_config: Any) -> CKernelFrontend:
    return CKernelFrontend()
