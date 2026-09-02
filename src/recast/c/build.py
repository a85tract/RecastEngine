"""Build specs, staging, and running for kernel directories.

A build spec is data a frontend puts in ``Unit.attrs``::

    {"dir": "data/src/jacobi-omp",          # relative to the tree root
     "steps": [["make", "-f", "Makefile.nvc", "CC={cc}", "SM={sm}", "clean"],
               ["make", "-f", "Makefile.nvc", "CC={cc}", "SM={sm}"]],
     "program": "main",                     # the executable the last step leaves
     "args": ["512", "input/temp_512"],     # what to run it with
     "sources": ["main.cpp"],               # the files a probe may be carried to
     "inputs": ["golden_labels/x/data"],    # files/dirs copied into the directory
     "stage": ["gate_sdk"]}                 # dirs copied to the same place under staging

``{name}`` placeholders come from the operator's ``toolchain`` table
(``{"cc": "nvc++", "sm": "cc80", ...}``) plus ``{staging}`` and ``{dir}``,
which the engine supplies. The table is free-form: a key the operator adds
is a placeholder a spec may use. Its ``env`` entry is merged into every
run's environment, which is where ``OMP_TARGET_OFFLOAD=MANDATORY`` goes.
``identity()`` -- the compiler's own ``--version`` line -- is what an oracle
folds into its key, because a different compiler is a different reference.

Staging copies a directory without its build leftovers (objects, binaries,
profiler traces, dangling links), and brings along the directories outside it
that its build reaches into (``-I../x`` in a Makefile, ``#include "../x"`` in
a source). Everything that leaves the process goes through the ``Executor``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recast.plugins.executor import Executor, Job, JobResult

__all__ = [
    "BUILD_ARTIFACT_SUFFIXES",
    "SOURCE_SUFFIXES",
    "Spec",
    "Toolchain",
    "build",
    "is_build_artifact",
    "run",
    "stage_directory",
    "stage_siblings",
]

BUILD_ARTIFACT_SUFFIXES = frozenset(
    {".o", ".nsys-rep", ".sqlite", ".bak", ".log", ".x", ".out", ".llvm", ".DS_Store"}
)
SOURCE_SUFFIXES = frozenset({".c", ".cpp", ".cc", ".cu", ".h", ".hpp"})
_ARTIFACT_NAMES = frozenset({"analysis.md", ".DS_Store"})
_SIBLING_INCLUDE = re.compile(r"-I\s*(\.\./[\w.\-/+]+)")
_PARENT_INCLUDE = re.compile(r'^\s*#\s*include\s+"((?:\.\./)+[\w.\-/+]+)"', re.MULTILINE)


@dataclass(frozen=True)
class Spec:
    """One side's build, as a frontend declared it."""

    dir: Path
    steps: tuple[tuple[str, ...], ...]
    program: str
    args: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @classmethod
    def from_attrs(cls, table: Mapping[str, Any]) -> Spec:
        return cls(
            dir=Path(str(table["dir"])),
            steps=tuple(tuple(str(a) for a in step) for step in table.get("steps", ())),
            program=str(table.get("program", "main")),
            args=tuple(str(a) for a in table.get("args", ())),
            sources=tuple(str(s) for s in table.get("sources", ())),
        )

    def resolve_dir(self, root: Path) -> Path:
        return self.dir if self.dir.is_absolute() else root / self.dir


class Toolchain:
    """The operator's table, with the substitutions it implies."""

    def __init__(self, table: Mapping[str, Any] | None = None) -> None:
        self.table: dict[str, Any] = {"cc": "cc", **dict(table or {})}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Toolchain:
        return cls(config.get("toolchain"))

    @property
    def cc(self) -> str:
        return str(self.table["cc"])

    @property
    def timeout_s(self) -> float:
        return float(self.table.get("timeout_s", 900.0))

    def identity(self) -> str:
        """What the compiler says it is, with the table's other values."""
        exe = self.cc.split()[0]
        path = shutil.which(exe) or exe
        try:
            out = subprocess.run(  # noqa: S603
                [path, "--version"], capture_output=True, text=True, timeout=30, check=False
            )
            first = (out.stdout or out.stderr).strip().splitlines()
            version = first[0] if first else "unknown"
        except (OSError, subprocess.TimeoutExpired):
            version = "unavailable"
        rest = {k: v for k, v in sorted(self.table.items()) if k not in {"env", "timeout_s"}}
        return f"{version} {rest!r}"

    def render(self, template: str, **extra: str) -> str:
        """``{name}`` placeholders filled from the table; a value may itself
        hold placeholders (``npb_cflags: "-gpu={sm}"``), so it settles."""
        values = {k: str(v) for k, v in self.table.items() if not isinstance(v, dict)}
        values.update(extra)
        out = template
        for _ in range(4):
            before = out
            for key, value in values.items():
                out = out.replace("{" + key + "}", value)
            if out == before:
                break
        return out

    def env(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        table_env = self.table.get("env") or {}
        env = {**os.environ, **{str(k): str(v) for k, v in dict(table_env).items()}}
        if extra:
            env.update(extra)
        return env


def _is_binary(path: Path) -> bool:
    try:
        with path.open("rb") as fh:
            return b"\0" in fh.read(512)
    except OSError:
        return False


def is_build_artifact(path: Path) -> bool:
    """A file a kernel directory carries that is not source or input data.

    An extensionless file counts only when it is binary: a Makefile or a
    ``run`` script with the executable bit set is not a build product, a
    stray binary from someone else's machine is.
    """
    if path.suffix in BUILD_ARTIFACT_SUFFIXES or path.name in _ARTIFACT_NAMES:
        return True
    if path.name.startswith("nsys_profile"):
        return True
    return path.suffix == "" and path.is_file() and _is_binary(path)


def stage_directory(
    source: Path, target: Path, *, replace: Mapping[str, bytes] | None = None
) -> None:
    """Copy a directory without its build leftovers, replacing named files."""

    def skip(directory: str, names: list[str]) -> set[str]:
        d = Path(directory)
        out = set()
        for n in names:
            p = d / n
            if (p.is_symlink() and not p.exists()) or (p.is_file() and is_build_artifact(p)):
                out.add(n)
        return out

    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    for item in sorted(source.iterdir()):
        if item.is_symlink() and not item.exists():
            continue
        if item.is_dir():
            shutil.copytree(item, target / item.name, symlinks=False, ignore=skip)
        elif not is_build_artifact(item):
            shutil.copy2(item, target / item.name)
    for name, content in (replace or {}).items():
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        (target / name).write_bytes(content)


def stage_siblings(source_dir: Path, target_dir: Path) -> list[str]:
    """Copy the directories outside ``source_dir`` that its build reaches into
    -- a Makefile's ``-I../x``, a source's ``#include "../x"`` -- to the same
    relative place beside the staged copy."""
    wanted: list[tuple[Path, Path]] = []
    for makefile in source_dir.glob("Makefile*"):
        for rel in _SIBLING_INCLUDE.findall(makefile.read_text(errors="replace")):
            wanted.append(((source_dir / rel).resolve(), (target_dir / rel).resolve()))
    for src in source_dir.rglob("*"):
        if src.suffix not in SOURCE_SUFFIXES or not src.is_file():
            continue
        for rel in _PARENT_INCLUDE.findall(src.read_text(errors="replace")):
            inside = src.parent.relative_to(source_dir)
            wanted.append(
                ((src.parent / rel).resolve().parent, (target_dir / inside / rel).resolve().parent)
            )
    staged = []
    for src, dst in wanted:
        if src.is_dir() and not dst.exists():
            stage_directory(src, dst)
            staged.append(str(src))
    return staged


def build(
    spec: Spec,
    kernel_dir: Path,
    staging: Path,
    toolchain: Toolchain,
    executor: Executor,
    label: str,
) -> JobResult:
    """Run the spec's steps in ``kernel_dir``; the last one's result is the build's.

    A step that is not the last may fail (``make clean`` on a fresh tree);
    the log carries every step's command and output.
    """
    env = toolchain.env()
    log: list[str] = []
    result = JobResult(0, "", "")
    for i, step in enumerate(spec.steps):
        # An argument that renders empty (a flag table left blank) is dropped
        # rather than passed as ``""``, which a compiler would read as a file.
        argv = [
            rendered
            for a in step
            if (rendered := toolchain.render(a, staging=str(staging), dir=str(kernel_dir)))
        ]
        result = executor.run(
            Job(
                argv=argv,
                cwd=kernel_dir,
                env=env,
                timeout_s=toolchain.timeout_s,
                label=f"{label}:{i}",
            )
        )
        log.append(" ".join(argv) + "\n" + result.stdout + result.stderr)
    return JobResult(result.returncode, "\n".join(log), result.stderr, result.artifacts)


def run(
    spec: Spec,
    kernel_dir: Path,
    toolchain: Toolchain,
    executor: Executor,
    label: str,
    *,
    args: Sequence[str] | None = None,
    prefix: Sequence[str] = (),
    env: Mapping[str, str] | None = None,
) -> JobResult:
    """Execute the built program, optionally under a profiler."""
    program = toolchain.render(spec.program)
    return executor.run(
        Job(
            argv=[*prefix, f"./{program}", *(spec.args if args is None else args)],
            cwd=kernel_dir,
            env=toolchain.env(env),
            timeout_s=toolchain.timeout_s,
            label=label,
        )
    )
