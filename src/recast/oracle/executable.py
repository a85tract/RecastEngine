"""``executable-golden``: a reference program, built and run, as the oracle.

The reference is whatever ``Unit.attrs["golden"]`` describes -- a directory
and a build spec (``recast.c.build``): the serial original of a kernel being
offloaded, or any program whose output is the truth. It is staged into the
workspace with the directories its build reaches into and the input files
``attrs["golden"]["inputs"]`` names, built with the operator's toolchain, and
run once. The staging, the binary and the stdout are the handle.

It never sees a candidate. A verifier that needs the reference re-run under
other conditions (with probes, under a profiler) copies the staging and does
that itself.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from recast.c.build import Spec, Toolchain, build, run, stage_directory, stage_siblings
from recast.c.frontend import source_files
from recast.errors import OracleUnavailable
from recast.model import Facts, OracleRef, Unit
from recast.plugins.executor import Executor
from recast.plugins.oracle import Oracle

__all__ = ["ExecutableGoldenOracle", "factory", "stage_golden"]


def stage_golden(unit: Unit, root: Path, staging: Path) -> tuple[Spec, Path]:
    """The golden directory under ``staging/golden/``, ready to build."""
    spec = Spec.from_attrs(unit.attrs["golden"])
    source = spec.resolve_dir(root)
    golden = staging / "golden" / source.name
    stage_directory(source, golden)
    stage_siblings(source, golden)
    for extra in unit.attrs["golden"].get("stage", ()):
        item = root / str(extra)
        if item.is_dir():
            stage_directory(item, staging / item.relative_to(root))
    for extra in unit.attrs["golden"].get("inputs", ()):
        item = root / str(extra)
        if item.is_symlink() and not item.exists():
            continue
        target = golden / item.name
        if item.is_dir():
            if not target.exists():
                stage_directory(item, target)
        elif item.is_file() and not target.exists():
            target.write_bytes(item.read_bytes())
    return spec, golden


class ExecutableGoldenOracle(Oracle):
    name = "executable-golden"
    cost = "build"

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        root = Path(config["root"])
        toolchain = Toolchain.from_config(config)
        spec = Spec.from_attrs(unit.attrs["golden"])
        source = spec.resolve_dir(root)
        h = hashlib.sha256()
        if source.is_dir():
            for path in source_files(source, recursive=True) + sorted(source.glob("Makefile*")):
                h.update(path.relative_to(source).as_posix().encode())
                h.update(path.read_bytes())
        h.update(json.dumps(unit.attrs["golden"], sort_keys=True, default=str).encode())
        h.update(json.dumps(facts.provenance, sort_keys=True, default=str).encode())
        h.update(toolchain.identity().encode())
        h.update(" ".join(config.get("run_args") or spec.args).encode())
        return f"{unit.uid}:{h.hexdigest()[:16]}"

    def materialize(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> OracleRef:
        root = Path(config["root"])
        toolchain = Toolchain.from_config(config)
        staging = workspace / "oracle"
        try:
            spec, golden = stage_golden(unit, root, staging)
        except OSError as error:
            raise OracleUnavailable(f"golden for {unit.uid} cannot be staged: {error}") from error
        args = tuple(config.get("run_args") or spec.args)
        program = toolchain.render(spec.program)
        try:
            built = build(spec, golden, staging, toolchain, executor, label=f"{unit.uid}:golden")
            (golden / "build.log").write_text(built.stdout + built.stderr)
            if not built.ok or not (golden / program).exists():
                raise OracleUnavailable(
                    f"golden for {unit.uid} did not build (rc={built.returncode}): "
                    + (built.stderr or built.stdout)[-600:]
                )
            ran = run(spec, golden, toolchain, executor, label=f"{unit.uid}:golden-run", args=args)
        except OracleUnavailable:
            raise
        except Exception as error:  # the executor refused, or could not start the job
            raise OracleUnavailable(f"golden for {unit.uid} could not run: {error}") from error
        (golden / "run.stdout").write_text(ran.stdout)
        (golden / "run.stderr").write_text(ran.stderr)
        if ran.returncode != 0:
            raise OracleUnavailable(
                f"golden for {unit.uid} exited {ran.returncode}: "
                + (ran.stderr or ran.stdout)[-600:]
            )
        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=self.key(unit, facts, config),
            handle={
                "dir": str(golden),
                "staging": str(staging),
                "program": program,
                "run_args": list(args),
                "stdout": ran.stdout,
                "toolchain": toolchain.identity(),
            },
            cost=self.cost,
        )


def factory(**_config: Any) -> ExecutableGoldenOracle:
    return ExecutableGoldenOracle()
