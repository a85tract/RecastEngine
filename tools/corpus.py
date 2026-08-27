#!/usr/bin/env python3
"""Hold the engine, alone, to the public translation corpus.

``corpus/`` carries twelve open-source Fortran libraries as pinned submodules
and ``corpus/cases.json`` says which of their files make one case. This tool
stages a case -- copies the files, runs cpp over the ones that need it --
walks the ``translate`` recipe over it with no domain extension installed,
and records what happened per unit: how many blocks the rules refused and
why, whether the static read/write check agreed, how far the recipe got.

The record, ``corpus/baseline.json``, is the engine's claim about itself and
the list of what to build next; a rule relayed from the translator either
moves a number here or was not needed. Reasons are normalised (names and
numbers elided) so the same refusal counts once however many times it fires.

    python tools/corpus.py stage minpack          # into corpus/.build/minpack
    python tools/corpus.py run                    # every case, rewrite baseline.json
    python tools/corpus.py run minpack fftpack
    python tools/corpus.py report                 # print the table from baseline.json
"""

from __future__ import annotations

import argparse
import ast
import collections
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORPUS = ROOT / "corpus"
BUILD = CORPUS / ".build"
CASES = json.loads((CORPUS / "cases.json").read_text())["cases"]
BASELINE = CORPUS / "baseline.json"


def stage(name: str) -> Path:
    """Copy the case's files into a build directory, preprocessing ``.F90``."""
    case = CASES[name]
    source_root = CORPUS / case["submodule"]
    if not any(source_root.iterdir()):
        sys.exit(f"{name}: submodule {case['submodule']} is empty -- git submodule update --init")
    out = BUILD / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    excluded = {
        Path(p)
        for pattern in case.get("exclude", [])
        for p in glob.glob(str(source_root / pattern), recursive=True)
    }
    files: list[Path] = []
    for pattern in case["files"]:
        matches = sorted(Path(p) for p in glob.glob(str(source_root / pattern), recursive=True))
        if not matches:
            sys.exit(f"{name}: {pattern} matched nothing under {source_root}")
        files.extend(m for m in matches if m not in excluded and m.is_file())
    cpp = case.get("cpp", {})
    for path in files:
        relative = path.relative_to(source_root)
        target = out / relative.name
        if path.suffix == ".F90":
            # Through the compiler's own preprocessor, the way a build would.
            command = ["gfortran", "-E", "-P", "-cpp", *(f"-D{d}" for d in cpp.get("defines", []))]
            command += [f"-I{source_root / inc}" for inc in cpp.get("include", [])]
            command += [f"-I{path.parent}", str(path)]
            # Our own argv, built above from the case table -- not user input.
            text = subprocess.run(command, check=True, capture_output=True, text=True).stdout  # noqa: S603
            target = out / (relative.stem + ".f90")
            target.write_text(text)
        else:
            shutil.copy2(path, target)
    return out


def _units(root: Path) -> list[str]:
    from recast.fortran.frontend import FortranFrontend

    found = FortranFrontend().discover(root)
    return sorted(u.uid for u in found if u.kind in ("module", "program"))


def _bare_files(root: Path) -> int:
    """Files of bare subprograms -- no module, no program -- which the
    translate recipe does not yet take as units. Counted so the record says
    what was *not* tried, not only what failed."""
    from recast.fortran.frontend import FortranFrontend

    return sum(1 for u in FortranFrontend().discover(root) if u.kind == "file")


def _normalise(reason: str) -> str:
    reason = re.sub(r"^[\w/]+/B\d+:\s*", "", reason)
    reason = re.sub(r"'[^']*'", "'X'", reason)
    reason = re.sub(r"\b\d+\b", "N", reason)
    return reason[:90]


def run_case(name: str) -> dict[str, Any]:
    from recast.cli import _recipe
    from recast.run import run_recipe

    root = stage(name)
    units = _units(root)
    recipe = _recipe("translate")
    # No ``workspace``: the default is ``output/<case>/``, outside the staged
    # tree, so a re-stage does not delete the last run's candidates and a
    # second run does not discover the first one's generated Python.
    run = run_recipe(recipe, root, {"units": units})
    record: dict[str, Any] = {
        "units": {},
        "status": run.status.value if hasattr(run.status, "value") else str(run.status),
    }
    reasons: collections.Counter[str] = collections.Counter()
    for unit_run in run.units:
        entry: dict[str, Any] = {"stopped_by": unit_run.stopped_by}
        entry["stages"] = {f"{o.kind}/{o.plugin}": o.status for o in unit_run.outcomes}
        if unit_run.candidate is not None:
            entry["deferred"] = len(unit_run.candidate.deferred)
            for reason in unit_run.candidate.deferred:
                reasons[_normalise(reason)] += 1
        for verdict in unit_run.verdicts:
            entry.setdefault("verdicts", {})[verdict.verifier] = {
                "passed": verdict.passed,
                "metrics": {
                    k: v for k, v in (verdict.metrics or {}).items() if not isinstance(v, list)
                },
                "detail": (verdict.detail or "")[:160],
            }
        record["units"][unit_run.unit.uid] = entry
    record["refusals"] = reasons.most_common()
    record["bare_files"] = _bare_files(root)
    _translated(name, root, run, record)
    record["blocks"] = sum(
        (e.get("verdicts", {}).get("static.rwset", {}).get("metrics", {}).get("blocks_checked", 0))
        for e in record["units"].values()
    )
    record["deferred"] = sum(e.get("deferred", 0) for e in record["units"].values())
    return record


def _translated(name: str, root: Path, run: Any, record: dict[str, Any]) -> None:
    """Write what was emitted, and say how far each unit's artifact got.

    Three bars, each stricter than the last and none of them the real one:

    *mechanical* -- the rules refused nothing, so the file has no
    ``NotImplementedError`` standing where code should be.
    *parses* -- the emitted text is Python. A translation that does not
    parse is not a translation, however few blocks it refused.
    *imports* -- it and everything it imports load, which needs its
    constants module, its companions' modules, and every shim it calls to
    exist. This is where a call to something nobody defines shows up.

    The real bar is the differential against a build of the same Fortran,
    and it is further on than any of these.
    """
    emitted = root / "translated"
    if emitted.exists():
        shutil.rmtree(emitted)
    emitted.mkdir()
    artifacts: dict[str, list[str]] = {}
    for unit_run in run.units:
        if unit_run.candidate is None:
            continue
        names = []
        for relative, text in unit_run.candidate.files.items():
            (emitted / Path(relative).name).write_bytes(text)
            names.append(Path(relative).name)
        artifacts[unit_run.unit.uid] = names

    for unit_run in run.units:
        entry = record["units"][unit_run.unit.uid]
        files = artifacts.get(unit_run.unit.uid)
        if not files:
            entry["artifact"] = "none"
            continue
        entry["mechanical"] = entry.get("deferred", 0) == 0
        module = next((f for f in files if f.endswith("_numpy.py")), None)
        try:
            for one in files:
                ast.parse((emitted / one).read_text())
            entry["parses"] = True
        except SyntaxError as error:
            entry["parses"] = False
            entry["syntax_error"] = f"{Path(error.filename or '').name}:{error.lineno}: {error.msg}"
            continue
        if module is None:
            continue
        entry["imports"], reason = _imports(emitted, module)
        if reason:
            entry["import_error"] = reason


def _imports(where: Path, module: str) -> tuple[bool, str]:
    """Import one emitted module in a subprocess, with its siblings beside it."""
    probe = subprocess.run(  # noqa: S603 -- our own argv
        [sys.executable, "-c", f"import {module[:-3]}"],
        cwd=where,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
    )
    if probe.returncode == 0:
        return True, ""
    last = [line for line in probe.stderr.strip().splitlines() if line.strip()]
    return False, (last[-1] if last else "no output")[:160]


def report(baseline: dict[str, Any]) -> None:
    print(
        f"{'case':14} {'units':>5} {'bare':>4} {'mech':>6} {'parse':>6} {'import':>7} "
        f"{'rwset':>7} {'deferred':>9}  top refusal"
    )
    for name, record in baseline["cases"].items():
        units = record["units"]
        rw = [e.get("verdicts", {}).get("static.rwset") for e in units.values()]
        rw = [v for v in rw if v]
        total = len(units)

        def tally(key: str, seen: dict[str, Any] = units, n: int = total) -> str:
            return f"{sum(1 for e in seen.values() if e.get(key)):>3}/{n}" if n else "  -"

        rw_text = f"{sum(1 for v in rw if v['passed'])}/{len(rw)}" if rw else "-"
        top = "; ".join(f"{n}x {r[:34]}" for r, n in list(record["refusals"])[:1])
        print(
            f"{name:14} {total:>5} {record.get('bare_files', 0):>4} {tally('mechanical'):>6} "
            f"{tally('parses'):>6} {tally('imports'):>7} {rw_text:>7} "
            f"{record['deferred']:>9}  {top}"
        )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["stage", "run", "report"])
    parser.add_argument("cases", nargs="*", help="case names; default: all")
    args = parser.parse_args(argv)
    names = args.cases or list(CASES)
    unknown = [n for n in names if n not in CASES]
    if unknown:
        sys.exit(f"unknown case(s): {unknown}; known: {list(CASES)}")
    if args.command == "stage":
        for name in names:
            print(f"{name}: staged at {stage(name).relative_to(ROOT)}")
        return 0
    if args.command == "report":
        report(json.loads(BASELINE.read_text()))
        return 0
    baseline: dict[str, Any] = {"schema": 1, "cases": {}}
    if BASELINE.is_file():
        baseline = json.loads(BASELINE.read_text())
    for name in names:
        print(f"== {name}", flush=True)
        try:
            baseline["cases"][name] = run_case(name)
        except Exception as error:  # the record is the point; one case must not end the run
            status = f"error: {type(error).__name__}: {error}"[:300]
            baseline["cases"][name] = {
                "units": {},
                "status": status,
                "refusals": [],
                "blocks": 0,
                "deferred": 0,
            }
            print(f"   error: {error}")
    baseline["cases"] = dict(sorted(baseline["cases"].items()))
    BASELINE.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
    report(baseline)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
