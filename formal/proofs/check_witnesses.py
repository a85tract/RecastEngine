#!/usr/bin/env python3
"""Check the Lean library and every defect witness against it.

Run from this directory after ``lake build``. Fails closed: a missing or
unlisted witness, a Lean error, a theorem whose axioms go beyond Lean's three
standard ones, or a ``sorry``/``admit``/``native_decide``/``axiom`` in the
library is a failure.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ALLOWED = {"propext", "Classical.choice", "Quot.sound"}
BANNED = re.compile(r"\b(?:sorry|admit|native_decide|axiom)\b")
REPORT = re.compile(
    r"^'([^']+)' (?:depends on axioms: \[([^\]]*)\]|does not depend on any axioms)",
    re.M,
)


def strip_comments(code: str) -> str:
    out, depth, i = [], 0, 0
    while i < len(code):
        if code.startswith("/-", i):
            depth, i = depth + 1, i + 2
        elif depth and code.startswith("-/", i):
            depth, i = depth - 1, i + 2
        elif not depth and code.startswith("--", i):
            end = code.find("\n", i)
            i = end if end >= 0 else len(code)
        else:
            if not depth:
                out.append(code[i])
            i += 1
    return "".join(out)


def check_library() -> list[str]:
    errors = []
    for path in sorted(HERE.glob("RecastProofs/**/*.lean")):
        for token in BANNED.findall(strip_comments(path.read_text())):
            errors.append(f"{path.relative_to(HERE)}: {token}")
    return errors


def check_witness(wid: str) -> list[str]:
    path = HERE / "witnesses" / f"{wid}.lean"
    expected = set(re.findall(r"^#print axioms (\S+)", path.read_text(), re.M))
    if not expected:
        return [f"{wid}: no #print axioms line"]
    proc = subprocess.run(  # noqa: S603 -- fixed argv, no shell
        ["lake", "env", "lean", str(path)],  # noqa: S607 -- lake from the toolchain on PATH
        cwd=HERE,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode:
        return [f"{wid}: lean exited {proc.returncode}\n{proc.stdout}"]
    axioms = {
        m[1]: {a.strip() for a in (m[2] or "").split(",") if a.strip()}
        for m in REPORT.finditer(proc.stdout)
    }
    errors = []
    if set(axioms) != expected:
        errors.append(f"{wid}: axioms reported for {sorted(axioms)}, expected {sorted(expected)}")
    for name, used in axioms.items():
        if used - ALLOWED:
            errors.append(f"{wid}: {name} depends on {sorted(used - ALLOWED)}")
    return errors


def main() -> int:
    listed = [w["id"] for w in json.loads((HERE / "witnesses.json").read_text())["witnesses"]]
    present = sorted(p.stem for p in (HERE / "witnesses").glob("*.lean"))
    errors = check_library()
    for w in sorted(set(listed) - set(present)):
        errors.append(f"{w}: listed in witnesses.json, no witnesses/{w}.lean")
    for w in sorted(set(present) - set(listed)):
        errors.append(f"{w}: witnesses/{w}.lean is not listed in witnesses.json")
    for wid in sorted(set(listed) & set(present)):
        errors += check_witness(wid)
        print(f"{wid}: {'FAIL' if errors and errors[-1].startswith(wid) else 'ok'}", flush=True)
    for e in errors:
        print(e, file=sys.stderr)
    print(f"{len(present)} witnesses, {len(errors)} errors")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
