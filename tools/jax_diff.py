#!/usr/bin/env python3
"""Check the migrated JAX backend against the collection it was migrated from.

Two comparisons exist on the port side and they must not be confused, because
their standards are opposite. The *scientific* one asks whether the ported
kernel computes what the Fortran computed; it cannot be bit-identical -- XLA
lowers transcendentals to its own implementations and fuses multiply-add --
and it is gated at the ULP tier by ``differential.tolerance``. This file is
the other one. It asks whether the migration changed anything, and there the
answer must be *nothing at all*: same inputs, same emitted bytes.

That standard is affordable because this compares text rather than numbers.
``build_module`` is a pure transformation of an interface record and a Python
AST, so no compiler, no libm and no device enters into it -- verified: the
survey below runs over the whole corpus with JAX not installed. Keeping the
migration check a *text* comparison is a design choice, not an accident: the
moment it becomes "run both and compare results", it inherits every source of
floating-point difference the scientific comparison has to live with, and
stops being the clean instrument that makes a 600-line migration reviewable.

Live against the collection's code, not against stored output, for the reason
``emit_diff.py`` gives about the translator: a golden file older than the code
that wrote it is how this repository once reported fixing a bug that was never
there.

What is compared, per module: the emitted pieces byte for byte; which
subprograms became JAX kernels, exactly; and which were host-delegated,
exactly. Delegation *placement* is strict. The *reason* is compared only as
far as its category -- ``[elig] derived type``, ``[emit] unsupported stmt
Raise`` -- because the tag and the category are a decision while the tail is a
diagnostic, and the two sides are entitled to word a diagnostic differently.

The module header is not compared, and neither side emits one here:
``build_module`` returns the pieces, and the header is written by whatever
calls it. That is the same carve-out ``emit_diff.py`` makes, for the same
reason -- the engine's header is real, typed code and deliberately differs
from a string constant in a script.

Corpus: every directory under ``--suite-dir`` holding both ``interface.json``
and ``<stem>_numpy.py``, discovered at run time so a new sweep widens this
check without anyone editing it.

Usage:
    python tools/jax_diff.py --collection <dir> --suite-dir <dir> --survey
    python tools/jax_diff.py --collection <dir> --suite-dir <dir>
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


class Emitter(Protocol):
    """What both sides have to offer. The migrated backend is expected to
    expose the same call, so that this file never learns two shapes."""

    def build_module(
        self, interface: dict[str, Any], tree: ast.Module
    ) -> tuple[list[str], Any, dict[str, str]]: ...


@dataclass
class Emission:
    """One module through one emitter."""

    pieces: list[str] = field(default_factory=list)
    jitted: list[str] = field(default_factory=list)
    delegated: dict[str, str] = field(default_factory=dict)
    error: str = ""

    @property
    def ok(self) -> bool:
        return not self.error


def load_original(collection: Path) -> Any:
    """Import the collection's ``jaxize.py`` by path, without installing it."""
    path = collection / "jaxize.py"
    if not path.is_file():
        raise SystemExit(f"no jaxize.py under {collection}")
    spec = importlib.util.spec_from_file_location("jaxize_original", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SystemExit(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_migrated() -> Any:
    """The engine's JAX backend, once it exists.

    Named here before it is written on purpose: this file is the check the
    migration has to pass, so the shape it expects is part of the migration's
    specification rather than something discovered afterwards.
    """
    try:
        from recast.transform.jax import backend  # type: ignore[import-not-found]
    except ImportError as error:
        raise SystemExit(
            "the migrated JAX backend is not importable "
            f"(recast.transform.jax.backend: {error}).\n"
            "It is not written yet; run with --survey to measure the corpus "
            "against the collection alone."
        ) from error
    return backend


def corpus(suite_dir: Path) -> list[Path]:
    return sorted(
        d
        for d in suite_dir.iterdir()
        if d.is_dir() and (d / "interface.json").is_file() and (d / f"{d.name}_numpy.py").is_file()
    )


def emit(emitter: Any, module_dir: Path) -> Emission:
    """Run one emitter over one suite directory. A raise is a result, not a
    crash: which modules an emitter refuses is part of what is being compared."""
    try:
        interface = json.loads((module_dir / "interface.json").read_text())
        tree = ast.parse((module_dir / f"{module_dir.name}_numpy.py").read_text())
        pieces, jitted, delegated = emitter.build_module(interface, tree)
    except Exception as error:
        return Emission(error=f"{type(error).__name__}: {error}")
    return Emission(
        pieces=list(pieces),
        jitted=sorted(jitted),
        delegated={str(k): str(v) for k, v in delegated.items()},
    )


def category(reason: str) -> str:
    """``[emit] unsupported stmt Raise: foo`` -> ``[emit] unsupported stmt Raise``."""
    return reason.split(":", 1)[0].strip()


def differences(name: str, left: Emission, right: Emission) -> list[str]:
    """Every way these two disagree about one module."""
    if left.error or right.error:
        if left.error == right.error:
            return []
        return [f"{name}: refused differently -- {left.error!r} vs {right.error!r}"]

    out: list[str] = []
    if left.jitted != right.jitted:
        out.append(
            f"{name}: kernels differ -- only original "
            f"{sorted(set(left.jitted) - set(right.jitted))}, only migrated "
            f"{sorted(set(right.jitted) - set(left.jitted))}"
        )
    if sorted(left.delegated) != sorted(right.delegated):
        out.append(
            f"{name}: host-delegation differs -- only original "
            f"{sorted(set(left.delegated) - set(right.delegated))}, only migrated "
            f"{sorted(set(right.delegated) - set(left.delegated))}"
        )
    for subprogram in sorted(set(left.delegated) & set(right.delegated)):
        one, two = category(left.delegated[subprogram]), category(right.delegated[subprogram])
        if one != two:
            out.append(f"{name}/{subprogram}: delegated for {one!r} vs {two!r}")

    if len(left.pieces) != len(right.pieces):
        out.append(f"{name}: {len(left.pieces)} emitted piece(s) vs {len(right.pieces)}")
        return out
    for index, (a, b) in enumerate(zip(left.pieces, right.pieces, strict=True)):
        if a != b:
            out.append(f"{name}: piece {index} differs\n{_first_difference(a, b)}")
    return out


def _first_difference(a: str, b: str) -> str:
    """The first line that is not identical, with both spellings."""
    left, right = a.splitlines(), b.splitlines()
    for number, (one, two) in enumerate(zip(left, right, strict=False), 1):
        if one != two:
            return f"      line {number}\n      original: {one!r}\n      migrated: {two!r}"
    return f"      one side has {len(left)} lines, the other {len(right)}"


def survey(emitter: Any, modules: list[Path]) -> int:
    """What the collection's backend does across the corpus, before anything
    is migrated. Not a check -- a measurement, and the baseline a reviewer
    needs to tell a migration regression from a limit that was always there."""
    built = refused = kernels = delegated = 0
    reasons: dict[str, int] = {}
    for module_dir in modules:
        result = emit(emitter, module_dir)
        if not result.ok:
            refused += 1
            print(f"  refused  {module_dir.name}: {result.error}")
            continue
        built += 1
        kernels += len(result.jitted)
        delegated += len(result.delegated)
        for reason in result.delegated.values():
            key = category(reason)
            reasons[key] = reasons.get(key, 0) + 1

    print(f"\n{len(modules)} module(s): {built} built, {refused} refused")
    print(f"{kernels} kernel(s) emitted, {delegated} host-delegated")
    print("\nwhy a subprogram was host-delegated:")
    for reason, count in sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0])):
        print(f"  {count:5}  {reason}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--collection",
        type=Path,
        required=True,
        help="the directory holding the collection's jaxize.py",
    )
    ap.add_argument(
        "--suite-dir",
        type=Path,
        required=True,
        help="directory of <stem>/ dirs with interface.json + <stem>_numpy.py",
    )
    ap.add_argument("--only", help="check just this module")
    ap.add_argument(
        "--survey",
        action="store_true",
        help="measure the collection's backend alone; no migrated side needed",
    )
    ns = ap.parse_args()

    original = load_original(ns.collection)
    modules = corpus(ns.suite_dir)
    if ns.only:
        modules = [m for m in modules if m.name == ns.only]
        if not modules:
            raise SystemExit(f"no module {ns.only!r} under {ns.suite_dir}")
    if not modules:
        raise SystemExit(f"no suite directories under {ns.suite_dir}")

    if ns.survey:
        return survey(original, modules)

    migrated = load_migrated()
    problems: list[str] = []
    for module_dir in modules:
        problems += differences(
            module_dir.name, emit(original, module_dir), emit(migrated, module_dir)
        )

    if problems:
        print(f"{len(problems)} disagreement(s) across {len(modules)} module(s):\n")
        for problem in problems:
            print(f"  {problem}")
        return 1
    print(f"{len(modules)} module(s) emit identically.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
