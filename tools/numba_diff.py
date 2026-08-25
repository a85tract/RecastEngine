#!/usr/bin/env python3
"""Check the migrated Numba backend against the pipeline it was migrated from.

``emit_diff.py`` does this for the NumPy emitter; this is the same check one
floor up. It imports the pipeline's ``numbaize.py`` and the engine's Numba
floors, runs both over the same live-extracted analysis of the same sources,
and compares each emitted kernel and host wrapper byte for byte.

Live against the pipeline's code, never against ``translated/*_njit.py``. That
directory holds three files, the newest of them older than several rules that
have changed since -- and one of them lists a kernel the pipeline's current
eligibility rule refuses. Diffing against it would measure the age of a file.

Everything ``emit_diff`` says about ``RECAST_INTRINSICS`` applies here
unchanged: without it the pipeline's ``ifx`` profile spells its transcendentals
through a maths library the engine does not, and every one of them reads as a
difference. The setup is imported from ``emit_diff`` rather than copied, so the
two harnesses cannot drift into extracting the analysis differently.

What is compared, per subprogram:

* **eligibility** -- the two must agree on whether it is a kernel at all;
* **delegation** -- if either refuses the body, both must, and the *reason
  category* must match (``[elig]`` versus ``[emit]``). The reason prose is
  normalized away, as it is in ``emit_diff``: a reason is a diagnostic;
* **the kernel** -- decorator, signature, docstring, prologue, block markers,
  body and trailing return, byte for byte;
* **the host wrapper** -- signature and the closure it fills, byte for byte.

Module headers are not compared, for the reason ``recast.transform.numpy.
modules`` gives for the NumPy one: the runtime here is real, typed, tested code
rather than a string constant, and its emitted text follows the code.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import emit_diff  # noqa: E402

from recast.fortran import constants as fconstants  # noqa: E402
from recast.fortran import interface as finterface  # noqa: E402
from recast.fortran._parse import f03, parse, walk  # noqa: E402
from recast.transform.numba.backend import Kernels, ineligible_reason  # noqa: E402
from recast.transform.numba.emitter import Emission, NumbaSubprograms  # noqa: E402
from recast.transform.numpy.statements import REFUSED  # noqa: E402
from recast.transform.profiles import PROFILES  # noqa: E402


def category(reason: str) -> str:
    """``[elig]`` or ``[emit]`` -- the part of a refusal that is a decision."""
    return "elig" if str(reason).startswith("[elig]") else "emit"


def kernel_of(emitter: Any, node: Any, record: dict[str, Any]) -> tuple[list[str] | None, str]:
    """The pipeline's kernel for one subprogram, or why it delegated."""
    try:
        return emitter.emit_kernel(node, record), ""
    except Exception as refusal:
        return None, f"[emit] {refusal}"


def engine_kernel(
    assembler: NumbaSubprograms, node: Any, name: str
) -> tuple[list[str] | None, str]:
    try:
        lines, _report = assembler.render(node, name)
    except (*REFUSED, KeyError) as refusal:
        return None, f"[emit] {refusal}"
    return lines, ""


def report(label: str, theirs: list[str] | None, ours: list[str] | None, verbose: bool) -> bool:
    """True when the two agree. Prints the first divergence when asked."""
    if theirs == ours:
        return True
    if not verbose:
        return False
    print(f"    --- {label}")
    left = theirs or ["<delegated>"]
    right = ours or ["<delegated>"]
    for at in range(max(len(left), len(right))):
        a = left[at] if at < len(left) else "<end>"
        b = right[at] if at < len(right) else "<end>"
        if a != b:
            print(f"      pipeline: {a}")
            print(f"      engine  : {b}")
            break
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--translator", type=Path, required=True)
    ap.add_argument("--only", help="check just this module")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--scratch", type=Path, default=None)
    ns = ap.parse_args()

    root = ns.translator.resolve()
    sys.path.insert(0, str(root / "pipeline"))
    import numbaize
    import translate as pipeline

    intrinsic_overrides = emit_diff._intrinsics()
    scratch = ns.scratch or Path(tempfile.mkdtemp(prefix="numba_diff_"))
    scratch.mkdir(parents=True, exist_ok=True)

    modules = list(emit_diff.MODULES)
    sweep = root / "extracted_auto"
    if sweep.is_dir():
        for swept in sorted(sweep.iterdir()):
            if not (swept / "interface.json").is_file():
                continue
            companions = swept / "companions.json"
            modules.append(
                (
                    swept.name,
                    None,
                    None,
                    None,
                    str(companions.relative_to(root)) if companions.is_file() else None,
                    None,
                )
            )

    totals = dict.fromkeys(["modules", "kernels", "delegated", "lines", "different", "error"], 0)

    for name, override, use_p, ext_p, comp_p, intents_p in modules:
        if ns.only and name != ns.only:
            continue
        gold = next(
            json.loads((root / area / name / "interface.json").read_text())
            for area in ("extracted", "extracted_auto")
            if (root / area / name / "interface.json").is_file()
        )
        source = root / (override or gold["source_file"])
        intents = json.loads((root / intents_p).read_text()) if intents_p else None
        interface = finterface.extract(
            source, kind_assumptions=emit_diff.CAM_KINDS, intent_overrides=intents
        )
        constants = fconstants.extract(source)
        use_params = emit_diff.load(root, use_p)
        externals = emit_diff.load(root, ext_p)
        fixed, records, remotes, companion_globals = emit_diff.companion_tables(
            root, comp_p, scratch
        )

        try:
            emitter = numbaize.NjitEmitter(
                interface,
                constants["literal_map"],
                {},
                externals=externals,
                use_params=use_params,
                companions_meta=fixed or None,
                profile="ifx",
            )
        except Exception as error:
            print(f"{name}: pipeline emitter failed to build: {error}")
            totals["error"] += 1
            continue

        aliased = {c["alias"]: r for c, r in zip(fixed, records, strict=True)}
        kernels = Kernels(
            record=interface, companions=aliased, remotes=remotes, externals=externals
        )
        emission = Emission(kernels=kernels)
        assembler = NumbaSubprograms(
            record=interface,
            constants=constants,
            profile=PROFILES["ifx"],
            companions=tuple(records),
            use_parameters=use_params,
            companion_globals=companion_globals,
            externals=externals,
            remotes=remotes,
            function_stubs=dict(pipeline.INFRA_FN_STUBS),
            statement_stubs=dict(pipeline.INFRA_STUBS),
            intrinsics=intrinsic_overrides,
            emission=emission,
        )

        tree = parse(source)
        parsed = walk(tree, f03.Module)
        scope = parsed[0] if parsed else tree
        emitter.prescan_module_allocates(scope)
        nodes = {}
        for subprogram in walk(scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
            heading = walk(subprogram, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
            nodes[str(heading.children[1]).lower()] = subprogram

        declared = [s["name"] for s in interface["subprograms"]]
        duplicated = {n for n in declared if declared.count(n) > 1}

        module_different = 0
        counts = {"kernels": 0, "delegated": 0, "lines": 0}
        for record in interface["subprograms"]:
            subprogram = record["name"]
            if subprogram in duplicated or subprogram not in nodes:
                continue
            theirs_eligible = subprogram in emitter.kernels
            ours_eligible = subprogram in kernels.names
            if theirs_eligible != ours_eligible:
                module_different += 1
                if ns.verbose:
                    print(
                        f"    ELIGIBILITY {subprogram}: pipeline "
                        f"{'kernel' if theirs_eligible else 'delegated'}, engine "
                        f"{'kernel' if ours_eligible else 'delegated'} "
                        f"({ineligible_reason(record, externals)})"
                    )
                continue
            if not theirs_eligible:
                counts["delegated"] += 1
                continue

            theirs, why_theirs = kernel_of(emitter, nodes[subprogram], record)
            ours, why_ours = engine_kernel(assembler, nodes[subprogram], subprogram)
            if theirs is None or ours is None:
                if (theirs is None) != (ours is None):
                    module_different += 1
                    if ns.verbose:
                        print(
                            f"    DELEGATION {subprogram}: pipeline "
                            f"{why_theirs or 'emitted'}, engine {why_ours or 'emitted'}"
                        )
                elif category(why_theirs) != category(why_ours):
                    module_different += 1
                    if ns.verbose:
                        print(f"    REASON {subprogram}: {why_theirs} vs {why_ours}")
                else:
                    counts["delegated"] += 1
                    emitter.kernels.discard(subprogram)
                    kernels.names.discard(subprogram)
                continue

            counts["kernels"] += 1
            counts["lines"] += len(ours)
            if not report(f"KERNEL {subprogram}", theirs, ours, ns.verbose):
                module_different += 1

        for record in interface["subprograms"]:
            subprogram = record["name"]
            if subprogram not in emitter.kernels or subprogram not in kernels.names:
                continue
            try:
                theirs = emitter.emit_wrapper(record)
            except Exception as error:
                theirs = [f"<failed: {error}>"]
            ours = assembler.wrapper(record)
            if not report(f"WRAPPER {subprogram}", theirs, ours, ns.verbose):
                module_different += 1

        totals["modules"] += 1
        totals["kernels"] += counts["kernels"]
        totals["delegated"] += counts["delegated"]
        totals["lines"] += counts["lines"]
        totals["different"] += module_different
        summary = (
            f"{name:<24} kernels={counts['kernels']}  delegated={counts['delegated']}  "
            f"lines={counts['lines']}"
        )
        if module_different:
            summary += f"  different={module_different}"
        print(summary)

    print()
    print("  ".join(f"{k}={v}" for k, v in totals.items()))
    return 1 if totals["different"] or totals["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
