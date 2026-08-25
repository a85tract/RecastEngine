#!/usr/bin/env python3
"""Check the migrated CUDA backend against the pipeline it was migrated from.

The third of the emission differentials, after ``emit_diff.py`` (NumPy) and
``numba_diff.py`` (Numba), and it works the same way: import the pipeline's
``cudaize.py``, run both emitters over the same live-extracted analysis of the
same sources, compare every emitted byte. Live against their code, never
against ``translated/wv_sat_methods_cuda.py``.

Device functions are compared for every module the other differentials cover,
which is wider than upstream ever ran this emitter -- it was built for one
module, and a rule that only ever saw one module is exactly the kind that
turns out to have been written against it. Both emitters are pointed at the
rest of the corpus here, and a disagreement anywhere is a difference.

Launchers are compared only where a launcher configuration exists, because
upstream's is hard-coded: which subprograms get one, and which closure entries
are integers, are facts about ``wv_sat_methods``. The engine takes them as
configuration, so this harness supplies the values upstream has inline --
which is also what makes the comparison meaningful rather than a comparison of
two different exclusion lists.
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
import numba_diff  # noqa: E402

from recast.fortran import constants as fconstants  # noqa: E402
from recast.fortran import interface as finterface  # noqa: E402
from recast.fortran._parse import f03, parse, walk  # noqa: E402
from recast.transform.cuda.emitter import CudaSubprograms  # noqa: E402
from recast.transform.cuda.translate import CudaTranslation  # noqa: E402
from recast.transform.numba.backend import Kernels  # noqa: E402
from recast.transform.numba.emitter import Emission  # noqa: E402
from recast.transform.numpy.statements import REFUSED  # noqa: E402
from recast.transform.profiles import PROFILES  # noqa: E402

LAUNCHER_CONFIG: dict[str, dict[str, Any]] = {
    "wvsat": {
        # ``cudaize.emit_launchers`` skips these inline: the four
        # interchangeable saturation-vapour-pressure schemes, which are
        # reached through a dispatch rather than called directly, and an
        # index lookup whose result is not a float array.
        "launchers_exclude": (
            "goffgratch_svp_water",
            "goffgratch_svp_ice",
            "murphykoop_svp_water",
            "murphykoop_svp_ice",
            "oldgoffgratch_svp_water",
            "oldgoffgratch_svp_ice",
            "bolton_svp_water",
            "wv_sat_valid_idx",
        ),
        "integer_state": ("default_idx",),
    }
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--translator", type=Path, required=True)
    ap.add_argument("--only", help="check just this module")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--scratch", type=Path, default=None)
    ns = ap.parse_args()

    root = ns.translator.resolve()
    sys.path.insert(0, str(root / "pipeline"))
    import cudaize
    import translate as pipeline

    refusal_types = (pipeline.AgentQueue,)

    intrinsic_overrides = emit_diff._intrinsics()
    scratch = ns.scratch or Path(tempfile.mkdtemp(prefix="cuda_diff_"))
    scratch.mkdir(parents=True, exist_ok=True)

    modules = list(emit_diff.MODULES)
    sweep = root / "extracted_auto"
    if sweep.is_dir():
        for swept in sorted(sweep.iterdir()):
            if (swept / "interface.json").is_file():
                modules.append((swept.name, None, None, None, None, None))

    totals = dict.fromkeys(
        ["modules", "devices", "delegated", "launchers", "different", "crashed"], 0
    )

    for name, override, _use_p, _ext_p, _comp_p, intents_p in modules:
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
        # No externals, on either side. ``cudaize.py``'s CLI has no
        # ``--externals`` flag and its emitter is constructed without one, so a
        # name the NumPy backend would send to an audited shim is an ordinary
        # intrinsic here. Handing the engine a table the pipeline does not have
        # would compare two different configurations.
        externals: dict[str, Any] = {}

        emitter = cudaize.CudaEmitter(interface, constants["literal_map"], {})
        kernels = Kernels(record=interface, externals=externals)
        emission = Emission(kernels=kernels)
        assembler = CudaSubprograms(
            record=interface,
            constants=constants,
            profile=PROFILES["ifx"],
            externals=externals,
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

        different = 0
        devices = 0
        delegated = 0
        crashed = 0
        for record in interface["subprograms"]:
            subprogram = record["name"]
            if subprogram in duplicated or subprogram not in nodes:
                continue
            if subprogram not in emitter.kernels or subprogram not in kernels.names:
                if (subprogram in emitter.kernels) != (subprogram in kernels.names):
                    different += 1
                    if ns.verbose:
                        print(f"    ELIGIBILITY {subprogram}")
                else:
                    delegated += 1
                continue
            try:
                theirs = emitter.emit_kernels(nodes[subprogram], record)
            except refusal_types as refusal:
                theirs, why_theirs = None, str(refusal)
            except Exception as error:
                # Not a refusal: their emitter raised. There is nothing to
                # compare a crash against, so it is counted apart from a
                # difference -- saying the engine disagrees would be false.
                crashed += 1
                if ns.verbose:
                    print(f"    PIPELINE CRASHED {subprogram}: {type(error).__name__}: {error}")
                continue
            else:
                why_theirs = ""
            try:
                ours, _ = assembler.render(nodes[subprogram], subprogram)
            except (*REFUSED, KeyError) as refusal:
                ours, why_ours = None, str(refusal)
            else:
                why_ours = ""
            if theirs is None or ours is None:
                if (theirs is None) != (ours is None):
                    different += 1
                    if ns.verbose:
                        print(
                            f"    DELEGATION {subprogram}: pipeline "
                            f"{why_theirs or 'emitted'}, engine {why_ours or 'emitted'}"
                        )
                else:
                    delegated += 1
                    emitter.kernels.discard(subprogram)
                    kernels.names.discard(subprogram)
                continue
            devices += 1
            if not numba_diff.report(f"DEVICE {subprogram}", theirs, ours, ns.verbose):
                different += 1

        launchers = 0
        configured = LAUNCHER_CONFIG.get(name)
        if configured is not None:
            theirs_l = cudaize.emit_launchers(emitter, interface)
            ours_l = CudaTranslation.launchers(
                interface,
                emission,
                exclude=configured["launchers_exclude"],
                integer_state=frozenset(configured["integer_state"]),
            )
            launchers = len(ours_l)
            if not numba_diff.report("LAUNCHERS", theirs_l, ours_l, ns.verbose):
                different += 1

        totals["modules"] += 1
        totals["devices"] += devices
        totals["delegated"] += delegated
        totals["launchers"] += launchers
        totals["different"] += different
        totals["crashed"] += crashed
        line = f"{name:<24} devices={devices}  delegated={delegated}"
        if launchers:
            line += f"  launchers={launchers}"
        if crashed:
            line += f"  pipeline-crashed={crashed}"
        if different:
            line += f"  different={different}"
        print(line)

    print()
    print("  ".join(f"{k}={v}" for k, v in totals.items()))
    return 1 if totals["different"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
