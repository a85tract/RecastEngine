#!/usr/bin/env python3
"""Check the migrated emitter against the pipeline it was migrated from.

``golden_diff.py`` says the *analysis* still answers the same; this says the
*emission* still writes the same. It imports the pipeline's ``translate.py``
and the engine's statement layer, runs both over the same live-extracted
analysis of the same sources, and compares the emitted Python statement by
statement -- byte for byte, because the pipeline's output is what the
bit-exact gates have been run against and a single reflowed parenthesis is
indistinguishable from a wrong number until run time.

The pipeline's ``ifx`` profile spells its transcendentals through a maths
library that is not the system one, and the table saying which lives with the
package that knows the build. Point ``RECAST_INTRINSICS`` at it
(``module`` or ``module:NAME``) or every such call reads as a difference, and
many subprograms fail to emit at all -- so the run reports *more* differences
over *fewer* subprograms, and the count is not comparable to one taken with
the variable set. A count from a run without it means nothing on its own.

Live against the pipeline's code, not against its stored output: comparing
against a golden file older than the code that wrote it is how this
repository once reported fixing a bug the pipeline did not have.

Whole subprograms are compared: signature, docstring, determinizing
prologue, block markers, the body, and the trailing return -- plus the block
report both emitters produce, which says which blocks are mechanical and
which are deferred. Refusal *placement* is compared strictly (the same
blocks must defer, with the same spans and line counts); refusal *prose* is
normalized away, because the two sides word their reasons differently and a
reason string is a diagnostic, not a number.

Above the subprograms, the whole module body -- type factories, module state
initialization, every subprogram in order -- and the embedded signature
table are compared against a real run of the pipeline's ``main()`` (as a
subprocess, patch-free, over the same live-extracted analysis), along with
the final block report at header-relative line numbers. The *headers* are
not compared: the engine's runtime is real, typed, tested code and its
emitted text deliberately differs from the pipeline's string constant --
see ``recast.transform.numpy.modules`` for that decision.

The corpus is the six schemes with full operator tables plus every module the
translator's batch sweep produced, discovered from ``extracted_auto/`` at run
time so a new sweep widens this check without anyone editing it. One skip: a
subprogram whose name appears twice in one file -- unpreprocessed ``#if``
variants -- is left out, because pairing records with definitions is ambiguous
there on both sides and a mismatched pairing measures the harness, not the
emitters.

Usage:
    uv run --extra fortran tools/emit_diff.py --translator ../<translator-checkout>

Exit status is 1 on any difference or error, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recast.fortran import constants as fconstants
from recast.fortran import interface as finterface
from recast.fortran._parse import f03, parse, walk
from recast.transform.numpy.constants import constants_module
from recast.transform.numpy.expressions import Remote
from recast.transform.numpy.modules import Modules
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.numpy.translate import companion_tables as companion_views
from recast.transform.profiles import PROFILES

REASON_IN_MARKER = re.compile(r"(# B\d{3} <- L\d+-L\d+ AGENT_QUEUE: ).*")
REASON_IN_RAISE = re.compile(r"(raise NotImplementedError\().*(\)  # B\d{3})")
SOURCE_PATH = re.compile(r"#\s+\S*/(\S+\.F90)(:\d+)", re.IGNORECASE)


def compared_against(root: Path) -> list[str]:
    """What this run measured, so the count it prints has something to pin it to.

    A differential's answer is a claim about two trees, and only one of them is
    in this repository. Without the other one's revision the number is
    unfalsifiable later: ``different=5`` recorded in a document says nothing
    about which upstream it was 5 against, and upstream moves. It moved under
    this repository once already -- a squashed single commit was replaced with
    249 real ones -- and nothing here could have detected it.
    """
    git_binary = shutil.which("git")

    def git(*arguments: str) -> str:
        if git_binary is None:
            return ""
        finished = subprocess.run(  # noqa: S603 -- git, resolved, on a path the caller named
            [git_binary, *arguments],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        return finished.stdout.strip() if finished.returncode == 0 else ""

    revision = git("rev-parse", "--short=9", "HEAD") or "unknown revision"
    if git("status", "--porcelain"):
        revision += " +dirty"
    return [f"compared against {root}@{revision}"]


def normalized(line: str) -> str:
    """One emitted line, with the two things that are not claims removed.

    Refusal prose, because the two sides word their reasons differently and a
    reason is a diagnostic. And the directory part of a source path: the
    pipeline stamped each hoisted constant with whatever path its command line
    was given, while the engine emits the file's name, because an absolute path
    in a generated file makes the artifact -- and the Candidate digest that
    identifies it -- differ between two machines that translated the same
    source. That is a reproducibility defect the engine deliberately declines
    to reproduce; the file name, the line, and the value still have to match.
    """
    line = SOURCE_PATH.sub(r"# \1\2", line)
    return REASON_IN_RAISE.sub(r"\1<reason>\2", REASON_IN_MARKER.sub(r"\1<reason>", line))


MODULES: list[tuple[str, str | None, str | None, str | None, str | None, str | None]] = [
    # name, source override, use_params, externals, companions, intent overrides
    # -- all paths relative to the translator checkout. These are the six
    # schemes with hand-maintained operator tables; the batch-swept modules
    # are discovered from extracted_auto/ at run time.
    ("mg_utils", None, None, "reports/mg/externals_utils.json", None, None),
    (
        "mg2",
        # The extracted golden's source_file predates the suite layout.
        "reports/suite/micro_mg2_0/micro_mg2_0_cpp.F90",
        None,
        None,
        "reports/mg/companions_mg2.json",
        None,
    ),
    (
        "wvsat",
        None,
        "extracted/wvsat/use_params.json",
        "reports/wvsat/externals_wvsat.json",
        "reports/wvsat/companions_wvsat.json",
        None,
    ),
    (
        "cwm",
        None,
        "extracted/cwm/use_params.json",
        "reports/cwm/externals.json",
        "reports/cwm/companions.json",
        None,
    ),
    (
        "cldfrc2m",
        None,
        None,
        "reports/cldfrc2m/externals.json",
        "reports/cldfrc2m/companions.json",
        None,
    ),
    (
        "zm_conv",
        None,
        "extracted/zm_conv/use_params.json",
        "reports/zm/externals.json",
        None,
        "reports/zm/intent_overrides.json",
    ),
]


def load(root: Path, relative: str | None) -> dict[str, Any]:
    if relative is None:
        return {}
    return {k.lower(): v for k, v in json.loads((root / relative).read_text()).items()}


# A companion interface whose ``source_file`` no longer says what the companion
# means. The translator's root ``extracted/interface.json`` was wv_sat_methods
# when ``companions_mg2.json`` and ``companions_wvsat.json`` were written, and
# a 2026-07-23 extraction overwrote it with dadadj without touching either
# config -- their ``module_py`` still says ``wv_sat_methods_numpy``. The
# companion's meaning is the config's, so the source is taken from here rather
# than from the overwritten file. Upstream's to fix; listed for the author.
def _intrinsics() -> dict[str, Any]:
    """The intrinsic spellings the pipeline emits under its ``ifx`` profile.

    A maths library that is not the system one is a fact about the build, and
    the package that knows the build ships the table -- so this harness is
    told where to find it rather than knowing: ``--intrinsics
    <module>:<name>``, or ``RECAST_INTRINSICS`` in the same form. Without one
    the comparison is against a translation that spells the transcendentals
    differently, and every one of them shows up as a difference.
    """
    import importlib

    spec = os.environ.get("RECAST_INTRINSICS", "")
    if not spec:
        return {}
    module, _, attribute = spec.partition(":")
    return dict(getattr(importlib.import_module(module), attribute or "OVERRIDES"))


CAM_KINDS = {
    "r8": "float64",
    "r4": "float32",
    "i8": "int64",
    "shr_kind_r8": "float64",
    "shr_kind_r4": "float32",
    "shr_kind_i8": "int64",
    "shr_kind_i4": "int32",
}
"""What the extension supplies at run time. Spelled out here because this
harness talks to the pipeline directly, with no plugin in the path."""

COMPANION_SOURCES: dict[str, str] = {
    "extracted/interface.json": "src_fortran/wv_sat_methods.F90",
}


def companion_tables(
    root: Path, config_path: str | None, scratch: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Remote], dict[str, str]]:
    """The four views of the companion modules, from one live extraction.

    The pipeline reads companion interfaces from JSON files, so each one is
    re-extracted from its source and written where the pipeline can read it
    -- the goldens may be stale, and a staleness difference is not an
    emission difference. Mirrors ``Translator.load_companions``.
    """
    if config_path is None:
        return [], [], {}, {}
    configured = json.loads((root / config_path).read_text())
    fixed, entries = [], []
    for companion in configured:
        stale = json.loads((root / companion["interface"]).read_text())
        source = COMPANION_SOURCES.get(companion["interface"], stale["source_file"])
        record = finterface.extract(root / source)
        constants = fconstants.extract(root / source)
        # The pipeline spells a companion's parameter the way the constants
        # module beside its interface.json spells it, so each live interface
        # gets a directory of its own with that file in it -- rendered by the
        # engine, whose constants emission emit_diff holds to the pipeline's
        # anyway. The file has to carry the stem the descriptor declares, not
        # a fixed "constants": the pipeline looks it up under
        # ``constants_stem``, and writing it anywhere else leaves that lookup
        # empty -- which is not a missing file to the pipeline but a companion
        # whose parameters are all unresolved, so every reference to one comes
        # out lower-cased and every line carrying one reads as a difference.
        live_dir = scratch / record["module"]
        live_dir.mkdir(exist_ok=True)
        live = live_dir / "interface.json"
        live.write_text(json.dumps(record))
        stem = companion.get("constants_stem") or "constants"
        (live_dir / f"{stem}.py").write_text(constants_module(constants))
        fixed.append({**companion, "interface": str(live)})
        entries.append({**companion, "record": record, "constants": constants})
    records, remotes, globals_, _imports = companion_views(entries)
    return fixed, list(records), remotes, globals_


def pipeline_module(
    root: Path,
    scratch: Path,
    name: str,
    source: Path,
    interface: dict[str, Any],
    literal_map: dict[str, Any],
    use_p: str | None,
    ext_p: str | None,
    fixed: list[dict[str, Any]],
) -> tuple[str, str, list[dict[str, Any]]]:
    """Run the pipeline's main() patch-free; -> (body, sigs line, final
    report). The body is everything after the _SIGNATURES paragraph."""
    (scratch / f"{name}_interface.json").write_text(json.dumps(interface))
    (scratch / f"{name}_literal_map.json").write_text(json.dumps(literal_map))
    out = scratch / f"{name}_numpy.py"
    report = scratch / f"{name}_report.json"
    command = [
        sys.executable,
        str(root / "pipeline" / "translate.py"),
        str(source),
        "--interface",
        str(scratch / f"{name}_interface.json"),
        "--literal-map",
        str(scratch / f"{name}_literal_map.json"),
        "-o",
        str(out),
        "--report",
        str(report),
        "--profile",
        "ifx",
    ]
    if use_p:
        command += ["--use-params", str(root / use_p)]
    if ext_p:
        command += ["--externals", str(root / ext_p)]
    if fixed:
        config = scratch / f"{name}_companions.json"
        config.write_text(json.dumps(fixed))
        command += ["--companions", str(config)]
    finished = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        command, cwd=scratch, capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        raise RuntimeError(f"pipeline main() failed: {finished.stderr.strip()[-500:]}")
    text = out.read_text()
    sig_start = text.index("_SIGNATURES = ")
    sig_end = text.index("\n", sig_start)
    body = text[sig_end + 2 :]
    return body, text[sig_start:sig_end], json.loads(report.read_text())


def pipeline_constants(root: Path, scratch: Path, name: str, source: Path) -> str:
    """Run the pipeline's extract_constants.py; -> the generated constants.py."""
    out = scratch / f"{name}_constants.py"
    command = [
        sys.executable,
        str(root / "pipeline" / "extract_constants.py"),
        str(source),
        "-o",
        str(out),
        "--map",
        str(scratch / f"{name}_constants_map.json"),
    ]
    finished = subprocess.run(  # noqa: S603 -- our own interpreter, our own script
        command, cwd=scratch, capture_output=True, text=True, check=False
    )
    if finished.returncode != 0:
        raise RuntimeError(f"extract_constants failed: {finished.stderr.strip()[-500:]}")
    return out.read_text()


def comparable(report: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The part of a block report that means the same thing on both sides.

    Dropped, and why:

    ``reason`` -- refusal wording, which the two spell differently for the
    same refusal and which nothing consumes by value.

    ``key`` -- the ``host/name`` form the engine records so an internal
    procedure can be addressed; the pipeline has no equivalent. A difference
    in what the record *can say*, not in the translation. The per-subprogram
    comparison in this file already dropped it; this path did not, and that
    alone marked thirteen modules different.

    ``py_lines`` -- each emitter's bookkeeping about its *own* file. They are
    not the same measurement: the engine's are absolute and land exactly on
    the block's ``# Bnnn <- ...`` marker, while the pipeline's first block
    reports line 6 of a file whose line 6 is the closing ``\"\"\"`` of the
    module docstring, and the gap between the two grows with each subprogram
    rather than staying at the header's height. So no single origin
    reconciles them and there is nothing here to compare. What both sides
    mean identically -- the block ids, their order, their status, and the
    Fortran ``src_span`` they came from -- is still compared.
    """
    return [
        {k: v for k, v in entry.items() if k not in ("reason", "key", "py_lines")}
        for entry in report
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--translator",
        type=Path,
        required=True,
        help="the translator checkout to compare against",
    )
    ap.add_argument("--only", help="check just this module")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every disagreement")
    ap.add_argument(
        "--scratch",
        type=Path,
        default=None,
        help="where the live companion interfaces are written for the pipeline to read",
    )
    ns = ap.parse_args()

    root = ns.translator.resolve()
    for line in compared_against(root):
        print(line)
    print()
    sys.path.insert(0, str(root / "pipeline"))
    import translate as pipeline

    intrinsic_overrides = _intrinsics()
    scratch = ns.scratch or Path(tempfile.mkdtemp(prefix="emit_diff_"))
    scratch.mkdir(parents=True, exist_ok=True)
    kinds = ["subprograms", "lines", "blocks", "deferred", "modules", "constants", "different"]
    totals = dict.fromkeys([*kinds, "skipped", "error"], 0)

    modules = list(MODULES)
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

        interface = finterface.extract(source, kind_assumptions=CAM_KINDS, intent_overrides=intents)
        constants = fconstants.extract(source)
        use_params = load(root, use_p)
        externals = load(root, ext_p)
        fixed, records, remotes, companion_globals = companion_tables(root, comp_p, scratch)

        translator = pipeline.Translator(
            interface,
            constants["literal_map"],
            patches={},
            use_params=use_params,
            externals=externals,
            companions=fixed,
            profile="ifx",
        )

        tree = parse(source)
        parsed = walk(tree, f03.Module)
        scope = parsed[0] if parsed else tree
        # The pipeline's ``main()`` runs this before translating anything, and
        # without it every reference to a module allocatable falls back to the
        # declared bounds -- so ``mam_idx(m, 0)``, allocated ``0:nspec_max``,
        # compares as ``[m - 1, 0 - 1]`` against a real run's ``[m - 1, 0]``.
        # Comparing against a state the pipeline is never in measures this
        # harness, not the emitters.
        translator.prescan_module_allocates(scope)
        nodes = {}
        for subprogram in walk(scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
            heading = walk(subprogram, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
            nodes[str(heading.children[1]).lower()] = subprogram

        declared = [s["name"] for s in interface["subprograms"]]
        duplicated = {n for n in declared if declared.count(n) > 1}
        if duplicated:
            print(f"{name}: skipping {sorted(duplicated)} (unpreprocessed #if variants)")

        assembler = Subprograms(
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
        )

        counts = dict.fromkeys(totals, 0)
        for record in interface["subprograms"]:
            node = nodes.get(record["name"])
            if node is None:
                continue
            if record["name"] in duplicated:
                counts["skipped"] += 1
                continue
            where = f"{name}/{record['name']}"
            try:
                want, want_report = translator.translate_subprogram(node, record)
            except Exception as error:  # a crash is a finding, not a refusal
                print(f"PIPELINE ERROR {where}: {type(error).__name__}: {error}")
                counts["error"] += 1
                continue
            try:
                got, got_report = assembler.render(node, record["name"])
            except Exception as error:
                print(f"ENGINE ERROR {where}: {type(error).__name__}: {error}")
                counts["error"] += 1
                continue

            counts["blocks"] += len(want_report)
            counts["deferred"] += sum(1 for e in want_report if e["status"] == "agent_queue")
            placements = [
                [{k: v for k, v in entry.items() if k not in ("reason", "key")} for entry in report]
                for report in (want_report, got_report)
            ]
            same_lines = list(map(normalized, want)) == list(map(normalized, got))
            if same_lines and placements[0] == placements[1]:
                counts["subprograms"] += 1
                counts["lines"] += len(want)
                continue
            counts["different"] += 1
            print(
                f"DIFFERENT {where}"
                + ("" if same_lines else " (lines)")
                + ("" if placements[0] == placements[1] else " (report)")
            )
            if ns.verbose:
                for a, b in zip(want, got, strict=False):
                    if normalized(a) != normalized(b):
                        print(f"  pipeline |{a}")
                        print(f"  engine   |{b}")
                for a in want[len(got) :]:
                    print(f"  pipeline only |{a}")
                for b in got[len(want) :]:
                    print(f"  engine only   |{b}")
        want_consts = pipeline_constants(root, scratch, name, source)
        got_consts = constants_module(fconstants.extract(source))
        marker = "# ----- module-level parameters"
        want_tail = [normalized(x) for x in want_consts[want_consts.index(marker) :].splitlines()]
        got_tail = [normalized(x) for x in got_consts[got_consts.index(marker) :].splitlines()]
        if want_tail == got_tail:
            counts["constants"] += 1
        else:
            counts["different"] += 1
            print(f"CONSTANTS DIFFERENT {name}")
            if ns.verbose:
                for a, b in zip(want_tail, got_tail, strict=False):
                    if a != b:
                        print(f"  pipeline |{a}")
                        print(f"  engine   |{b}")

        if not duplicated:
            renderer = Modules(
                subprograms=assembler,
                companion_imports=tuple(f"import {c['module_py']} as {c['alias']}" for c in fixed),
            )
            try:
                want_body, want_sigs, want_report = pipeline_module(
                    root,
                    scratch,
                    name,
                    source,
                    interface,
                    constants["literal_map"],
                    use_p,
                    ext_p,
                    fixed,
                )
                got_text, got_report = renderer.render(source)
                compile(got_text, f"{name}_numpy.py", "exec")
                got_body = "\n".join(renderer.body(nodes)[0])
                got_sigs = f"_SIGNATURES = {renderer._signatures()!r}"
            except Exception as error:
                print(f"MODULE ERROR {name}: {type(error).__name__}: {error}")
                counts["error"] += 1
            else:
                same_body = [normalized(l_) for l_ in want_body.splitlines()] == [
                    normalized(l_) for l_ in got_body.splitlines()
                ]
                same_report = comparable(want_report) == comparable(got_report)
                if same_body and want_sigs == got_sigs and same_report:
                    counts["modules"] += 1
                else:
                    counts["different"] += 1
                    print(
                        f"MODULE DIFFERENT {name}"
                        + ("" if same_body else " (body)")
                        + ("" if want_sigs == got_sigs else " (signatures)")
                        + ("" if same_report else " (report)")
                    )
                    if ns.verbose and not same_body:
                        for a, b in zip(
                            want_body.splitlines(), got_body.splitlines(), strict=False
                        ):
                            if normalized(a) != normalized(b):
                                print(f"  pipeline |{a}")
                                print(f"  engine   |{b}")
                                break
                    if ns.verbose and want_sigs != got_sigs:
                        # A signature table that disagrees: name the entries.
                        want_table = eval(want_sigs.split(" = ", 1)[1])  # noqa: S307
                        got_table = eval(got_sigs.split(" = ", 1)[1])  # noqa: S307
                        for key in sorted(set(want_table) | set(got_table)):
                            if want_table.get(key) != got_table.get(key):
                                print(f"  pipeline |{key}: {want_table.get(key)!r}")
                                print(f"  engine   |{key}: {got_table.get(key)!r}")
                    if ns.verbose and same_body and not same_report:
                        # The emitted Python agrees line for line and only the
                        # block bookkeeping does not. Printing nothing here --
                        # which is what this did -- makes a whole class of
                        # difference unreadable, and it is the largest class.
                        want_entries = comparable(want_report)
                        got_entries = comparable(got_report)
                        shown = 0
                        for a, b in zip(want_entries, got_entries, strict=False):
                            if a != b and shown < 3:
                                keys = sorted(set(a) | set(b))
                                where = ", ".join(
                                    f"{k}: {a.get(k)!r} != {b.get(k)!r}"
                                    for k in keys
                                    if a.get(k) != b.get(k)
                                )
                                site = f"{a.get('subprogram')}/{a.get('block')}"
                                print(f"  report   |{site}: {where}")
                                shown += 1
                        if len(want_entries) != len(got_entries):
                            print(
                                f"  report   |block count {len(want_entries)} != {len(got_entries)}"
                            )
        print(f"{name:<10} " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
        for key, value in counts.items():
            totals[key] += value

    print()
    print("  ".join(f"{k}={v}" for k, v in totals.items()))
    failed = totals["different"] + totals["error"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
