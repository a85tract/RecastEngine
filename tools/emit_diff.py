#!/usr/bin/env python3
"""Check the migrated emitter against the pipeline it was migrated from.

``golden_diff.py`` says the *analysis* still answers the same; this says the
*emission* still writes the same. It imports the pipeline's ``translate.py``
and the engine's statement layer, runs both over the same live-extracted
analysis of the same sources, and compares the emitted Python statement by
statement -- byte for byte, because the pipeline's output is what the
bit-exact gates have been run against and a single reflowed parenthesis is
indistinguishable from a wrong number until run time.

Live against the pipeline's code, not against its stored output: comparing
against a golden file older than the code that wrote it is how this
repository once reported fixing a bug the pipeline did not have.

Statements are compared at the top level of each subprogram's execution part,
in order -- order matters, because a statement function defined mid-body
changes how every later reference to its name renders. A construct compares
as its whole rendered body, so the statement count understates the coverage;
the line count is the honest number. A refusal only counts as agreement when
*both* sides refuse: one-sided refusals are the migration losing (or quietly
inventing) a rule.

The corpus is the six schemes with full operator tables plus every module the
translator's batch sweep produced, discovered from ``extracted_auto/`` at run
time so a new sweep widens this check without anyone editing it. One skip: a
subprogram whose name appears twice in one file -- unpreprocessed ``#if``
variants -- is left out, because pairing records with definitions is ambiguous
there on both sides and a mismatched pairing measures the harness, not the
emitters.

Usage:
    uv run --extra fortran tools/emit_diff.py --translator ../CESM-language-translator

Exit status is 1 on any difference, one-sided refusal, or error, 0 otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recast.fortran import constants as fconstants
from recast.fortran import interface as finterface
from recast.fortran._parse import f03, parse, walk
from recast.fortran.semantics import Unanalyzable, for_subprogram
from recast.transform.numpy.expressions import Expressions, Remote
from recast.transform.numpy.names import for_subprogram as names_for
from recast.transform.numpy.statements import Statements
from recast.transform.numpy.vocabulary import pysafe
from recast.transform.profiles import PROFILES
from recast.transform.rules import NoRule

ENGINE_REFUSED = (NoRule, Unanalyzable)

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
    fixed, records = [], []
    remotes: dict[str, Remote] = {}
    globals_: dict[str, str] = {}
    for companion in configured:
        stale = json.loads((root / companion["interface"]).read_text())
        record = finterface.extract(root / stale["source_file"])
        live = scratch / f"companion_{record['module']}.json"
        live.write_text(json.dumps(record))
        fixed.append({**companion, "interface": str(live)})
        records.append(record)
        alias = companion["alias"]
        subprograms = {s["name"]: s for s in record["subprograms"]}
        renames = {k.lower(): v.lower() for k, v in (companion.get("renames") or {}).items()}
        for local, remote in renames.items():
            if remote in subprograms:
                remotes[local] = Remote(alias, remote)
        for subprogram in record["subprograms"]:
            remotes.setdefault(subprogram["name"], Remote(alias, subprogram["name"]))
        for parameter in record["module_parameters"]:
            globals_.setdefault(parameter["name"], f"{alias}.{pysafe(parameter['name'].upper())}")
        for state in record["module_state"]:
            globals_.setdefault(state["name"], f"{alias}.{pysafe(state['name'])}")
    return fixed, records, remotes, globals_


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--translator",
        type=Path,
        required=True,
        help="the CESM-language-translator checkout to compare against",
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
    sys.path.insert(0, str(root / "pipeline"))
    import translate as pipeline

    scratch = ns.scratch or Path(tempfile.mkdtemp(prefix="emit_diff_"))
    scratch.mkdir(parents=True, exist_ok=True)
    kinds = ["lines", "identical", "different", "both refused", "pipeline only", "engine only"]
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

        interface = finterface.extract(source, intent_overrides=intents)
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
        modules = walk(tree, f03.Module)
        scope = modules[0] if modules else tree
        nodes = {}
        for subprogram in walk(scope, (f03.Subroutine_Subprogram, f03.Function_Subprogram)):
            heading = walk(subprogram, (f03.Subroutine_Stmt, f03.Function_Stmt))[0]
            nodes[str(heading.children[1]).lower()] = subprogram

        declared = [s["name"] for s in interface["subprograms"]]
        duplicated = {n for n in declared if declared.count(n) > 1}
        if duplicated:
            print(f"{name}: skipping {sorted(duplicated)} (unpreprocessed #if variants)")

        counts = dict.fromkeys(totals, 0)
        for record in interface["subprograms"]:
            node = nodes.get(record["name"])
            if node is None:
                continue
            if record["name"] in duplicated:
                counts["skipped"] += 1
                continue
            elemental = any("ELEMENTAL" in str(p).upper() for p in (record.get("prefixes") or []))

            translator.cur = record
            translator.alloc_lb = {}
            translator.stmt_funcs = set()
            translator.cur_elemental = elemental
            translator.scan_break_labels(node)

            semantics = for_subprogram(interface, record["name"], companions=tuple(records))
            names = names_for(
                semantics,
                constants,
                use_parameters=use_params,
                companion_globals=companion_globals,
            )
            expressions = Expressions(
                semantics,
                names,
                PROFILES["ifx"],
                externals=externals,
                remotes=remotes,
                stubs=dict(pipeline.INFRA_FN_STUBS),
                elemental=elemental,
            )
            statements = Statements(
                semantics,
                names,
                expressions,
                externals=externals,
                stubs=dict(pipeline.INFRA_STUBS),
            )
            statements.scan(node)

            execution = next((c for c in node.children if isinstance(c, f03.Execution_Part)), None)
            for statement in execution.children if execution is not None else []:
                where = f"{name}/{record['name']} [{type(statement).__name__}]"
                want = got = None
                try:
                    want = translator.stmt(statement, 1)
                except pipeline.AgentQueue:
                    pass
                except Exception as error:  # a crash is a finding, not a refusal
                    print(f"PIPELINE ERROR {where}: {type(error).__name__}: {error}")
                    counts["error"] += 1
                    continue
                try:
                    got = statements.render(statement, 1)
                except ENGINE_REFUSED:
                    pass
                except Exception as error:
                    print(f"ENGINE ERROR {where}: {type(error).__name__}: {error}")
                    counts["error"] += 1
                    continue
                if want is None and got is None:
                    counts["both refused"] += 1
                elif want is None:
                    counts["pipeline only"] += 1
                    print(f"PIPELINE-ONLY REFUSAL {where}")
                elif got is None:
                    counts["engine only"] += 1
                    print(f"ENGINE-ONLY REFUSAL {where}")
                elif want == got:
                    counts["identical"] += 1
                    counts["lines"] += len(want)
                else:
                    counts["different"] += 1
                    print(f"DIFFERENT {where}")
                    if ns.verbose:
                        for line in want:
                            print(f"  pipeline |{line}")
                        for line in got:
                            print(f"  engine   |{line}")
        print(f"{name:<10} " + "  ".join(f"{k}={v}" for k, v in counts.items() if v))
        for key, value in counts.items():
            totals[key] += value

    print()
    print("  ".join(f"{k}={v}" for k, v in totals.items()))
    failed = totals["different"] + totals["pipeline only"] + totals["engine only"] + totals["error"]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
