#!/usr/bin/env python3
"""Check a migrated frontend against the pipeline it was migrated from.

CESM-language-translator's ``pipeline/`` produced JSON for every scheme it
ran on. ``recast.fortran`` is that analysis, moved in and refactored. Running
both over the same sources and diffing the JSON is the only check that says
the move preserved behaviour, rather than merely that the result still runs.

This is not a one-time migration instrument. The translator keeps being
developed, so its sources and its output keep moving, and the migration is a
standing job rather than a finished one. That is what ``ACCEPTED`` below is
for: two of the differences are deliberate and will reappear on every run
forever, and a check that reports known-good divergence every time is a check
people stop reading. Each accepted entry pins the *exact* values it excuses,
so a change to the underlying source stops matching and the divergence comes
back for re-confirmation -- which is the correct behaviour, because a fact
confirmed against last year's source is not confirmed against this year's.

Not compared: ``use_params.json``, ``*_constants.py`` and ``stubs/`` in the
golden set. Those are emitter output, and emitting source is not a Frontend's
job -- the rendering left with the Transform that has a target language.

Usage:
    uv run --extra fortran tools/golden_diff.py --golden ../CESM-language-translator/extracted
    ... --map mg2=reports/suite/micro_mg2_0/micro_mg2_0_cpp.F90
    ... --intent-overrides zm_conv=reports/zm/intent_overrides.json

Exit status is 1 if any behaviour changed in a way ``ACCEPTED`` does not
excuse, and 0 otherwise. Additive keys never fail: nothing that read the old
output can notice a key the old output did not have.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from recast.fortran import constants as new_constants
from recast.fortran import interface as new_interface

ANY = object()
"""Matches any value in an ``ACCEPTED`` entry. Use only where the old value is
an artefact of where the golden set happened to live, never to wave through a
value that carries meaning."""

INDEX_RE = re.compile(r"\[\d+\]")

# --- deliberate divergences, each pinned to the exact values it excuses -------
#
# Adding an entry here is a claim that the new answer is better than the old
# one, and the reason is the whole content of that claim. Anything that cannot
# be defended in a sentence should be a bug report, not an entry.
ACCEPTED: list[tuple[str, Any, Any, str]] = [
    (
        "subprograms[].args[].intent_override",
        ANY,
        True,
        "the old record repeated the override file's path on every argument; the "
        "value is now a flag meaning 'not from the source', and where it did come "
        "from is recorded once in Facts.provenance",
    ),
    (
        "subprograms[].module_state_read",
        ["estbl"],
        [],
        "wv_sat_final: deallocate(estbl) is a write, not a read -- Fortran sets the "
        "allocation status to unallocated, which a translation spells estbl = None",
    ),
    (
        "subprograms[].module_state_written",
        [],
        ["estbl"],
        "the other half of the deallocate reclassification above",
    ),
]


def normalize(path: str) -> str:
    """``subprograms[3].args[12].intent`` -> ``subprograms[].args[].intent``."""
    return INDEX_RE.sub("[]", path).lstrip(".")


def excuse(path: str, old: Any, new: Any) -> str | None:
    """The reason this difference is accepted, or ``None`` if it is not."""
    key = normalize(path)
    for pattern, want_old, want_new, reason in ACCEPTED:
        if key != pattern:
            continue
        if want_old is not ANY and want_old != old:
            continue
        if want_new is not ANY and want_new != new:
            continue
        return reason
    return None


def compare(old: Any, new: Any, path: str = "") -> tuple[list[str], list[tuple[str, Any, Any]]]:
    """``(added paths, [(path, was, now)])``, recursing into dicts and lists.

    A key only the new record has is added. A key only the old record has is a
    change to ``<absent>``, because something that used to be reported and is
    not any more is a regression however it is spelled.
    """
    added: list[str] = []
    changed: list[tuple[str, Any, Any]] = []
    if isinstance(old, dict) and isinstance(new, dict):
        added += [f"{path}.{k}" for k in new.keys() - old.keys()]
        for k in old.keys() & new.keys():
            a, c = compare(old[k], new[k], f"{path}.{k}")
            added += a
            changed += c
        changed += [(f"{path}.{k}", old[k], "<absent>") for k in old.keys() - new.keys()]
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i, (o, n) in enumerate(zip(old, new, strict=True)):
            a, c = compare(o, n, f"{path}[{i}]")
            added += a
            changed += c
    elif old != new:
        changed.append((path, old, new))
    return added, changed


def locate(case: str, recorded: str, root: Path, mapped: dict[str, str]) -> Path:
    """Find the source a golden record was produced from.

    The recorded path is preferred and is usually right. When it is not -- the
    golden set outlived a reorganisation, or was produced from a working copy
    -- fall back to a search by file name, and refuse to guess between several
    candidates: they are different revisions, and diffing against the wrong one
    reports line-number noise that buries the real differences.
    """
    if case in mapped:
        return (root / mapped[case]).resolve()
    direct = root / recorded
    if direct.is_file():
        return direct.resolve()
    candidates = sorted(p for p in root.rglob(Path(recorded).name) if p.is_file())
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise SystemExit(f"{case}: no source found for {recorded!r} under {root}")
    listing = "\n".join(f"    --map {case}={p.relative_to(root)}" for p in candidates)
    raise SystemExit(
        f"{case}: {recorded!r} is gone and {len(candidates)} files share its name.\n"
        f"  Pick the revision the golden set was produced from:\n{listing}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--golden", type=Path, required=True, help="the pipeline's extracted/ directory"
    )
    ap.add_argument(
        "--sources",
        type=Path,
        help="root the golden set's source_file paths are relative to "
        "(default: the golden directory's parent)",
    )
    ap.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help="source for one case, when the recorded path is gone or ambiguous",
    )
    ap.add_argument(
        "--intent-overrides",
        action="append",
        default=[],
        metavar="CASE=PATH",
        help="intent override table the pipeline used for one case",
    )
    ap.add_argument(
        "-v", "--verbose", action="store_true", help="list added keys and excused diffs"
    )
    args = ap.parse_args()

    root = (args.sources or args.golden.parent).resolve()
    mapped = dict(entry.split("=", 1) for entry in args.map)
    intents = dict(entry.split("=", 1) for entry in args.intent_overrides)

    cases = sorted(d for d in args.golden.iterdir() if (d / "interface.json").is_file())
    if not cases:
        raise SystemExit(f"no case with an interface.json under {args.golden}")

    # Resolve every source before comparing anything, so an unusable mapping is
    # reported at once rather than after a partial run whose output looks like
    # a result.
    golds = {case: json.loads((case / "interface.json").read_text()) for case in cases}
    sources = {
        case: locate(case.name, gold.pop("source_file"), root, mapped)
        for case, gold in golds.items()
    }

    unexplained = 0
    for case in cases:
        gold = golds[case]
        src = sources[case]
        table = intents.get(case.name)
        got = new_interface.extract(
            src,
            intent_overrides=json.loads((root / table).read_text()) if table else None,
        )
        got.pop("source_file")

        added, changed = compare(gold, got)
        excused = [(p, o, n, r) for p, o, n in changed if (r := excuse(p, o, n))]
        real = [(p, o, n) for p, o, n in changed if not excuse(p, o, n)]
        unexplained += len(real)

        lit_gold = case / "literal_map.json"
        lit = "n/a"
        if lit_gold.is_file():
            same = json.loads(lit_gold.read_text()) == new_constants.extract(src)["literal_map"]
            lit = "ok" if same else "DIFF"
            unexplained += 0 if same else 1

        print(
            f"{case.name:<12} {len(got['subprograms']):>3} subs  literal_map={lit:<4} "
            f"added={len(added):<3} excused={len(excused):<3} unexplained={len(real)}"
        )
        for path, was, now in real:
            print(f"    {normalize(path)}: {was!r} -> {now!r}")
        if args.verbose:
            for key in sorted({normalize(p) for p in added}):
                print(f"    + {key}")
            for path, was, now in excused:
                print(f"    ~ {normalize(path)}: {was!r} -> {now!r}")

    print()
    if unexplained:
        print(f"{len(cases)} cases, {unexplained} unexplained difference(s).")
        print("Each is either a regression to fix or a claim to add to ACCEPTED with a reason.")
        return 1
    print(f"{len(cases)} cases, no unexplained differences.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
