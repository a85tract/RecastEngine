#!/usr/bin/env python3
"""Check the migrated dump parser against the script it was migrated from.

The fourth migration differential, and the one with the weakest material, which
is worth saying at the top rather than in a footnote.

``emit_diff``, ``numba_diff`` and ``cuda_diff`` run both sides over the real
corpus: 27 modules of CAM Fortran that are in the repository. **There is no
corpus here.** The dumps ``dump_verify.py`` was written against were written by
probes inside a production CESM run and live on scratch storage; none is
committed, in either repository. So this harness compares the two parsers over
dumps it *constructs* in the probe format, and the strength of the check is the
strength of that construction rather than of a real recording.

What that can and cannot establish:

* It **can** show the two parsers agree on the format's awkward parts, because
  those are what the cases are chosen to exercise -- Fortran ``G`` editing's
  implicit exponent, ``D`` exponents, header scalars that are also extents, an
  extent naming a header scalar, a declared shape that does not multiply out,
  ``order="F"`` walking, and a section terminated by the next header rather
  than by anything of its own.
* It **cannot** show they agree on a real dump. A production probe may write
  something no case here anticipates, and this harness would not notice. The
  honest reading of a green run is "these two parsers agree on everything this
  file knows to ask about", and the file says how many that is.

So the cases are the evidence, and they are listed rather than generated: a
random-case generator would raise the count without raising the confidence, and
would make the two numbers look alike when they are not.

Usage:
    python tools/dump_diff.py --translator ../cesm/<translator-checkout>
    python tools/dump_diff.py --translator <dir> -v
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from recast.oracle.dump_replay import parse_dump  # noqa: E402

CASES: list[tuple[str, str]] = [
    (
        "a plain rank-1 input and output",
        """# PROBE wv_sat_methods.wv_sat_svp_water: call=       1
# INPUT: t(3)
273.15
283.15
293.15
# OUTPUT: es(3)
611.2
1228.1
2338.8
""",
    ),
    (
        "a header scalar is metadata and an input at once",
        """# PROBE micro_mg_utils.size_dist_param_liq: call=       1
# mgncol =       4
# INPUT: qic(mgncol)
1.0e-5
2.0e-5
3.0e-5
4.0e-5
# OUTPUT: lamc(mgncol)
1.0e5
2.0e5
3.0e5
4.0e5
""",
    ),
    (
        "a rank-2 extent naming two header scalars, walked in Fortran order",
        """# PROBE micro_mg2_0.micro_mg_tend: call=       2
# mgncol =       2
# nlev =       3
# INPUT: qv(mgncol,nlev)
1.0
2.0
3.0
4.0
5.0
6.0
# OUTPUT: qvout(mgncol,nlev)
0.5
1.5
2.5
3.5
4.5
5.5
""",
    ),
    (
        "G editing drops the E when the exponent needs three digits",
        """# PROBE shr_spfn_mod.calerf_r8: call=       1
# INPUT: x(2)
1.0701116457083034-114
-2.3456789012345678+123
# OUTPUT: y(2)
0.0
1.0
""",
    ),
    (
        "a D exponent, in a data line and in a header scalar",
        """# PROBE wv_sat_methods.wv_sat_qsat_water: call=       1
# tmelt = 2.7315D2
# INPUT: p(2)
1.01325D5
5.0d4
# OUTPUT: qs(2)
1.0D-2
2.0d-2
""",
    ),
    (
        "a declared shape that does not multiply out to the values written",
        """# PROBE mo_airmas.airmas: call=       1
# INPUT: z(2,3)
1.0
2.0
3.0
4.0
# OUTPUT: airmas(4)
1.0
2.0
3.0
4.0
""",
    ),
    (
        "an extent that names nothing known falls back to the value count",
        """# PROBE cloud_fraction.cldfrc: call=       1
# INPUT: cld(pcols)
0.1
0.2
0.3
# OUTPUT: cldout(pcols)
0.4
0.5
0.6
""",
    ),
    (
        "an unparseable data line is skipped, not fatal",
        """# PROBE drydep_mod.drydep: call=       1
# INPUT: v(3)
1.0
not-a-number
3.0
# OUTPUT: vd(2)
1.0
2.0
""",
    ),
    (
        "a header scalar that is not a number at all",
        """# PROBE micro_mg2_0.micro_mg_init: call=       1
# precip_frac_method = max_overlap
# mgncol =       2
# INPUT: q(mgncol)
1.0
2.0
# OUTPUT: qout(mgncol)
3.0
4.0
""",
    ),
    (
        "a section with a name but no values, followed by one with values",
        """# PROBE ndrop.dropmixnuc: call=       1
# INPUT: empty(2)
# INPUT: real(2)
7.0
8.0
# OUTPUT: out(2)
9.0
10.0
""",
    ),
    (
        "an INPUT with no declared extents at all",
        """# PROBE mo_util.rebin: call=       1
# INPUT: scalarish
42.0
# OUTPUT: result
43.0
""",
    ),
    (
        "blank lines and a trailing comment that is not a scalar",
        """# PROBE quadrature_mod.gauss: call=       1

# INPUT: pts(2)
-0.5773502691896257

0.5773502691896257
# OUTPUT: wts(2)
1.0
1.0
# end of record
""",
    ),
]
"""Each case is one probe dump, chosen for a part of the format that is easy to
read two ways. A case that both parsers get trivially right adds nothing."""


def compare(theirs: Any, ours: Any, label: str, verbose: bool) -> list[str]:
    """Every difference between two parse results, as prose."""
    import numpy as np

    problems = []
    their_in, their_out = theirs
    our_in, our_out = ours
    for section, mine, yours in (("INPUT", our_in, their_in), ("OUTPUT", our_out, their_out)):
        if set(mine) != set(yours):
            only_ours = sorted(set(mine) - set(yours))
            only_theirs = sorted(set(yours) - set(mine))
            problems.append(
                f"{label}: {section} names differ -- engine only {only_ours}, "
                f"pipeline only {only_theirs}"
            )
            continue
        for key in sorted(mine):
            a, b = np.asarray(mine[key]), np.asarray(yours[key])
            if a.shape != b.shape:
                problems.append(f"{label}: {section} {key} shape {a.shape} vs {b.shape}")
                continue
            if a.dtype != b.dtype:
                problems.append(f"{label}: {section} {key} dtype {a.dtype} vs {b.dtype}")
                continue
            # Bit for bit. These are parsed literals, not computed values:
            # there is no rounding between the two sides to be tolerant of,
            # and a tolerance here would hide the digit-dropping bug this
            # harness exists to catch.
            same = (a == b) | (np.isnan(a) & np.isnan(b))
            if not bool(np.all(same)):
                problems.append(f"{label}: {section} {key} values differ: {a!r} vs {b!r}")
    if verbose and not problems:
        print(f"    ok  {label}")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--translator", type=Path, required=True)
    ap.add_argument("-v", "--verbose", action="store_true")
    ns = ap.parse_args()

    root = ns.translator.resolve()
    if not (root / "pipeline" / "dump_verify.py").is_file():
        # Exit 2, not 1, and the distinction is the point: nothing disagreed,
        # but the comparison did not happen. A differential that reports
        # success when it could not reach the thing it compares against is the
        # failure it exists to prevent. ``tools/ci_local.sh`` uses the same
        # three exits for the same reason.
        print(
            f"not run: {root} has no pipeline/dump_verify.py to compare against.\n"
            "Point --translator at a checkout of the translator. It is a\n"
            "private repository, which is why this runs from a local checkout and\n"
            "not from CI -- see docs/disclosure-ledger.md, row 10.",
            file=sys.stderr,
        )
        return 2
    sys.path.insert(0, str(root / "pipeline"))
    try:
        from dump_verify import parse_dump_file
    except ImportError as error:
        print(f"cannot import the pipeline's dump_verify: {error}", file=sys.stderr)
        return 2

    import tempfile

    problems: list[str] = []
    scratch = Path(tempfile.mkdtemp(prefix="dump_diff_"))
    for label, text in CASES:
        # Their parser takes a path; ours takes the text. Same bytes either way.
        path = scratch / "probe.txt"
        path.write_text(text)
        theirs = parse_dump_file(str(path))
        ours = parse_dump(text)
        problems.extend(compare(theirs, ours, label, ns.verbose))

    print(f"\ncases={len(CASES)}  different={len(problems)}")
    for problem in problems:
        print(f"  {problem}")
    if not problems:
        print(
            "No real dump is committed in either repository, so this is agreement "
            "on constructed cases, not on a production recording."
        )
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
