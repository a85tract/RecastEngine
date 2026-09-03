"""The probe protocol: checksums and statistics two programs print, compared.

A program instrumented for this gate prints one line per observed buffer::

    GATE:SUM name=<n> dtype=<t> algo=<a> value=<hex>
    GATE:STAT name=<n> dtype=f32|f64 n=<count> min=<v> max=<v> mean=<v> L1=<v> L2=<v>

The lines, their spelling and the tolerance are ParaCodex's ``gate.h`` /
``gate_harness.py`` (Kaplan et al., 2026, MIT; see NOTICE): checksums must
be identical, statistics agree within ``rtol = atol = 1e-2``, and the
candidate must be deterministic across ``runs`` executions. This module is
that logic returning a record instead of exiting.

``stdout_agree`` is the harness's other mode for programs without probes:
identical stdout, the same mismatch signature, or both reporting their own
success. Two readings are deliberately not the harness's: NPB's own
``Verification = SUCCESSFUL`` line decides for NPB output, and Rodinia's
``... Beyond Error Threshold ...`` sentence is not a failure.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "DEFAULT_ATOL",
    "DEFAULT_RTOL",
    "ProbeOutput",
    "ProbeResult",
    "compare",
    "is_success_output",
    "parse",
    "stdout_agree",
]

DEFAULT_RTOL = 1e-2
DEFAULT_ATOL = 1e-2

SUM_RE = re.compile(r"^\s*GATE:SUM name=(\S+) dtype=(\S+) algo=(\S+) value=([0-9a-fA-F]+)")
STAT_RE = re.compile(
    r"^\s*GATE:STAT name=(\S+) dtype=(f32|f64) n=(\d+) min=([^\s]+) max=([^\s]+) "
    r"mean=([^\s]+) L1=([^\s]+) L2=([^\s]+)"
)
STAT_FIELDS = ("min", "max", "mean", "L1", "L2")


@dataclass
class ProbeOutput:
    """What one execution printed through its probes."""

    sums: dict[str, str] = field(default_factory=dict)
    stats: dict[str, dict[str, float]] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def has_probes(self) -> bool:
        return bool(self.sums or self.stats)


def parse(stdout: str) -> ProbeOutput:
    """Read the probe lines out of a program's stdout, the harness's way."""
    out = ProbeOutput()
    for line in stdout.splitlines():
        m = SUM_RE.match(line)
        if m:
            name, dtype, _algo, value = m.groups()
            key = f"{name}:{dtype}"
            if key in out.sums:
                out.problems.append(f"duplicate checksum metric {key}")
            out.sums[key] = value.lower()
            continue
        m = STAT_RE.match(line)
        if m:
            name, dtype, n, *rest = m.groups()
            key = f"{name}:{dtype}"
            if key in out.stats:
                out.problems.append(f"duplicate stats metric {key}")
            try:
                values = [float(v) for v in rest]
            except ValueError:
                out.problems.append(f"non-numeric stat for {key}")
                continue
            if not all(math.isfinite(v) for v in values):
                out.problems.append(f"non-finite stat for {key}")
            if int(n) <= 0:
                out.problems.append(f"n<=0 for {key}")
            out.stats[key] = {"n": float(n), **dict(zip(STAT_FIELDS, values, strict=True))}
    return out


def _approx_eq(a: float, b: float, rtol: float, atol: float) -> bool:
    return abs(a - b) <= atol + rtol * max(1.0, abs(a), abs(b))


def _rel_err(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(a), abs(b))


@dataclass
class ProbeResult:
    passed: bool
    reason: str
    """The harness's own tag (``checksum``, ``determinism``, ...) or ``pass``."""

    checksums_compared: int = 0
    stats_compared: int = 0
    max_rel_err: float = 0.0
    failures: list[str] = field(default_factory=list)
    runs: int = 0

    def metrics(self) -> dict[str, Any]:
        return {
            "gate": self.reason,
            "checksums_compared": self.checksums_compared,
            "stats_compared": self.stats_compared,
            "max_rel_err": self.max_rel_err,
            "runs": self.runs,
            "failures": list(self.failures),
        }


def compare(
    reference: ProbeOutput,
    candidates: list[ProbeOutput],
    *,
    rtol: float = DEFAULT_RTOL,
    atol: float = DEFAULT_ATOL,
) -> ProbeResult:
    """The harness's verdict over one reference execution and ``runs`` candidate ones.

    Same order of checks as ``gate_harness.py``: candidate determinism across
    runs, presence of probes on both sides, every reference key present in the
    candidate, checksums identical, statistics within tolerance.
    """
    runs = len(candidates)
    failures: list[str] = []
    problems = reference.problems + [p for c in candidates for p in c.problems]
    if problems:
        return ProbeResult(False, "malformed_probe_output", failures=problems, runs=runs)
    if not candidates:
        return ProbeResult(False, "cand_nonzero_exit", failures=["no candidate run"], runs=0)

    first = candidates[0]
    for i, later in enumerate(candidates[1:], start=2):
        if later.sums != first.sums:
            failures.append(f"checksum changed between run 1 and run {i}")
        if set(later.stats) != set(first.stats):
            failures.append(f"stats key set changed between run 1 and run {i}")
        else:
            for key, ref_stat in first.stats.items():
                for f in STAT_FIELDS:
                    if not _approx_eq(ref_stat[f], later.stats[key][f], rtol, atol):
                        failures.append(
                            f"det:{key} {f}: run1={ref_stat[f]} run{i}={later.stats[key][f]}"
                        )
    if failures:
        return ProbeResult(False, "determinism", failures=failures, runs=runs)

    if not reference.has_probes and not first.has_probes:
        return ProbeResult(
            False, "no_gate_output", failures=["neither side printed probes"], runs=runs
        )
    if not reference.has_probes:
        return ProbeResult(
            False, "no_ref_gate_output", failures=["reference printed no probes"], runs=runs
        )
    if not first.has_probes:
        return ProbeResult(
            False, "no_cand_gate_output", failures=["candidate printed no probes"], runs=runs
        )

    missing_sums = sorted(set(reference.sums) - set(first.sums))
    missing_stats = sorted(set(reference.stats) - set(first.stats))
    if missing_sums:
        return ProbeResult(
            False,
            "missing_candidate_checksums",
            failures=[f"missing {k}" for k in missing_sums],
            runs=runs,
        )
    if missing_stats:
        return ProbeResult(
            False,
            "missing_candidate_stats",
            failures=[f"missing {k}" for k in missing_stats],
            runs=runs,
        )

    max_rel = 0.0
    for key, value in reference.sums.items():
        if first.sums[key] != value:
            failures.append(f"checksum {key}: ref={value} cand={first.sums[key]}")
    for key, ref_stat in reference.stats.items():
        cand_stat = first.stats[key]
        if ref_stat["n"] != cand_stat["n"]:
            failures.append(f"n {key}: ref={int(ref_stat['n'])} cand={int(cand_stat['n'])}")
        for f in STAT_FIELDS:
            max_rel = max(max_rel, _rel_err(ref_stat[f], cand_stat[f]))
            if not _approx_eq(ref_stat[f], cand_stat[f], rtol, atol):
                failures.append(
                    f"stat {key} {f}: ref={ref_stat[f]} cand={cand_stat[f]} "
                    f"(rtol={rtol}, atol={atol})"
                )
    if not failures:
        reason = "pass"
    elif any(f.startswith("checksum") for f in failures):
        reason = "checksum"
    else:
        reason = "stats"
    return ProbeResult(
        passed=not failures,
        reason=reason,
        checksums_compared=len(reference.sums),
        stats_compared=len(reference.stats),
        max_rel_err=max_rel,
        failures=failures,
        runs=runs,
    )


def is_success_output(output: str) -> bool:
    """A program's own verdict, read from its stdout."""
    lowered = output.lower()
    if "usage:" in lowered:
        return True
    # NPB prints ``Verification = SUCCESSFUL`` (or ``UNSUCCESSFUL``) and then
    # ``Error is <residual>``: its own verdict is the one that counts.
    if "verification" in lowered:
        return "successful" in lowered and "unsuccessful" not in lowered
    # Rodinia's own check prints ``... Beyond Error Threshold of 0.05 Percent: 0``;
    # the word "error" in that sentence is not a failure.
    lowered = lowered.replace("error threshold", "")
    for bad in ("fail", "error", "fatal"):
        if bad in lowered and "pass" not in lowered and "passed" not in lowered:
            return False
    return True


def _signature(output: str) -> dict[str, str]:
    sig: dict[str, str] = {}
    for line in output.splitlines():
        if "Non-Matching" in line and ":" in line:
            sig["non_matching"] = line.split(":", maxsplit=1)[1].strip()
    return sig


def stdout_agree(golden: str, candidate: str) -> tuple[bool, str]:
    """``stdout_compare.py``: identical stdout, the same mismatch signature,
    or both programs reporting their own success. Returns ``(agree, how)``."""
    if golden == candidate:
        return True, "identical"
    g, c = _signature(golden), _signature(candidate)
    if g and g == c:
        return True, "same_signature"
    if is_success_output(golden) and is_success_output(candidate):
        return True, "both_self_report_pass"
    return False, "mismatch"
