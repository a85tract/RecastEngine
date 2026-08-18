"""Tests for ``symbolic.notary`` and the ULP vocabulary.

The notarization triple is the pipeline's own self-test, kept verbatim: a
genuine algebraic expansion, a rewrite that drops a term, and an association
reorder -- the shape a GPU hand-reorder produces. The verdicts are the
contract: equivalent, algorithmic, equivalent.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from recast.errors import ConfigError
from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence
from recast.verify.ulp import ulp_audit, ulp_distance

pytest.importorskip("sympy", reason="needs recast-engine[verify]")

from recast.verify.notary import NotaryVerifier, notarize

RANGES = {"a": (-10.0, 10.0), "b": (-10.0, 10.0)}


# --- notarize ----------------------------------------------------------------


def test_an_algebraic_expansion_is_equivalent() -> None:
    result = notarize("(a + b)**2", "a**2 + 2*a*b + b**2", RANGES)
    assert result["verdict"] == "EQUIVALENT"
    assert result["worst_rel"] < 1e-45


def test_a_dropped_term_is_algorithmic() -> None:
    result = notarize("(a + b)**2", "a**2 + b**2", RANGES, samples=200)
    assert result["verdict"] == "ALGORITHMIC"
    assert result["worst_point"] is not None  # the counterexample is named


def test_an_association_reorder_is_equivalent() -> None:
    """Equivalent in exact arithmetic even though float64 would round the two
    orders differently -- which is exactly the distinction this exists to
    draw: the *rewrite* is sound, and the floating-point difference is then
    adjudicated by the differential gate, not waved through here."""
    result = notarize(
        "((f1 + f2) + f3) + f4",
        "f1 + (f2 + (f3 + f4))",
        {f"f{i}": (-50.0, 50.0) for i in range(1, 5)},
    )
    assert result["verdict"] == "EQUIVALENT"


def test_a_symbol_without_a_range_refuses() -> None:
    """Sampling an unphysical interval proves nothing; guessing one would
    launder that nothing into a verdict."""
    with pytest.raises(ConfigError, match="no physical range"):
        notarize("a + b", "b + a", {"a": (0.0, 1.0)})


def test_notarization_replays() -> None:
    first = notarize("(a + b)**2", "a**2 + b**2", RANGES, samples=100)
    second = notarize("(a + b)**2", "a**2 + b**2", RANGES, samples=100)
    assert first["worst_rel"] == second["worst_rel"]
    assert first["worst_point"] == second["worst_point"]


# --- the verifier ------------------------------------------------------------


def _candidate(notes: dict) -> Candidate:
    return Candidate(unit="fortran:demo", transform="translate.numpy", notes=notes)


def _check(candidate: Candidate) -> object:
    return NotaryVerifier().check(candidate.unit, candidate, Path("."), LocalExecutor(), {})


def test_zero_rewrites_pass_and_say_so() -> None:
    """The pipeline's production log recorded 'zero rewrites' explicitly
    rather than leaving it to be assumed."""
    verdict = _check(_candidate({}))
    assert verdict.confidence is Confidence.SYMBOLIC
    assert verdict.metrics["rewrites"] == 0
    assert "no rewrites" in verdict.detail


def test_an_algorithmic_rewrite_fails_the_gate() -> None:
    verdict = _check(
        _candidate(
            {
                "rewrites": [
                    {
                        "site": "demo/B001",
                        "old": "(a + b)**2",
                        "new": "a**2 + b**2",
                        "ranges": {"a": [-10, 10], "b": [-10, 10]},
                    }
                ]
            }
        )
    )
    assert verdict.confidence is Confidence.FAILED
    assert "demo/B001" in verdict.detail


def test_a_malformed_rewrite_fails_closed() -> None:
    verdict = _check(_candidate({"rewrites": [{"site": "demo/B002", "old": "a"}]}))
    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["judged"][0]["verdict"] == "FAILED"


# --- ULP ---------------------------------------------------------------------


def test_ulp_distance_counts_representable_doubles() -> None:
    one_up = math.nextafter(1.0, 2.0)
    assert ulp_distance(1.0, 1.0) == 0
    assert ulp_distance(1.0, one_up) == 1
    assert ulp_distance(-0.0, 0.0) == 0  # equal, however the bits differ
    assert ulp_distance(1.0, -1.0) > 2**62  # opposite signs span the line


def test_nan_agreement_is_exact_and_nan_mismatch_is_infinite() -> None:
    """Two NaNs mean both sides refused the same input the same way; one NaN
    means one side produced a value and the other did not, which no finite
    tolerance can excuse."""
    assert ulp_distance(math.nan, math.nan) == 0
    assert math.isinf(ulp_distance(math.nan, 1.0))


def test_ulp_audit_reports_the_reviewable_numbers() -> None:
    a = [1.0, 2.0, math.nan, 4.0]
    b = [1.0, math.nextafter(2.0, 3.0), 1.0, 4.0]
    audit = ulp_audit(a, b)
    assert audit["total_points"] == 4
    assert audit["bit_exact"] == 2
    assert audit["max_ulp"] == 1
    assert audit["nan_mismatch"] == 1
    assert audit["ulp_histogram"] == {0: 2, 1: 1, "inf": 1}
