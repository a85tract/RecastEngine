"""Tests for the gate's handling of a library procedure the candidate stood in
for and the reference did not (#109).

ELM's ``SoilTemperature`` ends in ``call dgbsv(...)``. The translation calls
recast's own reference implementation of it; the recording it is judged
against ran the system's LAPACK, whose blocked elimination does not round
like the textbook one. Of 1,929,936 points 3,244 differed, all at 1e-11, all
downstream of the solve -- and the verdict said "no rtol excuses them" with no
way to say why. These tests hold what it says now: the stand-in is named on
every verdict, the differences are attributed when every one of them is
downstream of it, and the operator's ``stood_in_rtol`` reaches exactly those
subprograms and nothing else.

No Fortran: the reference is a plain Python callable, as in the tolerance
tests, and the transform's note is written by hand.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence, OracleRef, Unit
from recast.verify.bitexact import BitexactVerifier

MODULE = """\
import numpy as np

_SIGNATURES = {{
    "solve": {{
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [{{"name": "x", "intent": "IN", "dtype": "float64"}}],
    }},
    "plain": {{
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [{{"name": "x", "intent": "IN", "dtype": "float64"}}],
    }},
}}


def solve(x):
    y = x * 3.0
{solve_edit}
    return y


def plain(x):
    y = x * 0.5
{plain_edit}
    return y
"""

LAST_BITS = "    y = y + y * 1e-13"


def _candidate(solve_edit: str = "", plain_edit: str = "", stood_in: bool = True) -> Candidate:
    return Candidate(
        unit="tier:band",
        transform="test.tier",
        files={
            Path("band_numpy.py"): MODULE.format(
                solve_edit=solve_edit or "    pass", plain_edit=plain_edit or "    pass"
            ).encode()
        },
        notes={"references": {"stood_in": {"dgbsv": ["solve"]}}} if stood_in else {},
    )


def _oracle(**handle: Any) -> OracleRef:
    return OracleRef(
        unit="tier:band",
        oracle="test.python-truth",
        key="k",
        handle={
            "module": SimpleNamespace(w_solve=lambda x: x * 3.0, w_plain=lambda x: x * 0.5),
            "wrappers": {"solve": "w_solve", "plain": "w_plain"},
            **handle,
        },
    )


def _judge(
    tmp_path: Path, candidate: Candidate, oracle: OracleRef, gate: Any = None, **config: Any
) -> Any:
    return (gate or BitexactVerifier()).verify(
        Unit(uid="tier:band", kind="subprogram"),
        candidate,
        oracle,
        tmp_path,
        LocalExecutor(),
        {"trials": 3, **config},
    )


def test_a_bit_exact_unit_still_names_what_the_candidate_stood_in_for(tmp_path: Path) -> None:
    verdict = _judge(tmp_path, _candidate(), _oracle())
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["stood_in"] == {
        "dgbsv": {
            "subprograms": ["solve"],
            "points": 3,
            "bit_exact": 3,
            "max_ulp": 0,
            "max_rel": 0.0,
        }
    }
    assert "dgbsv (reached by solve)" in verdict.detail
    assert "on the candidate side only" in verdict.detail


def test_a_difference_downstream_of_the_stand_in_is_attributed_and_still_fails(
    tmp_path: Path,
) -> None:
    """No tolerance was asked for, so the gate fails -- and says where the
    differences are and what would excuse them."""
    verdict = _judge(tmp_path, _candidate(solve_edit=LAST_BITS), _oracle())
    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["bit_exact"] == 3 and verdict.metrics["points"] == 6
    assert "every differing point is downstream of dgbsv" in verdict.detail
    assert "no stood_in_rtol excuses them" in verdict.detail
    assert "stood_in_rtol" not in verdict.metrics


def test_the_granted_tolerance_reaches_the_stand_ins_subprograms(tmp_path: Path) -> None:
    verdict = _judge(tmp_path, _candidate(solve_edit=LAST_BITS), _oracle(), stood_in_rtol=1e-12)
    assert verdict.confidence is Confidence.TOLERANCED, verdict.detail
    assert verdict.metrics["stood_in_rtol"] == 1e-12
    assert verdict.metrics["stood_in"]["dgbsv"]["bit_exact"] == 0
    assert verdict.metrics["stood_in"]["dgbsv"]["points"] == 3
    assert verdict.detail.startswith(
        "3 points across 1 subprogram(s) bit-exact; the 1 reaching dgbsv"
    )
    assert "within stood_in_rtol=1e-12" in verdict.detail
    # A unit whose every compared subprogram reaches the solve (ELM's
    # SoilTemperature: one probe) says that rather than "0 points bit-exact".
    candidate = _candidate(solve_edit=LAST_BITS)
    candidate.notes["references"]["stood_in"]["dgbsv"] = ["plain", "solve"]
    verdict = _judge(tmp_path, candidate, _oracle(), stood_in_rtol=1e-12)
    assert verdict.confidence is Confidence.TOLERANCED, verdict.detail
    assert verdict.detail.startswith("all 2 compared subprogram(s) reach dgbsv")
    # A relative tolerance the solve does not meet is not met.
    verdict = _judge(tmp_path, _candidate(solve_edit=LAST_BITS), _oracle(), stood_in_rtol=1e-15)
    assert verdict.confidence is Confidence.FAILED
    assert "stood_in_rtol=1e-15 does not reach them" in verdict.detail


def test_and_nothing_else(tmp_path: Path) -> None:
    """A difference in a subprogram that never reaches the solve is a
    translation defect; the granted tolerance does not touch it."""
    verdict = _judge(
        tmp_path,
        _candidate(solve_edit=LAST_BITS, plain_edit=LAST_BITS),
        _oracle(),
        stood_in_rtol=1e-12,
    )
    assert verdict.confidence is Confidence.FAILED
    assert "no rtol excuses them" in verdict.detail
    assert "every differing point is downstream" not in verdict.detail
    # ... nor does it when the solve itself is exact.
    verdict = _judge(tmp_path, _candidate(plain_edit=LAST_BITS), _oracle(), stood_in_rtol=1e-12)
    assert verdict.confidence is Confidence.FAILED


def test_a_flat_adapter_reaches_what_its_subprogram_reaches(tmp_path: Path) -> None:
    text = MODULE.format(solve_edit=LAST_BITS, plain_edit="    pass")
    text = text.replace('"solve":', '"solve_flat":').replace("def solve(", "def solve_flat(")
    candidate = Candidate(
        unit="tier:band",
        transform="test.tier",
        files={Path("band_numpy.py"): text.encode()},
        notes={"references": {"stood_in": {"dgbsv": ["solve"]}}},
    )
    oracle = _oracle(wrappers={"solve_flat": "w_solve", "plain": "w_plain"})
    verdict = _judge(tmp_path, candidate, oracle, stood_in_rtol=1e-12)
    assert verdict.confidence is Confidence.TOLERANCED, verdict.detail
    assert verdict.metrics["stood_in"]["dgbsv"]["subprograms"] == ["solve_flat"]


def test_a_stand_in_on_both_sides_is_not_attributed(tmp_path: Path) -> None:
    """The f2py oracle compiles the same reference implementation into the
    reference: both sides did the same arithmetic, the handle says so, and a
    difference downstream of it is as much a defect as any other."""
    verdict = _judge(
        tmp_path,
        _candidate(solve_edit=LAST_BITS),
        _oracle(substituted={"dgbsv": "recast's own"}),
        stood_in_rtol=1e-12,
    )
    assert verdict.confidence is Confidence.FAILED
    assert "stood_in" not in verdict.metrics
    assert "on both sides: dgbsv" in verdict.detail
    assert "candidate side only" not in verdict.detail


def test_a_candidate_that_stood_nothing_in_is_unchanged(tmp_path: Path) -> None:
    verdict = _judge(
        tmp_path, _candidate(solve_edit=LAST_BITS, stood_in=False), _oracle(), stood_in_rtol=1e-12
    )
    assert verdict.confidence is Confidence.FAILED
    assert "stood_in" not in verdict.metrics
    assert verdict.detail.endswith("no rtol excuses them")


def test_the_tolerance_gate_names_the_stand_in_and_keeps_its_own_tiers(tmp_path: Path) -> None:
    """``port-elm`` judges JAX at the ULP tiers already; the stand-in is
    named on its verdict and its policy is untouched. The tolerance gate is
    an optional module, so an installation without it skips this one."""
    ToleranceVerifier = pytest.importorskip(
        "recast.verify.tolerance", reason="the tolerance gate is not installed here"
    ).ToleranceVerifier
    verdict = _judge(
        tmp_path,
        _candidate(solve_edit="    y = np.nextafter(y, np.inf)"),
        _oracle(),
        gate=ToleranceVerifier(),
    )
    assert verdict.confidence is Confidence.ULP_BOUNDED, verdict.detail
    assert verdict.metrics["tier"] == "ulp"
    assert verdict.metrics["stood_in"]["dgbsv"]["subprograms"] == ["solve"]
    assert "on the candidate side only" in verdict.detail
    assert "stood_in_rtol" not in verdict.metrics
