"""Tests for ``differential.tolerance``, the gate for a backend that cannot be
bit-exact.

The whole point of the two tiers is that the *same* size of disagreement is a
defect in one place and physics in another, so most of these run one
perturbation at two indices and require two different verdicts.

No JAX here, and none needed: the gate compares two Python callables and does
not care which backend produced either. The oracle is a plain object with the
wrapper on it, which is all the harness ever asks of an ``OracleRef.handle``.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[translate]")

import numpy as np

from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence, OracleRef, Unit
from recast.plugins.executor import Executor
from recast.verify.tolerance import ToleranceVerifier

# exp(-2i) over eight points: index 0 is the whole signal, 4..7 are below the
# 1e-3 dominance line and are exactly the elements conditioning ruins.
DECAY = np.exp(-2.0 * np.arange(8, dtype=np.float64))

MODULE = """\
import numpy as np

_SIGNATURES = {{
    "spread": {{
        "kind": "function",
        "result": "y",
        "args": [{{"name": "x", "intent": "IN", "dtype": "float64"}}],
    }}
}}

_DECAY = np.exp(-2.0 * np.arange(8, dtype=np.float64))


def spread(x):
    y = x * _DECAY
{perturbation}
    return y
"""


def truth(x: Any) -> Any:
    return x * DECAY


@pytest.fixture
def judge(tmp_path: Path) -> Any:
    """Run the gate over a candidate that differs from truth by one edit."""

    def run(perturbation: str, **config: Any) -> Any:
        body = perturbation or "    pass"
        candidate = Candidate(
            unit="tier:spread",
            transform="test.tier",
            files={Path("tier_numpy.py"): MODULE.format(perturbation=body).encode()},
        )
        oracle = OracleRef(
            unit="tier:spread",
            oracle="test.python-truth",
            key="k",
            handle={"module": SimpleNamespace(w_spread=truth), "wrappers": {"spread": "w_spread"}},
        )
        executor: Executor = LocalExecutor()
        return ToleranceVerifier().verify(
            Unit(uid="tier:spread", kind="subprogram"),
            candidate,
            oracle,
            tmp_path,
            executor,
            config,
        )

    return run


def test_an_identical_candidate_does_not_need_the_tiers(judge: Any) -> None:
    verdict = judge("")
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["bit_exact"] == verdict.metrics["points"]


def test_drift_inside_the_ulp_bound_everywhere_is_a_stronger_claim(judge: Any) -> None:
    """``ULP_BOUNDED`` rather than ``TOLERANCED``: when even the negligible tail
    stays inside the bound, a relative tolerance is not what is holding the
    verdict up, and saying so is the difference the ladder exists to record."""
    verdict = judge("    y[7] = np.nextafter(np.nextafter(y[7], np.inf), np.inf)")
    assert verdict.confidence is Confidence.ULP_BOUNDED, verdict.detail
    assert verdict.metrics["max_ulp"] <= 32


def test_the_same_drift_is_tolerated_in_the_tail(judge: Any) -> None:
    verdict = judge("    y[7] *= 1.0 + 1e-13")
    assert verdict.confidence is Confidence.TOLERANCED, verdict.detail
    assert verdict.metrics["max_ulp"] > 32, "the tail really is outside the ULP bound"
    assert verdict.metrics["max_ulp_dominant"] == 0, "and no dominant element moved"


def test_and_fails_in_a_dominant_element(judge: Any) -> None:
    """Same edit, same magnitude, different index. This pair is the gate."""
    verdict = judge("    y[0] *= 1.0 + 1e-13")
    assert verdict.confidence is Confidence.FAILED
    assert "dominant element" in verdict.detail
    assert verdict.metrics["max_ulp_dominant"] > 32


def test_a_tail_that_drifts_too_far_is_not_excused(judge: Any) -> None:
    """The relative tier is a catch-all, not an amnesty."""
    verdict = judge("    y[7] *= 1.0 + 1e-9")
    assert verdict.confidence is Confidence.FAILED
    assert "rel_gate" in verdict.detail
    assert verdict.metrics["max_rel"] > 1e-12


def test_the_gates_are_configurable_and_reported(judge: Any) -> None:
    """An operator who widens a gate has widened it in the record too."""
    verdict = judge("    y[7] *= 1.0 + 1e-9", rel_gate=1e-6)
    assert verdict.confidence is Confidence.TOLERANCED, verdict.detail
    assert verdict.metrics["rel_gate"] == 1e-6
    assert verdict.metrics["ulp_gate"] == 32


def test_a_tightened_ulp_gate_catches_what_the_default_allows(judge: Any) -> None:
    verdict = judge("    y[0] = np.nextafter(y[0], np.inf)", ulp_gate=0)
    assert verdict.confidence is Confidence.FAILED
    assert "dominant element is 1 ULP out" in verdict.detail


def test_bitexact_is_unchanged_by_the_dominance_machinery() -> None:
    """The tiering is opt-in: the bit-exact gate declares no dominance and so
    computes none, and its metrics stay the shape the committed summaries
    already record."""
    from recast.verify.bitexact import BitexactVerifier

    assert BitexactVerifier.dominant_at is None
    assert ToleranceVerifier.dominant_at == 1e-3
