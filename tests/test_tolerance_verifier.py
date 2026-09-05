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
        "result_dtype": "float64",
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


def test_integer_mismatch_cannot_climb_the_tolerance_ladder(tmp_path: Path) -> None:
    """No relative or ULP gate can excuse a discrete integer disagreement."""
    candidate = Candidate(
        unit="tier:integer",
        transform="test.tier",
        files={
            Path("integer_numpy.py"): b"""\
_SIGNATURES = {
    "measure": {
        "kind": "function",
        "result": "y",
        "result_dtype": "int64",
        "args": [],
    }
}

def measure():
    return 9007199254740993
"""
        },
    )
    oracle = OracleRef(
        unit=candidate.unit,
        oracle="test.python-truth",
        key="k",
        handle={
            "module": SimpleNamespace(w_measure=lambda: 9007199254740992),
            "wrappers": {"measure": "w_measure"},
        },
    )
    verdict = ToleranceVerifier().verify(
        Unit(uid=candidate.unit, kind="subprogram"),
        candidate,
        oracle,
        tmp_path,
        LocalExecutor(),
        {"rel_gate": 1e100, "ulp_gate": 2**63 - 1},
    )

    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["integer_mismatch"] == 10
    assert "cannot be tolerance-excused" in verdict.detail


def test_bitexact_is_unchanged_by_the_dominance_machinery() -> None:
    """The tiering is opt-in: the bit-exact gate declares no dominance and so
    computes none, and its metrics stay the shape the committed summaries
    already record."""
    from recast.verify.bitexact import BitexactVerifier

    assert BitexactVerifier.dominant_at is None
    assert ToleranceVerifier.dominant_at == 1e-3


def test_the_device_each_side_ran_on_is_recorded_when_declared(tmp_path: Path) -> None:
    """A ULP bound between a GPU and a CPU is a different claim from one
    between two CPUs, and the verdict has to say which it was.

    Declared rather than detected: the emitted module says ``_DEVICE`` and the
    oracle puts one on its handle, so the core never imports an accelerator to
    find out.
    """
    module = MODULE.format(perturbation="    pass") + '\n_DEVICE = "gpu:0"\n'
    candidate = Candidate(
        unit="tier:spread",
        transform="test.tier",
        files={Path("tier_numpy.py"): module.encode()},
    )
    oracle = OracleRef(
        unit="tier:spread",
        oracle="test.python-truth",
        key="k",
        handle={
            "module": SimpleNamespace(w_spread=truth),
            "wrappers": {"spread": "w_spread"},
            "device": "cpu",
        },
    )
    verdict = ToleranceVerifier().verify(
        Unit(uid="tier:spread", kind="subprogram"),
        candidate,
        oracle,
        tmp_path,
        LocalExecutor(),
        {},
    )
    assert verdict.metrics["candidate_device"] == "gpu:0"
    assert verdict.metrics["reference_device"] == "cpu"


def test_nothing_is_recorded_when_neither_side_says(judge: Any) -> None:
    """Silence stays silence: the committed run summaries filter metrics by
    type, so inventing an empty key here would put ``None`` in every one."""
    verdict = judge("")
    assert "candidate_device" not in verdict.metrics
    assert "reference_device" not in verdict.metrics


def test_a_routine_forwarded_to_the_host_is_not_a_lowered_kernel(tmp_path: Path) -> None:
    """The JAX module binds what it could not lower to the NumPy anchor
    (``f = _host.f``) and lists what it did lower in ``_JAX_KERNELS``. The
    gate judged the forwarded function -- the anchor's own code -- and
    awarded the port a bit-exact verdict on ELM's hydraulic-stress kernel,
    which it had never emitted. A name outside the lowered list fails by
    name, with the backend's reason."""

    def run(kernels: list[str]) -> Any:
        text = MODULE.format(perturbation="    pass") + f"\n_JAX_KERNELS = {kernels!r}\n"
        candidate = Candidate(
            unit="tier:spread",
            transform="test.tier",
            files={Path("tier_numpy.py"): text.encode()},
            notes={"jax": {"delegated": {"spread": "a helper needs a state this plan does not carry"}}},
        )
        oracle = OracleRef(
            unit="tier:spread",
            oracle="test.python-truth",
            key="k",
            handle={"module": SimpleNamespace(w_spread=truth), "wrappers": {"spread": "w_spread"}},
        )
        return ToleranceVerifier().verify(
            Unit(uid="tier:spread", kind="subprogram"),
            candidate,
            oracle,
            tmp_path,
            LocalExecutor(),
            {},
        )

    forwarded = run([])
    assert forwarded.confidence is Confidence.FAILED
    assert "spread: not lowered by this backend, forwarded to its host module" in forwarded.detail
    assert "a helper needs a state this plan does not carry" in forwarded.detail
    assert run(["spread"]).confidence is Confidence.BIT_EXACT
