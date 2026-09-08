"""Tests for how ``differential.bitexact`` compares a subprogram whose
derived-type dummy the reference wrapper spells component by component.

f2py cannot marshal a derived type. The oracle takes one of scalar
components as a flat dummy per component and puts the plan on its handle
(``recast.oracle.f2py.flattened_dummies``); the verifier splits the
candidate's argument the same way, so every component is a drawn input and
a compared output, and nothing about the comparison itself changes.
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
from recast.verify.bitexact import BitexactVerifier, flatten_derived

PLAN = {
    "state": {
        "type": "state_t",
        "components": [
            {"name": "state_a", "component": "a", "dtype": "float64"},
            {"name": "state_i", "component": "i", "dtype": "int32"},
        ],
    }
}

SIGNATURE = {
    "kind": "subroutine",
    "public": True,
    "args": [
        {"name": "n", "intent": "IN", "dtype": "int32", "optional": False},
        {
            "name": "x",
            "intent": "INOUT",
            "dtype": "float64",
            "optional": False,
            "dims": [{"lb": "1", "ub": "n"}],
        },
        {"name": "state", "intent": "INOUT", "dtype": "UNKNOWN(TYPE(STATE_T))", "optional": False},
        {"name": "y", "intent": "OUT", "dtype": "float64", "optional": False},
    ],
}

MODULE = """\
import numpy as np

_SIGNATURES = {"step": SIGNATURE_LITERAL}


class _new_derived:
    pass


def _make_state_t():
    o = _new_derived()
    o.a = 0.0
    o.i = 0
    return o


def step(n, x, state, y=None):
    x[...] = x * 2.0
    state.a = state.a + float(np.sum(x))
    state.i = state.i + int(n)
    y = state.a * 0.5
    return x, state, y
""".replace("SIGNATURE_LITERAL", repr(SIGNATURE))


def test_flatten_derived_splits_the_signature_and_the_returns() -> None:
    """The flat signature carries the components in the argument's place
    with its intent; the flat function assembles the object from those
    draws and hands its components back where the object was returned."""
    seen: dict[str, Any] = {}

    def step(n: Any, x: Any, state: Any) -> Any:
        seen["state"] = state
        state.a = state.a + 1.0
        state.i = state.i * 2
        return x, state, 7.0

    flat_sub, flat_fn = flatten_derived(SIGNATURE, step, PLAN)
    assert [a["name"] for a in flat_sub["args"]] == ["n", "x", "state_a", "state_i", "y"]
    assert [a["intent"] for a in flat_sub["args"]][2:4] == ["INOUT", "INOUT"]
    assert flat_sub["args"][3]["dtype"] == "int32"
    out = flat_fn(n=np.int32(2), x=np.zeros(2), state_a=np.float64(0.5), state_i=np.int32(3))
    assert isinstance(seen["state"], SimpleNamespace)
    assert out[1] == 1.5 and out[2] == 6 and out[3] == 7.0 and len(out) == 4


def test_flatten_derived_leaves_a_signature_without_derived_dummies_alone() -> None:
    plain = {**SIGNATURE, "args": [a for a in SIGNATURE["args"] if a["name"] != "state"]}

    def step(**kwargs: Any) -> Any:
        return None

    assert flatten_derived(plain, step, PLAN) == (plain, step)


def test_a_flattened_derived_type_compares_bit_exact_end_to_end(tmp_path: Path) -> None:
    """The reference takes the flat scalars f2py's way -- INOUT rank-0
    buffers updated in place, the OUT returned -- and the candidate takes
    the object its translation was emitted with; the gate compares every
    component as a point of its own."""

    def w_step(n: Any, x: Any, state_a: Any, state_i: Any) -> Any:
        x[...] = x * 2.0
        state_a[...] = state_a + float(np.sum(x))
        state_i[...] = state_i + int(n)
        return float(state_a) * 0.5

    candidate = Candidate(
        unit="fortran:m", transform="test.derived", files={Path("m_numpy.py"): MODULE.encode()}
    )
    oracle = OracleRef(
        unit="fortran:m",
        oracle="test.python-truth",
        key="k",
        handle={
            "module": SimpleNamespace(w_step=w_step),
            "wrappers": {"step": "w_step"},
            "flattened": {"step": PLAN},
        },
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:m", kind="module"),
        candidate,
        oracle,
        tmp_path,
        LocalExecutor(),
        {"trials": 4},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    outcome = verdict.metrics["subprograms"]["step"]
    # x (n cells), state_a, state_i and y per trial, the integer component exactly.
    assert outcome["integer_points"] == 4
    assert outcome["points"] > 4 * 3
