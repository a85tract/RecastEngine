"""Tests for what ``differential.bitexact`` does with a draw it cannot use.

Generated inputs are not always inputs the subprogram takes. Fortran source
says so in three ways, and none of them is a difference between the two sides:

* ``ERROR STOP`` -- the source rejecting its own arguments. The reference says
  the same by ending the process, taking every other unit's verdict with it,
  so it must not be called on that draw at all.
* a subscript past a dummy array's declared extent -- the reference, compiled
  without bounds checking, reads memory the call does not own.
* NaN. Fortran does not say what MIN and MAX return for a NaN operand and
  gfortran's answer is whichever operand its register allocator made the
  second one, so a NaN-tainted trial held to the bit compares the compiler's
  scheduling rather than the translation.

Each is a draw to make again, not a comparison that failed -- and the bound on
how many times is the part that keeps it from being a way to narrow the gate:
a subprogram whose every draw is refused fails by name.
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
from recast.verify.bitexact import BitexactVerifier


def judge(tmp_path: Path, module: str, truth: Any, **config: Any) -> Any:
    """Run the gate over one emitted module against one Python reference."""
    candidate = Candidate(
        unit="draw:m",
        transform="test.draw",
        files={Path("m_numpy.py"): module.encode()},
    )
    oracle = OracleRef(
        unit="draw:m",
        oracle="test.python-truth",
        key="k",
        handle={"module": truth, "wrappers": {"probe": "w_probe"}},
    )
    return BitexactVerifier().verify(
        Unit(uid="draw:m", kind="subprogram"),
        candidate,
        oracle,
        tmp_path,
        LocalExecutor(),
        config,
    )


MODE = """\
_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [
            {"name": "mode", "intent": "IN", "dtype": "int32"},
            {"name": "x", "intent": "IN", "dtype": "float64"},
        ],
    }
}


def probe(mode, x):
    if int(mode) not in (1, 2):
        raise SystemExit("invalid mode in probe")
    return x * 2.0
"""


def test_a_draw_the_source_stops_on_is_drawn_again(tmp_path: Path) -> None:
    """``mode`` is sampled from the harness's default integer range and only
    two of those values are ones the subprogram takes."""

    def w_probe(mode: Any, x: Any) -> Any:
        # An ERROR STOP on the reference side ends the process; the harness
        # must never reach one. Standing in for it with an exception is how
        # this test can tell that it did not.
        assert int(mode) in (1, 2), "the reference was called on a refused draw"
        return x * 2.0

    verdict = judge(tmp_path, MODE, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] > 0


def test_every_draw_refused_is_a_subprogram_that_could_not_be_compared(tmp_path: Path) -> None:
    """The bound is the point: redrawing is not a way to make a gate green,
    because a subprogram no draw satisfies still fails, by name."""
    always = MODE.replace("if int(mode) not in (1, 2):", "if True:")
    verdict = judge(tmp_path, always, SimpleNamespace(w_probe=lambda mode, x: x * 2.0), draws=4)
    assert verdict.confidence is Confidence.FAILED
    assert "probe: no draw this harness could compare in 4 attempt(s)" in (verdict.detail or "")


PACKED = """\
_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [
            {"name": "n", "intent": "IN", "dtype": "int32"},
            {"name": "lr", "intent": "IN", "dtype": "int32"},
            {
                "name": "r",
                "intent": "IN",
                "dtype": "float64",
                "dims": [{"lb": "1", "ub": "lr"}],
            },
        ],
    }
}


def probe(n, lr, r):
    return float(r[(int(n) * (int(n) + 1)) // 2 - 1])
"""


def test_an_extent_too_small_for_the_body_is_drawn_again(tmp_path: Path) -> None:
    """Every unpinned extent defaults to the same number, so a packed
    triangular workspace -- ``n*(n+1)/2`` long for an order ``n`` -- is a
    subscript past the end. A refusal for a *shape* moves the extents; the
    values alone would never have got there."""
    verdict = judge(
        tmp_path,
        PACKED,
        SimpleNamespace(w_probe=lambda n, lr, r: float(r[(int(n) * (int(n) + 1)) // 2 - 1])),
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] > 0


NAN = """\
import numpy as np

_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [{"name": "x", "intent": "IN", "dtype": "float64"}],
    }
}


def probe(x):
    with np.errstate(invalid="ignore"):
        return np.sqrt(x)
"""


def test_a_nan_tainted_draw_is_drawn_again(tmp_path: Path) -> None:
    """Both sides compute the NaN; what they do with it afterwards is the
    compiler's business and not the translation's, so the trial is not one to
    hold either side to. The reference here disagrees on exactly those draws
    and agrees on every other, which is the shape of the real case."""

    def w_probe(x: Any) -> Any:
        return np.sqrt(x) if x >= 0.0 else np.float64(0.0)

    verdict = judge(tmp_path, NAN, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] > 0
    assert verdict.metrics["nan_mismatch"] == 0


def test_a_draw_that_needs_no_redrawing_is_the_one_the_seed_names(tmp_path: Path) -> None:
    """The first draw of every trial is unchanged -- same seed, same extents --
    so a run that never has to redraw compares exactly what it compared
    before."""
    plain = NAN.replace("return np.sqrt(x)", "return x * 2.0")
    verdict = judge(tmp_path, plain, SimpleNamespace(w_probe=lambda x: x * 2.0))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] == 0
