"""Tests for what ``differential.bitexact`` does with a draw it cannot use.

Generated inputs are not always inputs the subprogram takes. Fortran source
says so in three ways, and none of them is a difference between the two sides:

* ``ERROR STOP`` -- the source rejecting its own arguments. The reference says
  the same by ending the process, taking every other unit's verdict with it,
  so it must not be called on that draw at all.
* a subscript past a dummy array's declared extent -- the reference, compiled
  without bounds checking, reads memory the call does not own.
* NaN on both sides. Fortran does not say what MIN and MAX return for a NaN
  operand and gfortran's answer is whichever operand its register allocator
  made the second one, so a NaN-tainted trial held to the bit compares the
  compiler's scheduling rather than the translation.

Each is a draw to make again, not a comparison that failed -- and the bounds
are the part that keeps it from being a way to narrow the gate: a subprogram
whose every draw is refused fails by name, a NaN on one side only is a
mismatch and not a redraw, and a subprogram compared mostly on extents the
redraw moved to fails by name as well.
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


def test_a_subprogram_compared_mostly_on_moved_extents_fails_by_name(tmp_path: Path) -> None:
    """Every unpinned extent defaults to the same number, so a packed
    triangular workspace -- ``n*(n+1)/2`` long for an order ``n`` -- is a
    subscript past the end. A refusal for a *shape* moves the extents, and
    the draws that then fit are the small orders: a pass on those is evidence
    about n = 1, not about the extents the run was configured with, so the
    subprogram fails by name and says which extents to pin."""
    verdict = judge(
        tmp_path,
        PACKED,
        SimpleNamespace(w_probe=lambda n, lr, r: float(r[(int(n) * (int(n) + 1)) // 2 - 1])),
    )
    assert verdict.confidence is Confidence.FAILED
    detail = verdict.detail or ""
    # ``n`` is a value, not an extent: the three trials that fit as drawn did
    # so on an ``n`` the seed made small. ``lr`` is the extent that moved.
    assert "probe: 7 of 10 trial(s) were compared only after the free extent(s) lr" in detail
    assert "Pin `dims`" in detail


def test_pinned_extents_the_body_takes_are_not_redrawn(tmp_path: Path) -> None:
    """The same packed workspace at extents that fit -- ``lr`` pinned to
    ``n(n+1)/2`` for the pinned ``n`` -- is compared as drawn, no redraw and
    nothing moved."""
    verdict = judge(
        tmp_path,
        PACKED,
        SimpleNamespace(w_probe=lambda n, lr, r: float(r[(int(n) * (int(n) + 1)) // 2 - 1])),
        dims={"n": 4, "lr": 10},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] == 0
    assert verdict.metrics["subprograms"]["probe"]["reshaped"] == 0


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


def test_a_draw_both_sides_take_to_nan_is_drawn_again(tmp_path: Path) -> None:
    """Both sides compute the NaN; what they do with it afterwards is the
    compiler's business and not the translation's, so the trial is not one to
    hold either side to -- and a NaN agreeing with a NaN is not a point of
    evidence either, so the trial is drawn again rather than counted."""

    def w_probe(x: Any) -> Any:
        with np.errstate(invalid="ignore"):
            return np.sqrt(x)

    verdict = judge(tmp_path, NAN, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] > 0
    assert verdict.metrics["nan_mismatch"] == 0


def test_a_nan_on_one_side_only_is_a_mismatch_not_a_redraw(tmp_path: Path) -> None:
    """The candidate goes to NaN where the reference has a number. That is
    the two sides disagreeing -- a variable read before it was assigned, a
    guard one side has and the other lost -- and redrawing it away would be
    exactly the narrowing the bound exists to prevent."""

    def w_probe(x: Any) -> Any:
        return np.sqrt(x) if x >= 0.0 else np.float64(0.0)

    verdict = judge(tmp_path, NAN, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["nan_mismatch"] > 0
    assert "where one side produced NaN and the other a number" in (verdict.detail or "")


def test_a_draw_that_needs_no_redrawing_is_the_one_the_seed_names(tmp_path: Path) -> None:
    """The first draw of every trial is unchanged -- same seed, same extents --
    so a run that never has to redraw compares exactly what it compared
    before."""
    plain = NAN.replace("return np.sqrt(x)", "return x * 2.0")
    verdict = judge(tmp_path, plain, SimpleNamespace(w_probe=lambda x: x * 2.0))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] == 0


# -- the project's input profile ----------------------------------------------


def profile(tmp_path: Path, body: str) -> Path:
    """Write ``recast_inputs.py`` at a project root and return that root."""
    root = tmp_path / "project"
    root.mkdir(exist_ok=True)
    (root / "recast_inputs.py").write_text(body)
    return root


PACKED_PROFILE = """\
import numpy as np


def prepare(unit, subprogram, inputs, rng):
    assert unit == "draw:m" and subprogram == "probe"
    n = int(inputs["n"])
    lr = n * (n + 1) // 2
    inputs["lr"] = np.int32(lr)
    inputs["r"] = np.asfortranarray(rng.uniform(-1.0, 1.0, size=lr))
    return inputs
"""


def test_a_shaped_draw_is_compared_as_shaped_and_never_redrawn(tmp_path: Path) -> None:
    """The packed workspace that fails by name under the generated rules is
    compared as drawn once the project says how ``lr`` follows ``n``: no
    redraw, nothing moved, and the trials are recorded as shaped."""
    root = profile(tmp_path, PACKED_PROFILE)
    verdict = judge(
        tmp_path,
        PACKED,
        SimpleNamespace(w_probe=lambda n, lr, r: float(r[(int(n) * (int(n) + 1)) // 2 - 1])),
        root=str(root),
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    probe = verdict.metrics["subprograms"]["probe"]
    assert (probe["redrawn"], probe["reshaped"], probe["shaped"]) == (0, 0, 10)
    assert verdict.metrics["input_profile"] == "recast_inputs.py"
    assert verdict.metrics["shaped"] == ["probe"]
    # The root is a checkout whose cleanliness is checked: reading the
    # profile must not drop bytecode into it.
    assert sorted(entry.name for entry in root.iterdir()) == ["recast_inputs.py"]


def test_a_candidate_that_refuses_a_shaped_draw_has_failed(tmp_path: Path) -> None:
    """Under the generated rules a translated ERROR STOP is a draw to make
    again. Under a profile that fixed ``mode`` to a value the source takes,
    the reference answers and the candidate stops: that is the translation
    refusing inputs the source accepts, and it fails by name."""
    root = profile(
        tmp_path,
        "import numpy as np\n"
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    inputs['mode'] = np.int32(1)\n"
        "    return inputs\n",
    )
    wrong = MODE.replace("if int(mode) not in (1, 2):", "if int(mode) != 2:")
    verdict = judge(
        tmp_path, wrong, SimpleNamespace(w_probe=lambda mode, x: x * 2.0), root=str(root)
    )
    assert verdict.confidence is Confidence.FAILED
    detail = verdict.detail or ""
    assert "probe: candidate raised on shaped inputs the reference took: SystemExit" in detail
    assert "redrawn" not in detail


def test_shaped_inputs_the_reference_refuses_are_the_profiles_fault(tmp_path: Path) -> None:
    """The profile asserts the reference takes the draw, so the reference is
    called first. When it refuses, nothing about the candidate is being
    judged: the gate stops with the profile, subprogram and trial named."""
    from recast.errors import InputProfileError

    root = profile(
        tmp_path,
        "import numpy as np\n"
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    inputs['mode'] = np.int32(7)\n"
        "    return inputs\n",
    )

    def w_probe(mode: Any, x: Any) -> Any:
        # Standing in for an ERROR STOP the reference would end the process on.
        if int(mode) not in (1, 2):
            raise ValueError("invalid mode in probe")
        return x * 2.0

    with pytest.raises(InputProfileError) as caught:
        judge(tmp_path, MODE, SimpleNamespace(w_probe=w_probe), root=str(root))
    message = str(caught.value)
    assert message.startswith("recast_inputs.py: prepare('draw:m', 'probe') at trial 0")
    assert "the reference does not take: ValueError: invalid mode in probe" in message


def test_a_reference_nan_on_shaped_inputs_is_the_profiles_fault(tmp_path: Path) -> None:
    """Both sides going to NaN is a redraw under the generated rules. A
    shaped draw is not drawn again, so a reference NaN on it is the profile
    having put the source outside its numeric domain."""
    from recast.errors import InputProfileError

    root = profile(
        tmp_path,
        "import numpy as np\n"
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    inputs['x'] = np.float64(-4.0)\n"
        "    return inputs\n",
    )

    def w_probe(x: Any) -> Any:
        with np.errstate(invalid="ignore"):
            return np.sqrt(x)

    with pytest.raises(InputProfileError, match="the reference produced NaN in y"):
        judge(tmp_path, NAN, SimpleNamespace(w_probe=w_probe), root=str(root))


def test_a_profile_that_returns_none_leaves_the_draw_to_the_generated_rules(
    tmp_path: Path,
) -> None:
    root = profile(
        tmp_path,
        "def prepare(unit, subprogram, inputs, rng):\n    return None\n",
    )

    def w_probe(mode: Any, x: Any) -> Any:
        assert int(mode) in (1, 2), "the reference was called on a refused draw"
        return x * 2.0

    verdict = judge(tmp_path, MODE, SimpleNamespace(w_probe=w_probe), root=str(root))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    probe = verdict.metrics["subprograms"]["probe"]
    assert probe["redrawn"] > 0 and probe["shaped"] == 0
    assert verdict.metrics["input_profile"] == "recast_inputs.py"
    assert verdict.metrics["shaped"] == []


def test_a_profile_that_edits_in_place_and_returns_none_is_refused(tmp_path: Path) -> None:
    """The profile sees a copy; the only way its work reaches the comparison
    is by returning it. Otherwise an edited draw would be judged under the
    unshaped rules with nobody told."""
    from recast.errors import InputProfileError

    root = profile(
        tmp_path,
        "import numpy as np\n"
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    inputs['mode'] = np.int32(1)\n",
    )
    with pytest.raises(InputProfileError, match="edited mode in place and returned None"):
        judge(tmp_path, MODE, SimpleNamespace(w_probe=lambda mode, x: x * 2.0), root=str(root))


def test_a_profile_that_renames_the_arguments_is_refused(tmp_path: Path) -> None:
    from recast.errors import InputProfileError

    root = profile(
        tmp_path,
        "def prepare(unit, subprogram, inputs, rng):\n"
        "    return {'mode': inputs['mode'], 'xx': inputs['x']}\n",
    )
    with pytest.raises(InputProfileError, match="missing x; unknown xx"):
        judge(tmp_path, MODE, SimpleNamespace(w_probe=lambda mode, x: x * 2.0), root=str(root))


def test_a_profile_that_does_not_import_stops_the_gate(tmp_path: Path) -> None:
    """A tree with a profile that cannot be read is not a tree without one."""
    from recast.errors import InputProfileError

    root = profile(tmp_path, "def prepare(unit, subprogram, inputs, rng)\n    return None\n")
    with pytest.raises(InputProfileError, match=r"recast_inputs\.py does not import: SyntaxError"):
        judge(tmp_path, MODE, SimpleNamespace(w_probe=lambda mode, x: x * 2.0), root=str(root))
    root = profile(tmp_path, "PREPARE = None\n")
    with pytest.raises(InputProfileError, match="defines no callable prepare"):
        judge(tmp_path, MODE, SimpleNamespace(w_probe=lambda mode, x: x * 2.0), root=str(root))


def test_a_root_without_a_profile_is_the_generated_path(tmp_path: Path) -> None:
    root = tmp_path / "bare"
    root.mkdir()
    verdict = judge(
        tmp_path,
        NAN.replace("return np.sqrt(x)", "return x * 2.0"),
        SimpleNamespace(w_probe=lambda x: x * 2.0),
        root=str(root),
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["input_profile"] is None
