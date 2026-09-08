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
    """``mode`` is drawn from 1 to 3 and only two of those values are ones the
    subprogram takes: the third is declined and drawn again, and the verdict
    says how many times."""

    def w_probe(mode: Any, x: Any) -> Any:
        # An ERROR STOP on the reference side ends the process; the harness
        # must never reach one. Standing in for it with an exception is how
        # this test can tell that it did not.
        assert int(mode) in (1, 2), "the reference was called on a refused draw"
        return x * 2.0

    verdict = judge(tmp_path, MODE, SimpleNamespace(w_probe=w_probe), ranges={"mode": (1, 3)})
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    redrawn = verdict.metrics["subprograms"]["probe"]["redrawn"]
    assert redrawn > 0
    assert verdict.metrics["subprograms"]["probe"]["declined"] == {"error stop": redrawn}
    assert f"{redrawn} draw(s) declined and drawn again ({redrawn} error stop)" in verdict.detail


def test_a_subprogram_whose_draws_are_mostly_declined_fails_by_name(tmp_path: Path) -> None:
    """``mode`` from the default integer range, 1 to 8, and the subprogram
    takes two of them: three draws in four are declined. The survivors are a
    minority of the configured draw, and a candidate that stopped on inputs
    the source accepts would pass on that minority one survivor at a time --
    so it fails by name, with the count, the reason and the remedy."""
    verdict = judge(tmp_path, MODE, SimpleNamespace(w_probe=lambda mode, x: x * 2.0))
    assert verdict.confidence is Confidence.FAILED
    detail = verdict.detail or ""
    assert "probe: " in detail and "draw(s) were declined (" in detail
    assert "error stop) to compare 10 trial(s)" in detail
    assert "Narrow the draw with `ranges`" in detail
    assert verdict.metrics["subprograms"]["probe"] == {
        "error": verdict.metrics["subprograms"]["probe"]["error"]
    }


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


def test_a_packed_workspace_is_grown_to_a_shape_the_body_takes(tmp_path: Path) -> None:
    """Every unpinned extent defaults to the same number, so a packed
    triangular workspace -- ``n*(n+1)/2`` long for an order ``n`` -- is a
    subscript past the end at the default, and no order it goes with makes
    it not one. An extent nobody pinned is this harness's own choice, so the
    choice is *grown* -- once, before the trials, and only upward -- until
    the body's subscripts fit, and every trial is then compared at that one
    shape rather than at whatever a per-trial redraw landed on. The metrics
    say which extent grew and to what, because a reader told the points were
    bit-exact is owed the shape they were bit-exact at."""
    verdict = judge(
        tmp_path,
        PACKED,
        SimpleNamespace(w_probe=lambda n, lr, r: float(r[(int(n) * (int(n) + 1)) // 2 - 1])),
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    probe = verdict.metrics["subprograms"]["probe"]
    # ``n`` is a value, not an extent, and is drawn across its whole range;
    # ``lr`` is the extent, and 64 covers the largest order the range holds.
    assert probe["extents"] == {"lr": 64}
    assert (probe["points"], probe["redrawn"], probe["reshaped"]) == (10, 0, 0)


SHORT = """\
_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [
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

WEIGHT = [0.25, 0.5, 0.75, 1.0]


def probe(lr, r):
    return float(r[0] * WEIGHT[int(lr) - 1])
"""


def _short(lr: Any, r: Any) -> float:
    return float(r[0] * [0.25, 0.5, 0.75, 1.0][int(lr) - 1])


def test_a_subprogram_compared_mostly_on_moved_extents_fails_by_name(tmp_path: Path) -> None:
    """Growth only goes up, and a body that indexes a fixed table of four by
    its workspace's extent takes no shape above four -- so no growth reaches
    one, and the trials fall back to the shape redraw. The draws that then
    fit are the short ones: a pass on those is evidence about a workspace of
    one or two, not about the extents the run was configured with, so the
    subprogram fails by name and says which extents to pin."""
    verdict = judge(tmp_path, SHORT, SimpleNamespace(w_probe=_short))
    assert verdict.confidence is Confidence.FAILED
    detail = verdict.detail or ""
    assert "probe: 10 of 10 trial(s) were compared only after the free extent(s) lr" in detail
    assert "Pin `dims`" in detail


NEGATIVE_SUBSCRIPT = """\
_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [
            {"name": "n", "intent": "IN", "dtype": "int32"},
            {
                "name": "x",
                "intent": "IN",
                "dtype": "float64",
                "dims": [{"lb": "1", "ub": None}],
            },
        ],
    }
}


def probe(n, x):
    return float(x[int(n) - 2])
"""


def test_a_subscript_below_the_lower_bound_is_drawn_again_not_wrapped(tmp_path: Path) -> None:
    """PCHIP's ``dpchkt`` forms ``x(n-1)`` and is only ever called with N>=2,
    so ``n=1`` reads ``x(0)`` -- one before the dummy's declared lower bound.
    Plain ndarray wraps a negative Python index to the array's *last*
    element instead of refusing it, so the candidate would silently compare
    the wrong value instead of the reference never being called on a draw
    outside its own domain. It must be drawn again like any other refused
    value -- and quietly: growing an extent never changes whether an index
    is negative, so it must not be counted as the ``reshaped`` extent-moved
    failure ``test_a_subprogram_compared_mostly_on_moved_extents_fails_by_name``
    covers."""

    def w_probe(n: Any, x: Any) -> Any:
        assert int(n) >= 2, "the reference was called on n=1, which reads before x's start"
        return float(x[int(n) - 2])

    verdict = judge(tmp_path, NEGATIVE_SUBSCRIPT, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    probe = verdict.metrics["subprograms"]["probe"]
    assert probe["redrawn"] > 0
    assert probe["reshaped"] == 0


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
        return np.sqrt(x + 500.0)
"""


def test_a_draw_both_sides_take_to_nan_is_drawn_again(tmp_path: Path) -> None:
    """Both sides compute the NaN; what they do with it afterwards is the
    compiler's business and not the translation's, so the trial is not one to
    hold either side to -- and a NaN agreeing with a NaN is not a point of
    evidence either, so the trial is drawn again rather than counted."""

    def w_probe(x: Any) -> Any:
        with np.errstate(invalid="ignore"):
            return np.sqrt(x + 500.0)

    verdict = judge(tmp_path, NAN, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    redrawn = verdict.metrics["subprograms"]["probe"]["redrawn"]
    assert redrawn > 0
    assert verdict.metrics["subprograms"]["probe"]["declined"] == {"NaN on both sides": redrawn}
    assert verdict.metrics["nan_mismatch"] == 0
    assert "NaN on both sides" in verdict.detail


def test_a_subprogram_that_mostly_goes_to_nan_on_both_sides_fails_by_name(tmp_path: Path) -> None:
    """Both sides agree on the NaN, on three draws in four. A NaN agreeing with
    a NaN is not evidence, and the draws that did compare are a minority of
    the configured range: the operator has to narrow the range, and the
    verdict says so rather than passing on the quarter that fit."""

    def w_probe(x: Any) -> Any:
        with np.errstate(invalid="ignore"):
            return np.sqrt(x - 500.0)

    verdict = judge(
        tmp_path, NAN.replace("x + 500.0", "x - 500.0"), SimpleNamespace(w_probe=w_probe)
    )
    assert verdict.confidence is Confidence.FAILED
    detail = verdict.detail or ""
    assert "draw(s) were declined (" in detail and "NaN on both sides) to compare" in detail
    assert "Narrow the draw with `ranges`" in detail


def test_a_nan_on_one_side_only_is_a_mismatch_not_a_redraw(tmp_path: Path) -> None:
    """The candidate goes to NaN where the reference has a number. That is
    the two sides disagreeing -- a variable read before it was assigned, a
    guard one side has and the other lost -- and redrawing it away would be
    exactly the narrowing the bound exists to prevent."""

    def w_probe(x: Any) -> Any:
        return np.sqrt(x + 500.0) if x >= -500.0 else np.float64(0.0)

    verdict = judge(tmp_path, NAN, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["nan_mismatch"] > 0
    assert "where one side produced NaN and the other a number" in (verdict.detail or "")


def test_a_draw_that_needs_no_redrawing_is_the_one_the_seed_names(tmp_path: Path) -> None:
    """The first draw of every trial is unchanged -- same seed, same extents --
    so a run that never has to redraw compares exactly what it compared
    before."""
    plain = NAN.replace("return np.sqrt(x + 500.0)", "return x * 2.0")
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


LAYOUT = """\
import numpy as np

_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [
            {"name": "k", "intent": "IN", "dtype": "int32"},
            {
                "name": "a",
                "intent": "IN",
                "dtype": "float64",
                "dims": [{"lb": "1", "ub": "3"}, {"lb": "1", "ub": "4"}],
            },
        ],
    }
}


def probe(k, a):
    return float(a[1, 2]) * float(k)
"""

LAYOUT_PROFILE = """\
import numpy as np


def prepare(unit, subprogram, inputs, rng):
    inputs["k"] = np.int32(2)  # shaped: the rest of the draw is returned as offered
    return inputs
"""


def test_a_shaped_draw_keeps_the_layout_of_what_the_profile_left_alone(tmp_path: Path) -> None:
    """The profile sees a copy of the draw. A copy in C order of a
    Fortran-ordered array is still a copy, but not one f2py takes for an
    array dummy: CLUBB's advance_helper_module, whose profile shapes the
    grid and returns the rest untouched, was refused by the reference with
    "input not fortran contiguous". The copy keeps the draw's layout."""
    seen: list[bool] = []

    def w_probe(k, a):
        seen.append(bool(a.flags.f_contiguous))
        return float(a[1, 2]) * float(k)

    root = profile(tmp_path, LAYOUT_PROFILE)
    verdict = judge(tmp_path, LAYOUT, SimpleNamespace(w_probe=w_probe), root=str(root))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert seen and all(seen), "the reference saw the draw's Fortran layout"


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

    verdict = judge(
        tmp_path, MODE, SimpleNamespace(w_probe=w_probe), root=str(root), ranges={"mode": (1, 3)}
    )
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
        NAN.replace("return np.sqrt(x + 500.0)", "return x * 2.0"),
        SimpleNamespace(w_probe=lambda x: x * 2.0),
        root=str(root),
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["input_profile"] is None


SHOUT = """\
_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [{"name": "x", "intent": "IN", "dtype": "float64"}],
    },
    "shout": {
        "kind": "subroutine",
        "result": None,
        "result_dtype": None,
        "args": [{"name": "msg", "intent": "IN", "dtype": "str"}],
    },
}


def probe(x):
    return x * 2.0


def shout(msg):
    print(msg)
"""


def test_a_subprogram_the_harness_has_no_draw_for_is_ungated_by_name(tmp_path: Path) -> None:
    """numfor's ``print_msg`` takes a message and writes it to stderr. The
    harness generates no character argument, so it was never compared -- and
    the coverage gate, judging against what was translated, failed the unit
    for the silence. Not compared is right; silent is not: it is ungated by
    name, with the reason, beside what the oracle and the operator declare."""
    verdict = judge(tmp_path, SHOUT, SimpleNamespace(w_probe=lambda x: x * 2.0))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["uncovered"] == []
    assert verdict.metrics["ungated"] == {
        "shout": "character argument msg: no generated draw for one"
    }
    assert "shout (character argument msg" in (verdict.detail or "")


def test_a_reference_that_aborts_declines_the_draw(tmp_path: Path) -> None:
    """The reference ending its process on a draw (a Fortran ERROR STOP, now
    an exception from its own process) is the source refusing the inputs,
    the way the candidate's SystemExit is: the draw is declined, drawn
    again, and the verdict says so -- the run is not taken down (#21)."""
    from recast.oracle.isolated import ReferenceAborted

    plain = NAN.replace("return np.sqrt(x + 500.0)", "return x * 2.0")
    calls = {"n": 0}

    def w_probe(x):
        calls["n"] += 1
        if calls["n"] % 3 == 1:
            raise ReferenceAborted("reference m.w_probe ended the process (exit 2)")
        return x * 2.0

    verdict = judge(tmp_path, plain, SimpleNamespace(w_probe=w_probe))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    probe = verdict.metrics["subprograms"]["probe"]
    assert probe["redrawn"] > 0
    assert "reference error stop" in (verdict.detail or ""), verdict.detail


BUFFER = """\
import numpy as np

_SIGNATURES = {
    "fill": {
        "kind": "subroutine",
        "args": [
            {
                "name": "x",
                "intent": "IN",
                "dtype": "float64",
                "dims": [{"lb": "1", "ub": None}],
            },
            {
                "name": "y",
                "intent": "OUT",
                "dtype": "float64",
                "dims": [{"lb": "1", "ub": None}],
                "buffer": True,
            },
        ],
    }
}


def fill(x, y):
    y[0] = x[0] * 2.0
    return y
"""


def test_a_caller_buffer_out_array_is_handed_to_the_reference_too(tmp_path: Path) -> None:
    """``y`` is the caller's storage on both sides: the callee writes one cell
    of it and leaves the rest as the caller had it. So the gate generates it,
    hands the same values to the reference and the candidate, and reads the
    reference's answer back out of the array it passed -- an f2py wrapper
    spells such a dummy ``intent(in out)`` and returns nothing for it. Handed
    to the candidate only, the reference is called an argument short, and the
    cells it never writes are compared against a fresh allocation."""
    seen: dict[str, Any] = {}

    def w_fill(x: Any, y: Any) -> None:
        seen["y"] = np.copy(y)
        y[0] = x[0] * 2.0

    verdict = BitexactVerifier().verify(
        Unit(uid="draw:m", kind="subprogram"),
        Candidate(
            unit="draw:m", transform="test.draw", files={Path("m_numpy.py"): BUFFER.encode()}
        ),
        OracleRef(
            unit="draw:m",
            oracle="test.python-truth",
            key="k",
            handle={"module": SimpleNamespace(w_fill=w_fill), "wrappers": {"fill": "w_fill"}},
        ),
        tmp_path,
        LocalExecutor(),
        {"draws": 2},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert seen["y"].any(), "the reference was handed a fresh buffer, not the caller's"


LOOPS = """\
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
    while int(mode) > 4:
        pass
    return x * 2.0
"""


def test_a_draw_the_source_never_returns_from_is_drawn_again(tmp_path: Path) -> None:
    """A fourth way a draw is not one the subprogram takes, and the only one
    with nothing to raise: the source's own loop never ends on it --
    ``do while (b - a > tol)`` under a negative tolerance. The reference is
    the same loop, so it must not be called on that draw either."""

    def w_probe(mode: Any, x: Any) -> Any:
        assert int(mode) <= 4, "the reference was called on a draw that never returns"
        return x * 2.0

    verdict = judge(tmp_path, LOOPS, SimpleNamespace(w_probe=w_probe), call_seconds=0.25, trials=4)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["subprograms"]["probe"]["redrawn"] > 0


OPTIONAL = """\
_SIGNATURES = {
    "probe": {
        "kind": "function",
        "result": "y",
        "result_dtype": "float64",
        "args": [
            {"name": "x", "intent": "IN", "dtype": "float64"},
            {"name": "maxiter", "intent": "UNKNOWN", "dtype": "int32", "optional": True},
        ],
    }
}


def probe(x, maxiter=None):
    return x * 2.0
"""


def test_an_optional_argument_of_unknown_intent_does_not_stop_the_comparison(
    tmp_path: Path,
) -> None:
    """``integer, optional :: maxiter`` declares no intent, and neither side
    is passed it: the wrapper drops an optional dummy and the translation
    spells it as a keyword sentinel. An intent nothing reads decides nothing,
    and refusing over it cost ``secant`` its comparison."""
    verdict = judge(tmp_path, OPTIONAL, SimpleNamespace(w_probe=lambda x: x * 2.0))
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail


def test_a_required_argument_of_unknown_intent_still_stops_it(tmp_path: Path) -> None:
    """The refusal it is a relaxation of: an argument both sides are passed
    whose post-call value may or may not be an output is not comparable."""
    required = OPTIONAL.replace(', "optional": True', "").replace("maxiter=None", "maxiter")
    verdict = judge(tmp_path, required, SimpleNamespace(w_probe=lambda x, maxiter: x * 2.0))
    assert verdict.confidence is Confidence.FAILED
    assert "maxiter have UNKNOWN intent" in (verdict.detail or "")


WRITER = """\
_SIGNATURES = {
    "probe": {
        "kind": "subroutine",
        "result": None,
        "result_dtype": None,
        "args": [
            {"name": "filename", "intent": "IN", "dtype": "str", "path": "created"},
            {"name": "x", "intent": "IN", "dtype": "float64"},
        ],
    }
}


def probe(filename, x):
    with open(filename, "wb") as handle:
        handle.write(b"%a" % float(x))
"""


def test_a_subprogram_whose_only_output_is_a_file_is_compared_on_that_file(
    tmp_path: Path,
) -> None:
    """No OUT argument and no result: without the file there is nothing to
    pair, and the gate would have to call this uncompared. Each side is given
    a scratch path of its own -- one path and the second call would overwrite
    what the comparison is about to read -- and the bytes are compared."""
    written: list[str] = []

    def w_probe(filename: Any, x: Any) -> None:
        written.append(str(filename))
        with open(str(filename), "wb") as handle:
            handle.write(b"%a" % float(x))

    verdict = judge(tmp_path, WRITER, SimpleNamespace(w_probe=w_probe), trials=2)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["integer_points"] == verdict.metrics["points"] > 0
    assert len(set(written)) == len(written), "each trial draws its own path"


def test_a_file_that_differs_by_one_byte_is_not_a_pass(tmp_path: Path) -> None:
    """The bar for a file is the bar for anything else: the same bytes. A
    candidate that opened the file and wrote nothing passed every structural
    check there is, which is what this gate is for."""

    def w_probe(filename: Any, x: Any) -> None:
        with open(str(filename), "wb") as handle:
            handle.write(b"%a " % float(x))

    verdict = judge(tmp_path, WRITER, SimpleNamespace(w_probe=w_probe), trials=1)
    assert verdict.confidence is Confidence.FAILED
    assert "filename (file)" in (verdict.detail or "")


def test_a_side_that_wrote_no_file_is_named(tmp_path: Path) -> None:
    """An absent file is not an empty file to compare against an empty file:
    the source's OPEN creates one, so nothing there is the call having done
    nothing."""
    verdict = judge(tmp_path, WRITER, SimpleNamespace(w_probe=lambda filename, x: None), trials=1)
    assert verdict.confidence is Confidence.FAILED
    assert "the oracle left no file at the path it was given" in (verdict.detail or "")


def test_a_shape_the_body_checks_for_is_the_shape_it_is_drawn_at() -> None:
    """``if (size(c,1) /= 5) call stop_error(...)`` is not a diagnostic aside:
    it is the declaration the language had no way to make about an
    assumed-shape dummy. Without it every extent is ``default_dim``, the body
    stops on every draw, and the subprogram is reported as one nothing could
    compare -- which says nothing about the translation."""
    from recast.verify.bitexact import DEFAULT_DIMENSION, _guarded_shapes

    required = [
        {"name": "xi", "dims": [{"lb": "1", "ub": None}]},
        {"name": "c", "dims": [{"lb": "0", "ub": None}, {"lb": "1", "ub": None}]},
        {"name": "val"},
    ]
    guards = [
        {"arg": "c", "axis": 0, "extent": "5"},
        {"arg": "c", "axis": 1, "extent": "size(xi,0) - 1"},
    ]
    assert _guarded_shapes(required, guards, {}) == {
        "xi": [DEFAULT_DIMENSION],
        "c": [5, DEFAULT_DIMENSION - 1],
    }


def test_a_shape_guard_that_resolves_to_nothing_leaves_the_default_alone() -> None:
    """An extent this cannot resolve, or one no array can have, is worse than
    the default it would replace: the subprogram refusing the default says so
    where a shape nobody can name would not."""
    from recast.verify.bitexact import DEFAULT_DIMENSION, _guarded_shapes

    required = [{"name": "a", "dims": [{"lb": "1", "ub": None}]}]
    guards = [
        {"arg": "a", "axis": 0, "extent": "size(missing,0)"},
        {"arg": "a", "axis": 0, "extent": "0"},
    ]
    assert _guarded_shapes(required, guards, {}) == {"a": [DEFAULT_DIMENSION]}


def test_the_reference_is_handed_only_the_buffers_its_wrapper_takes() -> None:
    """The f2py wrapper spells an OUT buffer ``inout`` only where it cannot
    size it -- an axis of no declared extent, or an allocatable -- and
    sizes and returns every other OUT array. The gate once handed every
    buffer, and under CLUBB's convention (every OUT array the caller's)
    each of its explicit-shape outputs was one keyword argument more than
    the wrapper took."""
    from recast.verify.bitexact import _reference_takes

    explicit = {"intent": "OUT", "buffer": True, "dims": [{"ub": "ngrdcol"}, {"ub": "nzm"}]}
    assumed = {"intent": "OUT", "buffer": True, "dims": [{"ub": None}]}
    star = {"intent": "OUT", "buffer": True, "dims": [{"ub": "n"}, {"assumed_size": True}]}
    allocatable = {"intent": "OUT", "allocatable": True, "dims": [{"ub": None}]}
    scalar = {"intent": "OUT", "buffer": True, "dims": None}
    inout = {"intent": "INOUT", "dims": [{"ub": "n"}]}
    assert not _reference_takes(explicit)
    assert _reference_takes(assumed)
    assert _reference_takes(star)
    assert _reference_takes(allocatable)
    assert not _reference_takes(scalar)
    assert not _reference_takes(inout)  # not an OUT: handed as an input anyway
