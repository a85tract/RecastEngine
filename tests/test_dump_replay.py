"""The replay oracle, and the direction it makes the differential gate run.

Two things are under test and they are different in kind. The parser is a
*relay* -- it must read the probe format the way the script it came from reads
it, and ``tools/dump_diff.py`` is what holds it there over constructed cases.
What is here instead is the part that has no upstream to be held to: the
inversion. Every other oracle answers inputs the harness chose; this one
supplies them, and the tests below are about what that costs and what the gate
must refuse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recast.errors import ConfigError, OracleUnavailable
from recast.model import Candidate, Confidence, Facts, OracleRef, Unit
from recast.oracle.dump_replay import DumpReplayOracle, parse_dump, parse_probe_header

np = pytest.importorskip("numpy")

PROBE = """# PROBE toy.scale_by_two: call=       1
# n =       3
# INPUT: x(n)
1.0
2.0
3.0
# OUTPUT: y(n)
2.0
4.0
6.0
"""


# -- the parser ---------------------------------------------------------------


def test_a_header_scalar_is_both_an_extent_and_an_input() -> None:
    """``n`` shapes the arrays below it and is also an argument.

    Relayed deliberately: a probe writes the extent once and the subprogram
    takes it, so a parser that treated it as metadata only would drop a
    required input.
    """
    inputs, outputs = parse_dump(PROBE)
    assert inputs["n"] == 3
    assert inputs["x"].tolist() == [1.0, 2.0, 3.0]
    assert outputs["y"].tolist() == [2.0, 4.0, 6.0]


def test_g_editing_drops_the_e_when_the_exponent_is_three_digits() -> None:
    """``1.07...-114`` is a number, not a subtraction. Python cannot read it."""
    inputs, _ = parse_dump("# INPUT: v(1)\n1.0701116457083034-114\n")
    assert inputs["v"][0] == pytest.approx(1.0701116457083034e-114, rel=0, abs=0)


def test_a_rank_two_array_is_walked_in_fortran_order() -> None:
    text = "# a =       2\n# b =       3\n# INPUT: m(a,b)\n" + "\n".join("123456") + "\n"
    inputs, _ = parse_dump(text)
    assert inputs["m"].shape == (2, 3)
    # Fortran order: the first index moves fastest.
    assert inputs["m"][0, 0] == 1.0
    assert inputs["m"][1, 0] == 2.0
    assert inputs["m"][0, 1] == 3.0


def test_a_shape_that_does_not_multiply_out_keeps_the_flat_array() -> None:
    """Inherited, not chosen: upstream keeps the data and drops the shape.

    The probe's declared extents are a claim about the data; when the two
    disagree the data is what was actually written, and refusing here would
    turn a recoverable recording into no recording at all.
    """
    inputs, _ = parse_dump("# INPUT: z(2,3)\n1.0\n2.0\n3.0\n4.0\n")
    assert inputs["z"].shape == (4,)


def test_an_integer_array_is_read_as_integers() -> None:
    # The recorder writes integer arrays with i0 and reals with an exponent;
    # a declared-integer output compared as float64 is refused by the gate.
    inputs, outputs = parse_dump(
        "# INPUT: nbot(1)\n-9999\n# INPUT: x(2)\n1.0\n2.0\n# OUTPUT: nbot(1)\n2\n"
    )
    assert inputs["nbot"].dtype == np.int32 and inputs["nbot"].tolist() == [-9999]
    assert inputs["x"].dtype == np.float64
    assert outputs["nbot"].dtype == np.int32 and outputs["nbot"].tolist() == [2]


def test_the_probe_header_names_what_was_recorded() -> None:
    """The line upstream's parser drops, and the reason the fuzzy matcher existed."""
    assert parse_probe_header(PROBE) == ("toy", "scale_by_two")
    assert parse_probe_header("# INPUT: x(1)\n1.0\n") is None


# -- the oracle ---------------------------------------------------------------


def _unit() -> Unit:
    return Unit(uid="toy", kind="module", sources=(Path("toy.F90"),))


def _facts() -> Facts:
    return Facts(
        unit="toy",
        interface={"module": "toy", "subprograms": []},
        provenance={"digest": "d"},
    )


def _dumps(tmp_path: Path, *files: tuple[str, str]) -> dict[str, Any]:
    for name, text in files:
        (tmp_path / name).write_text(text)
    return {"dumps": str(tmp_path)}


def test_a_missing_dumps_config_is_a_config_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        DumpReplayOracle().key(_unit(), _facts(), {})


def test_an_empty_directory_is_unavailable_not_empty_evidence(tmp_path: Path) -> None:
    """Fail closed: no recording is not the same as a recording that agreed."""
    with pytest.raises(OracleUnavailable):
        DumpReplayOracle().materialize(
            _unit(), _facts(), tmp_path, _executor(), {"dumps": str(tmp_path)}
        )


def test_a_dump_with_no_probe_header_is_refused_rather_than_guessed(tmp_path: Path) -> None:
    """Which subprogram a recording is of is the one thing it cannot infer."""
    config = _dumps(tmp_path, ("a.txt", "# INPUT: x(1)\n1.0\n# OUTPUT: y(1)\n2.0\n"))
    with pytest.raises(OracleUnavailable) as caught:
        DumpReplayOracle().materialize(_unit(), _facts(), tmp_path, _executor(), config)
    assert "PROBE" in str(caught.value)


def test_the_key_moves_with_the_recording_and_not_with_its_path(tmp_path: Path) -> None:
    """A dump carries no build identity, so its content is the whole key."""
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    (first / "a.txt").write_text(PROBE)
    (second / "a.txt").write_text(PROBE)
    oracle = DumpReplayOracle()
    assert oracle.key(_unit(), _facts(), {"dumps": str(first)}) == oracle.key(
        _unit(), _facts(), {"dumps": str(second)}
    )
    (second / "a.txt").write_text(PROBE.replace("2.0\n4.0", "2.0\n4.5"))
    assert oracle.key(_unit(), _facts(), {"dumps": str(first)}) != oracle.key(
        _unit(), _facts(), {"dumps": str(second)}
    )


def test_the_handle_declares_that_it_supplies_the_inputs(tmp_path: Path) -> None:
    config = _dumps(tmp_path, ("a.txt", PROBE))
    ref = DumpReplayOracle().materialize(_unit(), _facts(), tmp_path, _executor(), config)
    assert ref.handle["input_source"] == "recorded"
    assert ref.handle["module"] is None
    assert ref.handle["return_convention"] == "recorded"
    assert [s["subprogram"] for s in ref.handle["samples"]] == ["scale_by_two"]
    # Nothing claims to know which machine produced the recording.
    assert ref.handle["device"] is None


# -- the gate, running backwards ----------------------------------------------

SIGNATURES = {
    "scale_by_two": {
        "kind": "subroutine",
        "args": [
            {"name": "n", "dtype": "int32", "intent": "IN", "dims": None},
            {
                "name": "x",
                "dtype": "float64",
                "intent": "IN",
                "dims": [{"lb": "1", "ub": "n"}],
            },
            {
                "name": "y",
                "dtype": "float64",
                "intent": "OUT",
                "dims": [{"lb": "1", "ub": "n"}],
            },
        ],
    }
}


def _candidate_module(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "toy_numpy.py"
    path.write_text(
        f"import numpy as np\n_SIGNATURES = {SIGNATURES!r}\ndef scale_by_two(n, x):\n{body}\n"
    )
    return path


def _executor() -> Any:
    from recast.executors.local import LocalExecutor

    return LocalExecutor()


def _verdict(tmp_path: Path, body: str, dump: str = PROBE) -> Any:
    from recast.verify.bitexact import BitexactVerifier

    _candidate_module(tmp_path, body)
    candidate = Candidate(
        unit="toy",
        transform="translate.numpy",
        files={Path("toy_numpy.py"): (tmp_path / "toy_numpy.py").read_bytes()},
    )
    dumps = tmp_path / "dumps"
    dumps.mkdir(exist_ok=True)
    (dumps / "a.txt").write_text(dump)
    ref = DumpReplayOracle().materialize(
        _unit(), _facts(), tmp_path, _executor(), {"dumps": str(dumps)}
    )
    return BitexactVerifier().verify(_unit(), candidate, ref, tmp_path, _executor(), {})


def test_a_replay_is_bit_exact_when_the_candidate_reproduces_the_recording(
    tmp_path: Path,
) -> None:
    verdict = _verdict(tmp_path, "    return x * 2.0")
    assert verdict.confidence is Confidence.BIT_EXACT
    assert verdict.metrics["points"] == 3
    assert verdict.metrics["bit_exact"] == 3


def test_a_replay_fails_when_the_candidate_does_not(tmp_path: Path) -> None:
    """The gate can fail. A harness that cannot is not a gate."""
    verdict = _verdict(tmp_path, "    return x * 2.5")
    assert verdict.confidence is Confidence.FAILED


def test_the_points_are_the_recorded_ones_and_not_a_trial_count(tmp_path: Path) -> None:
    """``trials`` is a sampling parameter and a recording is not sampled.

    Ten trials against a three-value recording would be either ten copies of
    the same comparison or seven invented ones. It is three.
    """
    verdict = _verdict(tmp_path, "    return x * 2.0")
    assert verdict.metrics["points"] == 3


def test_a_required_argument_the_recording_does_not_name_is_refused(tmp_path: Path) -> None:
    """No substring match, no zero fill -- the two things the source script did.

    This is the one place the migration departs from what it relayed, and it
    departs toward refusing.
    """
    without_n = PROBE.replace("# n =       3\n", "")
    verdict = _verdict(tmp_path, "    return x * 2.0", dump=without_n)
    assert verdict.confidence is Confidence.FAILED
    assert "records no value for" in verdict.detail


def test_an_oracle_that_supplies_no_samples_fails_closed(tmp_path: Path) -> None:
    from recast.verify.bitexact import BitexactVerifier

    _candidate_module(tmp_path, "    return x * 2.0")
    candidate = Candidate(
        unit="toy",
        transform="translate.numpy",
        files={Path("toy_numpy.py"): (tmp_path / "toy_numpy.py").read_bytes()},
    )
    empty = OracleRef(
        unit="toy",
        oracle="dump-replay",
        key="k",
        handle={"input_source": "recorded", "samples": [], "module": None},
    )
    verdict = BitexactVerifier().verify(_unit(), candidate, empty, tmp_path, _executor(), {})
    assert verdict.confidence is Confidence.FAILED
    assert "handed over no samples" in verdict.detail


def test_a_candidate_hook_cannot_edit_the_recorded_inputs(tmp_path: Path) -> None:
    """``_PREPARE_INPUTS`` shapes *generated* inputs and must not touch these.

    The hook ships inside the artifact under test. Letting it rewrite the
    production run's own numbers before the artifact is judged on them would
    let the candidate choose its own exam.
    """
    path = tmp_path / "toy_numpy.py"
    path.write_text(
        "import numpy as np\n"
        f"_SIGNATURES = {SIGNATURES!r}\n"
        "def _PREPARE_INPUTS(name, inputs, rng):\n"
        "    inputs['x'] = inputs['x'] * 0.0\n"
        "def scale_by_two(n, x):\n"
        "    return x * 2.0\n"
    )
    candidate = Candidate(
        unit="toy",
        transform="translate.numpy",
        files={Path("toy_numpy.py"): path.read_bytes()},
    )
    dumps = tmp_path / "dumps"
    dumps.mkdir(exist_ok=True)
    (dumps / "a.txt").write_text(PROBE)
    ref = DumpReplayOracle().materialize(
        _unit(), _facts(), tmp_path, _executor(), {"dumps": str(dumps)}
    )
    from recast.verify.bitexact import BitexactVerifier

    verdict = BitexactVerifier().verify(_unit(), candidate, ref, tmp_path, _executor(), {})
    # If the hook had run, x would be zeros and the recording's 2/4/6 would
    # not be reproduced.
    assert verdict.confidence is Confidence.BIT_EXACT


# -- the example, end to end --------------------------------------------------


def test_the_shipped_example_replays_bit_exact(tmp_path: Path) -> None:
    """The port side's cheap spine, on material that is in the repository.

    ``examples/toy_physics/dumps/`` is synthetic and says so in every file --
    no production dump is committed in either repository, and a test that
    waited for one would never run. What it can still establish is the whole
    chain in one go: frontend, the NumPy transform, the replay oracle, and the
    gate running backwards, on a unit whose recorded answers were computed
    from the Fortran's own arithmetic.
    """
    pytest.importorskip("fparser")
    from recast.registry import REGISTRY
    from recast.verify.bitexact import BitexactVerifier

    root = Path(__file__).resolve().parent.parent / "examples" / "toy_physics"
    if not (root / "dumps").is_dir():  # pragma: no cover - wheel install
        pytest.skip("examples/ is not present in an installed wheel")

    frontend = REGISTRY.get("frontend", "fortran")()
    unit = min(frontend.discover(root), key=lambda u: len(u.uid))
    facts = frontend.analyze(unit, root)
    config = {"root": str(root)}
    candidate = REGISTRY.get("transform", "translate.numpy")().apply(unit, facts, config)
    for name, blob in candidate.files.items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(blob)

    ref = DumpReplayOracle().materialize(
        unit, facts, tmp_path, _executor(), {**config, "dumps": str(root / "dumps")}
    )
    verdict = BitexactVerifier().verify(unit, candidate, ref, tmp_path, _executor(), config)
    assert verdict.confidence is Confidence.BIT_EXACT
    # Three recordings of a subprogram with two out-intent arguments over
    # columns of 4, 3 and 2 levels, 2*(4+3+2), plus three recordings of a
    # scalar function over the same columns, 3*1. Every translated
    # subprogram of the unit is recorded: one that was not would be an
    # uncovered translation, and the gate refuses to call that a pass.
    assert verdict.metrics["points"] == 21
    assert verdict.metrics["bit_exact"] == 21
    assert verdict.metrics["max_ulp"] == 0
    compared = verdict.metrics["subprograms"]
    assert set(compared) == {"settle", "column_mass"}
    assert all(outcome["points"] > 0 for outcome in compared.values())
    assert verdict.metrics["uncovered"] == []


def test_a_logical_header_scalar_is_an_input() -> None:
    """CLUBB's ``l_implemented``: the recorder writes a logical ``T``/``F``.
    The parser took only numbers, so the replay had no value for it."""
    inputs, _ = parse_dump("# PROBE m.s: call=1\n# l_on = T\n# l_off = F\n# INPUT: x(1)\n1.0\n")
    assert inputs["l_on"] is not None and bool(inputs["l_on"]) is True
    assert bool(inputs["l_off"]) is False
    assert inputs["l_on"].dtype == np.bool_


def test_a_zero_extent_array_is_a_value() -> None:
    """A component the run never allocated (CLUBB's scalar tracers under
    ``sclr_dim = 0``) is written ``name(1,3,0)`` with nothing under it.
    Dropped, the replay said the record carried no value for it."""
    text = "# PROBE m.s: call=1\n# INPUT: s(1,3,0)\n# OUTPUT: t(0)\n# OUTPUT: y(1)\n2.0\n"
    inputs, outputs = parse_dump(text)
    assert inputs["s"].shape == (1, 3, 0)
    assert outputs["t"].shape == (0,)
    assert outputs["y"].tolist() == [2.0]
