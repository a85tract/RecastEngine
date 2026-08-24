"""Tests for the ``f2py-golden`` oracle and the ``differential.bitexact`` gate.

Two layers. The mechanism tests need no compiler: wrapper text, cache keys,
and every fail-closed path of the verifier. The end-to-end test compiles a
real toy module with gfortran and walks the whole translate spine --
frontend, transform, rwset gate, oracle, bit-exact gate -- and then breaks
the candidate on purpose, because a gate that has never been seen to fail
proves nothing by passing.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.errors import ConfigError
from recast.executors.local import LocalExecutor
from recast.fortran.frontend import FortranFrontend
from recast.model import Candidate, Confidence, OracleRef, Unit
from recast.oracle.f2py import F2pyGoldenOracle, wrappers_for
from recast.transform.numpy.translate import NumpyTranslation
from recast.verify.bitexact import BitexactVerifier
from recast.verify.rwset import ReadWriteSetVerifier

GFORTRAN = shutil.which("gfortran")
MESON = importlib.util.find_spec("mesonbuild") is not None
"""f2py's build backend, carried by the verify extra. CI's test matrix has a
compiler (the runner image ships one) but not the backend, and the spine job
has both -- so the guard must check both, or the matrix runs half a build."""

RECORD = {
    "module": "demo_mod",
    "generics": {"scale": ["scale_r"]},
    "subprograms": [
        {
            "name": "settle",
            "kind": "subroutine",
            "args": [
                {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
                {
                    "name": "rho",
                    "dtype": "float64",
                    "intent": "IN",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
                {
                    "name": "p",
                    "dtype": "float64",
                    "intent": "OUT",
                    "optional": False,
                    "dims": [{"lb": "1", "ub": "n"}],
                },
                {"name": "extra", "dtype": "float64", "intent": "OUT", "optional": True},
            ],
        },
        {
            "name": "scale_r",
            "kind": "function",
            "args": [{"name": "x", "dtype": "float64", "intent": "IN", "optional": False}],
            "result": "y",
            "result_dtype": "float64",
        },
    ],
}


# --- wrapper text ------------------------------------------------------------


def test_wrappers_drop_optionals_and_route_generics() -> None:
    text, names = wrappers_for(RECORD, ["settle", "scale_r"])
    assert names == ["w_settle", "w_scale_r"]
    assert "extra" not in text  # optional: the wrapper compares the required surface
    # A specific of a generic is private; the call goes through the generic name.
    assert "use demo_mod, only: scale" in text
    assert "res = scale(x)" in text
    assert "real(8), intent(out) :: p(n)" in text  # dims spelled so f2py can size them


def test_a_dtype_the_wrapper_cannot_spell_refuses() -> None:
    broken = {
        "module": "m",
        "generics": {},
        "subprograms": [
            {
                "name": "s",
                "kind": "subroutine",
                "args": [
                    {
                        "name": "grid",
                        "dtype": "UNKNOWN(TYPE(GRID_T))",
                        "intent": "IN",
                        "optional": False,
                    }
                ],
            }
        ],
    }
    with pytest.raises(ConfigError, match="cannot spell"):
        wrappers_for(broken, ["s"])


# --- the verifier fails closed -----------------------------------------------


def _bare_candidate() -> Candidate:
    return Candidate(unit="fortran:demo_mod", transform="translate.numpy")


def _no_oracle() -> OracleRef:
    return OracleRef(unit="fortran:demo_mod", oracle="f2py-golden", key="x", handle=None)


def test_an_oracle_without_a_module_fails_closed(tmp_path: Path) -> None:
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:demo_mod", kind="module"),
        _bare_candidate(),
        _no_oracle(),
        tmp_path,
        LocalExecutor(),
        {},
    )
    assert verdict.confidence is Confidence.FAILED
    assert "no compiled module" in verdict.detail


def test_a_candidate_without_files_fails_closed(tmp_path: Path) -> None:
    ref = OracleRef(
        unit="fortran:demo_mod", oracle="f2py-golden", key="x", handle={"module": object()}
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:demo_mod", kind="module"),
        _bare_candidate(),
        ref,
        tmp_path,
        LocalExecutor(),
        {},
    )
    assert verdict.confidence is Confidence.FAILED
    assert "does not import" in verdict.detail


# --- the whole spine, against a real compiler --------------------------------

SOURCE = """\
module toy_physics
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: gravity = 9.80616_r8

contains

  subroutine settle(n, rho, dz, w, p)
    integer, intent(in) :: n
    real(r8), intent(in) :: rho(n)
    real(r8), intent(in) :: dz(n)
    real(r8), intent(inout) :: w(n)
    real(r8), intent(out) :: p(n)
    integer :: i
    p(1) = rho(1) * gravity * dz(1)
    do i = 2, n
      p(i) = p(i-1) + rho(i) * gravity * dz(i)
      w(i) = w(i) - dz(i) / (1.0_r8 + rho(i))
    end do
  end subroutine settle

  function column_mass(n, rho, dz) result(m)
    integer, intent(in) :: n
    real(r8), intent(in) :: rho(n)
    real(r8), intent(in) :: dz(n)
    real(r8) :: m
    integer :: i
    m = 0.0_r8
    do i = 1, n
      m = m + rho(i) * dz(i)
    end do
  end function column_mass
end module toy_physics
"""


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_translate_spine_ends_bit_exact(tmp_path: Path) -> None:
    (tmp_path / "toy_physics.f90").write_text(SOURCE)
    workspace = tmp_path / "work"
    workspace.mkdir()
    executor = LocalExecutor()

    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.kind == "module")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": tmp_path})
    assert candidate.deferred == []

    gate = ReadWriteSetVerifier().check(unit, candidate, workspace, executor, {})
    assert gate.passed, gate.detail

    config = {
        "root": tmp_path,
        "fc": GFORTRAN,
        "trials": 5,
        "dims": {"n": 8},
        "ranges": {"rho": (0.1, 2.0), "dz": (10.0, 100.0), "w": (-5.0, 5.0)},
    }
    oracle = F2pyGoldenOracle()
    key = oracle.key(unit, facts, config)
    assert key == oracle.key(unit, facts, config)  # stable
    assert key != oracle.key(unit, facts, {**config, "fflags": "-O2"})  # flags move it

    ref = oracle.materialize(unit, facts, workspace, executor, config)
    verdict = BitexactVerifier().verify(unit, candidate, ref, workspace, executor, config)
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail
    assert verdict.metrics["points"] > 0
    assert verdict.metrics["bit_exact"] == verdict.metrics["points"]

    # A gate that has never failed proves nothing by passing: corrupt one
    # constant in the candidate and the same comparison must say FAILED.
    module_path = next(p for p in candidate.files if str(p).endswith("_numpy.py"))
    broken = Candidate(
        unit=candidate.unit,
        transform=candidate.transform,
        files={
            **candidate.files,
            module_path: candidate.files[module_path].replace(b"GRAVITY", b"(GRAVITY * 1.0000001)"),
        },
        deferred=list(candidate.deferred),
        notes=dict(candidate.notes),
    )
    broken_workspace = tmp_path / "broken"
    broken_workspace.mkdir()
    failed = BitexactVerifier().verify(unit, broken, ref, broken_workspace, executor, config)
    assert failed.confidence is Confidence.FAILED
    assert failed.metrics["bit_exact"] < failed.metrics["points"]

    # ...unless the operator explicitly asked for a tolerance that excuses it.
    excused = BitexactVerifier().verify(
        unit, broken, ref, broken_workspace, executor, {**config, "rtol": 1e-3}
    )
    assert excused.confidence is Confidence.TOLERANCED


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_example_runs_through_the_cli(tmp_path: Path) -> None:
    """The roadmap's P2 claim, literally: `recast run translate examples/...`
    walks every stage and leaves evidence manifests behind."""
    import json
    import shutil as _shutil

    from recast.cli import main

    example = Path(__file__).resolve().parent.parent / "examples" / "toy_physics"
    staged = tmp_path / "toy_physics"
    _shutil.copytree(example, staged, ignore=_shutil.ignore_patterns(".recast"))

    code = main(["run", "translate", str(staged), "--config", str(staged / "recast.json")])
    assert code == 0
    manifests = list((staged / ".recast" / "evidence").rglob("*.json"))
    assert len(manifests) == 3  # rwset, bitexact, notary
    results = {json.loads(m.read_text())["result"]["verdict"] for m in manifests}
    assert results == {"sampled", "bit_exact", "symbolic"}


def test_the_oracle_defaults_to_public_subprograms() -> None:
    """The wrappers `use` the module, and a private symbol is not importable
    -- one private specific in the list fails the whole build."""
    from recast.model import Facts

    facts = Facts(
        unit="fortran:m",
        interface={
            "module": "m",
            "subprograms": [
                {"name": "api", "public": True},
                {"name": "detail", "public": False},
            ],
        },
    )
    assert F2pyGoldenOracle._subprograms(facts, {}) == ["api"]
    # Explicit config still wins, and then fails loudly if it names a private.
    assert F2pyGoldenOracle._subprograms(facts, {"subprograms": ["detail"]}) == ["detail"]


def test_wrappers_serve_a_file_of_bare_subprograms() -> None:
    """A file with no module borrows its stem for a name, so a `use` line
    would not compile -- the callee is an external. Dimension names the file
    use-imports arrive as local PARAMETERs, which is what lets f2py fold the
    declared shapes."""
    record = {
        "module": "dadadj",
        "is_module": False,
        "generics": {},
        "subprograms": [
            {
                "name": "dadadj_native",
                "kind": "subroutine",
                "args": [
                    {
                        "name": "t",
                        "dtype": "float64",
                        "intent": "INOUT",
                        "optional": False,
                        "dims": [{"lb": "1", "ub": "pcols"}, {"lb": "1", "ub": "pver"}],
                    }
                ],
            }
        ],
    }
    text, _ = wrappers_for(record, ["dadadj_native"], parameters={"pcols": 8, "pver": 30})
    assert "use dadadj" not in text
    assert "external dadadj_native" in text
    assert "integer, parameter :: pcols = 8" in text
    assert "real(8), intent(inout) :: t(pcols, pver)" in text


def test_the_gate_lets_a_candidate_shape_its_own_inputs(tmp_path: Path) -> None:
    """Per-name ranges cannot express structure -- a monotone pressure
    column, a consistent thickness field. A candidate may carry
    ``_PREPARE_INPUTS`` the way it carries ``_SIGNATURES``; both sides then
    receive the same shaped arrays, so it chooses the sampled region without
    touching the verdict."""
    import numpy as np

    module = tmp_path / "candidate"
    module.mkdir()
    (module / "shaped_numpy.py").write_text(
        """
import numpy as np

_SIGNATURES = {
    "step": {
        "kind": "subroutine",
        "args": [
            {"name": "x", "dtype": "float64", "intent": "IN", "optional": False,
             "dims": [{"lb": "1", "ub": "n"}]},
            {"name": "y", "dtype": "float64", "intent": "OUT", "optional": False,
             "dims": [{"lb": "1", "ub": "n"}]},
        ],
        "result": None, "result_dtype": None,
    }
}
SEEN = []


def _PREPARE_INPUTS(name, inputs, rng):
    inputs["x"][:] = 2.0        # every trial sees the same shaped input


def step(x):
    SEEN.append(float(x[0]))
    return np.asarray(x) * 3.0
"""
    )

    class Truth:
        @staticmethod
        def w_step(x):
            return np.asarray(x) * 3.0

    candidate = Candidate(
        unit="fortran:shaped",
        transform="t",
        files={Path("shaped_numpy.py"): (module / "shaped_numpy.py").read_bytes()},
    )
    ref = OracleRef(
        unit="fortran:shaped",
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"step": "w_step"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:shaped", kind="module"),
        candidate,
        ref,
        tmp_path / "ws",
        LocalExecutor(),
        {"trials": 3, "dims": {"n": 4}, "ranges": {"x": (100.0, 200.0)}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT
    staged = tmp_path / "ws" / "candidate"
    sys.path.insert(0, str(staged))
    try:
        import shaped_numpy

        # The hook ran: every trial saw 2.0, not a value from the range.
        assert shaped_numpy.SEEN and all(v == 2.0 for v in shaped_numpy.SEEN)
    finally:
        sys.path.remove(str(staged))
        sys.modules.pop("shaped_numpy", None)


def test_the_oracle_side_is_called_with_lowercased_names(tmp_path: Path) -> None:
    """Fortran is case-insensitive and f2py lowercases every dummy name, so
    a candidate reporting `sl_prePBL` must still reach the same oracle
    argument. The source's spelling is not a fact about the interface."""
    staged = tmp_path / "cand"
    staged.mkdir()
    (staged / "mixed_numpy.py").write_text(
        """
import numpy as np

_SIGNATURES = {
    "step": {
        "kind": "subroutine",
        "args": [
            {"name": "inVal", "dtype": "float64", "intent": "IN", "optional": False},
            {"name": "outVal", "dtype": "float64", "intent": "OUT", "optional": False},
        ],
        "result": None, "result_dtype": None,
    }
}


def step(inVal):
    return inVal * 2.0
"""
    )

    class Truth:
        @staticmethod
        def w_step(**kwargs):
            # f2py's own convention: lowercase only.
            return kwargs["inval"] * 2.0

    candidate = Candidate(
        unit="fortran:mixed",
        transform="t",
        files={Path("mixed_numpy.py"): (staged / "mixed_numpy.py").read_bytes()},
    )
    ref = OracleRef(
        unit="fortran:mixed",
        oracle="f2py-golden",
        key="k",
        handle={"module": Truth(), "wrappers": {"step": "w_step"}},
    )
    verdict = BitexactVerifier().verify(
        Unit(uid="fortran:mixed", kind="module"),
        candidate,
        ref,
        tmp_path / "ws",
        LocalExecutor(),
        {"trials": 2, "ranges": {"inval": (1.0, 2.0)}},
    )
    assert verdict.confidence is Confidence.BIT_EXACT, verdict.detail


@pytest.mark.skipif(GFORTRAN is None, reason="the cache key asks the compiler its version")
def test_a_refused_build_fails_this_stage_and_not_the_run(tmp_path: Path) -> None:
    """An executor that will not run the build is an unavailable oracle.

    ``run_recipe`` catches ``RecastError`` and marks the unit's oracle stage
    failed; anything else escapes it. A refusal that arrives as a bare
    ``RuntimeError`` therefore costs every *other* unit its verdict too, which
    is a much larger blast radius than the one build that could not run.
    """
    from recast.conformance.doubles import RefusingExecutor
    from recast.errors import OracleUnavailable

    (tmp_path / "toy_physics.f90").write_text(SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:toy_physics")
    facts = frontend.analyze(unit, tmp_path)
    with pytest.raises(OracleUnavailable, match="did not run the f2py build"):
        F2pyGoldenOracle().materialize(
            unit,
            facts,
            tmp_path / "work",
            RefusingExecutor(),
            {"root": str(tmp_path)},
        )


KINDS_SOURCE = """\
module toy_kinds
  use, intrinsic :: iso_fortran_env
  implicit none
  integer, parameter :: wp = real64
end module toy_kinds
"""

SPLIT_SOURCE = """\
module toy_split
  use toy_kinds, only: wp
  implicit none
contains
  subroutine scale_all(n, a, x)
    integer, intent(in) :: n
    real(wp), intent(in) :: a
    real(wp), intent(inout) :: x(*)
    integer :: i
    do i = 1, n
      x(i) = a * x(i)
    end do
  end subroutine scale_all
end module toy_split
"""


def _split_tree(tmp_path: Path) -> tuple[Unit, object]:
    (tmp_path / "toy_kinds.f90").write_text(KINDS_SOURCE)
    (tmp_path / "toy_split.f90").write_text(SPLIT_SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:toy_split")
    return unit, frontend.analyze(unit, tmp_path)


def test_the_reference_names_the_siblings_the_unit_uses(tmp_path: Path) -> None:
    """A module that takes its precision from a kinds module one file over
    does not compile alone -- gfortran wants a ``.mod`` nobody built. The
    frontend already resolved the sibling, so the build asks the facts rather
    than making the operator list it by hand."""
    from recast.oracle.f2py import companion_sources

    _unit, facts = _split_tree(tmp_path)
    assert companion_sources(facts, tmp_path) == [(tmp_path / "toy_kinds.f90").resolve()]


def test_a_changed_sibling_moves_the_cache_key(tmp_path: Path) -> None:
    """The reference is only a reference if everything that can change what it
    computes is in its key. A kinds module edited from real64 to real32 is a
    different reference, not the same one."""
    unit, facts = _split_tree(tmp_path)
    oracle = F2pyGoldenOracle()
    config = {"root": tmp_path, "fc": GFORTRAN or "gfortran"}
    before = oracle.key(unit, facts, config)
    (tmp_path / "toy_kinds.f90").write_text(KINDS_SOURCE.replace("real64", "real32"))
    assert oracle.key(unit, facts, config) != before


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_reference_builds_across_two_files(tmp_path: Path) -> None:
    """The same case, actually compiled. Every library in the public corpus
    that keeps its working precision in its own module failed here, on a
    ``Cannot open module file`` that named a file sitting beside the source."""
    unit, facts = _split_tree(tmp_path)
    workspace = tmp_path / "work"
    workspace.mkdir()
    ref = F2pyGoldenOracle().materialize(
        unit, facts, workspace, LocalExecutor(), {"root": tmp_path, "fc": GFORTRAN}
    )
    assert ref.handle["wrappers"]["scale_all"] == "w_scale_all"
