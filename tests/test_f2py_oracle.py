"""Tests for the ``f2py-golden`` oracle and the ``differential.bitexact`` gate.

Two layers. The mechanism tests need no compiler: wrapper text, cache keys,
and every fail-closed path of the verifier. The end-to-end test compiles a
real toy module with gfortran and walks the whole translate spine --
frontend, transform, rwset gate, oracle, bit-exact gate -- and then breaks
the candidate on purpose, because a gate that has never been seen to fail
proves nothing by passing.
"""

from __future__ import annotations

import shutil
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


@pytest.mark.skipif(GFORTRAN is None, reason="needs a Fortran compiler on PATH")
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
