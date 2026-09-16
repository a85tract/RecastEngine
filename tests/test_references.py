"""Tests for the reference implementations recast supplies for a missing library.

The whole value of :mod:`recast.references` is that its two spellings of one
procedure -- the Fortran the oracle build compiles and the Python the
translation calls -- produce the same bits. Nothing structural can check that:
it is a claim about rounding, and the only way to hold it is to build both and
run them over the same draws, which is what the last test here does.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[verify]")

import numpy as np

from recast import references

GFORTRAN = shutil.which("gfortran")
MESON = importlib.util.find_spec("mesonbuild") is not None

WRAPPER = """\
subroutine w_dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
  implicit none
  integer, intent(in) :: n, nrhs, lda, ldb
  integer, intent(inout) :: ipiv(lda), info
  double precision, intent(inout) :: a(lda,lda), b(ldb,nrhs)
  call dgesv(n, nrhs, a, lda, ipiv, b, ldb, info)
end subroutine w_dgesv

subroutine w_dgbsv(n, kl, ku, nrhs, ab, ldab, m, ipiv, b, ldb, info)
  implicit none
  integer, intent(in) :: n, kl, ku, nrhs, ldab, ldb, m
  integer, intent(inout) :: ipiv(n), info
  double precision, intent(inout) :: ab(ldab,m), b(ldb,nrhs)
  call dgbsv(n, kl, ku, nrhs, ab, ldab, ipiv, b, ldb, info)
end subroutine w_dgbsv
"""


def _python_side() -> Any:
    """The emitted Python, run the way a generated module runs it."""
    namespace: dict[str, Any] = {"np": np}
    text = "\n".join(references.python_for(references.SUPPORTED))
    exec(compile(text, "<refs>", "exec"), namespace)
    return namespace


# --- what is supplied and what is not ----------------------------------------


def test_only_the_named_procedures_are_supplied() -> None:
    """A name recast has no implementation for is still declared, still
    undefined, and still disclaims its callers -- an audited shim is what
    covers those, and it is the operator's to write."""
    assert references.supported(["DGESV", "dgbsv", "dsyevd", "ilaenv"]) == ["dgbsv", "dgesv"]
    assert references.supported(["dsyevd"]) == []
    assert references.fortran_for(["dsyevd"]) == ""
    assert references.python_for(["dsyevd"]) == []


def test_the_fortran_defines_a_global_symbol_not_a_module() -> None:
    """The callers were compiled against the interface their own tree
    declared; what they need resolved is the external symbol of that name."""
    text = references.fortran_for(["dgesv"])
    assert "\nsubroutine dgesv(" in text
    assert "module" not in text.replace("implementations", "")
    assert "recast_ref_gepp" in text, "the shared elimination has to come with it"


def test_the_python_defines_the_names_the_translation_calls() -> None:
    """``call dgbsv(...)`` in a module that ``use``s the interface module is
    emitted as ``_lapack.dgbsv(...)``, so the definition has to land in the
    translation of *that* module under exactly that name."""
    side = _python_side()
    assert callable(side["dgesv"])
    assert callable(side["dgbsv"])


# --- the two sides are one implementation ------------------------------------


def test_a_singular_matrix_refuses_rather_than_returning_a_number() -> None:
    """A translated call cannot write INFO back into its caller's scalar, so
    the caller's ``if (info /= 0) call stop_error(...)`` cannot fire on this
    side. Stopping the way a translated ERROR STOP stops is the honest
    remainder: the differential calls the candidate first and draws again."""
    side = _python_side()
    a = np.asfortranarray(np.zeros((2, 2)))
    b = np.asfortranarray(np.ones((2, 1)))
    with pytest.raises(SystemExit):
        side["dgesv"](2, 1, a, 2, np.zeros(2, dtype=np.int32), b, 2, 0)


def test_a_dummy_whose_leading_dimension_is_not_its_extent_is_refused() -> None:
    """The Python reads a Fortran ``(LD,*)`` dummy as a rank-2 array whose own
    first extent is LD. Anything else is a sequence association it cannot see
    through, and reading the wrong element silently is the one outcome worth
    refusing."""
    side = _python_side()
    a = np.asfortranarray(np.eye(3))
    b = np.asfortranarray(np.ones((3, 1)))
    with pytest.raises(SystemExit):
        side["dgesv"](2, 1, a, 2, np.zeros(3, dtype=np.int32), b, 3, 0)


@pytest.mark.skipif(
    GFORTRAN is None or not MESON,
    reason="needs a Fortran compiler and the meson backend (recast-engine[verify])",
)
def test_the_two_sides_are_bit_for_bit_one_implementation(tmp_path: Path) -> None:
    """The only property that makes this worth having.

    If the Fortran and the Python drift by a single ULP, every subprogram
    that reaches one of these fails the differential -- and it fails as a
    translation defect, pointing at the caller rather than at this file. So
    both are built and run over the same draws, and the comparison is on the
    bytes.
    """
    (tmp_path / "refs.f90").write_text(references.fortran_for(references.SUPPORTED))
    (tmp_path / "wrap.f90").write_text(WRAPPER)
    flags = "-O1 -fno-fast-math -ffp-contract=off"
    built = subprocess.run(
        [
            sys.executable,
            "-m",
            "numpy.f2py",
            "-c",
            "--build-dir",
            "build",
            "wrap.f90",
            "refs.f90",
            "-m",
            "refs",
            f"--f90flags={flags}",
            "--backend",
            "meson",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert built.returncode == 0, built.stdout[-3000:] + built.stderr[-3000:]

    sys.path.insert(0, str(tmp_path))
    try:
        fortran = importlib.import_module("refs")
    finally:
        sys.path.remove(str(tmp_path))
    python = _python_side()

    rng = np.random.default_rng(20260905)
    for _ in range(40):
        n = int(rng.integers(1, 7))
        a = np.asfortranarray(rng.uniform(-10.0, 10.0, size=(n, n)))
        b = np.asfortranarray(rng.uniform(-10.0, 10.0, size=(n, 1)))
        ipiv = np.zeros(n, dtype=np.int32)
        mine = (a.copy(order="F"), b.copy(order="F"), ipiv.copy())
        fortran.w_dgesv(n, a, ipiv, b, np.array(0, dtype=np.int32))
        python["dgesv"](n, 1, mine[0], n, mine[2], mine[1], n, 0)
        assert a.tobytes() == mine[0].tobytes()
        assert b.tobytes() == mine[1].tobytes()
        assert (ipiv == mine[2]).all()

    for _ in range(40):
        n = int(rng.integers(2, 10))
        kl, ku = int(rng.integers(0, 3)), int(rng.integers(0, 3))
        ldab = 2 * kl + ku + 1
        ab = np.asfortranarray(rng.uniform(-10.0, 10.0, size=(ldab, n)))
        b = np.asfortranarray(rng.uniform(-10.0, 10.0, size=(n, 1)))
        ipiv = np.zeros(n, dtype=np.int32)
        mine = (ab.copy(order="F"), b.copy(order="F"), ipiv.copy())
        fortran.w_dgbsv(kl, ku, ab, ipiv, b, np.array(0, dtype=np.int32))
        python["dgbsv"](n, kl, ku, 1, mine[0], ldab, mine[2], mine[1], n, 0)
        assert ab.tobytes() == mine[0].tobytes(), "the factors, over the band pivoting fills"
        assert b.tobytes() == mine[1].tobytes()
        assert (ipiv == mine[2]).all()


# --- a call with no interface anywhere ---------------------------------------


def test_a_bare_call_binds_against_lapacks_argument_list() -> None:
    """ELM's BandDiagonalMod writes ``call dgbsv(...)`` with the name declared
    nowhere. The record such a call binds against is LAPACK's argument list
    with no intents -- passed whole, updated in place, like a call through an
    interface block without them -- and ``WRITTEN`` says which actuals the
    JAX runtime hands back instead."""
    record = references.interface("DGBSV")
    assert record["kind"] == "subroutine"
    assert [a["name"] for a in record["args"]] == [
        "n", "kl", "ku", "nrhs", "ab", "ldab", "ipiv", "b", "ldb", "info",
    ]  # fmt: skip
    assert {a["intent"] for a in record["args"]} == {"UNKNOWN"}
    assert references.WRITTEN["dgbsv"] == ("ab", "ipiv", "b")
    assert [a["name"] for a in references.interface("dgesv")["args"]][:3] == ["n", "nrhs", "a"]
    with pytest.raises(KeyError):
        references.interface("dsyevd")


def test_a_rank_one_right_hand_side_is_the_one_column_of_the_contract() -> None:
    """``result(n)`` handed to ``b(ldb,*)`` with ``nrhs = 1`` is Fortran
    sequence association for the one column of a ``(LDB,1)`` array. The Python
    reads it as that column, through a view, so the solution lands in the
    caller's array; with more right-hand sides than one it stays the rank-2
    contract and a rank-1 actual is refused."""
    side = _python_side()
    ab = np.asfortranarray(
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 1.0], [4.0, 4.0, 4.0], [1.0, 1.0, 0.0]])
    )
    rhs = np.array([1.0, 2.0, 3.0])
    ipiv = np.zeros(3, dtype=np.int32)
    ab_col, b_col = ab.copy(order="F"), np.asfortranarray(rhs.copy().reshape(3, 1))
    side["dgbsv"](3, 1, 1, 1, ab_col, 4, ipiv.copy(), b_col, 3, 0)
    ab_vec, b_vec = ab.copy(order="F"), rhs.copy()
    side["dgbsv"](3, 1, 1, 1, ab_vec, 4, ipiv.copy(), b_vec, 3, 0)
    assert b_vec.tolist() == b_col[:, 0].tolist()
    assert ab_vec.tolist() == ab_col.tolist()
    assert not np.array_equal(b_vec, rhs), "the solution was written into the caller's array"
    with pytest.raises(SystemExit):
        side["dgbsv"](3, 1, 1, 2, ab.copy(order="F"), 4, ipiv.copy(), rhs.copy(), 3, 0)
