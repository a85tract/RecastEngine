"""Stores the ASSOCIATE, SAVE and intent fixes still lost, found by review.

The fixes for FNP-D0015 (a store through an ASSOCIATE name), FNP-D0014 (a
SAVEd local kept between calls) and FNP-D0031 / FNP-D0047 (the intent an
un-INTENTed dummy is given) each read the body's stores through one list of
what a statement can store to. An independent review ran gfortran against
the translation and found the stores that list left out: one made through
an associate name, an internal WRITE, a function's ``intent(inout)`` actual,
and the elements a CYCLE or an EXIT skips. Every test translates a small
module, runs the emitted Python, and asserts what gfortran prints for the
same source.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
np = pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from tests.test_fnp_statements import body_of, rwset_verdict, translate  # noqa: E402

REVIEWED = """\
module rv_mod
  implicit none
  real(8) :: g = 1.0d0
contains
  subroutine noint(x)
    real(8) :: x
    associate (t => x)
      t = 5.0d0
    end associate
  end subroutine noint

  subroutine counted(c)
    integer, intent(out) :: c
    integer :: cnt = 0
    associate (t => cnt)
      t = t + 1
    end associate
    c = cnt
  end subroutine counted

  subroutine bumpg()
    associate (t => g)
      t = t + 1.0d0
    end associate
  end subroutine bumpg

  subroutine fmtw(s, n)
    character(len=8), intent(inout) :: s
    integer, intent(in) :: n
    associate (c => s)
      write (c, '(i4)') n
    end associate
  end subroutine fmtw

  subroutine logged(u, n)
    integer, intent(in) :: u, n
    associate (k => u)
      write (k, *) n
    end associate
  end subroutine logged

  subroutine cyc(a, n)
    integer :: n
    real(8) :: a(n)
    integer :: i
    do i = 1, n
      if (i == 2) cycle
      a(i) = 1.0d0
    end do
  end subroutine cyc

  subroutine ext(a, n)
    integer :: n
    real(8) :: a(n)
    integer :: i
    do i = 1, n
      if (i > 2) exit
      a(i) = 1.0d0
    end do
  end subroutine ext

  subroutine full(a, n)
    integer :: n
    real(8) :: a(n)
    integer :: i
    do i = 1, n
      a(i) = 1.0d0
    end do
  end subroutine full

  integer function incf(j)
    integer, intent(inout) :: j
    j = j + 1
    incf = j
  end function incf

  subroutine viaf(k)
    integer, intent(out) :: k
    integer :: cnt = 0
    integer :: z
    z = incf(cnt)
    k = cnt
  end subroutine viaf
end module rv_mod
"""


@pytest.fixture(scope="module")
def reviewed(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("reviewed"), REVIEWED, "rv_mod")


def test_a_dummy_written_only_through_an_associate_name_is_handed_back(reviewed: Any) -> None:
    """``associate (t => x); t = 5`` stores to ``x``. Counted as a store to
    ``t``, the un-INTENTed ``x`` had none and stayed UNKNOWN: passed in,
    never returned, and the caller kept 1.0 where gfortran gives 5.0."""
    module, text, _ = reviewed
    assert module.noint(1.0) == 5.0
    assert "return x" in body_of(text, "noint")


def test_a_saved_local_changed_through_an_associate_name_is_kept(reviewed: Any) -> None:
    """gfortran counts 1, 2; with ``cnt`` read as never changed, its
    initializer ran at every entry and the translation counted 1, 1."""
    module, _, _ = reviewed
    assert [module.counted(), module.counted()] == [1, 2]


def test_a_module_variable_written_through_an_associate_name_is_global(reviewed: Any) -> None:
    """The store is spelled ``g = (g + 1.0)``; without ``g`` on the ``global``
    line that made ``g`` a local and raised ``UnboundLocalError``."""
    module, text, _ = reviewed
    module.bumpg()
    assert module.g == 2.0
    assert "global g" in body_of(text, "bumpg")


def test_an_internal_write_through_an_associate_name_fills_its_variable(reviewed: Any) -> None:
    """gfortran: ``[  42    ]``. The WRITE to ``c`` was a log that stored
    nothing, and the caller kept ``xxxxxxxx``."""
    module, text, _ = reviewed
    assert module.fmtw("xxxxxxxx", 42) == "  42    "
    assert "log" not in body_of(text, "fmtw")


def test_a_write_to_a_unit_number_through_an_associate_name_is_still_a_log(
    reviewed: Any,
) -> None:
    _, text, candidate = reviewed
    assert not any(entry.startswith("logged/") for entry in candidate.deferred)
    assert "log" in body_of(text, "logged")


def test_a_loop_that_cycles_or_exits_leaves_the_callers_elements(reviewed: Any) -> None:
    """The array was inferred ``intent(out)``, so the caller's ``7 7 7 7``
    was never passed and the skipped elements came back as zeros. gfortran:
    ``1 7 1 1`` and ``1 1 7 7``."""
    module, _, _ = reviewed
    assert list(module.cyc(np.full(4, 7.0), 4)) == [1.0, 7.0, 1.0, 1.0]
    assert list(module.ext(np.full(4, 7.0), 4)) == [1.0, 1.0, 7.0, 7.0]


def test_a_loop_over_every_element_is_still_out(reviewed: Any) -> None:
    module, text, _ = reviewed
    assert "def full(n):" in text
    assert list(module.full(3)) == [1.0, 1.0, 1.0]


def test_a_saved_local_changed_through_a_function_argument_is_kept(reviewed: Any) -> None:
    """``z = incf(cnt)`` changes ``cnt`` through an ``intent(inout)`` dummy.
    gfortran counts 1, 2; the translation re-initialized ``cnt`` and
    counted 1, 1."""
    module, _, _ = reviewed
    assert [module.viaf(), module.viaf()] == [1, 2]


def test_the_read_write_gate_agrees_with_every_reviewed_block(
    reviewed: Any, tmp_path: Path
) -> None:
    _, _, candidate = reviewed
    verdict = rwset_verdict(candidate, tmp_path)
    assert not verdict.metrics["failures"], verdict.metrics["failures"]
