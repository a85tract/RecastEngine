"""An array's bounds, as they were when the array came into being.

Two defects the RecastEngine-Pro-Lean audit's review found in how the
translation spells an array's bounds: an ALLOCATE of a zero-sized axis, and
bounds re-read from variables the body has since redefined. Each case is
translated whole, the emitted module imported and run, and the value asserted
is the one gfortran (``-O0 -fcheck=all``) prints for the same program.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

import numpy as np

from tests.test_fnp_intrinsics import dataflow_agrees, translate

# --- FNP-D0050: an ALLOCATE whose upper bound is below its lower one ---------

EMPTY = """\
module zero_extent_mod
  implicit none
contains
  subroutine basic(n, x1, x2, lb, ub, sz, s)
    integer, intent(in) :: n
    real(8), intent(out) :: x1, x2, s
    integer, intent(out) :: lb, ub, sz
    real(8), allocatable :: a(:)
    integer :: i
    allocate(a(-n:n))
    do i = -n, n
      a(i) = 10d0 * i
    end do
    x1 = 0d0; x2 = 0d0
    if (n >= 0) then
      x1 = a(-n)
      x2 = a(0)
    end if
    lb = lbound(a, 1); ub = ubound(a, 1); sz = size(a); s = sum(a(-n:0))
    deallocate(a)
  end subroutine basic
  subroutine one(n, sz, ub)
    integer, intent(in) :: n
    integer, intent(out) :: sz, ub
    real(8), allocatable :: b(:), c(:)
    allocate(b(n), c(4))
    sz = size(b) + size(c); ub = ubound(b, 1)
    deallocate(b, c)
  end subroutine one
end module zero_extent_mod
"""


def test_an_allocate_below_its_lower_bound_is_empty(tmp_path: Path) -> None:
    """FNP-D0050 (rule FNP-R0160). ``allocate(a(-n:n))`` with n = -1 is a
    zero-sized ``a`` (F2018 9.7.1.2); gfortran prints 0 0 1 0 0 0.0 for
    ``basic(-1)`` -- LBOUND 1 and UBOUND 0 on the empty axis -- where the
    shape ``((n) - ((-n)) + 1,)``, -1, was a ValueError from NumPy."""
    module, text, deferred = translate(tmp_path, EMPTY)
    assert not deferred
    assert "np.zeros((max(0, (n) - ((-n)) + 1),)" in text
    assert module.basic(-1) == (0.0, 0.0, 1, 0, 0, 0.0)


def test_an_allocate_that_is_not_empty_keeps_its_values(tmp_path: Path) -> None:
    """``basic(2)``: gfortran prints -20.0 0.0 -2 2 5 -30.0."""
    module, _, _ = translate(tmp_path, EMPTY)
    assert module.basic(2) == (-20.0, 0.0, -2, 2, 5, -30.0)


def test_an_allocate_based_at_one_is_empty_below_one(tmp_path: Path) -> None:
    """``allocate(b(n))`` with n = -3 is empty too: gfortran's ``one(-3)``
    is 4 0. A literal extent cannot go negative and is written as it was."""
    module, text, _ = translate(tmp_path, EMPTY)
    assert "np.zeros((max(0, n),)" in text
    assert "np.zeros((I_4,)" in text
    assert module.one(-3) == (4, 0)
    assert module.one(3) == (7, 3)


# --- FNP-D0049: bounds over a variable the body redefines ---------------------

MOVED = """\
module moved_bounds_mod
  implicit none
contains
  subroutine redef(n0, x, lb, ub)
    integer, intent(in) :: n0
    real(8), intent(out) :: x
    integer, intent(out) :: lb, ub
    real(8), allocatable :: a(:)
    integer :: n, i
    n = n0
    allocate(a(-n:n))
    do i = -n, n
      a(i) = 10d0 * i
    end do
    n = n + 1
    x = a(0); lb = lbound(a, 1); ub = ubound(a, 1)
    deallocate(a)
  end subroutine redef
  subroutine d(n, a, x, lb, ub, s)
    integer, intent(inout) :: n
    real(8), intent(in) :: a(-n:n)
    real(8), intent(out) :: x, s
    integer, intent(out) :: lb, ub
    n = n + 1
    x = a(0); lb = lbound(a, 1); ub = ubound(a, 1); s = sum(a(-1:))
  end subroutine d
  subroutine auto(n, x, lb, ub)
    integer, intent(inout) :: n
    real(8), intent(out) :: x
    integer, intent(out) :: lb, ub
    real(8) :: w(-n:n)
    integer :: i
    do i = -n, n
      w(i) = 10d0 * i
    end do
    n = n + 1
    x = w(0); lb = lbound(w, 1); ub = ubound(w, 1)
  end subroutine auto
  subroutine twice(n0, x, lb, ub)
    integer, intent(in) :: n0
    real(8), intent(out) :: x
    integer, intent(out) :: lb, ub
    real(8), allocatable :: a(:)
    integer :: n, k, i
    x = 0d0
    n = n0
    do k = 1, 2
      allocate(a(-n:n))
      do i = -n, n
        a(i) = 10d0 * i + k
      end do
      n = n + 1
      x = x + a(1 - n + 1)
      lb = lbound(a, 1); ub = ubound(a, 1)
      deallocate(a)
    end do
  end subroutine twice
  subroutine kept(n, a, x, lb, ub)
    integer, intent(in) :: n
    real(8), intent(in) :: a(-n:n)
    real(8), intent(out) :: x
    integer, intent(out) :: lb, ub
    x = a(0); lb = lbound(a, 1); ub = ubound(a, 1)
  end subroutine kept
end module moved_bounds_mod
"""

TEN_I = np.arange(-20.0, 21.0, 10.0)  # a(i) = 10 i over -2:2


def test_an_allocated_bound_is_the_value_at_the_allocate(tmp_path: Path) -> None:
    """FNP-D0049 (rules FNP-R0040, FNP-R0043, FNP-R0049). ``allocate(a(-n:n))``
    with n = 2, then ``n = n + 1``: ``a`` is still ``a(-2:2)`` (F2018
    9.7.1.2), and gfortran's ``redef(2)`` is 0.0 -2 2. The shift re-read
    ``n`` -- ``a[(0) - (- n)]`` -- and answered 10.0 -3 1."""
    module, text, deferred = translate(tmp_path, MOVED)
    assert not deferred
    assert "_lb_a_1 = (-n)" in text
    assert "x = a[(0) - (_lb_a_1)]" in text
    assert module.redef(2) == (0.0, -2, 2)


def test_a_dummys_bound_is_its_value_on_entry(tmp_path: Path) -> None:
    """``a(-n:n)`` of an intent(inout) n, then ``n = n + 1``: the bounds
    stay -2:2 (F2018 10.1.11). gfortran's ``d`` over a(i) = 10 i is n = 3,
    x = 0.0, -2 2, and ``sum(a(-1:))`` = 20.0."""
    module, _, _ = translate(tmp_path, MOVED)
    assert module.d(2, TEN_I.copy()) == (3, 0.0, -2, 2, 20.0)


def test_an_automatic_locals_bound_is_its_value_on_entry(tmp_path: Path) -> None:
    """``real(8) :: w(-n:n)`` sized on entry: gfortran's ``auto(2)`` is
    n = 3, 0.0 -2 2."""
    module, _, _ = translate(tmp_path, MOVED)
    assert module.auto(2) == (3, 0.0, -2, 2)


def test_every_allocate_of_the_array_sets_its_bound_again(tmp_path: Path) -> None:
    """An ALLOCATE in a loop, its bound variable moving between passes:
    ``a(-2:2)`` then ``a(-3:3)``, reading ``a(-1)`` then ``a(-2)``.
    gfortran's ``twice(2)`` is -27.0 -3 3."""
    module, _, _ = translate(tmp_path, MOVED)
    assert module.twice(2) == (-27.0, -3, 3)


def test_the_held_bounds_pass_the_read_write_gate(tmp_path: Path) -> None:
    """The local a bound is held in is the translation's own: the source
    reads the bound's variables where the array comes into being, and a
    subscript over the bound reads none of them."""
    dataflow_agrees(tmp_path, MOVED)


def test_a_bound_nothing_redefines_is_spelled_as_it_was(tmp_path: Path) -> None:
    """``kept``: n is intent(in), so its bound text is its value, and the
    text is what it was before the fix. gfortran: 0.0 -2 2."""
    module, text, _ = translate(tmp_path, MOVED)
    kept = text[text.index("def kept(") :]
    assert "_lb_" not in kept
    assert "x = a[(0) - (- n)]" in kept
    assert "lb = _f_lbound(a, 1, (- n,))" in kept
    assert module.kept(2, TEN_I.copy()) == (0.0, -2, 2)


# --- FNP-D0050: an automatic array whose upper bound is below its lower one --

AUTOMATIC_EMPTY = """\
module auto_zero_mod
  implicit none
  type pt
    real(8) :: v
  end type pt
contains
  subroutine centred(n, lb, ub, sz, s)
    integer, intent(in) :: n
    integer, intent(out) :: lb, ub, sz
    real(8), intent(out) :: s
    real(8) :: w(-n:n)
    integer :: i
    do i = -n, n
      w(i) = 10d0 * i + 1d0
    end do
    lb = lbound(w, 1); ub = ubound(w, 1); sz = size(w); s = sum(w)
  end subroutine centred
  subroutine based(m, lb, ub, sz, sk)
    integer, intent(in) :: m
    integer, intent(out) :: lb, ub, sz, sk
    real(8) :: w(m)
    integer :: k(m, 3), c(3)
    w = 1d0; k = 2; c = 3
    lb = lbound(w, 1); ub = ubound(w, 1); sz = size(w)
    sk = size(k) + sum(k) + sum(c)
  end subroutine based
  subroutine sized(n, x, sz, ub, s)
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    integer, intent(out) :: sz, ub
    real(8), intent(out) :: s
    real(8) :: w(size(x)), v(n)
    w = x; v = 2d0 * x
    sz = size(w) + size(v); ub = ubound(v, 1); s = sum(w) + sum(v)
  end subroutine sized
  subroutine objects(n, sz)
    integer, intent(in) :: n
    integer, intent(out) :: sz
    type(pt) :: p(n)
    sz = size(p)
  end subroutine objects
end module auto_zero_mod
"""


def test_an_automatic_array_below_its_lower_bound_is_empty(tmp_path: Path) -> None:
    """FNP-D0050 (rule FNP-R0160 and the automatic-array prologue).
    ``real(8) :: w(-n:n)`` entered with n = -1 is a zero-sized ``w``
    (F2018 8.5.8.2); gfortran's ``centred(-1)`` is 1 0 0 0.0 -- LBOUND 1
    and UBOUND 0 on the empty axis -- where the prologue's
    ``np.zeros(((n) - (- n) + 1,), ...)``, -1, was a ValueError."""
    module, text, deferred = translate(tmp_path, AUTOMATIC_EMPTY)
    assert not deferred
    assert "w = np.zeros((max(0, (n) - (- n) + 1),), dtype=np.float64)" in text
    assert module.centred(-1) == (1, 0, 0, 0.0)
    assert module.centred(2) == (-2, 2, 5, 5.0)


def test_an_automatic_array_based_at_one_is_empty_below_one(tmp_path: Path) -> None:
    """``w(m)`` and ``k(m, 3)`` with m = -2 are empty: gfortran's
    ``based(-2)`` is 1 0 0 9, and ``based(2)`` 1 2 2 27. The literal
    extents, 3 on ``k``'s second axis and ``c(3)``, are written as they
    were."""
    module, text, _ = translate(tmp_path, AUTOMATIC_EMPTY)
    assert "w = np.zeros((max(0, m),), dtype=np.float64)" in text
    assert "k = np.zeros((max(0, m), I_3,), dtype=np.int32)" in text
    assert "c = np.zeros((I_3,), dtype=np.int32)" in text
    assert module.based(-2) == (1, 0, 0, 9)
    assert module.based(2) == (1, 2, 2, 27)


def test_an_automatic_array_sized_off_a_dummy_is_empty_with_it(tmp_path: Path) -> None:
    """``v(n)`` beside a dummy ``x(n)``, n = -2: both empty, and gfortran's
    ``sized(-2, x)`` is 0 0 0.0; ``sized(3, [1, 2, 3])`` is 6 3 18.0.
    ``w(size(x))`` cannot go negative and is written as it was."""
    module, text, _ = translate(tmp_path, AUTOMATIC_EMPTY)
    assert "w = np.zeros((np.size(x),), dtype=np.float64)" in text
    assert "v = np.zeros((max(0, n),), dtype=np.float64)" in text
    assert module.sized(-2, np.zeros(0)) == (0, 0, 0.0)
    assert module.sized(3, np.array([1.0, 2.0, 3.0])) == (6, 3, 18.0)


def test_an_automatic_array_of_a_derived_type_is_empty_below_one(tmp_path: Path) -> None:
    """``type(pt) :: p(n)``: gfortran's ``objects(-1)`` is 0, ``objects(2)``
    2."""
    module, _, _ = translate(tmp_path, AUTOMATIC_EMPTY)
    assert module.objects(-1) == 0
    assert module.objects(2) == 2
