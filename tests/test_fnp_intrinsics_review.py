"""The intrinsics fixes of the RecastEngine-Pro-Lean audit, held to what
gfortran computes where an independent review ran it against them.

Each test is a case the first round of fixes (``test_fnp_intrinsics``) left
wrong or open: a construct translated into Python that computes a different
number from the compiled Fortran, or that fails where the Fortran runs. The
expected values are gfortran's (``-O0 -fcheck=all``), not the runtime's.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

import numpy as np

from recast.transform.numpy import runtime
from tests.test_fnp_intrinsics import translate

# --- FNP-D0007/8/9/27: an INTEGER kind the frontend could not resolve ---------

UNRESOLVED = """\
module unresolved_mod
  implicit none
  integer, parameter :: ik = kind(1_8)
  integer, parameter :: i4 = selected_int_kind(9)
contains
  subroutine wide(a, r, h, b, k, d)
    integer(ik), intent(in) :: a
    integer(8), intent(out) :: r, h
    integer, intent(out) :: b, k, d
    r = ishft(a, 40)
    h = huge(a)
    b = bit_size(a)
    k = kind(a)
    d = digits(a)
  end subroutine wide
  subroutine default(h, m, b)
    integer, intent(out) :: h, m, b
    integer :: n
    integer(i4) :: n4
    n = 1
    n4 = 1
    h = huge(n)
    m = huge(n4)
    b = bit_size(n)
  end subroutine default
  integer function top(x)
    integer, intent(in) :: x
    top = x
    top = huge(top)
  end function top
end module unresolved_mod
"""


def test_an_unresolved_integer_kind_is_read_from_its_spelling(tmp_path: Path) -> None:
    """``integer(kind(1_8))`` reaches the emitter as ``int32``, the dtype the
    frontend gives every INTEGER kind it cannot resolve. Taken at its word,
    ``ishft(a, 40)`` of 1 was shifted within 32 bits to 0, and HUGE,
    BIT_SIZE, KIND and DIGITS answered for a default INTEGER; gfortran says
    1099511627776, 9223372036854775807, 64, 8 and 63."""
    module, text, deferred = translate(tmp_path, UNRESOLVED)
    assert not deferred, deferred
    assert "_f_ishft(a, I_40, 64)" in text
    assert module.wide(np.int64(1)) == (2**40, 2**63 - 1, 64, 8, 63)
    assert module.wide(1) == (2**40, 2**63 - 1, 64, 8, 63)


def test_a_default_integer_keeps_its_answers(tmp_path: Path) -> None:
    """No kind, a kind parameter that names the 4-byte kind, and a function
    result: the default INTEGER, as before."""
    module, text, deferred = translate(tmp_path, UNRESOLVED)
    assert not deferred, deferred
    assert "h = _f_huge(n, 'int32')" in text
    assert module.default() == (2**31 - 1, 2**31 - 1, 32)
    assert module.top(3) == 2**31 - 1


@pytest.mark.parametrize("inquiry", ["huge", "digits", "kind", "bit_size", "range"])
def test_a_kind_neither_four_nor_eight_is_refused(tmp_path: Path, inquiry: str) -> None:
    """``integer(2) :: k2`` is ``int32`` to the frontend as well, and
    ``huge(k2)`` answered 2147483647 where gfortran says 32767 (DIGITS 15,
    KIND 2, BIT_SIZE 16, RANGE 4). No dtype here holds a 2-byte INTEGER."""
    source = f"""\
module short_mod
  implicit none
contains
  subroutine short(h)
    integer, intent(out) :: h
    integer(2) :: k2
    k2 = 1
    h = {inquiry}(k2)
  end subroutine short
end module short_mod
"""
    _, _, deferred = translate(tmp_path, source)
    assert len(deferred) == 1 and f"{inquiry}(k2)" in deferred[0]


def test_selected_int_kind_is_the_kind_gfortran_selects(tmp_path: Path) -> None:
    """``selected_int_kind(4)`` is gfortran's 2-byte kind; read as "4 unless
    it needs 8", ``int(x, i2)`` was emitted as a default conversion where
    ``int(x, 2)`` is refused. ``selected_int_kind(9)`` stays the default
    kind and ``selected_int_kind(18)`` the 8-byte one; ``huge(1_i2)`` asks
    about the 2-byte kind too."""
    source = """\
module sik_mod
  implicit none
  integer, parameter :: i2 = selected_int_kind(4)
  integer, parameter :: i4 = selected_int_kind(9)
  integer, parameter :: i8 = selected_int_kind(18)
contains
  subroutine named(x, k)
    real(8), intent(in) :: x
    integer, intent(out) :: k
    k = int(x, i2)
  end subroutine named
  subroutine literal(k)
    integer, intent(out) :: k
    k = huge(1_i2)
  end subroutine literal
  subroutine kept(x, k4, k8)
    real(8), intent(in) :: x
    integer, intent(out) :: k4
    integer(8), intent(out) :: k8
    k4 = int(x, i4)
    k8 = int(x, i8)
  end subroutine kept
end module sik_mod
"""
    module, _, deferred = translate(tmp_path, source)
    assert len(deferred) == 2, deferred
    assert "kind 2" in deferred[0] and "huge(1_i2)" in deferred[1]
    assert module.kept(3.0e10) == (-(2**31), 30000000000)


# --- FNP-D0004/5: a default-kind NINT is the low half of a 64-bit lround ------

NINT = """\
module nintwrap_mod
  implicit none
contains
  subroutine rounded(x, v, n4, n8, w)
    real(8), intent(in) :: x, v(3)
    integer, intent(out) :: n4, w(3)
    integer(8), intent(out) :: n8
    n4 = nint(x)
    n8 = nint(x, kind=8)
    w = nint(v)
  end subroutine rounded
end module nintwrap_mod
"""

NINT_CASES = [  # x, gfortran's nint(x), gfortran's nint(x, kind=8)
    (3.0e10, -64771072, 30000000000),
    (-3.0e10, 64771072, -30000000000),
    (4503599627370497.0, 1, 4503599627370497),
    (float("nan"), 0, -(2**63)),
    (1.0e19, 0, -(2**63)),
    (-2.5e9, 1794967296, -2500000000),
    (2.0**31, -(2**31), 2**31),
    (-(2.0**31) - 0.5, 2**31 - 1, -(2**31) - 1),
    (2.5, 3, 3),
    (-0.49999999999999994, 0, 0),
]


@pytest.mark.parametrize(("x", "default", "wide"), NINT_CASES)
def test_nint_wraps_as_gfortran_does(tmp_path: Path, x: float, default: int, wide: int) -> None:
    """gfortran computes a default-kind NINT as a 64-bit ``lround`` and
    keeps its low 32 bits. The conversion saturated at -2147483648 instead,
    for a NaN and for every value past the int32 range: ``nint(3.0d10)`` is
    -64771072 and ``nint`` of a NaN 0. The 8-byte kind was right already,
    and a REAL array rounds element by element the same way."""
    module, _, deferred = translate(tmp_path, NINT)
    assert not deferred
    n4, n8, w = module.rounded(x, np.array([x, 0.5, x]))
    assert (n4, n8) == (default, wide)
    assert list(w) == [default, 1, default]


# --- FNP-D0005: FLOOR and CEILING keep their KIND over an array ---------------

ROUNDING = """\
module floor8_mod
  implicit none
contains
  subroutine rounding(x, f, c, f4, g, h)
    real(8), intent(in) :: x(2)
    integer(8), intent(out) :: f(2), c(2), g, h
    integer, intent(out) :: f4(2)
    f = floor(x, kind=8)
    c = ceiling(x, 8)
    f4 = floor(x / 1.0d4)
    g = floor(x(1), kind=8)
    h = ceiling(x(2), 8)
  end subroutine rounding
end module floor8_mod
"""


def test_floor_and_ceiling_of_an_array_keep_their_kind(tmp_path: Path) -> None:
    """``floor(x, kind=8)`` of ``[3d10, -3d10]`` is ``[30000000000,
    -30000000000]``; the array spelling dropped the KIND and converted into
    int32, the int32 edge for both. The default kind and the scalar
    spelling keep what they had."""
    module, text, deferred = translate(tmp_path, ROUNDING)
    assert not deferred
    assert "_f_vfloor(x, 8)" in text and "_f_vceil(x, 8)" in text
    assert "g = math.floor(x[0])" in text
    f, c, f4, g, h = module.rounding(np.array([3.0e10 + 0.5, -3.0e10 - 0.5]))
    assert list(f) == [30000000000, -30000000001]
    assert list(c) == [30000000001, -30000000000]
    assert list(f4) == [3000000, -3000001]
    assert (g, h) == (30000000000, -30000000000)


# --- FNP-D0036/9: the elementwise mapping over an empty or integer array ------

EMPTY = """\
module empty_mod
  implicit none
contains
  subroutine mapped(n, a, i, t, m, s)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n)
    integer, intent(in) :: i(n)
    real(8), intent(out) :: t(n)
    integer, intent(out) :: m(n), s(n)
    t = tan(a)
    m = mod(i, 3)
    s = ishft(i, 1)
  end subroutine mapped
end module empty_mod
"""


def test_a_mapped_intrinsic_of_a_zero_size_array_is_empty(tmp_path: Path) -> None:
    """``t = tan(a)`` with ``n = 0`` is nothing at all in gfortran; the
    mapping raised "cannot call vectorize on size 0 inputs unless otypes is
    set". Non-empty, the values are as before."""
    module, text, deferred = translate(tmp_path, EMPTY)
    assert not deferred
    assert "_f_ecall(math.tan, a)" in text
    t, m, s = module.mapped(0, np.zeros(0), np.zeros(0, dtype=np.int32))
    assert t.size == m.size == s.size == 0
    t, m, s = module.mapped(2, np.array([0.5, 1.0]), np.array([7, -(2**30)], dtype=np.int32))
    assert list(t) == [np.tan(0.5), np.tan(1.0)] and list(m) == [1, -1]
    assert list(s) == [14, -(2**31)]


def test_a_mapped_bit_intrinsic_stays_in_its_operands_width() -> None:
    """``np.vectorize`` hands the helpers Python ints, so a width the
    emitter did not pass could not be read off a dtype, and the result came
    back int64 for int32 operands."""
    i = np.array([1, 2**30], dtype=np.int32)
    shifted = runtime._f_ecall(runtime._f_ishft, i, 1)
    assert shifted.dtype == np.int32 and list(shifted) == [2, -(2**31)]
    given = runtime._f_ecall(runtime._f_ior, i, 1, 32)
    assert given.dtype == np.int32 and list(given) == [1, 2**30 + 1]
    wide = runtime._f_ecall(runtime._f_ishft, i.astype(np.int64), 33)
    assert wide.dtype == np.int64 and list(wide) == [2**33, -(2**63)]
