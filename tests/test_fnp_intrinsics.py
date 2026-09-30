"""Intrinsics and arithmetic the formal audit refuted, held to Fortran's value.

Each test here is one defect of the RecastEngine-Pro-Lean audit (``FNP-Dxxxx``,
against the rule ``FNP-Rxxxx`` it refutes): a construct the translation
accepted and emitted as runnable Python that computes a different number from
the Fortran. They translate a small module through the real frontend and
Transform and run what comes out, because the emitted spelling is the claim
-- a runtime helper that is right on its own proves nothing if the emitter
never calls it with what it needs.
"""

from __future__ import annotations

import importlib
import itertools
import math
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

import numpy as np

from recast.fortran.frontend import FortranFrontend
from recast.transform.numpy import runtime
from recast.transform.numpy.translate import NumpyTranslation

_serial = itertools.count()


def candidate_for(tmp_path: Path, source: str, *siblings: str) -> tuple[Any, Any, Path]:
    """``source`` (one module) through the real frontend and Transform, with
    ``siblings`` beside it in the tree: the unit, the candidate, and the
    directory it was written to."""
    name = source.split()[1].lower()
    tree = tmp_path / f"src{next(_serial)}"
    tree.mkdir()
    for text in (source, *siblings):
        (tree / f"{text.split()[1].lower()}.f90").write_text(text)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tree) if u.uid == f"fortran:{name}")
    candidate = NumpyTranslation().apply(unit, frontend.analyze(unit, tree), {"root": tree})
    out = tree / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    return unit, candidate, out


def translate(tmp_path: Path, source: str, *siblings: str) -> tuple[ModuleType, str, list[str]]:
    """``source`` translated and imported, with its text and deferred list:
    what the Transform actually hands on."""
    name = source.split()[1].lower()
    _, candidate, out = candidate_for(tmp_path, source, *siblings)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module(f"{name}_numpy")
    finally:
        sys.path.remove(str(out))
        sys.modules.pop(f"{name}_numpy", None)
    return module, (out / f"{name}_numpy.py").read_text(), list(candidate.deferred)


def dataflow_agrees(tmp_path: Path, source: str) -> None:
    """The static read/write gate over the translation of ``source``: a fix
    that reads a name the source does not -- a lower bound, say -- stops a
    unit at the gate however right its numbers are."""
    from recast.executors.local import LocalExecutor
    from recast.model import Confidence
    from recast.verify.rwset import ReadWriteSetVerifier

    unit, candidate, _ = candidate_for(tmp_path, source)
    verdict = ReadWriteSetVerifier().check(unit, candidate, Path("."), LocalExecutor(), {})
    assert verdict.confidence is Confidence.SAMPLED, verdict.detail


def same(a: Any, b: Any) -> bool:
    """Equal, and equal in sign for zeros and in NaN-ness: ``anint(-0.3)`` is
    ``-0.0``, which ``==`` cannot tell from ``0.0``."""
    if isinstance(b, float) and math.isnan(b):
        return math.isnan(a)
    return bool(a == b) and math.copysign(1.0, a) == math.copysign(1.0, b)


# --- FNP-D0004: ANINT rounds halves away from zero ----------------------------

ANINT = """\
module anint_mod
  implicit none
contains
  subroutine rounded(x, v, r, w)
    real(8), intent(in) :: x, v(4)
    real(8), intent(out) :: r, w(4)
    r = anint(x)
    w = anint(v)
  end subroutine rounded
end module anint_mod
"""

ANINT_CASES = [
    (2.5, 3.0),
    (-2.5, -3.0),
    (0.5, 1.0),
    (-0.5, -1.0),
    (1.5, 2.0),
    (0.49999999999999994, 0.0),
    (-0.3, -0.0),
    (2.0**52 + 1, 2.0**52 + 1),
    (1e300, 1e300),
    (math.inf, math.inf),
    (math.nan, math.nan),
]


@pytest.mark.parametrize(("x", "expected"), ANINT_CASES)
def test_anint_is_round_not_round_half_even(x: float, expected: float) -> None:
    """``np.round`` rounds a half to even; C's ``round``, which gfortran
    calls for ANINT, rounds it away from zero. ``floor(x + 0.5)`` is wrong
    too, just less often: the addition rounds 0.49999999999999994 up."""
    assert same(runtime._f_anint(x), expected)
    assert same(runtime._f_anint(np.array([x]))[0], expected)


def test_anint_is_emitted_as_fortran_rounds_it(tmp_path: Path) -> None:
    module, text, deferred = translate(tmp_path, ANINT)
    assert not deferred
    assert "r = _f_anint(x)" in text and "w[...] = _f_anint(v)" in text
    r, w = module.rounded(2.5, np.array([2.5, -2.5, 0.5, -0.49999999999999994]))
    assert r == 3.0
    assert list(w) == [3.0, -3.0, 1.0, -0.0]


def test_nint_does_not_round_a_value_below_a_half_up() -> None:
    """The same ``floor(x + 0.5)`` in NINT: 0.49999999999999994 is below a
    half, and gfortran's ``lround`` answers 0."""
    assert runtime._f_nint(0.49999999999999994) == 0
    assert runtime._f_nint(-0.49999999999999994) == 0
    assert runtime._f_nint(2.5) == 3 and runtime._f_nint(-2.5) == -3


# --- FNP-D0005: INT and NINT keep their KIND ----------------------------------

INT_KIND = """\
module intkind_mod
  implicit none
  integer, parameter :: i8 = selected_int_kind(18)
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine widths(x, k8, k4, kn, kp, kr, ka)
    real(8), intent(in) :: x
    integer(8), intent(out) :: k8, kn, kp, kr, ka
    integer, intent(out) :: k4
    k8 = int(x, kind=8)
    k4 = int(x)
    kn = nint(x, 8)
    kp = int(x, i8)
    kr = int(x, kind=r8)
    ka = x
  end subroutine widths
end module intkind_mod
"""


def test_a_64_bit_int_holds_what_a_default_one_cannot(tmp_path: Path) -> None:
    """``int(x, kind=8)`` of 3.0d10 is 30000000000; the kind was dropped and
    the default conversion saturated it at the int32 edge. A REAL stored
    into an ``integer(8)`` converts into that kind as well."""
    module, text, deferred = translate(tmp_path, INT_KIND)
    assert not deferred
    assert "k4 = _f_int(x)\n" in text  # the default kind keeps its spelling
    k8, k4, kn, kp, kr, ka = module.widths(3.0e10)
    assert (k8, kn, kp, kr, ka) == (30000000000,) * 5
    assert k4 == -(2**31)  # the hardware conversion's answer, as before
    assert module.widths(2.5)[2] == 3  # nint rounds, whatever the kind


def test_a_kind_nothing_here_can_evaluate_is_refused(tmp_path: Path) -> None:
    source = INT_KIND.replace("k4 = int(x)", "k4 = int(x, kind=2)")
    _, _, deferred = translate(tmp_path, source)
    assert len(deferred) == 1 and "kind 2" in deferred[0]


# --- FNP-D0006: LBOUND and UBOUND report the declared bounds -------------------

BOUNDS = """\
module bounds_mod
  implicit none
contains
  subroutine bounds(a, c, n, lo, hi, cl, cu, sz, lov, hiv, w, total)
    integer, intent(in) :: n
    real(8), intent(in) :: a(0:5), c(-1:n, 2:3)
    integer, intent(out) :: lo, hi, cl, cu, sz, lov(2), hiv(2)
    real(8), intent(out) :: w(0:n), total
    integer :: i
    lo = lbound(a, 1)
    hi = ubound(a, 1)
    cl = lbound(c, dim=2)
    cu = ubound(c, 1)
    sz = size(a)
    lov = lbound(c)
    hiv = ubound(c)
    total = 0.0d0
    do i = lbound(w, 1), ubound(w, 1)
      w(i) = i
      total = total + w(i)
    end do
  end subroutine bounds
end module bounds_mod
"""


def test_bounds_are_the_declared_ones_not_one_and_the_extent(tmp_path: Path) -> None:
    """``a(0:5)``: Fortran says LBOUND 0 and UBOUND 5, the translation said 1
    and 6. A loop over ``lbound:ubound`` has to index what it did in the
    source, so the bounds are the ones the subscripts are shifted by."""
    module, text, deferred = translate(tmp_path, BOUNDS)
    assert not deferred
    assert "lo = _f_lbound(a, 1, (0,))" in text
    lo, hi, cl, cu, sz, lov, hiv, w, total = module.bounds(np.zeros(6), np.zeros((5, 2)), 3)
    assert (lo, hi, cl, cu, sz) == (0, 5, 2, 3, 6)
    assert list(lov) == [-1, 2] and list(hiv) == [3, 3]
    assert list(w) == [0.0, 1.0, 2.0, 3.0] and total == 6.0
    dataflow_agrees(tmp_path, BOUNDS.replace("-1:n", "-n:n"))  # a bound that reads a name


def test_a_bound_based_at_one_keeps_its_spelling(tmp_path: Path) -> None:
    source = """\
module plain_mod
  implicit none
contains
  subroutine plain(a, m, lo, hi, both)
    real(8), intent(in) :: a(:), m(3, 4)
    integer, intent(out) :: lo, hi, both(2)
    lo = lbound(a, 1)
    hi = ubound(a, 1)
    both = ubound(m)
  end subroutine plain
end module plain_mod
"""
    module, text, deferred = translate(tmp_path, source)
    assert not deferred
    assert "lo = _f_lbound(a, 1)" in text and "hi = np.size(a, axis=(1) - 1)" in text
    lo, hi, both = module.plain(np.zeros(7), np.zeros((3, 4)))
    # UBOUND without DIM is a vector of bounds; it used to be the element count.
    assert (lo, hi, list(both)) == (1, 7, [3, 4])


def test_an_axis_of_zero_extent_has_bounds_one_and_zero() -> None:
    empty = np.zeros((0, 3))
    assert list(runtime._f_lbound(empty, None, (5, 2))) == [1, 2]
    assert list(runtime._f_ubound(empty, None, (5, 2))) == [0, 4]


# --- FNP-D0007, FNP-D0008, FNP-D0039: kind inquiries read the declaration -----

INQUIRIES = """\
module inquire_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine inquiries(x4, x8, k8, kx4, kx8, kk8, hk8, hk4, px4, px8, ex4, tx8, hx, hl)
    real(4), intent(in) :: x4
    real(r8), intent(in) :: x8
    integer(8), intent(in) :: k8
    integer, intent(out) :: kx4, kx8, kk8, px4, px8
    integer(8), intent(out) :: hk8
    integer, intent(out) :: hk4
    real(8), intent(out) :: ex4, tx8, hx, hl
    real(8) :: x
    x = 0
    kx4 = kind(x4)
    kx8 = kind(x8)
    kk8 = kind(k8)
    hk8 = huge(k8)
    hk4 = huge(0)
    px4 = precision(x4)
    px8 = precision(x8)
    ex4 = epsilon(1.0)
    tx8 = tiny(x8)
    hx = huge(x)
    hl = huge(1.0_r8)
  end subroutine inquiries
end module inquire_mod
"""


def test_kind_inquiries_answer_for_the_declared_kind(tmp_path: Path) -> None:
    """The runtime guessed the kind from the Python type: ``kind(x4)`` was 8,
    ``huge(k8)`` the int32 maximum, and a ``real(8)`` assigned the integer 0
    -- a Python ``0`` -- answered HUGE with 2147483647 (the audit's
    ``hugeint`` probe). The arguments are passed as the drivers of the audit
    passed them: ``k8`` a NumPy int64, the reals plain floats."""
    module, text, deferred = translate(tmp_path, INQUIRIES)
    assert not deferred
    assert "hk8 = _f_huge(k8, 'int64')" in text
    kx4, kx8, kk8, hk8, hk4, px4, px8, ex4, tx8, hx, hl = module.inquiries(1.0, 1.0, 1)
    assert (kx4, kx8, kk8) == (4, 8, 8)
    assert hk8 == 2**63 - 1 and hk4 == 2**31 - 1
    assert (px4, px8) == (6, 15)
    assert ex4 == 2.0**-23 and tx8 == 2.0**-1022
    assert hx == hl == np.finfo(np.float64).max


def test_a_companion_parameter_answers_for_its_declared_kind(tmp_path: Path) -> None:
    """SLSQP's ``huge(one)``, ``one`` a parameter of ``slsqp_support``."""
    kinds = """\
module kinds_mod
  use, intrinsic :: iso_fortran_env, only: real64
  implicit none
  integer, parameter :: wp = real64
end module kinds_mod
"""
    support = """\
module support_mod
  use kinds_mod, only: wp
  implicit none
  real(wp), parameter :: one = 1.0_wp
end module support_mod
"""
    user = """\
module user_mod
  use kinds_mod, only: wp
  use support_mod, only: one
  implicit none
contains
  subroutine bound(h, e)
    real(8), intent(out) :: h, e
    h = huge(one)
    e = epsilon(one)
  end subroutine bound
end module user_mod
"""
    _, text, deferred = translate(tmp_path, user, support, kinds)
    assert not deferred, deferred
    assert "'float64'" in text


def test_a_kind_inquiry_on_an_unresolved_kind_is_refused(tmp_path: Path) -> None:
    source = """\
module unknown_mod
  use elsewhere, only: wp
  implicit none
contains
  subroutine eps(x, e)
    real(wp), intent(in) :: x
    real(wp), intent(out) :: e
    e = epsilon(x)
  end subroutine eps
end module unknown_mod
"""
    _, _, deferred = translate(tmp_path, source)
    assert len(deferred) == 1 and "epsilon" in deferred[0]


# --- FNP-D0027: DIGITS and the other model inquiries, and no subscript --------

MODEL = """\
module model_mod
  implicit none
contains
  subroutine model(x, k, d8, dk, r8, rk, mx, mn, bs)
    real(8), intent(in) :: x
    integer(8), intent(in) :: k
    integer, intent(out) :: d8, dk, r8, rk, mx, mn, bs
    d8 = digits(x)
    dk = digits(k)
    r8 = range(x)
    rk = range(k)
    mx = maxexponent(x)
    mn = minexponent(x)
    bs = bit_size(k)
  end subroutine model
end module model_mod
"""


def test_digits_and_the_model_inquiries_are_spelled(tmp_path: Path) -> None:
    """``digits(x)`` had no spelling and fell through to the subscript
    fallback: ``n = digits[x - 1]``, a NameError where Fortran says 53."""
    module, text, deferred = translate(tmp_path, MODEL)
    assert not deferred
    assert "digits[" not in text
    assert module.model(1.0, 1) == (53, 63, 307, 18, 1024, -1021, 64)
    dataflow_agrees(tmp_path, MODEL)


def test_a_standard_intrinsic_without_a_spelling_is_refused(tmp_path: Path) -> None:
    source = """\
module frac_mod
  implicit none
contains
  subroutine frac(x, f)
    real(8), intent(in) :: x
    real(8), intent(out) :: f
    f = fraction(x)
  end subroutine frac
end module frac_mod
"""
    _, text, deferred = translate(tmp_path, source)
    assert len(deferred) == 1 and "intrinsic 'fraction'" in deferred[0]
    assert "fraction[" not in text


def test_a_name_a_bare_use_may_import_is_still_read_as_an_array(tmp_path: Path) -> None:
    """``use grid_mod`` may bring an array ``scale`` in, and an integer
    subscript of it is what the fallback exists for."""
    source = """\
module scaled_mod
  use grid_mod
  implicit none
contains
  subroutine scaled(i, v)
    integer, intent(in) :: i
    real(8), intent(out) :: v
    v = scale(i)
  end subroutine scaled
end module scaled_mod
"""
    _, candidate, out = candidate_for(tmp_path, source)
    assert not candidate.deferred
    assert "scale[i - 1]" in (out / "scaled_mod_numpy.py").read_text()


# --- FNP-D0009: bit intrinsics work within the declared kind -------------------

BITS = """\
module bits_mod
  implicit none
contains
  subroutine bits(i, k, s1, s2, s3, s4, s5)
    integer, intent(in) :: i
    integer(8), intent(in) :: k
    integer, intent(out) :: s1, s2, s4, s5
    integer(8), intent(out) :: s3
    s1 = ishft(1, 31)
    s2 = ishft(i, 31)
    s3 = ishft(k, 40)
    s4 = ior(i, ishft(1, 31))
    s5 = ieor(-1, ishft(i, shift=30))
  end subroutine bits
end module bits_mod
"""


def test_a_shift_stays_in_the_operands_width(tmp_path: Path) -> None:
    """``ishft(1, 31)`` is -2147483648 in a default INTEGER; a literal ``1``
    has no NumPy dtype for the runtime to read the width off, and the shift
    went unbounded. So did a Python int ``i`` a caller passed."""
    module, text, deferred = translate(tmp_path, BITS)
    assert not deferred
    assert "_f_ishft(1, I_31, 32)" in text
    assert module.bits(1, 1) == (-(2**31), -(2**31), 2**40, -(2**31) + 1, -(2**30) - 1)


# --- FNP-D0012: INTEGER ** a negative exponent is an integer -------------------

POWERS = """\
module ipow_mod
  implicit none
contains
  subroutine powers(j, n, x, v, k, m, p, q, r, w)
    integer, intent(in) :: j, n, v(3)
    real(8), intent(in) :: x
    integer, intent(out) :: k, m, p, q, w(3)
    real(8), intent(out) :: r
    k = 2**(-1)
    m = j**(-1)
    p = j**n
    q = j**3
    r = x**(-2)
    w = v**(-1)
  end subroutine powers
end module ipow_mod
"""


def test_an_integer_to_a_negative_power_is_integer_division(tmp_path: Path) -> None:
    """Fortran's ``2**(-1)`` is zero, and ``j**(-1)`` is zero but for j = +-1;
    Python said 0.5. A REAL base and a non-negative literal are unchanged."""
    module, text, deferred = translate(tmp_path, POWERS)
    assert not deferred
    assert "q = (j ** 3)" in text and "r = (x ** -2)" in text
    k, m, p, q, r, w = module.powers(2, -2, 2.0, np.array([1, -1, 2], dtype=np.int32))
    assert (k, m, p, q, r) == (0, 0, 0, 8, 0.25)
    assert list(w) == [1, -1, 0] and w.dtype == np.int32
    assert module.powers(-1, -3, 2.0, np.array([1, 1, 1], dtype=np.int32))[2] == -1
    assert module.powers(3, 2, 2.0, np.array([1, 1, 1], dtype=np.int32))[2] == 9


def test_zero_to_a_negative_power_traps_as_the_reference_does() -> None:
    with pytest.raises(ZeroDivisionError):
        runtime._f_ipow(0, -1)
    with pytest.raises(ZeroDivisionError):
        runtime._f_ipow(np.array([1, 0]), -1)


# --- FNP-D0036 (and FNP-D0035): integer division and scalar-only intrinsics
#     over arrays ------------------------------------------------------------

OVER_ARRAYS = """\
module arrays_mod
  implicit none
contains
  subroutine over(a, b, x, c, t, m, k, n, s)
    integer, intent(in) :: a(3), b(3)
    real(8), intent(in) :: x(3)
    integer, intent(out) :: c(3), m(3), k(3), n(3), s(3)
    real(8), intent(out) :: t(3)
    c = a / b
    t = tan(x)
    m = mod(a, b)
    k = int(x)
    n = nint(x)
    s = ishft(a, 1)
  end subroutine over
end module arrays_mod
"""


def test_integer_division_and_scalar_spellings_take_arrays(tmp_path: Path) -> None:
    """``c = a / b`` over integer arrays was ``_f_int_div(a, b)``, which
    called ``int`` on an array; ``tan``, ``mod`` and ``ishft`` had only
    scalar spellings; and ``nint`` of a REAL array was ``np.int32`` of it,
    which truncates -- the one of these that did not raise."""
    module, _, deferred = translate(tmp_path, OVER_ARRAYS)
    assert not deferred
    a = np.array([7, -7, 7], dtype=np.int32)
    b = np.array([2, 2, -2], dtype=np.int32)
    x = np.array([0.5, 2.5, -2.5])
    c, t, m, k, n, s = module.over(a, b, x)
    assert list(c) == [3, -3, -3]
    assert list(m) == [1, -1, 1]
    assert list(k) == [0, 2, -2]
    assert list(n) == [1, 3, -3]
    assert list(s) == [14, -14, 14]
    assert list(t) == [math.tan(v) for v in x]


def test_integer_division_is_exact_past_two_to_the_53() -> None:
    """``int(a / b)`` rounds the quotient through a double."""
    assert runtime._f_int_div(2**53 + 1, 1) == 2**53 + 1
    assert runtime._f_int_div(-(2**62) - 1, 3) == -((2**62 + 1) // 3)
    assert type(runtime._f_int_div(np.int32(7), np.int32(2))) is int


def test_a_substring_argument_keeps_the_scalar_spelling(tmp_path: Path) -> None:
    """``s(i:i)`` ranks as a section, and is a scalar string."""
    source = """\
module parity_mod
  implicit none
contains
  subroutine parity(s, i, p)
    character(len=4), intent(in) :: s
    integer, intent(in) :: i
    integer, intent(out) :: p
    p = mod(ichar(s(i:i)), 2)
  end subroutine parity
end module parity_mod
"""
    module, text, deferred = translate(tmp_path, source)
    assert not deferred and "_f_ecall" not in text[text.index("def parity") :]
    assert module.parity("abcd", 1) == 1


# --- FNP-D0040: DOT_PRODUCT of INTEGER arrays is an INTEGER --------------------

DOT = """\
module dotint_mod
  implicit none
contains
  subroutine idot(n, m, a, b, c, d, v)
    integer, intent(in) :: n, m
    integer, intent(in) :: a(n), b(n)
    real(8), intent(in) :: c(m)
    integer, intent(out) :: d
    real(8), intent(out) :: v
    d = dot_product(a, b)
    v = c(dot_product(a, b))
  end subroutine idot
end module dotint_mod
"""


def test_an_integer_dot_product_can_subscript(tmp_path: Path) -> None:
    """The accumulator was seeded with ``0.0``, so ``c(dot_product(a, b))``
    subscripted with a float and NumPy raised (the audit's dotint probe)."""
    module, _, deferred = translate(tmp_path, DOT)
    assert not deferred
    a = np.array([1, 2], dtype=np.int32)
    b = np.array([3, 4], dtype=np.int32)
    c = np.array([10.0 * i for i in range(1, 12)])
    assert module.idot(2, 11, a, b, c) == (11, 110.0)
    assert isinstance(runtime._f_vdot(a, b), np.integer)


def test_a_real_dot_product_is_unchanged_to_the_bit() -> None:
    rng = np.random.default_rng(7)
    a, b = rng.standard_normal(97), rng.standard_normal(97)
    s = 0.0
    for x, y in zip(a, b, strict=True):
        s += x * y
    assert runtime._f_vdot(a, b).tobytes() == np.float64(s).tobytes()


def test_a_complex_dot_product_conjugates_its_first_operand() -> None:
    a = np.array([1j, 2.0])
    b = np.array([1j, 1.0])
    assert runtime._f_vdot(a, b) == 1.0 + 2.0
