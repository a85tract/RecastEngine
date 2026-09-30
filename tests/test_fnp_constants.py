"""The constants pipelines against the defects the Lean audit refuted them on.

Each test names its record in the RecastEngine-Pro-Lean audit (``FNP-Dxxxx``)
and asserts the value gfortran gives the constant, not the spelling that
produced it: the defects were all spellings that ran and were a different
number.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser")
pytest.importorskip("numpy")

from recast.fortran import constants, interface
from recast.fortran.tree import integer_parameters
from recast.fortran.use import resolve
from recast.transform.numpy.constants import constants_module, use_constants_module
from recast.transform.numpy.modules import Modules
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.profiles import PROFILES

QUOTIENTS = """\
module quotients_mod
  implicit none
  integer, parameter :: a = -7
  integer, parameter :: m = a / 2
  integer, parameter :: m2 = (-7) / 2
  integer, parameter :: m3 = 7 / (-2)
  integer, parameter :: m4 = (-7) / (-2)
  integer(8), parameter :: big = 4611686018427387903_8
  integer(8), parameter :: third = big / 3_8
end module quotients_mod
"""


def _use_constants(tmp_path: Path, source: str, names: list[str]) -> dict[str, object]:
    """Render the use-constants module for ``names`` and import it."""
    path = tmp_path / "use_mod.f90"
    path.write_text(source)
    text = use_constants_module(resolve(names, [path]), "use_mod")
    scope: dict[str, object] = {}
    exec(compile(text, "use_constants.py", "exec"), scope)
    return scope


def _constants(tmp_path: Path, source: str) -> dict[str, object]:
    """Render a module's own constants file and import it."""
    path = tmp_path / "own_mod.f90"
    path.write_text(source)
    text = constants_module(constants.extract(path))
    scope: dict[str, object] = {}
    exec(compile(text, "constants.py", "exec"), scope)
    return scope


def test_a_use_constant_quotient_truncates_toward_zero(tmp_path: Path) -> None:
    """FNP-D0013: ``a / 2`` over ``a = -7`` is -3 in Fortran; the fold spelled
    it ``//``, which floors, and the module said -4. Every sign combination
    truncates, and a quotient past 2**53 stays the exact integer ``//`` was
    -- ``int(a / b)`` would have rounded it through binary64."""
    scope = _use_constants(tmp_path, QUOTIENTS, ["m", "m2", "m3", "m4", "third"])
    assert (scope["M"], scope["M2"], scope["M3"], scope["M4"]) == (-3, -3, -3, 3)
    assert scope["THIRD"] == 4611686018427387903 // 3


def test_a_file_without_a_quotient_carries_no_helper(tmp_path: Path) -> None:
    """The truncating helper is emitted only into a file that calls it."""
    path = tmp_path / "plain_mod.f90"
    path.write_text("module plain_mod\n  integer, parameter :: n = 3 * 4\nend module plain_mod\n")
    text = use_constants_module(resolve(["n"], [path]), "plain_mod")
    assert "_f_int_div" not in text


def test_an_extent_folded_from_a_quotient_truncates(tmp_path: Path) -> None:
    """The tree's integer evaluator -- what sizes an extent written over a
    module parameter -- folds the same tree and had the same ``//``."""
    (tmp_path / "quotients_mod.f90").write_text(QUOTIENTS)
    values = integer_parameters(["m", "m3", "m4"], tmp_path, frozenset({"quotients_mod"}))
    assert values == {"m": -3, "m3": -3, "m4": 3}


ARITHMETIC = """\
module arithmetic_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: seven = 7
  integer, parameter :: minus = -1
  real(r8), parameter :: x = 1.0_r8 * (7/2)
  real(r8), parameter :: y = real(7/2, r8)
  real(r8), parameter :: z = 2.0_r8 * (seven / 2) + 1.0_r8 / 4.0_r8
  real(r8), parameter :: w = 1.0_r8 * ((-7) / 2)
  integer, parameter :: n = 2 ** (-1)
  integer, parameter :: n1 = (-1) ** (-3)
  integer, parameter :: n2 = seven ** minus
  integer, parameter :: n3 = 2 ** 3
  real(r8), parameter :: v = 1.0_r8 * 2 ** (-1)
contains
  subroutine noop(k)
    integer, intent(out) :: k
    integer, parameter :: half = seven / 2
    real(r8), parameter :: part = 0.5_r8 * (seven / 2)
    k = half
  end subroutine noop
end module arithmetic_mod
"""


def test_an_integer_quotient_inside_a_real_constant_truncates(tmp_path: Path) -> None:
    """FNP-D0023: ``1.0_r8 * (7/2)`` is 3.0 -- ``7/2`` is an integer
    quotient whatever it is multiplied by -- and the token route emitted
    ``np.float64('1.0') * ( 7 / 2 )``, 3.5. A quotient of integer
    constants, one under a conversion, and one with a negative operand
    truncate; a quotient with a real operand stays real division; and a
    subprogram's own parameters, which have no integer-quotient rule of
    their own, are typed the same way."""
    scope = _constants(tmp_path, ARITHMETIC)
    assert scope["X"] == 3.0 and scope["Y"] == 3.0 and scope["W"] == -3.0
    assert scope["Z"] == 6.25
    assert scope["NOOP__HALF"] == 3 and scope["NOOP__PART"] == 1.5


def test_an_integer_power_with_a_negative_exponent_is_an_integer(tmp_path: Path) -> None:
    """FNP-D0024: ``2 ** (-1)`` is 0 in Fortran and was emitted as Python's
    ``2 ** ( - 1 )``, 0.5; an INTEGER constant raised to one raised
    outright, NumPy refusing a negative integer power. ``(-1) ** (-3)`` is
    -1, and a power with a literal non-negative exponent is spelled as it
    always was."""
    path = tmp_path / "own_mod.f90"
    path.write_text(ARITHMETIC)
    text = constants_module(constants.extract(path))
    assert "N3 = 2 ** 3" in text
    scope = _constants(tmp_path, ARITHMETIC)
    assert (scope["N"], scope["N1"], scope["N2"], scope["N3"]) == (0, -1, 0, 8)
    assert scope["V"] == 0.0


def test_an_expression_without_integer_arithmetic_keeps_its_text(tmp_path: Path) -> None:
    """Only a quotient or power over two integers is respelled: a constant
    without one is the text the token route always gave it, and the file
    defines no helper it does not call."""
    path = tmp_path / "own_mod.f90"
    path.write_text(
        "module plain_mod\n  integer, parameter :: r8 = selected_real_kind(12)\n"
        "  real(r8), parameter :: a = 2.0_r8 / 3.0_r8 + 1.0_r8\nend module plain_mod\n"
    )
    text = constants_module(constants.extract(path))
    assert "A = np.float64('2.0') / np.float64('3.0') + np.float64('1.0')" in text
    assert "_f_int_div" not in text and "_f_ipow" not in text


def test_a_use_constant_integer_power_is_an_integer(tmp_path: Path) -> None:
    """FNP-D0024 on the use-constants route: the same ``2 ** (-1)``."""
    source = (
        "module pow_mod\n  integer, parameter :: p = 2 ** (-1), q = 3 ** 2\nend module pow_mod\n"
    )
    scope = _use_constants(tmp_path, source, ["p", "q"])
    assert (scope["P"], scope["Q"]) == (0, 9)


ROUNDING = """\
module rounding_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: i8 = selected_int_kind(18)
  integer, parameter :: up = nint(2.5_r8), down = nint(-2.5_r8)
  integer, parameter :: under = nint(0.49999999999999994_r8), odd = nint(-3.5_r8)
  integer(i8), parameter :: wide = nint(3.0e10_r8, i8)
  integer, parameter :: m = mod(-7, 3), mo = modulo(-7, 3)
  real(r8), parameter :: rm = mod(-7.5_r8, 2.0_r8), rmo = modulo(-7.5_r8, 2.0_r8)
  integer, parameter :: s1 = sign(3, -2), s2 = sign(-3, 0)
  real(r8), parameter :: s3 = sign(3.0_r8, -1.0_r8)
end module rounding_mod
"""


def test_nint_mod_and_sign_in_a_constant_are_fortrans(tmp_path: Path) -> None:
    """FNP-D0025: ``nint`` was ``np.rint`` -- half to even, and a float --
    and ``mod`` was ``np.mod``, the floored remainder MODULO is. gfortran
    folds ``nint(2.5)`` to 3, ``nint(0.49999999999999994)`` to 0 (where
    ``floor(x + 0.5)`` says 1), ``mod(-7, 3)`` to -1 and ``mod(-7.5, 2.0)``
    to -1.5; ``modulo`` stays floored. A kind on ``nint`` is the result's.
    ``sign`` was ``np.sign``, which reads its second argument as an output
    array and raised on import."""
    scope = _constants(tmp_path, ROUNDING)
    assert (scope["UP"], scope["DOWN"], scope["UNDER"], scope["ODD"]) == (3, -3, 0, -4)
    assert scope["UP"].dtype.name == "int32" and scope["WIDE"].dtype.name == "int64"
    assert scope["WIDE"] == 30000000000
    assert (scope["M"], scope["MO"], scope["RM"], scope["RMO"]) == (-1, 2, -1.5, 0.5)
    assert (scope["S1"], scope["S2"], scope["S3"]) == (-3, 3, -3.0)


INQUIRIES = """\
module inquiries_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: i8 = selected_int_kind(18)
  integer, parameter :: big = huge(0), d = digits(0), d8 = digits(0_i8)
  integer, parameter :: dr = digits(1.0_r8), ds = digits(1.0), dx = digits(x0)
  integer(i8), parameter :: big8 = huge(0_i8)
  real(r8), parameter :: hr = huge(1.0_r8), er = epsilon(1.0_r8)
  real(r8), parameter :: x0 = 1.0_r8
  integer, parameter :: unplaced = huge(big)
  real(r8), parameter :: invalid = epsilon(0)
end module inquiries_mod
"""


def test_an_inquiry_about_an_integer_literal_answers_for_the_integer(tmp_path: Path) -> None:
    """FNP-D0026 and FNP-D0046: ``huge(0)`` and ``digits(0)`` ask about the
    default INTEGER, and the classifier read the literal as a default REAL
    -- ``huge(0)`` was ``np.finfo(np.float32).max``, ``digits(0)`` the
    double's 53. gfortran: 2147483647, 31, 63 for ``digits(0_i8)``, and 24
    for a single's DIGITS. An integer name, whose record has no kind, and
    EPSILON of an integer, which Fortran does not define, refuse."""
    import numpy as np

    scope = _constants(tmp_path, INQUIRIES)
    assert (scope["BIG"], scope["D"], scope["D8"]) == (2147483647, 31, 63)
    assert (scope["DR"], scope["DS"], scope["DX"]) == (53, 24, 53)
    assert scope["BIG8"] == 9223372036854775807
    assert scope["HR"] == np.finfo(np.float64).max and scope["ER"] == np.finfo(np.float64).eps
    path = tmp_path / "own_mod.f90"
    text = constants_module(constants.extract(path))
    assert "# SKIPPED UNPLACED" in text and "# SKIPPED INVALID" in text


STATE = """\
module state_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8) :: third = 1/3
  real(r8) :: neg = -7/2
  real(r8) :: half = 1.0_r8/2.0_r8
  integer :: k = 7/2
  integer :: kr = 7.0_r8/2.0_r8
  integer(8) :: k8 = -7_8/2_8
contains
  subroutine noop()
  end subroutine noop
end module state_mod
"""


def _renderer(tmp_path: Path, source: str) -> tuple[Modules, Path]:
    path = tmp_path / "mod.f90"
    path.write_text(source)
    renderer = Modules(
        subprograms=Subprograms(
            record=interface.extract(path),
            constants=constants.extract(path),
            profile=PROFILES["ifx"],
        )
    )
    return renderer, path


def test_a_module_variables_integer_quotient_initializer_truncates(tmp_path: Path) -> None:
    """FNP-D0029: ``real(r8) :: third = 1/3`` divides two integers, 0, and
    converts the 0 -- gfortran starts ``third`` at 0.0, and the state was
    ``np.float64(0.3333333333333333)``. The quotient truncates toward zero,
    a real quotient stays real, an INTEGER variable takes the quotient
    truncated, and ``-7/2`` -- which fparser writes ``- 7 / 2`` -- is -3
    rather than a ``float('- 7')`` that took the translation down."""
    import numpy as np

    renderer, path = _renderer(tmp_path, STATE)
    body, _report = renderer.body(renderer._subprogram_nodes(path))
    state = {}
    for line in body:
        if "# module state" in line:
            name, value = line.split("  #")[0].split(" = ", 1)
            state[name] = eval(value, {"np": np})
    assert (state["third"], state["neg"], state["half"]) == (0.0, -3.0, 0.5)
    assert (state["k"], state["kr"], state["k8"]) == (3, 3, -3)


DEFAULTS = """\
module defaults_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: nlev = 3
  real(r8), parameter :: grav = 9.81_r8
  type t
    real(r8) :: tol = 1.0d-6
    integer :: k = -3
    real(r8) :: v
    real(r8) :: w = 2
    logical :: on = .true.
    real(r8) :: a(nlev) = 0.5_r8
    integer :: b(3) = (/1, 2, 3/)
    real(r8) :: g = grav
    real(r8) :: third = 1/3
    real(r8), pointer :: p(:) => null()
    real(r8) :: e = 2.0_r8 * grav
    logical(kind=4) :: ini = .false._4
    integer :: cnt = nlev + 1
    real(r8) :: elsewhere = not_a_parameter
  end type t
contains
  subroutine noop()
  end subroutine noop
end module defaults_mod
"""


def test_a_derived_type_factory_starts_components_at_their_defaults(tmp_path: Path) -> None:
    """FNP-D0030: ``real(r8) :: tol = 1.0d-6`` in a type definition is where
    every object of the type starts, and the factory started it at 0.0 --
    the frontend's record carried no initializer. Scalars, a broadcast
    array, an integer constructor (integer), a parameter name, an integer
    quotient, a REAL component given an integer, a LOGICAL of a named kind
    and an expression over the module's parameters (mt19937's ``cnt = n +
    1_wi`` sentinel) are what gfortran starts them at; a pointer's ``=>
    null()`` is ``None`` as before; and an initializer the factory cannot
    render is ``None``, never the zero it used to guess."""
    import numpy as np

    renderer, path = _renderer(tmp_path, DEFAULTS)
    body, _report = renderer.body(renderer._subprogram_nodes(path))
    at = body.index("def _make_t():")
    factory = body[at : body.index("    return o", at)]
    assert "    o.tol = np.float64('1.0e-6')" in factory
    assert "    o.k = -3" in factory and "    o.v = 0.0" in factory
    assert "    o.w = np.float64(2)" in factory and "    o.on = True" in factory
    assert "    o.a = np.full((NLEV,), np.float64('0.5'), dtype=np.float64)" in factory
    assert "    o.b = np.array([1, 2, 3], dtype=np.int32)" in factory
    assert "    o.g = GRAV" in factory and "    o.third = np.float64(0.0)" in factory
    assert "    o.p = None" in factory and "    o.e = np.float64('2.0') * GRAV" in factory
    assert "    o.ini = False" in factory and "    o.cnt = NLEV + 1" in factory
    assert any(
        line.startswith("    o.elsewhere = None  # default initialization") for line in factory
    )
    scope = {"np": np, "NLEV": 3, "GRAV": 9.81}
    values = {
        line.split(" = ", 1)[0].strip()[2:]: eval(line.split(" = ", 1)[1], scope)
        for line in factory
        if line.startswith("    o.") and " = " in line
    }
    assert values["b"].tolist() == [1, 2, 3] and values["a"].tolist() == [0.5, 0.5, 0.5]
