"""CHARACTER lengths on the store paths the first fix of FNP-D0021 missed.

The RecastEngine-Pro-Lean audit's FNP-D0021 is a ``character(len=n)``
variable that did not hold ``n`` characters. Fitting a store to a local
scalar or element fixed the audit's probe; an independent review then ran
gfortran against four more ways a CHARACTER variable is stored -- the
caller's variable after a CALL to an ``intent(out)`` ``len=*`` dummy, a
derived-type component, a module variable, and an array stored from an
array -- and each still held the value's own length. Every test here
translates a small module, runs the emitted Python and asserts what
gfortran prints for the same source.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from tests.test_fnp_statements import body_of, rwset_verdict, translate

STORES = """\
module cs_mod
  implicit none
  integer, parameter :: nl = 5
  type named
    character(len=8) :: nm
  end type named
  type rec
    character(len=nl) :: tag
    character(len=3) :: codes(2)
    character(len=4) :: init = 'xy'
  end type rec
  character(len=6) :: ms
  character(len=nl) :: mp
  character(len=4) :: mi = 'q'
  character(len=3) :: ma(2)
contains
  subroutine setstar(s)
    character(len=*), intent(out) :: s
    s = 'hello'
  end subroutine setstar

  subroutine setnoint(s)
    character(len=*) :: s
    s = 'hey'
  end subroutine setnoint

  subroutine two(s, n)
    character(len=*), intent(out) :: s
    integer, intent(out) :: n
    s = 'ab'
    n = 7
  end subroutine two

  subroutine callstar(n, r)
    integer, intent(out) :: n
    character(len=9), intent(out) :: r
    character(len=8) :: s
    call setstar(s)
    n = len(s)
    r = s // '|'
  end subroutine callstar

  subroutine callnoint(n, r)
    integer, intent(out) :: n
    character(len=9), intent(out) :: r
    character(len=8) :: s
    call setnoint(s)
    n = len(s)
    r = s // '|'
  end subroutine callnoint

  subroutine multi(n, r, k)
    integer, intent(out) :: n, k
    character(len=9), intent(out) :: r
    character(len=nl) :: s
    call two(s, k)
    n = len(s)
    r = s // '|'
  end subroutine multi

  subroutine relay(s, n)
    character(len=*), intent(inout) :: s
    integer, intent(out) :: n
    call two(s, n)
  end subroutine relay

  subroutine comp(n, r)
    integer, intent(out) :: n
    character(len=9), intent(out) :: r
    type(named) :: t
    t%nm = 'ab'
    n = len(t%nm)
    r = t%nm // '|'
  end subroutine comp

  subroutine comps(r1, r2, r3, n)
    character(len=9), intent(out) :: r1, r2, r3
    integer, intent(out) :: n
    type(rec) :: t
    r3 = t%init // '|'
    t%tag = 'x'
    t%codes(2) = 'abcdef'
    t%codes(1) = 'z'
    r1 = t%tag // '|'
    r2 = t%codes(1) // t%codes(2) // '|'
    n = len(t%tag)
  end subroutine comps

  subroutine modvar(n, r)
    integer, intent(out) :: n
    character(len=7), intent(out) :: r
    ms = 'ab'
    n = len(ms)
    r = ms // '|'
  end subroutine modvar

  subroutine mods(r1, r2, r3)
    character(len=9), intent(out) :: r1, r2, r3
    r1 = mi // '|'
    r3 = ma(1) // '|'
    mp = 'abcdefgh'
    r2 = mp // '|'
  end subroutine mods

  subroutine arrs(r1, r2)
    character(len=5), intent(out) :: r1, r2
    character(len=4) :: a(2)
    character(len=2) :: b(2)
    b(1) = 'xy'
    b(2) = 'zw'
    a = b
    r1 = a(1) // '|'
    a = ['ab', 'cd']
    r2 = a(2) // '|'
  end subroutine arrs

  subroutine sect(r1, r2)
    character(len=9), intent(out) :: r1, r2
    character(len=4) :: a(3)
    character(len=2) :: b(2)
    a = 'qqqqqq'
    b = ['xy', 'zw']
    a(2:3) = b
    r1 = a(1) // a(2) // '|'
    a(1:2) = 'k'
    r2 = a(1) // a(3) // '|'
  end subroutine sect

  subroutine lennoint(s, n)
    character(len=*) :: s
    integer, intent(out) :: n
    s = 'hey'
    n = len(s)
  end subroutine lennoint

  subroutine padnoint(s, r)
    character(len=*) :: s
    character(len=12), intent(out) :: r
    s = 'hey'
    r = s // '|'
  end subroutine padnoint

  subroutine calllen(n, m, r)
    integer, intent(out) :: n, m
    character(len=12), intent(out) :: r
    character(len=8) :: s
    call lennoint(s, m)
    n = len(s)
    call padnoint(s, r)
  end subroutine calllen

  subroutine wel(r)
    character(len=9), intent(out) :: r
    character(len=6) :: a(2)
    a = 'zzzzzz'
    associate (c => a(2))
      write (c, '(i3)') 42
    end associate
    r = a(2) // '|'
  end subroutine wel
end module cs_mod
"""


@pytest.fixture(scope="module")
def stores(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("stores"), STORES, "cs_mod")


def test_the_callers_variable_keeps_its_length_after_an_out_assumed_length_dummy(
    stores: Any,
) -> None:
    """gfortran: ``8 [hello   |]``. The callee stores five characters into
    a dummy whose length the return convention never hands in, and the
    caller bound them unfitted: 5 and ``hello|``."""
    module, _, _ = stores
    assert module.callstar() == (8, "hello   |")


def test_an_assumed_length_dummy_without_intent_keeps_the_callers_length(stores: Any) -> None:
    """``setnoint`` declares no intent; every path stores ``s``. It is kept
    ``intent(inout)``, so the caller's length reaches the store.
    gfortran: ``8 [hey     |]``."""
    module, _, _ = stores
    assert module.callnoint() == (8, "hey     |")


def test_the_fitted_output_among_several_and_a_named_length(stores: Any) -> None:
    """gfortran: ``multi 5 [ab   |   ] 7``: the CHARACTER output is fitted to
    ``nl`` and the INTEGER one beside it is bound as it was."""
    module, _, _ = stores
    assert module.multi() == (5, "ab   |   ", 7)


def test_a_relayed_assumed_length_is_the_callers_callers(stores: Any) -> None:
    """gfortran: ``relay [ab    ] 7`` for a six-character actual."""
    module, _, _ = stores
    assert module.relay("zzzzzz") == ("ab    ", 7)


def test_a_character_component_holds_its_declared_length(stores: Any) -> None:
    """gfortran: ``8 [ab      |]``; the ``Data_Ref`` store was never fitted
    and the Python said 2 and ``ab|``."""
    module, _, _ = stores
    assert module.comp() == (8, "ab      |")


def test_component_elements_named_lengths_and_default_initialization(stores: Any) -> None:
    """gfortran: ``[x    |   ] [z  abc|  ] [xy  |    ] 5`` -- a component of
    a named-constant length, elements of a CHARACTER array component, and a
    component's literal default initialization at its length."""
    module, _, _ = stores
    assert module.comps() == ("x    |   ", "z  abc|  ", "xy  |    ", 5)


def test_a_module_variable_holds_its_declared_length(stores: Any) -> None:
    """gfortran: ``6 [ab    |]``. The store was not fitted, and the module
    bound the variable to ``None`` rather than six blanks."""
    module, _, _ = stores
    assert module.modvar() == (6, "ab    |")


def test_module_initial_values_are_at_the_declared_length(stores: Any) -> None:
    """gfortran: ``[q   |    ] [abcde|   ] [   |     ]`` -- a literal
    initializer, a store cut to a named length, and an array's blanks."""
    module, text, _ = stores
    assert module.mods() == ("q   |    ", "abcde|   ", "   |     ")
    assert "ms = '      '" in text


def test_an_array_stored_from_an_array_is_fitted_element_by_element(stores: Any) -> None:
    """gfortran: ``[xy  |] [cd  |]`` for ``a = b`` and ``a = ['ab', 'cd']``
    into a ``len=4`` ``a``; the elements kept their two characters."""
    module, text, _ = stores
    assert module.arrs() == ("xy  |", "cd  |")
    assert "np.frompyfunc" in body_of(text, "arrs")


def test_a_section_is_fitted_from_an_array_and_from_a_scalar(stores: Any) -> None:
    """gfortran: ``[qqqqxy  |] [k   zw  |]``."""
    module, _, _ = stores
    assert module.sect() == ("qqqqxy  |", "k   zw  |")


def test_an_assumed_length_dummy_without_intent_has_the_actuals_length_inside(
    stores: Any,
) -> None:
    """gfortran: ``8 8 [hey     |   ]``. Inferred ``intent(out)``, the dummy
    was never passed in, so its length never reached the callee: ``len(s)``
    there was 3 and ``s // '|'`` ``hey|``. Kept ``intent(inout)``, it is."""
    module, text, _ = stores
    assert module.calllen() == (8, 8, "hey     |   ")
    assert "def lennoint(s):" in text


def test_an_internal_write_to_an_element_still_fills_it(stores: Any) -> None:
    """gfortran: ``[ 42   |  ]``. An internal WRITE through an associate
    name for an element is one record, padded to the array's length; read
    as a write to an array, whose records are its elements, it was left
    unfitted, `` 42|``."""
    module, _, _ = stores
    assert module.wel() == " 42   |  "


def test_the_read_write_gate_counts_what_the_new_fittings_read(stores: Any, tmp_path: Path) -> None:
    """A fitted component, section or CALL output reads its length's named
    constants, or -- an assumed length -- the variable's own."""
    _, _, candidate = stores
    verdict = rwset_verdict(candidate, tmp_path)
    assert not verdict.metrics["failures"], verdict.metrics["failures"]
