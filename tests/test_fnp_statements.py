"""Statement-level defects the formal audit of this backend refuted.

The RecastEngine-Pro-Lean audit states, rule by rule, what a translation must
compute for the Fortran it accepts, and records a defect (``FNP-Dxxxx``)
wherever the emitted Python computes something else. The ones here are
statements -- FORALL, ASSOCIATE, a CHARACTER assignment, a SAVEd local, the
intents a dummy is given -- and every test translates a small module, runs
the emitted Python, and asserts the number gfortran prints for the same
source, so reverting the fix fails with the defect's own symptom.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
np = pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.fortran.frontend import FortranFrontend  # noqa: E402
from recast.transform.numpy.translate import NumpyTranslation  # noqa: E402


def translate(tmp_path: Path, source: str, module: str) -> tuple[ModuleType, str, Any]:
    """Translate ``module`` out of ``source`` and import what was emitted.

    Returns the imported module, its text and the candidate. Imported from
    its own directory under a name no other test uses, and dropped from
    ``sys.modules`` again, so two tests translating a module of one name do
    not see each other's.
    """
    root = tmp_path / "src"
    root.mkdir()
    (root / f"{module}.f90").write_text(source)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(root) if u.uid == f"fortran:{module}")
    facts = frontend.analyze(unit, root)
    candidate = NumpyTranslation().apply(unit, facts, {"root": root})
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    text = (out / f"{module}_numpy.py").read_text()
    sys.path.insert(0, str(out))
    try:
        for stale in [m for m in sys.modules if m.startswith(module)]:
            del sys.modules[stale]
        imported = importlib.import_module(f"{module}_numpy")
    finally:
        sys.path.remove(str(out))
        for loaded in [m for m in sys.modules if m.startswith(module)]:
            del sys.modules[loaded]
    return imported, text, candidate


def body_of(text: str, name: str) -> str:
    """One emitted function, from its ``def`` to the next one."""
    start = text.index(f"\ndef {name}(")
    end = text.find("\ndef ", start + 1)
    return text[start : end if end >= 0 else len(text)]


# -- FORALL (FNP-D0002, FNP-D0003; rule FNP-R0121) -----------------------------

FORALL = """\
module fa_mod
  implicit none
contains
  subroutine shift(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer :: i
    forall (i = 2:n) a(i) = a(i-1)
  end subroutine shift

  subroutine masked(a, b, n)
    integer, intent(in) :: n
    real(8), intent(in) :: a(n)
    real(8), intent(inout) :: b(n)
    integer :: i
    forall (i = 1:n, a(i) > 0.0d0) b(i) = a(i)
  end subroutine masked

  subroutine twice(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer :: i
    forall (i = 1:n) a(i) = 2.0d0 * a(i)
  end subroutine twice

  subroutine flip(a, b, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n), b(n)
    integer :: i
    forall (i = 1:n)
      a(i) = dble(i)
      b(i) = a(n + 1 - i)
    end forall
  end subroutine flip

  subroutine guarded(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer :: i
    forall (i = 2:n, a(i-1) > 0.0d0) a(i) = -a(i-1)
  end subroutine guarded

  subroutine grid(c, n, m)
    integer, intent(in) :: n, m
    real(8), intent(inout) :: c(n, m)
    integer :: i, j
    forall (i = 1:n, j = 2:m, c(i, j) >= 0.0d0) c(i, j) = c(i, j-1) + 1.0d0
  end subroutine grid

  subroutine nested(c, n, m)
    integer, intent(in) :: n, m
    real(8), intent(inout) :: c(n, m)
    integer :: i, j
    forall (i = 2:n)
      forall (j = 1:m) c(i, j) = c(i-1, j)
    end forall
  end subroutine nested
end module fa_mod
"""


@pytest.fixture(scope="module")
def forall(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("forall"), FORALL, "fa_mod")


def test_forall_evaluates_every_value_before_storing_any(forall: Any) -> None:
    """``forall (i = 2:n) a(i) = a(i-1)`` shifts the array by one; the
    loops it was emitted as read each ``a(i-1)`` after the previous trip had
    overwritten it and copied ``a(1)`` everywhere (FNP-D0002)."""
    module, _, _ = forall
    shifted = module.shift(np.array([1.0, 2.0, 3.0, 4.0]), 4)
    assert list(shifted) == [1.0, 1.0, 2.0, 3.0]


def test_forall_honours_its_mask(forall: Any) -> None:
    """The scalar mask was dropped, so every element was stored (FNP-D0003)."""
    module, _, _ = forall
    stored = module.masked(np.array([1.0, -1.0, 2.0, -2.0]), np.full(4, 9.0), 4)
    assert list(stored) == [1.0, 9.0, 2.0, 9.0]


def test_a_forall_reading_only_what_it_overwrites_stays_a_loop(forall: Any) -> None:
    """``a(i) = 2*a(i)`` reads the element the same trip stores and nothing
    else, so the loops mean what the FORALL means and are still what is
    emitted: no gathered values, and the numbers of a plain loop."""
    module, text, _ = forall
    assert "_fa" not in body_of(text, "twice")
    assert list(module.twice(np.array([1.0, 2.0, 3.0]), 3)) == [2.0, 4.0, 6.0]
    assert "if (a[_fi_i - 1] > 0.0):" in body_of(text, "masked")


def test_each_assignment_of_a_construct_completes_before_the_next(forall: Any) -> None:
    """The second assignment reads the first's array at another element: it
    sees every value the first stored, not only the trips before its own."""
    module, _, _ = forall
    a, b = module.flip(np.zeros(4), np.zeros(4), 4)
    assert list(a) == [1.0, 2.0, 3.0, 4.0]
    assert list(b) == [4.0, 3.0, 2.0, 1.0]


def test_the_mask_is_evaluated_once_before_any_store(forall: Any) -> None:
    """``a(i-1) > 0`` is asked of the array as it was: the store to ``a(2)``
    does not switch ``a(3)``'s trip off. gfortran: 1 -1 -2 4."""
    module, _, _ = forall
    assert list(module.guarded(np.array([1.0, 2.0, -3.0, 4.0]), 4)) == [1.0, -1.0, -2.0, 4.0]


def test_a_two_index_forall_with_a_mask(forall: Any) -> None:
    module, _, _ = forall
    c = np.array([[0.0, 5.0, -1.0], [1.0, 1.0, 1.0]], order="F")
    got = module.grid(c, 2, 3)
    # Every value from the array before the FORALL: c(1,3) < 0 is masked out.
    assert got.tolist() == [[0.0, 1.0, -1.0], [1.0, 2.0, 2.0]]


def test_a_dependent_forall_with_a_body_that_is_not_assignments_is_refused(forall: Any) -> None:
    _, text, candidate = forall
    assert any(entry.startswith("nested/") and "FORALL" in entry for entry in candidate.deferred)
    assert "raise NotImplementedError" in body_of(text, "nested")


def rwset_verdict(candidate: Any, workspace: Path) -> Any:
    """``static.rwset`` on a candidate: the read/write sets the source says
    each block has, against the ones its translation has."""
    from recast.executors.local import LocalExecutor
    from recast.model import Unit
    from recast.verify.rwset import factory

    return factory().check(
        Unit(uid=candidate.unit, kind="module"), candidate, workspace, LocalExecutor(), {}
    )


# -- ASSOCIATE (FNP-D0015; rule FNP-R0124) -------------------------------------

ASSOCIATE = """\
module as_mod
  implicit none
  type box_t
    real(8) :: dx
    real(8) :: cells(3)
  end type box_t
contains
  subroutine bump(x, y)
    real(8), intent(inout) :: x
    real(8), intent(out) :: y
    associate (t => x)
      t = t + 1.0d0
    end associate
    y = x
  end subroutine bump

  subroutine follow(x, y)
    real(8), intent(inout) :: x
    real(8), intent(out) :: y
    associate (t => x)
      x = 5.0d0
      y = t
    end associate
  end subroutine follow

  subroutine element(a)
    real(8), intent(inout) :: a(3)
    associate (t => a(2))
      t = 7.0d0
    end associate
  end subroutine element

  subroutine component(g)
    type(box_t), intent(inout) :: g
    associate (t => g%dx, c => g%cells)
      t = 3.0d0
      c(1) = t
    end associate
  end subroutine component

  subroutine reads(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    associate (t => x)
      y = 2.0d0 * t
    end associate
  end subroutine reads

  subroutine moving(a, i)
    real(8), intent(inout) :: a(3)
    integer, intent(inout) :: i
    associate (t => a(i))
      i = i + 1
      t = 0.0d0
    end associate
  end subroutine moving

  subroutine valued(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    associate (t => x + 1.0d0)
      y = t
    end associate
  end subroutine valued
end module as_mod
"""


@pytest.fixture(scope="module")
def associated(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("associate"), ASSOCIATE, "as_mod")


def test_a_store_through_an_associate_name_reaches_its_variable(associated: Any) -> None:
    """``associate (t => x); t = t + 1`` stores to ``x``. Bound as the local
    ``t = x``, the store rebound ``t`` and ``x`` stayed 1.0 (FNP-D0015)."""
    module, text, _ = associated
    assert module.bump(1.0) == (2.0, 2.0)
    assert "t = x" not in body_of(text, "bump")


def test_a_store_to_the_variable_is_seen_through_the_name(associated: Any) -> None:
    module, _, _ = associated
    assert module.follow(1.0) == (5.0, 5.0)


def test_an_element_and_a_component_are_written_through(associated: Any) -> None:
    module, _, _ = associated
    assert list(module.element(np.array([1.0, 2.0, 3.0]))) == [1.0, 7.0, 3.0]

    class Box:
        dx = 0.0
        cells = np.zeros(3)

    box = module.component(Box())
    assert box.dx == 3.0
    assert list(box.cells) == [3.0, 0.0, 0.0]


def test_a_name_only_read_is_still_bound_once(associated: Any) -> None:
    module, text, _ = associated
    assert module.reads(1.5) == 3.0
    assert "t = x" in body_of(text, "reads")
    assert module.valued(1.0) == 2.0


def test_an_element_whose_subscript_the_body_changes_is_refused(associated: Any) -> None:
    """``a(i)`` is the element of ``i`` as it was at the ASSOCIATE; a
    respelled ``a[i - 1]`` would store to the next one."""
    _, _, candidate = associated
    assert any(entry.startswith("moving/") for entry in candidate.deferred)


def test_the_read_write_gate_agrees_with_the_respelled_body(
    associated: Any, tmp_path: Path
) -> None:
    """The source's sets now say the name is its variable too: a store to
    ``t`` writes ``x``. Every associate block above agrees with its
    translation."""
    _, _, candidate = associated
    verdict = rwset_verdict(candidate, tmp_path)
    failures = [f["block"] for f in verdict.metrics["failures"]]
    assert not [f for f in failures if not f.startswith("moving/")], verdict.metrics["failures"]


# -- CHARACTER assignment (FNP-D0021; rule FNP-R0111) --------------------------

CHARACTERS = """\
module ch_mod
  implicit none
  integer, parameter :: nl = 6
contains
  subroutine lens(n1, n2, same)
    integer, intent(out) :: n1, n2
    logical, intent(out) :: same
    character(len=10) :: s
    character(len=3) :: t
    s = 'ab'
    t = 'abcdef'
    n1 = len(s)
    n2 = len_trim(t)
    same = (s == 'ab')
  end subroutine lens

  subroutine old_style(c, n)
    character*4, intent(out) :: c
    integer, intent(out) :: n
    character u
    c = 'q'
    u = 'yz'
    n = len(c) + len(u)
  end subroutine old_style

  subroutine assumed(a)
    character(len=*), intent(inout) :: a
    a = 'z'
  end subroutine assumed

  subroutine elements(w)
    character(len=3), intent(out) :: w(2)
    w(1) = 'a'
    w(2) = 'abcdef'
  end subroutine elements

  subroutine named(n, t)
    integer, intent(out) :: n
    character(len=nl), intent(out) :: t
    t = 'x'
    n = len(t)
  end subroutine named

  subroutine internal(s, n)
    character(len=8), intent(out) :: s
    integer, intent(out) :: n
    write(s, '(i3)') 42
    n = len(s)
  end subroutine internal

  subroutine initialized(u, n)
    character(len=5), intent(out) :: u
    integer, intent(out) :: n
    character(len=5) :: s = 'ab'
    u = s
    n = len(s)
  end subroutine initialized

  subroutine unassigned(flag, b)
    logical, intent(in) :: flag
    character(len=4), intent(out) :: b
    if (flag) b = 'x'
  end subroutine unassigned

  subroutine automatic(s, n)
    character(len=*), intent(in) :: s
    integer, intent(out) :: n
    character(len=len(s)) :: t
    t = s
    n = len(t)
  end subroutine automatic
end module ch_mod
"""


@pytest.fixture(scope="module")
def characters(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("characters"), CHARACTERS, "ch_mod")


def test_a_character_assignment_pads_and_truncates_to_the_length(characters: Any) -> None:
    """gfortran: ``lens 10 3 T``. The translation kept the bare strings, so
    ``len(s)`` was 2 and ``len_trim(t)`` 6 (FNP-D0021)."""
    module, _, _ = characters
    assert module.lens() == (10, 3, True)


def test_the_f77_length_spellings_are_lengths_too(characters: Any) -> None:
    """``character*4`` and a bare ``character`` (length 1) were recorded with
    no length at all."""
    module, _, _ = characters
    assert module.old_style() == ("q   ", 5)


def test_an_assumed_length_dummy_keeps_the_callers_length(characters: Any) -> None:
    module, _, _ = characters
    assert module.assumed("abcde") == "z    "


def test_an_element_of_a_character_array_is_fitted(characters: Any) -> None:
    module, _, _ = characters
    assert list(module.elements()) == ["a  ", "abc"]


def test_a_length_named_by_a_constant(characters: Any) -> None:
    module, _, _ = characters
    assert module.named() == (6, "x     ")


def test_an_internal_write_fills_the_variable(characters: Any) -> None:
    module, _, _ = characters
    assert module.internal() == (" 42     ", 8)


def test_a_declared_initializer_is_its_literal_at_the_length(characters: Any) -> None:
    """``character(len=5) :: s = 'ab'`` was initialized to the empty string:
    the initializer was dropped for every CHARACTER local."""
    module, _, _ = characters
    assert module.initialized() == ("ab   ", 5)


def test_an_unassigned_out_character_is_blanks_of_its_length(characters: Any) -> None:
    module, _, _ = characters
    assert module.unassigned(True) == "x   "
    assert module.unassigned(False) == "    "


def test_a_length_computed_by_a_call_is_left_as_it_was(characters: Any) -> None:
    """``len=len(s)`` is evaluated where ``t`` is created; the declared text
    is not a length this can respell, and a respelling of it emitted
    ``len_(s`` -- a file that does not parse -- on fortran-utils."""
    module, text, _ = characters
    assert module.automatic("abc") == 3
    assert "ljust" not in body_of(text, "automatic")


def test_the_read_write_gate_counts_what_a_fitted_length_reads(
    characters: Any, tmp_path: Path
) -> None:
    """``(value).ljust(nl)`` reads ``nl`` and ``ljust(len(a))`` reads ``a``;
    the source's sets say so too, or every fitted store would disagree."""
    _, _, candidate = characters
    verdict = rwset_verdict(candidate, tmp_path)
    assert not verdict.metrics["failures"], verdict.metrics["failures"]


# -- SAVEd locals (FNP-D0014; rules FNP-R0143, FNP-R0165) ----------------------

SAVED = """\
module sv_mod
  implicit none
contains
  subroutine count(c)
    integer, intent(out) :: c
    integer :: calls = 0
    calls = calls + 1
    c = calls
  end subroutine count

  subroutine remember(n, prev)
    integer, intent(in) :: n
    integer, intent(out) :: prev
    integer, save :: last = -1
    prev = last
    last = n
  end subroutine remember

  subroutine tally(x, s)
    real(8), intent(in) :: x
    real(8), intent(out) :: s
    real(8) :: acc
    data acc /0.0d0/
    acc = acc + x
    s = acc
  end subroutine tally

  subroutine every(n, total)
    integer, intent(in) :: n
    integer, intent(out) :: total
    integer :: acc
    save
    if (n == 0) acc = 0
    acc = acc + n
    total = acc
  end subroutine every

  subroutine grow(n, m)
    integer, intent(in) :: n
    integer, intent(out) :: m
    real(8), allocatable, save :: buf(:)
    if (.not. allocated(buf)) then
      allocate(buf(n))
      buf = 0.0d0
    end if
    buf(1) = buf(1) + 1.0d0
    m = int(buf(1))
  end subroutine grow

  subroutine scaled(x, y)
    real(8), intent(in) :: x
    real(8), intent(out) :: y
    real(8) :: half = 0.5d0
    y = x * half
  end subroutine scaled

  subroutine mixed(k)
    integer, intent(out) :: k
    integer :: a, b
    data a, b /1, 2/
    a = a + b
    k = a
  end subroutine mixed
end module sv_mod
"""


@pytest.fixture(scope="module")
def saved(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("saved"), SAVED, "sv_mod")


def test_an_initialized_local_keeps_its_value_between_calls(saved: Any) -> None:
    """``integer :: calls = 0`` is SAVEd by its initializer. gfortran counts
    1, 2; re-initialized at every entry the translation answered 1, 1
    (FNP-D0014)."""
    module, _, _ = saved
    assert [module.count(), module.count(), module.count()] == [1, 2, 3]


def test_a_save_attribute_keeps_the_last_value(saved: Any) -> None:
    module, _, _ = saved
    assert [module.remember(5), module.remember(7)] == [-1, 5]


def test_a_data_initialized_local_is_initialized_once(saved: Any) -> None:
    module, text, _ = saved
    assert [module.tally(1.0), module.tally(2.0)] == [1.0, 3.0]
    assert "if _fresh:" in body_of(text, "tally")


def test_a_bare_save_statement_saves_every_local(saved: Any) -> None:
    module, _, _ = saved
    assert [module.every(0), module.every(4), module.every(5)] == [0, 4, 9]


def test_a_saved_allocatable_stays_allocated(saved: Any) -> None:
    module, _, _ = saved
    assert [module.grow(3), module.grow(3)] == [1, 2]


def test_a_saved_local_the_body_never_changes_is_still_a_local(saved: Any) -> None:
    """Re-initializing a value nothing changes gives the same value on every
    call, so the emission is what it always was."""
    module, text, _ = saved
    assert module.scaled(3.0) == 1.5
    assert "_saved_" not in body_of(text, "scaled")


def test_a_data_statement_naming_kept_and_unchanged_locals_is_refused(saved: Any) -> None:
    _, _, candidate = saved
    assert any(entry.startswith("mixed/D001") for entry in candidate.deferred)


def test_the_read_write_gate_reads_a_kept_local_back_as_itself(saved: Any, tmp_path: Path) -> None:
    _, _, candidate = saved
    verdict = rwset_verdict(candidate, tmp_path)
    failures = [f["block"] for f in verdict.metrics["failures"]]
    assert not [f for f in failures if not f.startswith("mixed/")], verdict.metrics["failures"]


# -- Intents a dummy is given (FNP-D0031, FNP-D0047; FNP-R0140, FNP-R0141) ------

INTENTS = """\
module in_mod
  implicit none
  type pair_t
    real(8) :: a
    real(8) :: b
  end type pair_t
contains
  subroutine chk(x, ierr)
    real(8) :: x
    integer :: ierr
    if (x < 0d0) ierr = 1
  end subroutine chk

  subroutine setk(k)
    integer :: k
    k = 3
  end subroutine setk

  subroutine early(x, k)
    real(8) :: x
    integer :: k
    if (x < 0d0) return
    k = 5
  end subroutine early

  subroutine setname(s, n)
    character(len=8) :: s
    integer :: n
    s = 'hello'
    n = 42
  end subroutine setname

  subroutine drive(s, n)
    character(len=8), intent(out) :: s
    integer, intent(out) :: n
    s = 'xxxxxxxx'
    n = -1
    call setname(s, n)
  end subroutine drive

  subroutine component(o)
    type(pair_t) :: o
    o%a = 1d0
  end subroutine component

  subroutine formatted(s, n)
    character(len=6) :: s
    integer :: n
    write (s, '(i3)') n
  end subroutine formatted

  subroutine sometimes(flag, a)
    logical :: flag
    real(8) :: a(3)
    if (flag) a = 1d0
  end subroutine sometimes
end module in_mod
"""


@pytest.fixture(scope="module")
def intents(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("intents"), INTENTS, "in_mod")


def test_a_conditional_write_leaves_the_callers_value(intents: Any) -> None:
    """``if (x < 0) ierr = 1`` writes ``ierr`` on one path only. Inferred
    ``intent(out)``, the other path returned the prologue's 0 where gfortran
    leaves the caller's 7 (FNP-D0031)."""
    module, _, _ = intents
    assert module.chk(1.0, 7) == 7
    assert module.chk(-1.0, 7) == 1


def test_a_write_on_every_path_is_still_out(intents: Any) -> None:
    module, text, _ = intents
    assert module.setk() == 3
    assert "def setk():" in text


def test_a_return_before_the_write_leaves_the_callers_value(intents: Any) -> None:
    module, _, _ = intents
    assert module.early(-1.0, 9) == 9
    assert module.early(1.0, 9) == 5


def test_a_character_dummy_with_no_intent_hands_its_value_back(intents: Any) -> None:
    """gfortran: ``drive -> [hello   ] 42``. The CHARACTER dummy had no
    intent anything inferred, so it was passed in and never returned, and
    the caller kept ``xxxxxxxx`` (FNP-D0047)."""
    module, _, _ = intents
    assert module.drive() == ("hello   ", 42)


def test_a_component_write_keeps_the_rest_of_the_structure(intents: Any) -> None:
    module, _, _ = intents

    class Pair:
        a = 0.0
        b = 2.0

    pair = module.component(Pair())
    assert (pair.a, pair.b) == (1.0, 2.0)


def test_an_internal_write_is_a_write_of_its_unit(intents: Any) -> None:
    module, _, _ = intents
    assert module.formatted(42) == " 42   "


def test_a_conditional_whole_array_write_leaves_the_callers_array(intents: Any) -> None:
    module, _, _ = intents
    assert list(module.sometimes(False, np.full(3, 5.0))) == [5.0, 5.0, 5.0]
    assert list(module.sometimes(True, np.full(3, 5.0))) == [1.0, 1.0, 1.0]
