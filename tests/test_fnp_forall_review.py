"""FORALL shapes the review of the FNP-D0002/FNP-D0003 fix found still wrong.

``tests/test_fnp_statements.py`` covers the FORALL the audit refuted: an
element-wise right-hand side, a positive stride. The review ran gfortran on
the shapes around it -- a stride counting down, a row or column shifted as a
section, an index of the same name as a variable outside -- and each came
out of the translation with another number. Every test here translates a
small module, runs the emitted Python and asserts what gfortran prints for
the same source.
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
np = pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from tests.test_fnp_statements import body_of, translate  # noqa: E402

STRIDES = """\
module fs_mod
  implicit none
contains
  subroutine down(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer :: i
    forall (i = n:1:-1) a(i) = 10.0d0 * a(i)
  end subroutine down

  subroutine downshift(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer :: i
    forall (i = n:2:-1) a(i) = a(i-1)
  end subroutine downshift

  subroutine stepped(a, n, s)
    integer, intent(in) :: n, s
    real(8), intent(inout) :: a(n)
    integer :: i
    forall (i = n:1:s) a(i) = a(i) + 1.0d0
  end subroutine stepped
end module fs_mod
"""


@pytest.fixture(scope="module")
def strides(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("strides"), STRIDES, "fs_mod")


def test_a_forall_counting_down_reaches_its_last_index(strides: Any) -> None:
    """``forall (i = n:1:-1)`` visits ``i = 1``; the stop edge ``1 + 1``
    left it out. gfortran: 10 20 30 40."""
    module, _, _ = strides
    assert list(module.down(np.array([1.0, 2.0, 3.0, 4.0]), 4)) == [10.0, 20.0, 30.0, 40.0]


def test_a_dependent_forall_counting_down_reaches_its_last_index(strides: Any) -> None:
    """The gathered path listed its combinations from the same range.
    gfortran: 1 1 2 3."""
    module, _, _ = strides
    assert list(module.downshift(np.array([1.0, 2.0, 3.0, 4.0]), 4)) == [1.0, 1.0, 2.0, 3.0]


def test_a_forall_step_known_only_at_run_time(strides: Any) -> None:
    """``s = -2`` visits 4 and 2 (gfortran: 1 3 3 5); ``s = 1`` from 4 to 1
    visits nothing."""
    module, _, _ = strides
    assert list(module.stepped(np.array([1.0, 2.0, 3.0, 4.0]), 4, -2)) == [1.0, 3.0, 3.0, 5.0]
    assert list(module.stepped(np.array([1.0, 2.0, 3.0, 4.0]), 4, 1)) == [1.0, 2.0, 3.0, 4.0]


SECTIONS = """\
module fc_mod
  implicit none
contains
  subroutine rows(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n, 4)
    integer :: i
    forall (i = 2:n) a(i, :) = a(i-1, :)
  end subroutine rows

  subroutine cols(a, m)
    integer, intent(in) :: m
    real(8), intent(inout) :: a(2, m)
    integer :: j
    forall (j = 2:m) a(:, j) = a(:, j-1)
  end subroutine cols

  subroutine spread(a, n)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n, 4)
    integer :: i
    forall (i = 2:n) a(i, :) = a(i-1, 1)
  end subroutine spread
end module fc_mod
"""


@pytest.fixture(scope="module")
def sections(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("sections"), SECTIONS, "fc_mod")


def rows_of_tens() -> Any:
    """``a(r, k) = 10 r + k``, four by four."""
    return np.array([[10.0 * r + k for k in range(1, 5)] for r in range(1, 5)], order="F")


def test_a_forall_shifting_rows_gathers_their_values(sections: Any) -> None:
    """``a(i, :) = a(i-1, :)`` gathered views of the rows it went on to
    store into, and copied row 1 everywhere (FNP-D0002 at rank 2)."""
    module, _, _ = sections
    got = module.rows(rows_of_tens(), 4)
    assert got.tolist() == [[11.0, 12.0, 13.0, 14.0]] * 2 + [
        [21.0, 22.0, 23.0, 24.0],
        [31.0, 32.0, 33.0, 34.0],
    ]


def test_a_forall_shifting_columns_gathers_their_values(sections: Any) -> None:
    """gfortran: 1 5 | 1 5 | 2 6 | 3 7, column by column."""
    module, _, _ = sections
    got = module.cols(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], order="F"), 4)
    assert got.tolist() == [[1.0, 1.0, 2.0, 3.0], [5.0, 5.0, 6.0, 7.0]]


def test_a_forall_spreading_an_element_over_a_section(sections: Any) -> None:
    """An element stored into a section is broadcast, gathered or not.
    gfortran: row 1 as it was, then 11, 21 and 31 across."""
    module, _, _ = sections
    got = module.spread(rows_of_tens(), 4)
    assert got.tolist() == [[11.0, 12.0, 13.0, 14.0], [11.0] * 4, [21.0] * 4, [31.0] * 4]


SCOPES = """\
module fi_mod
  implicit none
contains
  subroutine gathered(a, n, k)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer, intent(out) :: k
    integer :: i
    i = 100
    forall (i = 2:n) a(i) = a(i-1)
    k = i
  end subroutine gathered

  subroutine looped(a, n, k)
    integer, intent(in) :: n
    real(8), intent(inout) :: a(n)
    integer, intent(out) :: k
    integer :: i
    i = 100
    forall (i = 1:n) a(i) = 2.0d0 * a(i)
    k = i
  end subroutine looped

  subroutine inner(c, n, m, k)
    integer, intent(in) :: n, m
    real(8), intent(inout) :: c(n, m)
    integer, intent(out) :: k
    integer :: i, j
    j = 7
    forall (i = 1:n)
      forall (j = 1:m) c(i, j) = dble(i + j)
    end forall
    k = j
  end subroutine inner
end module fi_mod
"""


@pytest.fixture(scope="module")
def scopes(tmp_path_factory: pytest.TempPathFactory) -> tuple[ModuleType, str, Any]:
    return translate(tmp_path_factory.mktemp("scopes"), SCOPES, "fi_mod")


def test_a_gathered_forall_leaves_the_variable_of_its_index_alone(scopes: Any) -> None:
    """The store loop rebound the Python local ``i``, so ``k = i`` read the
    last index, 4, where gfortran gives 100."""
    module, _, _ = scopes
    a, k = module.gathered(np.array([1.0, 2.0, 3.0, 4.0]), 4)
    assert list(a) == [1.0, 1.0, 2.0, 3.0]
    assert k == 100


def test_a_looped_forall_leaves_the_variable_of_its_index_alone(scopes: Any) -> None:
    module, _, _ = scopes
    a, k = module.looped(np.array([1.0, 2.0, 3.0, 4.0]), 4)
    assert list(a) == [2.0, 4.0, 6.0, 8.0]
    assert k == 100


def test_a_nested_forall_index_is_scoped_to_its_own_construct(scopes: Any) -> None:
    """gfortran: c(i, j) = i + j, and ``j`` still 7 after both."""
    module, _, _ = scopes
    c, k = module.inner(np.zeros((2, 3), order="F"), 2, 3)
    assert c.tolist() == [[2.0, 3.0, 4.0], [3.0, 4.0, 5.0]]
    assert k == 7


def test_the_read_write_gate_reads_a_forall_index_as_its_source_name(scopes: Any) -> None:
    """``_fi_i`` is the source's ``i`` spelled apart, not a name of its own:
    the gate sees the loop read and write what it did before the rename."""
    import ast

    from recast.verify.rwset import Protocol, span_rwset

    _, text, _ = scopes
    code = body_of(text, "looped").lstrip("\n")
    assert "for _fi_i in" in code
    reads, writes = span_rwset(ast.parse(code), 2, code.count("\n") + 1, Protocol())
    assert "i" in writes and "i" in reads
    assert not any(name.startswith("_fi_") for name in reads | writes)
