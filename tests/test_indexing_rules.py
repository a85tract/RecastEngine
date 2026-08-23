"""Tests for the subscript rules.

Plans, not text. Every case here is one where the plain shift-by-one is wrong,
or where Fortran can say something a zero-based slice cannot say back and the
rule has to refuse rather than approximate.

The refusals are the part worth having tests for: across the five translated
CAM modules there are 3,883 subscripts and not one of them takes a refusing
path, so the corpus says nothing about whether they are right.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran._parse import f03, parser
from recast.transform.rules import NoRule, indexing

SCALAR = frozenset({"0", "1", "2"})


@pytest.fixture(autouse=True)
def _fparser_patterns() -> None:
    """Constructing a node from text needs the parser to have been created
    once; these tests never open a file, so nothing else would do it."""
    parser()


def _subs(text: str) -> Any:
    """``"i, 1:n"`` -> the subscript list node it parses to."""
    return f03.Section_Subscript_List(text)


def _dims(*bounds: tuple[str, str]) -> list[dict[str, Any]]:
    return [{"lb": lb, "ub": ub} for lb, ub in bounds]


def _scalar(_node: Any) -> int:
    """Everything is rank 0 -- the usual case, and the one that isolates the
    rules from whatever is answering rank questions."""
    return 0


# --- shape -------------------------------------------------------------------


def test_a_scalar_subscript_is_an_index_and_a_triplet_is_a_range() -> None:
    positions = indexing.describe(_subs("i, 1:n"), None, rank_of=_scalar)
    assert [p.kind for p in positions] == [indexing.Kind.INDEX, indexing.Kind.RANGE]
    assert str(positions[0].index) == "i"
    assert (str(positions[1].lower), str(positions[1].upper)) == ("1", "n")
    assert positions[1].step is None


def test_an_integer_array_subscript_is_a_gather() -> None:
    """It looks exactly like a scalar index until something says what rank it
    has, which is why ``rank_of`` is an argument rather than an assumption."""
    positions = indexing.describe(_subs("idx"), None, rank_of=lambda _n: 1)
    assert positions[0].kind is indexing.Kind.VECTOR


def test_an_omitted_range_bound_stays_omitted(_=None) -> None:
    positions = indexing.describe(_subs(":"), None, rank_of=_scalar)
    assert positions[0].kind is indexing.Kind.RANGE
    assert positions[0].lower is None and positions[0].upper is None


def test_a_stride_is_carried_through() -> None:
    positions = indexing.describe(_subs("1:n:2"), None, rank_of=_scalar)
    assert str(positions[0].step) == "2"


# --- lower bounds ------------------------------------------------------------


def test_a_dimension_with_no_declared_bound_starts_at_one() -> None:
    positions = indexing.describe(_subs("i"), None, rank_of=_scalar)
    assert positions[0].origin == "1"
    assert positions[0].shifts_by_one


def test_a_declared_lower_bound_is_reported_verbatim() -> None:
    """``real :: q(0:n)`` indexes from zero already, so ``q(i)`` is ``q[i]``
    and not ``q[i-1]``. A bound may be an expression over other arguments, so
    it comes back as source text for the backend to render."""
    positions = indexing.describe(_subs("i, j"), _dims(("0", "n"), ("lo", "hi")), rank_of=_scalar)
    assert positions[0].origin == "0" and not positions[0].shifts_by_one
    assert positions[1].origin == "lo"


def test_an_assumed_shape_bound_counts_as_unit() -> None:
    positions = indexing.describe(_subs("i"), _dims(("1", None)), rank_of=_scalar)  # type: ignore[arg-type]
    assert positions[0].shifts_by_one


# --- literal folding ---------------------------------------------------------


def test_a_literal_folds_while_the_result_stays_whitelisted() -> None:
    """``a(3)`` may be written ``a[2]``: 2 is structural."""
    positions = indexing.describe(_subs("1, 2, 3"), None, rank_of=_scalar)
    assert [indexing.fold(p, SCALAR) for p in positions] == [0, 1, 2]


def test_a_literal_that_would_introduce_a_magic_number_does_not_fold() -> None:
    """``a(17)`` must not become ``a[16]``. The zero-literal rule says every
    non-structural number in the output is a named constant, and folding here
    would put back exactly what hoisting took out."""
    positions = indexing.describe(_subs("17"), None, rank_of=_scalar)
    assert indexing.fold(positions[0], SCALAR) is None


def test_nothing_folds_off_a_declared_lower_bound(_=None) -> None:
    """The amount to subtract is the bound, which may not even be a number."""
    positions = indexing.describe(_subs("3"), _dims(("0", "n")), rank_of=_scalar)
    assert indexing.fold(positions[0], SCALAR) is None


def test_only_an_index_folds() -> None:
    positions = indexing.describe(_subs("1:2"), None, rank_of=_scalar)
    assert indexing.fold(positions[0], SCALAR) is None


# --- what has no rule --------------------------------------------------------


def test_a_negative_stride_is_a_range_whatever_its_bounds() -> None:
    """Counting down, the stop for reaching the first element is off the end
    of the axis, and there is no index meaning "one before the start" -- so
    it is not a slice literal, and the emitter hands the edges and the axis
    origin to the runtime instead. A re-based axis and an implied edge are
    that same case, not a refusal."""
    rebased = indexing.describe(_subs("n:1:-1"), _dims(("0", "n")), rank_of=_scalar)
    assert rebased[0].kind is indexing.Kind.RANGE
    assert rebased[0].origin == "0"

    implied = indexing.describe(_subs(":1:-1"), None, rank_of=_scalar)
    assert implied[0].kind is indexing.Kind.RANGE
    assert implied[0].lower is None

    positions = indexing.describe(_subs("n:1:-1"), None, rank_of=_scalar)
    assert positions[0].kind is indexing.Kind.RANGE


def test_a_gather_off_a_declared_bound_refuses() -> None:
    """Gathering and re-basing at once has no single form."""
    with pytest.raises(NoRule, match="vector subscript"):
        indexing.describe(_subs("idx"), _dims(("0", "n")), rank_of=lambda _n: 1)


def test_a_keyword_argument_is_not_a_subscript() -> None:
    with pytest.raises(NoRule, match="keyword"):
        indexing.describe(f03.Actual_Arg_Spec_List("dim = 2"), None, rank_of=_scalar)


# --- sequence association ----------------------------------------------------


def test_an_element_at_the_lower_bound_takes_whole_leading_axes() -> None:
    """``call f(arr(1, k))`` onto ``real :: a(:)`` is ``arr[:, k-1]``: the
    callee sees the first column entire. The cheap case, and the common one."""
    subs = list(_subs("1, k").children)
    assoc = indexing.associate(subs, _dims(("1", "n"), ("1", "m")), formal_rank=1)
    assert assoc.whole_leading_axes == 1
    assert [str(t) for t in assoc.trailing] == ["k"]


def test_an_element_anywhere_else_needs_the_general_form() -> None:
    """Starting mid-column, the memory the callee sees is not any slice, so
    the answer is flatten, offset, reshape. Correct everywhere and worth
    avoiding where it is not needed."""
    subs = list(_subs("2, k").children)
    assoc = indexing.associate(subs, _dims(("1", "n"), ("1", "m")), formal_rank=1)
    assert assoc.whole_leading_axes == 0
    assert len(assoc.trailing) == 2


def test_a_non_literal_leading_subscript_needs_the_general_form() -> None:
    subs = list(_subs("i, k").children)
    assert indexing.associate(subs, _dims(("1", "n"), ("1", "m")), 1).whole_leading_axes == 0


def test_a_lower_bound_that_is_not_one_is_still_recognised() -> None:
    subs = list(_subs("0, k").children)
    assoc = indexing.associate(subs, _dims(("0", "n"), ("1", "m")), formal_rank=1)
    assert assoc.whole_leading_axes == 1


def test_a_rank_mismatch_refuses() -> None:
    """A wrong answer here hands the callee a differently shaped view of the
    right memory, which produces numbers rather than an error."""
    subs = list(_subs("1, k").children)
    with pytest.raises(NoRule, match="subscript per dimension"):
        indexing.associate(subs, _dims(("1", "n")), formal_rank=1)


def test_a_formal_of_higher_rank_than_the_actual_refuses() -> None:
    subs = list(_subs("1").children)
    with pytest.raises(NoRule, match="higher rank"):
        indexing.associate(subs, _dims(("1", "n")), formal_rank=2)


def test_an_all_slice_actual_has_no_element_to_start_from() -> None:
    subs = list(_subs(":, :").children)
    with pytest.raises(NoRule, match="no scalar subscript"):
        indexing.associate(subs, _dims(("1", "n"), ("1", "m")), formal_rank=1)
