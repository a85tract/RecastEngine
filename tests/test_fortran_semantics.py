"""Tests for the type and shape questions the frontend answers.

Everything here would have the same answer for a Julia or C++ backend, which
is the test for whether it belongs in the frontend at all. Where a method
refuses, the refusal is the behaviour being tested: guessing scalar produces a
translation that compiles, runs, and broadcasts one element where the Fortran
worked on a whole array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import interface, semantics
from recast.fortran._parse import f03, parse, walk

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


SOURCE = """\
module shapes_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  real(r8), parameter :: pi = 3.14159_r8
  integer :: counter
  real(r8) :: field(10)

  type grid_t
    real(r8), allocatable :: lat(:)
    real(r8) :: dx
  end type grid_t

  interface scale_it
    module procedure scale_scalar, scale_vector
  end interface scale_it
contains
  subroutine drive(v, n, g, label)
    real(r8), intent(inout) :: v(:)
    integer, intent(in) :: n
    type(grid_t), intent(in) :: g
    character(len=8), intent(in) :: label
    real(r8), parameter :: table(4) = [1._r8, 2._r8, 3._r8, 4._r8]
    real(r8) :: s
    integer :: i
    s = table(1) + v(n) + g % dx + pi
    i = n / 2
    call scale_it(v, s)
  end subroutine drive

  subroutine scale_scalar(x, f)
    real(r8), intent(inout) :: x
    real(r8), intent(in) :: f
    x = x * f
  end subroutine scale_scalar

  subroutine scale_vector(x, f)
    real(r8), intent(inout) :: x(:)
    real(r8), intent(in) :: f
    x = x * f
  end subroutine scale_vector

  elemental function twice(x) result(y)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = 2._r8 * x
  end function twice
end module shapes_mod
"""


@pytest.fixture
def sem(tmp_path: Path) -> semantics.Semantics:
    src = tmp_path / "shapes.f90"
    src.write_text(SOURCE)
    return semantics.for_subprogram(interface.extract(src, kind_assumptions=KINDS), "drive")


@pytest.fixture
def body(tmp_path: Path) -> Any:
    src = tmp_path / "shapes.f90"
    src.write_text(SOURCE)
    sub = next(
        s
        for s in walk(parse(src), f03.Subroutine_Subprogram)
        if str(walk(s, f03.Subroutine_Stmt)[0].children[1]).lower() == "drive"
    )
    return next(c for c in sub.children if isinstance(c, f03.Execution_Part))


def _find(node: Any, text: str) -> Any:
    """The first node in ``node`` whose source text matches, spacing ignored."""
    return next(n for n in walk(node) if str(n).replace(" ", "") == text.replace(" ", ""))


def _name(text: str) -> Any:
    """A bare identifier node, for asking about a symbol not in the body."""
    from fparser.two.Fortran2003 import Name

    return Name(text)


# --- declarations ------------------------------------------------------------


def test_a_local_shadows_module_state(sem) -> None:
    assert sem.declaration("s")["dtype"] == "float64"
    assert sem.declaration("counter")["name"] == "counter", "module state is still visible"
    assert sem.declaration("nothing_here") is None


def test_a_local_parameter_array_answers_array_to_one_question_and_scalar_to_the_other(
    sem,
) -> None:
    """The pipeline this came from looked for local parameters in its type and
    array queries and not in its shape query, so a 16-element lookup table is
    an array to ``is_array`` and a scalar to ``rank``.

    Reproduced rather than resolved. These are the answers a bit-exact gate has
    been run against, and nothing in CAM distinguishes them -- every use of
    such a table there is subscripted, which goes through ``is_array``.
    """
    assert sem.is_array("table")
    assert sem.rank(_name("table")) == 0


# --- rank --------------------------------------------------------------------


def test_a_literal_and_a_scalar_are_rank_zero(sem, body) -> None:
    assert sem.rank(_find(body, "pi")) == 0
    assert sem.rank(_find(body, "2")) == 0


def test_subscripting_drops_a_rank(sem, body) -> None:
    assert sem.rank(_find(body, "v(n)")) == 0, "one scalar index into a rank-1 array"


def test_an_assumed_shape_dummy_keeps_its_rank(sem) -> None:
    assert sem.rank(_name("v")) == 1


def test_a_bare_derived_type_component_is_assumed_scalar(sem, body) -> None:
    """Its declared shape is in the type, which this answer does not consult.
    An array component used in a scalar context fails loudly at run time,
    which is the failure mode to prefer over a quiet wrong broadcast."""
    assert sem.rank(_find(body, "g % dx")) == 0


def test_an_unknown_reference_refuses(sem) -> None:
    """Scalar is the answer that would produce working, wrong code."""
    from fparser.two.Fortran2003 import Part_Ref

    with pytest.raises(semantics.Unanalyzable, match="unknown reference"):
        sem.rank(Part_Ref("mystery(1)"))


def test_an_elemental_function_broadcasts_its_actuals(tmp_path: Path) -> None:
    """A non-elemental function returns its declared result rank; an elemental
    one takes the rank of what it is given."""
    src = tmp_path / "shapes.f90"
    src.write_text(SOURCE)
    record = interface.extract(src, kind_assumptions=KINDS)
    sem = semantics.for_subprogram(record, "drive")
    from fparser.two.Fortran2003 import Part_Ref

    assert sem.rank(Part_Ref("twice(v)")) == 1
    assert sem.rank(Part_Ref("twice(pi)")) == 0


def test_a_call_to_the_modules_own_generic_interface_refuses_a_rank(tmp_path: Path) -> None:
    """A *companion* module's generic interfaces get an answer and the
    module's own do not -- an asymmetry the pipeline this came from has, and
    29 sites in five CAM modules where it declines an answer it knows.

    Kept. Those sites went to the agent queue and were written by hand, so
    answering here would translate mechanically what the gated artifact has
    written another way.
    """
    src = tmp_path / "shapes.f90"
    src.write_text(SOURCE)
    sem = semantics.for_subprogram(interface.extract(src, kind_assumptions=KINDS), "drive")
    from fparser.two.Fortran2003 import Part_Ref

    with pytest.raises(semantics.Unanalyzable):
        sem.rank(Part_Ref("scale_it(v, pi)"))
    assert sem.dispatch("scale_it", list(f03.Actual_Arg_Spec_List("v, pi").children)) == (
        "scale_vector"
    ), "dispatch is a separate question and still has an answer"


# --- type --------------------------------------------------------------------


def test_integer_division_needs_both_operands_integer(sem, body) -> None:
    """The one rule where a wrong type answer changes arithmetic rather than
    spelling: Fortran's ``/`` truncates between two integers."""
    assert sem.is_integer(_find(body, "n / 2"))
    assert not sem.is_integer(_find(body, "pi"))


def test_an_unresolvable_expression_is_reported_not_integer(sem) -> None:
    """The safe direction: real division stays visible, truncation does not."""
    from fparser.two.Fortran2003 import Part_Ref

    assert not sem.is_integer(Part_Ref("mystery(1)"))


def test_a_character_dummy_is_recognised(sem) -> None:
    assert sem.is_character(_name("label"))
    assert not sem.is_character(_name("s"))


def test_a_constant_expression_is_literals_and_parameters(sem, body) -> None:
    assert sem.is_constant(_find(body, "pi"))
    assert not sem.is_constant(_name("s")), "a local variable is not constant"


def test_an_array_constructor_does_not_crash_the_constant_test(tmp_path: Path) -> None:
    """fparser puts operators and nodes in the same child list, and several
    node types are unhashable, so testing operator membership without a type
    check raises on an array constructor. Found by running over real source,
    not by reading."""
    src = tmp_path / "ctor.f90"
    src.write_text(
        "module c_mod\n  implicit none\ncontains\n"
        "  subroutine go(out)\n    real, intent(out) :: out(3)\n"
        "    out = [1.0, 2.0, 3.0]\n  end subroutine go\nend module c_mod\n"
    )
    sem = semantics.for_subprogram(interface.extract(src, kind_assumptions=KINDS), "go")
    sub = walk(parse(src), f03.Subroutine_Subprogram)[0]
    part = next(c for c in sub.children if isinstance(c, f03.Execution_Part))
    for node in walk(part):
        sem.is_constant(node)
        sem.is_integer(node)


def test_an_integer_literal_is_read_through_parens_and_a_sign(sem, body) -> None:
    from fparser.two.Fortran2003 import Level_2_Unary_Expr

    assert sem.integer_literal(_find(body, "2")) == 2
    assert sem.integer_literal(Level_2_Unary_Expr("- 3")) == -3
    assert sem.integer_literal(_name("n")) is None


# --- derived types -----------------------------------------------------------


def test_a_component_is_found_through_its_declared_type(sem) -> None:
    assert sem.derived_type_of("g") == "grid_t"
    assert sem.component("g", "dx")["dtype"] == "float64"
    assert sem.component("g", "lat")["allocatable"] is True
    assert sem.component("g", "nope") is None
    assert sem.derived_type_of("s") is None


# --- dispatch ----------------------------------------------------------------


def test_a_generic_call_resolves_on_argument_rank(sem, body) -> None:
    call = walk(body, f03.Call_Stmt)[0]
    actuals = list(call.children[1].children)
    assert sem.dispatch("scale_it", actuals) == "scale_vector"


def test_a_generic_call_with_a_scalar_picks_the_scalar_overload(sem) -> None:
    from fparser.two.Fortran2003 import Actual_Arg_Spec_List

    actuals = list(Actual_Arg_Spec_List("s, pi").children)
    assert sem.dispatch("scale_it", actuals) == "scale_scalar"


def test_a_generic_call_that_matches_nothing_refuses(sem) -> None:
    """Refusing is the point. Two implementations of this existed and the other
    one scored the candidates and took the best -- and an overload picked
    wrongly changes which arguments are written, which nothing downstream
    re-checks."""
    from fparser.two.Fortran2003 import Actual_Arg_Spec_List

    actuals = list(Actual_Arg_Spec_List("s").children)
    with pytest.raises(semantics.AmbiguousDispatch, match="no match"):
        sem.dispatch("scale_it", actuals)


def test_a_name_that_is_not_generic_refuses(sem) -> None:
    with pytest.raises(semantics.AmbiguousDispatch, match="not a generic"):
        sem.dispatch("scale_scalar", [])
