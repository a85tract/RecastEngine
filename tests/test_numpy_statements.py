"""Tests for the statement layer.

The differential against the pipeline says the accepting paths match: 998
top-level statements across six CAM modules, 4,606 emitted lines, byte
identical. What the corpus cannot say is whether the refusals are right --
its only refusing statements are five goto shapes -- and the refusals are
where the plain Python lookalike would run and return wrong numbers. So the
accepting tests here are a sketch, and the refusing ones are the point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import constants, interface
from recast.fortran._parse import f03, f08, parse, walk
from recast.fortran.semantics import for_subprogram
from recast.transform.numpy.expressions import Expressions, Remote
from recast.transform.numpy.names import for_subprogram as names_for
from recast.transform.numpy.statements import REFUSED, Statements
from recast.transform.profiles import PROFILES

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


SOURCE = """\
module emit_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  real(r8), parameter :: pi = 3.14159_r8
  real(r8) :: state(10)

  type grid_t
    real(r8) :: cells(5)
    real(r8) :: dx
  end type grid_t

  interface scale_it
    module procedure scale_scalar, scale_vector
  end interface scale_it

contains

  subroutine assigns(a, g, h, s)
    real(r8), intent(inout) :: a(10)
    type(grid_t), intent(inout) :: g, h
    real(r8), intent(inout) :: s
    s = 0.5_r8
    a = 1.0_r8
    g = h
    g % cells = 0.0_r8
    a(2) = s
  end subroutine assigns

  subroutine masked(a, c)
    real(r8), intent(inout) :: a(10), c(10)
    where (a > 0.0_r8) a = 0.0_r8
    where (a > 0.0_r8)
      c = 1.0_r8
      where (c > 0.5_r8)
        c = 0.5_r8
      end where
    elsewhere
      c = 0.0_r8
    end where
    where (a > 0.0_r8)
      c = 1.0_r8
    elsewhere (a < 0.0_r8)
      c = 0.0_r8
    end where
  end subroutine masked

  subroutine loops(a, n, j, s)
    real(r8), intent(inout) :: a(10)
    integer, intent(in) :: n, j
    real(r8), intent(inout) :: s
    integer :: i
    do i = 1, n
      if (a(i) > 0.0_r8) cycle
      a(i) = 0.0_r8
    end do
    do i = n, 1, -1
      a(i) = 0.0_r8
    end do
    do i = 1, n, j
      a(i) = 0.0_r8
    end do
    do while (s < 1.0_r8)
      s = s + 0.5_r8
    end do
    do i = 1, n
      if (a(i) > 0.0_r8) go to 20
      a(i) = 1.0_r8
    end do
20  continue
    go to 30
    s = 0.0_r8
30  continue
  end subroutine loops

  subroutine alloc(n, i)
    integer, intent(in) :: n, i
    real(r8), allocatable :: buf(:)
    integer, allocatable :: idx(:)
    real(r8), allocatable :: off(:)
    allocate(buf(n))
    allocate(idx(n))
    allocate(off(0:n))
    off(i) = 0.0_r8
    deallocate(buf, idx)
    allocate(off(2:n))
  end subroutine alloc

  subroutine calls(a, s, t, c, n, j, w, flat)
    real(r8), intent(inout) :: a(10), c(10)
    real(r8), intent(inout) :: s, t
    integer, intent(in) :: n, j
    real(r8), intent(in) :: w(4, 3)
    real(r8), intent(in) :: flat(8)
    call scale_it(a, s)
    call helper(s, c(1))
    call helper(s, c(1), extra=t)
    call helper(s)
    call e_scale(a, s)
    call outfld('X', a)
    call mystery(s)
    call ext_sub(s, c)
    call vec2(w(1, j))
    call consume(n, j, flat)
  end subroutine calls

  subroutine initialised(x)
    real(r8), intent(out) :: x
    real(r8) :: table(4)
    integer :: counter
    data counter /0/
    data table /3*1.0_r8, 0.0_r8/
    x = table(1) + counter
  end subroutine initialised

  subroutine sections(a, b, n)
    real(r8), intent(inout) :: a(10), b(0:9)
    integer, intent(in) :: n
    integer :: ks(3)
    a(n:1:-1) = 0.0_r8
    a(:n:-1) = 1.0_r8
    b(9:0:-1) = 2.0_r8
    ks = (/ (2*n, n = 1, 3) /)
  end subroutine sections

  subroutine backward(n, s)
    integer, intent(in) :: n
    real(r8), intent(inout) :: s
    integer :: i
    i = 0
40  continue
    i = i + 1
    s = s + 1.0_r8
    if (i < n) go to 40
  end subroutine backward

  subroutine cycling(n, a)
    integer, intent(in) :: n
    real(r8), intent(inout) :: a(:)
    integer :: i
    do 50 i = 1, n
      if (a(i) < 0.0_r8) go to 50
      a(i) = a(i) * 2.0_r8
50  continue
  end subroutine cycling

  subroutine constructs(a, n, s)
    real(r8), intent(inout) :: a(:)
    integer, intent(in) :: n
    real(r8), intent(inout) :: s
    complex(r8) :: z
    integer :: i
    z = (1.0_r8, -2.5_r8)
    i = 0
    do
      i = i + 1
      if (i >= n) exit
    end do
    associate (scaled => 2.0_r8 * s)
      a(1) = scaled
    end associate
    block
      integer :: k
      real(r8) :: acc = 1.5_r8
      k = 2
      a(k) = acc
    end block
  end subroutine constructs

  subroutine handles(s, n, g)
    real(r8), intent(inout) :: s
    integer, intent(in) :: n
    type(grid_t), intent(inout) :: g
    integer :: idx, other
    call register(idx)
    if (idx > 0) s = 0.0_r8
    if (idx >= 1) s = 1.0_r8
    if (idx > n) s = 2.0_r8
    other = lookup('f')
    if (other > 0) s = 3.0_r8
    s = g % pack(s)
  end subroutine handles

  subroutine io(s, a, ios)
    real(r8), intent(inout) :: s
    real(r8), intent(inout) :: a(10)
    integer, intent(out) :: ios
    character(len=32) :: line
    write(*,*) s
    write(line,*) s, a(1)
    write(11,*,iostat=ios) s
    stop 'boom'
    return
  end subroutine io

  subroutine switch(s, n, label, buf, idx)
    real(r8), intent(inout) :: s
    integer, intent(in) :: n
    character(len=8), intent(in) :: label
    real(r8), allocatable, intent(inout) :: buf(:)
    integer, allocatable, intent(inout) :: idx(:)
    if (s > 0.0_r8) then
      s = 0.0_r8
    else if (s < 0.0_r8) then
      s = 1.0_r8
    else
      s = 0.5_r8
    end if
    if (s > 0.0_r8) s = 0.0_r8
    if (s > 0.0_r8) deallocate(buf, idx)
    select case (label)
    case ('x')
      s = 0.0_r8
    case default
      s = 1.0_r8
    end select
    select case (n)
    case (1:2)
      s = 0.0_r8
    end select
  end subroutine switch

  subroutine framework(s, name_out)
    real(r8), intent(inout) :: s
    character(len=8), intent(in) :: name_out
    if (hist_fld_active(name_out)) s = 0.0_r8
    if (hist_fld_active('X')) s = 0.0_r8
  end subroutine framework

  subroutine stfunc(s, t)
    real(r8), intent(inout) :: s
    real(r8), intent(in) :: t
    real(r8) :: half
    real(r8) :: u
    half(u) = u * 0.5_r8
    s = half(t)
  end subroutine stfunc

  subroutine helper(x, y, extra)
    real(r8), intent(in) :: x
    real(r8), intent(out) :: y
    real(r8), intent(out), optional :: extra
    y = x
    if (present(extra)) extra = x
  end subroutine helper

  elemental subroutine e_scale(x, f)
    real(r8), intent(inout) :: x
    real(r8), intent(in) :: f
    x = x * f
  end subroutine e_scale

  subroutine vec2(x)
    real(r8), intent(in) :: x(4)
    state(1) = x(1)
  end subroutine vec2

  subroutine consume(m, k, x)
    integer, intent(in) :: m, k
    real(r8), intent(in) :: x(m, k)
    state(1) = x(1, 1)
  end subroutine consume

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

  subroutine based(v, n)
    use grid_mod, only: lo
    real(r8), intent(inout) :: v(1-lo:n)
    integer, intent(in) :: n
    v(1-lo:n) = 0.0_r8
  end subroutine based

  subroutine seeded()
    integer :: tab(4), grid(2,3), i, j
    data (tab(i), i=1,4) /10, 20, 30, 40/
    data ((grid(i,j), i=1,2), j=1,3) /1,2,3,4,5,6/
    j = tab(1)
  end subroutine seeded

  subroutine io_edges(u, ok)
    integer, intent(in) :: u
    logical, intent(out) :: ok
    rewind(u)
    backspace(u)
    inquire(unit=u, opened=ok)
    error stop 'nothing to do'
  end subroutine io_edges
end module emit_mod
"""

COMPANION = """\
module sibling_mod
  use precision_mod, only: r8 => wp_r8
  implicit none

  interface cscale
    module procedure cscale_scalar, cscale_vector
  end interface cscale

contains

  subroutine cscale_scalar(x, f)
    real(r8), intent(inout) :: x
    real(r8), intent(in) :: f
    x = x * f
  end subroutine cscale_scalar

  subroutine cscale_vector(x, f)
    real(r8), intent(inout) :: x(:)
    real(r8), intent(in) :: f
    x = x * f
  end subroutine cscale_vector

  function rise(x) result(y)
    real(r8), intent(in) :: x
    real(r8) :: y
    y = x + 1.0_r8
  end function rise
end module sibling_mod
"""

CALLER = """\
module caller_mod
  use precision_mod, only: r8 => wp_r8
  use sibling_mod, only: cscale, rise
  implicit none
contains
  subroutine drive(a, s)
    real(r8), intent(inout) :: a(10)
    real(r8), intent(inout) :: s
    call cscale(a, s)
    s = rise(s)
  end subroutine drive
end module caller_mod
"""


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("emit")
    paths = {}
    for name, text in (("emit_mod", SOURCE), ("sibling_mod", COMPANION), ("caller_mod", CALLER)):
        paths[name] = root / f"{name}.f90"
        paths[name].write_text(text)
    return paths


def build(
    src: Path,
    name: str,
    *,
    companions: tuple[dict[str, Any], ...] = (),
    remotes: dict[str, Remote] | None = None,
    externals: dict[str, dict[str, Any]] | None = None,
    stubs: dict[str, str] | None = None,
    function_stubs: dict[str, str] | None = None,
    call_transforms: dict[str, Any] | None = None,
    function_transforms: dict[str, Any] | None = None,
    handle_producers: frozenset[str] = frozenset(),
    type_bound: frozenset[str] = frozenset(),
) -> tuple[Statements, list[Any]]:
    """A ``Statements`` for one subprogram, plus its executable nodes."""
    record = interface.extract(src, kind_assumptions=KINDS)
    semantics = for_subprogram(record, name, companions=companions)
    names = names_for(semantics, constants.extract(src))
    expressions = Expressions(
        semantics,
        names,
        PROFILES["ifx"],
        externals=externals or {},
        remotes=remotes or {},
        stubs=function_stubs or {},
        function_transforms=function_transforms or {},
        handle_producers=handle_producers,
        type_bound=type_bound,
    )
    statements = Statements(
        semantics,
        names,
        expressions,
        externals=externals or {},
        stubs=stubs or {},
        call_transforms=call_transforms or {},
    )
    subprogram = next(
        sub
        for sub in walk(parse(src), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == name
    )
    statements.scan(subprogram)
    execution = next(c for c in subprogram.children if isinstance(c, f03.Execution_Part))
    return statements, list(execution.children)


def pick(nodes: list[Any], kind: type, ordinal: int = 0) -> Any:
    return [n for n in nodes if isinstance(n, kind)][ordinal]


# --- assignment --------------------------------------------------------------


def test_a_whole_array_assignment_fills_the_buffer(sources: dict[str, Path]) -> None:
    """``a = 1`` writes every element of ``a``'s storage; a plain Python
    assignment would rebind the name and leave the argument untouched."""
    statements, nodes = build(sources["emit_mod"], "assigns")
    assert statements.render(nodes[0], 1) == ["    s = 0.5"]
    assert statements.render(nodes[1], 1) == ["    a[...] = 1.0"]


def test_a_whole_derived_type_assignment_is_a_deep_copy(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "assigns")
    assert statements.render(nodes[2], 1) == ["    g = _copy_derived(h)"]


def test_a_whole_array_component_target_also_fills(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "assigns")
    assert statements.render(nodes[3], 1) == ["    g.cells[...] = 0.0"]


# --- WHERE -------------------------------------------------------------------


def test_a_where_statement_gathers_through_its_mask(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "masked")
    where = pick(nodes, f03.Where_Stmt)
    assert statements.render(where, 1) == [
        "    _wm = (a > 0.0)",
        "    a[...][_wm] = 0.0",
    ]


def test_a_nested_where_ands_the_outer_mask_in(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "masked")
    construct = pick(nodes, f03.Where_Construct)
    lines = statements.render(construct, 1)
    assert "    _wm2 = (c > 0.5)" in lines
    assert "    c[...][_wm & _wm2] = 0.5" in lines
    assert "    c[...][(~_wm)] = 0.0" in lines  # the ELSEWHERE branch


def test_a_masked_elsewhere_takes_from_what_is_left(sources: dict[str, Path]) -> None:
    """ELSEWHERE with its own condition selects from the elements no earlier
    branch claimed, not from the whole array: the running remainder is
    narrowed by each masked branch in turn, so a later one cannot reach an
    element an earlier one already assigned."""
    statements, nodes = build(sources["emit_mod"], "masked")
    assert statements.render(pick(nodes, f03.Where_Construct, 1), 1) == [
        "    _wm = (a > 0.0)",
        "    _wn = (~_wm)",
        "    c[...][_wm] = 1.0",
        "    _we0_1 = (_wn & (a < 0.0))",
        "    _wn = (_wn & (~(a < 0.0)))",
        "    c[...][_we0_1] = 0.0",
    ]


def test_data_becomes_assignments_with_its_repeats_written_out(
    sources: dict[str, Path],
) -> None:
    """DATA is a static initialisation in the specification part, so its
    assignments belong before any statement can read the names; ``3*1.5``
    is three elements, not a multiplication."""
    from recast.fortran._parse import parse as parse_source

    statements, _ = build(sources["emit_mod"], "initialised")
    subprogram = next(
        sub
        for sub in walk(
            parse_source(sources["emit_mod"]),
            (f03.Subroutine_Subprogram, f03.Function_Subprogram),
        )
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower()
        == "initialised"
    )
    data = walk(subprogram, f03.Data_Stmt)
    assert statements.data_statement(data[0], 1) == ["    counter = 0"]
    assert statements.data_statement(data[1], 1) == [
        "    table[:] = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)"
    ]


def test_a_descending_section_carries_its_declared_lower_bound(
    sources: dict[str, Path],
) -> None:
    """The stop edge underflows at the first element, and an array declared
    from 0 shifts by 0, not by 1 -- so the runtime is handed the bound and
    either edge may be left implied."""
    statements, nodes = build(sources["emit_mod"], "sections")
    rendered = [statements.render(node, 1)[0] for node in nodes[:3]]
    assert rendered[0] == "    a[_f_rstep_lb(n, 1, (-1), 1)] = 0.0"
    assert rendered[1] == "    a[_f_rstep_lb(None, n, (-1), 1)] = 1.0"
    assert rendered[2] == "    b[_f_rstep_lb(I_9, 0, (-1), 0)] = F_2P0"


def test_an_implied_do_in_an_array_constructor_is_a_comprehension(
    sources: dict[str, Path],
) -> None:
    statements, nodes = build(sources["emit_mod"], "sections")
    assert statements.render(nodes[3], 1) == [
        "    ks[...] = np.array([[(2 * n) for n in range(1, I_3 + 1)]])"
    ]


# --- loops and gotos ---------------------------------------------------------


def test_do_bounds_shift_by_the_sign_of_the_step(sources: dict[str, Path]) -> None:
    """Fortran's do reaches its last element; the exclusive stop edge moves
    the other way when counting down, and a variable step only knows its
    direction at run time."""
    statements, nodes = build(sources["emit_mod"], "loops")
    do = pick(nodes, f03.Block_Nonlabel_Do_Construct)
    assert statements.render(do, 1)[0] == "    for i in range(1, n + 1):"
    down = pick(nodes, f03.Block_Nonlabel_Do_Construct, 1)
    assert statements.render(down, 1)[0] == "    for i in range(n, 1 - 1, (-1)):"
    variable = pick(nodes, f03.Block_Nonlabel_Do_Construct, 2)
    assert (
        statements.render(variable, 1)[0]
        == "    for i in range(1, (n) + (1 if (j) > 0 else -1), j):"
    )


def test_cycle_and_a_do_while_translate_directly(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "loops")
    do = pick(nodes, f03.Block_Nonlabel_Do_Construct)
    assert "            continue" in statements.render(do, 1)
    while_ = pick(nodes, f03.Block_Nonlabel_Do_Construct, 3)
    assert statements.render(while_, 1)[0] == "    while (s < 1.0):"


def test_a_goto_to_the_label_after_end_do_is_exit(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "loops")
    do = pick(nodes, f03.Block_Nonlabel_Do_Construct, 4)
    lines = statements.render(do, 1)
    assert "            break  # goto 20 == exit (label follows end do)" in lines


def test_a_forward_goto_becomes_a_labelled_exception_region(sources: dict[str, Path]) -> None:
    """The jump can come from any nesting depth, which no break can express;
    the try/except region can."""
    statements, nodes = build(sources["emit_mod"], "loops")
    region_start = next(
        at for at, n in enumerate(nodes) if isinstance(n, f03.Goto_Stmt) and "30" in str(n)
    )
    lines = statements.sequence(nodes[region_start:], 1)
    assert lines[0] == "    try:  # forward-goto region (label 30)"
    assert "        raise _FGoto('30')  # goto 30" in lines
    assert "    except _FGoto as _g:" in lines


def test_a_backward_goto_becomes_a_loop_that_restarts_at_its_label(
    sources: dict[str, Path],
) -> None:
    """A label with a goto to it further down is a loop: everything from the
    label to the last such goto runs again, and the exception carries the
    jump out of whatever depth raised it."""
    statements, nodes = build(sources["emit_mod"], "backward")
    lines = statements.sequence(nodes, 1)
    assert "    while True:  # backward-goto region (label 40)" in lines
    assert "            break  # natural exit" in lines
    assert "            pass  # 40 (loop restart)" in lines
    assert any("raise _FGoto('40')" in line for line in lines)


def test_a_goto_to_a_labeled_do_terminator_is_a_cycle(sources: dict[str, Path]) -> None:
    """`do 50 ... / 50 continue`: a goto to the terminator from inside the
    body skips the rest of the iteration, which is `continue`, not a break
    and not a region."""
    statements, nodes = build(sources["emit_mod"], "cycling")
    lines = statements.render(pick(nodes, f03.Block_Label_Do_Construct, 0), 1)
    assert any("continue  # goto 50 == cycle (labeled-DO terminator)" in line for line in lines)


def test_a_goto_with_no_structuring_pattern_is_refused(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "loops")
    goto = next(n for n in nodes if isinstance(n, f03.Goto_Stmt) and "30" in str(n))
    with pytest.raises(REFUSED):
        statements.render(goto, 1)  # outside its region, nothing catches it


# --- allocation --------------------------------------------------------------


def test_allocate_takes_the_declared_dtype(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "alloc")
    assert statements.render(nodes[0], 1) == ["    buf = np.empty((n,), dtype=np.float64)"]
    assert statements.render(nodes[1], 1) == ["    idx = np.empty((n,), dtype=np.int32)"]


def test_an_allocated_lower_bound_shifts_later_subscripts(sources: dict[str, Path]) -> None:
    """``allocate(off(0:n))`` re-bases the array; ``off(i)`` afterwards must
    shift by 0, not by the 1 its declaration would suggest."""
    statements, nodes = build(sources["emit_mod"], "alloc")
    assert statements.render(nodes[2], 1) == [
        "    off = np.empty(((n) - (0) + 1,), dtype=np.float64)"
    ]
    assert statements.render(nodes[3], 1) == ["    off[(i) - (0)] = 0.0"]


def test_conflicting_allocate_lower_bounds_are_refused(sources: dict[str, Path]) -> None:
    """One name, two origins: every subscript after the second allocate would
    shift by whichever one was recorded, and half would be wrong."""
    statements, nodes = build(sources["emit_mod"], "alloc")
    statements.render(nodes[2], 1)
    with pytest.raises(REFUSED):
        statements.render(nodes[5], 1)


def test_deallocate_returns_the_names_to_none(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "alloc")
    assert statements.render(nodes[4], 1) == ["    buf = None", "    idx = None"]


# --- constructs the pipeline had rules for and this backend did not ------------


def test_a_complex_literal_is_written_as_a_python_complex(sources: dict[str, Path]) -> None:
    """Not through the literal table: the zero-literal rule hoists reals, not
    pairs of them, and a kind suffix is not part of the value."""
    statements, nodes = build(sources["emit_mod"], "constructs")
    assert statements.render(pick(nodes, f03.Assignment_Stmt, 0), 1) == [
        "    z = complex(1.0, -2.5)"
    ]


def test_a_do_with_no_control_is_an_unbounded_loop(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "constructs")
    rendered = statements.render(pick(nodes, f03.Block_Nonlabel_Do_Construct, 0), 1)
    assert rendered[0] == "    while True:"
    assert "            break" in rendered


def test_associate_binds_its_names_then_runs_the_body(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "constructs")
    assert statements.render(pick(nodes, f03.Associate_Construct, 0), 1) == [
        "    scaled = (F_2P0 * s)",
        "    a[0] = scaled",
    ]


def test_a_block_construct_declares_its_locals_and_runs_its_body(
    sources: dict[str, Path],
) -> None:
    """Python has no block scope, so the declarations become locals at the
    enclosing indent -- initialised, because Fortran leaves them undefined."""
    statements, nodes = build(sources["emit_mod"], "constructs")
    assert statements.render(pick(nodes, f08.Block_Construct, 0), 1) == [
        "    k = 0",
        "    acc = F_1P5",
        "    k = 2",
        "    a[k - 1] = acc",
    ]


# --- calls -------------------------------------------------------------------


def test_a_whole_array_out_intent_is_copied_into_the_buffer(sources: dict[str, Path]) -> None:
    """An inout array actual appears on both sides, and the target is the
    buffer -- the caller may be aliasing it."""
    statements, nodes = build(sources["emit_mod"], "calls")
    generic = pick(nodes, f03.Call_Stmt)
    assert statements.render(generic, 1) == ["    _f_copy_out(a, scale_vector(a, s))"]


def test_an_unsupplied_optional_out_still_occupies_the_tuple(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 1), 1) == ["    c[0], _ = helper(s)"]
    assert statements.render(pick(nodes, f03.Call_Stmt, 2), 1) == [
        "    c[0], t = helper(s, want_extra=True)"
    ]


def test_a_missing_required_actual_is_refused(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "calls")
    with pytest.raises(REFUSED):
        statements.render(pick(nodes, f03.Call_Stmt, 3), 1)


def test_an_elemental_call_over_an_array_actual_broadcasts(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 4), 1) == [
        "    _f_copy_out(a, _f_ecall(e_scale, a, s))"
    ]


def test_a_stubbed_framework_call_emits_its_stub(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "calls", stubs={"outfld": "pass"})
    assert statements.render(pick(nodes, f03.Call_Stmt, 5), 1) == [
        "    pass  # outfld (infra stub)"
    ]


def test_a_call_transform_answers_before_anything_else_is_consulted(
    sources: dict[str, Path],
) -> None:
    """A call whose meaning is a framework's is neither translatable nor
    stubbable: the answer depends on the call's own arguments. The domain
    package supplies a callable, and it is asked first -- before the stub
    table, before this module's own procedures."""

    def transform(site: Any) -> list[str]:
        return [f"{site.pad}{site.value(0)} = scaled_by({site.value(1)})  # scale_it"]

    statements, nodes = build(
        sources["emit_mod"],
        "calls",
        stubs={"scale_it": "pass"},
        call_transforms={"scale_it": transform},
    )
    assert statements.render(pick(nodes, f03.Call_Stmt, 0), 1) == [
        "    a = scaled_by(s)  # scale_it"
    ]


def test_a_call_transform_may_refuse_like_any_rule(sources: dict[str, Path]) -> None:
    def transform(site: Any) -> list[str]:
        raise REFUSED[0](f"{site.name} needs an argument it was not given")

    statements, nodes = build(sources["emit_mod"], "calls", call_transforms={"scale_it": transform})
    with pytest.raises(REFUSED):
        statements.render(pick(nodes, f03.Call_Stmt, 0), 1)


def test_a_function_transform_answers_a_reference_the_stub_table_cannot(
    sources: dict[str, Path],
) -> None:
    """The reference-side twin: a fixed string cannot answer a query whose
    answer depends on what was passed."""
    statements, nodes = build(
        sources["emit_mod"],
        "framework",
        function_transforms={"hist_fld_active": lambda args: f"_active({args[0]})"},
    )
    assert statements.render(nodes[1], 1) == ["    if _active('X'):", "        s = 0.0"]


def test_a_handle_answers_a_numeric_test_as_a_presence_question(
    sources: dict[str, Path],
) -> None:
    """A framework that hands out registrations gives Fortran an integer
    index, tested with ``idx > 0``. A transform that assigns something else
    -- a dictionary key -- says so, and the test comes out as the question
    it is rather than as arithmetic on a string."""

    def register(site: Any) -> list[str]:
        site.holds_handle(site.value(0))
        return [f"{site.pad}{site.value(0)} = 'field'"]

    statements, nodes = build(
        sources["emit_mod"], "handles", call_transforms={"register": register}
    )
    assert statements.render(nodes[0], 1) == ["    idx = 'field'"]
    assert statements.render(nodes[1], 1) == ["    if bool(idx):", "        s = 0.0"]
    assert statements.render(nodes[2], 1) == ["    if bool(idx):", "        s = 1.0"]
    assert statements.render(nodes[3], 1) == ["    if (idx > n):", "        s = F_2P0"]


def test_a_name_assigned_from_a_handle_producer_is_one_too(
    sources: dict[str, Path],
) -> None:
    statements, nodes = build(
        sources["emit_mod"], "handles", handle_producers=frozenset({"lookup"})
    )
    assert statements.render(nodes[4], 1) == ["    other = lookup('f')"]
    assert statements.render(nodes[5], 1) == ["    if bool(other):", "        s = F_3P0"]


def test_a_type_bound_procedure_is_a_call_not_a_subscript(
    sources: dict[str, Path],
) -> None:
    """Only the domain package knows which components are procedures: the
    type is declared somewhere this file never sees."""
    statements, nodes = build(sources["emit_mod"], "handles", type_bound=frozenset({"pack"}))
    assert statements.render(nodes[6], 1) == ["    s = g.pack(s)"]


def test_a_stub_wins_over_a_registered_external_of_the_same_name(sources: dict[str, Path]) -> None:
    """The pipeline asks its stub table before anything else, so a framework
    call that is both stubbed and registered as an external is the stub."""
    statements, nodes = build(
        sources["emit_mod"],
        "calls",
        externals={"ext_sub": {"kind": "subroutine", "out_positions": [1]}},
        stubs={"ext_sub": "pass"},
    )
    assert statements.render(pick(nodes, f03.Call_Stmt, 7), 1) == [
        "    pass  # ext_sub (infra stub)"
    ]


def test_a_call_to_nothing_known_is_refused(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "calls")
    with pytest.raises(REFUSED):
        statements.render(pick(nodes, f03.Call_Stmt, 6), 1)


def test_a_registered_external_reads_its_out_positions(sources: dict[str, Path]) -> None:
    statements, nodes = build(
        sources["emit_mod"],
        "calls",
        externals={"ext_sub": {"kind": "subroutine", "out_positions": [1]}},
    )
    assert statements.render(pick(nodes, f03.Call_Stmt, 7), 1) == ["    c[...] = _ext.ext_sub(s)"]


def test_sequence_association_takes_leading_axes_whole(sources: dict[str, Path]) -> None:
    """``w(1, j)`` to a rank-1 formal is the whole column at ``j``, and a
    rank-1 actual to a rank-2 formal refills it in column-major order."""
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 8), 1) == ["    vec2(w[:, j - 1])"]
    assert statements.render(pick(nodes, f03.Call_Stmt, 9), 1) == [
        "    consume(n, j, np.reshape(flat, (n, j,), order='F'))"
    ]


# --- companions --------------------------------------------------------------


def test_a_companion_generic_dispatches_to_its_specific(sources: dict[str, Path]) -> None:
    """The generic lives in a sibling translated module; the overload is
    picked here and the call goes through the sibling's alias."""
    sibling = interface.extract(sources["sibling_mod"], kind_assumptions=KINDS)
    remotes = {s["name"]: Remote("_sib", s["name"]) for s in sibling["subprograms"]}
    statements, nodes = build(
        sources["caller_mod"], "drive", companions=(sibling,), remotes=remotes
    )
    assert statements.render(pick(nodes, f03.Call_Stmt), 1) == [
        "    _f_copy_out(a, _sib.cscale_vector(a, s))"
    ]
    assert statements.render(nodes[1], 1) == ["    s = _sib.rise(s)"]


# --- I/O and control ---------------------------------------------------------


def test_writes_split_on_whether_dataflow_survives(sources: dict[str, Path]) -> None:
    """A log write carries nothing a differential can compare; an internal
    write assigns to a character variable, which is real dataflow."""
    statements, nodes = build(sources["emit_mod"], "io")
    assert statements.render(nodes[0], 1) == ["    pass  # write(*,...) log — no dataflow"]
    assert statements.render(nodes[1], 1) == ["    line = _f_list_write(s, a[0])"]
    with pytest.raises(REFUSED):
        statements.render(nodes[2], 1)  # iostat= is control flow, not logging


def test_return_carries_the_out_arguments(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "io")
    return_ = pick(nodes, f03.Return_Stmt)
    assert statements.render(return_, 1) == ["    return s, a, ios"]


def test_an_if_construct_keeps_its_branches(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "switch")
    lines = statements.render(pick(nodes, f03.If_Construct), 1)
    assert lines[0] == "    if (s > 0.0):"
    assert "    elif (s < 0.0):" in lines
    assert "    else:" in lines


def test_a_single_line_if_indents_however_many_lines_its_action_takes(
    sources: dict[str, Path],
) -> None:
    """``if (c) action``: the action may need several lines of its own -- a
    masked assignment, a stubbed call -- and they all belong under the
    branch."""
    statements, nodes = build(sources["emit_mod"], "switch")
    assert statements.render(pick(nodes, f03.If_Stmt), 1) == [
        "    if (s > 0.0):",
        "        s = 0.0",
    ]
    lines = statements.render(pick(nodes, f03.If_Stmt, 1), 1)
    assert lines[0].startswith("    if ")
    assert len(lines) > 2
    assert all(line.startswith("        ") for line in lines[1:])


def test_a_character_case_compares_with_blank_padding(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "switch")
    lines = statements.render(pick(nodes, f03.Case_Construct), 1)
    assert lines[0] == "    if _fstr_eq(label, 'x'):"
    assert "    else:" in lines


def test_a_case_value_range_slips_past_the_refusal(sources: dict[str, Path]) -> None:
    """Reproduced, not endorsed. The pipeline means to refuse ``case (1:2)``,
    but its check looks at the selector's children, which hold a
    ``Case_Value_Range_List`` rather than a bare range -- so the refusal never
    fires and the range's endpoints come out as equality tests. Right for
    ``(1:2)`` by luck, wrong for any wider range. Kept identical because the
    pipeline's output is the one the bit-exact gates have run against; this
    test is the record that the behaviour is inherited rather than intended."""
    statements, nodes = build(sources["emit_mod"], "switch")
    lines = statements.render(pick(nodes, f03.Case_Construct, 1), 1)
    assert lines[0] == "    if (n == 1) or (n == 2):"


def test_a_stub_answers_only_where_the_pipeline_answers(sources: dict[str, Path]) -> None:
    """``hist_fld_active(name_out)`` parses as a plain reference, and the stub
    table is not consulted for that shape -- the pipeline consults it only for
    the structure-constructor parse, which the same call over a character
    literal produces. The plain reference falls through to being read as a
    subscript, which is the pipeline's answer for a name it has no
    declaration for."""
    statements, nodes = build(
        sources["emit_mod"], "framework", function_stubs={"hist_fld_active": "False"}
    )
    assert statements.render(nodes[0], 1) == [
        "    if hist_fld_active[name_out - 1]:",
        "        s = 0.0",
    ]
    assert statements.render(nodes[1], 1) == ["    if False:", "        s = 0.0"]


# --- statement functions -----------------------------------------------------


def test_a_statement_function_defines_and_then_applies(sources: dict[str, Path]) -> None:
    """Before the definition renders, ``half(t)`` is an unknown reference;
    after it, a call. The registration is what the later statements read."""
    statements, nodes = build(sources["emit_mod"], "stfunc")
    assert statements.render(nodes[0], 1) == [
        "    def half(u):  # statement function",
        "        return (u * 0.5)",
    ]
    assert statements.render(nodes[1], 1) == ["    s = half(t)"]


# --- I/O edges ---------------------------------------------------------------


def test_error_stop_ends_the_program_the_way_stop_does(sources: dict[str, Path]) -> None:
    """ERROR STOP differs from STOP only in the exit status a compiler is
    asked to produce, and nothing downstream of a SystemExit compares
    anything."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    node = pick(nodes, f08.Error_Stop_Stmt)
    # The stop code keeps its Fortran quotes inside the Python string, which
    # is what the STOP rule has always done and what the pipeline does.
    assert statements.render(node, 1) == ["    raise SystemExit(\"'nothing to do'\")  # ERROR STOP"]


def test_file_positioning_carries_no_dataflow(sources: dict[str, Path]) -> None:
    """REWIND and BACKSPACE move a file pointer and write no variable, so
    there is nothing for a read/write gate to compare and nothing lost by
    dropping them -- the same reading OPEN and CLOSE already get."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    assert statements.render(pick(nodes, f03.Rewind_Stmt), 1) == ["    pass  # REWIND (I/O stub)"]
    assert statements.render(pick(nodes, f03.Backspace_Stmt), 1) == [
        "    pass  # BACKSPACE (I/O stub)"
    ]


def test_inquire_is_refused_because_its_specifiers_are_writes(sources: dict[str, Path]) -> None:
    """The one I/O statement in this group that is not a stub, and where the
    pipeline this was migrated from differs: it renders INQUIRE as ``pass``.
    ``opened=ok`` writes ``ok``, and a ``pass`` leaves it at whatever it held
    while the read/write gate is told nothing happened."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    with pytest.raises(REFUSED, match="OPENED="):
        statements.render(pick(nodes, f03.Inquire_Stmt), 1)


def test_a_data_implied_do_is_expanded_in_definition_order(sources: dict[str, Path]) -> None:
    """DATA pairs objects with values positionally, and an implied-do stands
    for as many objects as it has iterations -- so the list has to be
    flattened before anything can be paired with it. A run that is contiguous
    in the last dimension collapses to one slice, which is what a lookup table
    of four hundred elements needs to stay readable."""
    statements, _ = build(sources["emit_mod"], "seeded")
    node = next(
        d
        for d in walk(_specification_of(sources["emit_mod"], "seeded"), f03.Data_Stmt)
        if "tab" in str(d)
    )
    assert statements.data_statement(node, 1) == [
        "    tab[0:4] = np.array([I_10, I_20, I_30, I_40], dtype=np.int32)"
    ]


def test_a_nested_implied_do_varies_the_inner_index_fastest(sources: dict[str, Path]) -> None:
    """``((grid(i,j), i=1,2), j=1,3)`` is Fortran's column-major order, and
    getting it backwards would fill the table transposed -- silently."""
    statements, _ = build(sources["emit_mod"], "seeded")
    node = next(
        d
        for d in walk(_specification_of(sources["emit_mod"], "seeded"), f03.Data_Stmt)
        if "grid" in str(d)
    )
    assert statements.data_statement(node, 1) == [
        "    grid[0, 0] = 1",
        "    grid[1, 0] = 2",
        "    grid[0, 1] = I_3",
        "    grid[1, 1] = I_4",
        "    grid[0, 2] = I_5",
        "    grid[1, 2] = I_6",
    ]


def test_a_legacy_entry_statement_does_nothing_where_it_stands(sources: dict[str, Path]) -> None:
    """A second entry point into a subprogram, deleted in F2018. The callers
    this translates reach the primary entry."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    del nodes
    from recast.fortran._parse import parse as parse_file

    entry = walk(parse_file(sources["emit_mod"]), f03.Entry_Stmt)
    if entry:
        assert statements.render(entry[0], 1) == ["    pass  # ENTRY (legacy)"]


def _specification_of(source: Path, name: str) -> Any:
    from recast.fortran._parse import parse as parse_file

    subprogram = next(
        sub
        for sub in walk(parse_file(source), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == name
    )
    return next(c for c in subprogram.children if isinstance(c, f03.Specification_Part))
