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
from recast.transform.numpy.statements import INT_SENTINEL, REFUSED, Statements
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

  subroutine calls(a, s, t, c, n, j, w, flat, m2)
    real(r8), intent(inout) :: a(10), c(10)
    real(r8), intent(inout) :: s, t
    integer, intent(in) :: n, j
    real(r8), intent(in) :: w(4, 3)
    real(r8), intent(in) :: flat(8)
    real(r8), intent(inout) :: m2(4, 3)
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
    call spread_it(2, j, flat)
    call pick_it(flat)
    call fillv(a(n))
    call fillm(a(n))
    s = pick_norm(2, w(1, j))
    call tailv(a(n))
    call tailm(a(n))
    s = tail_norm(a(n))
    call tailv(m2(2, j))
    call tailm(m2(2, j))
    call tailv(m2)
    s = tail_norm(m2(2, j))
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
    character :: ccode
    integer :: u, offset
    open(newunit=u, file='out.dat', status='replace')
    write(*,*) s
    write(line,*) s, a(1)
    write(11,*,iostat=ios) s
    read(u, pos=offset-1) ccode
    write(u, '(a1)', advance='no') ccode
    write(line, '(a1)', advance='no') ccode
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

  subroutine spread_it(Lda, k, x)
    integer, intent(in) :: Lda, k
    real(r8), intent(in) :: x(Lda, k)
    state(1) = x(1, 1)
  end subroutine spread_it

  subroutine pick_it(x, m)
    real(r8), intent(in) :: x(m, 2)
    integer, intent(in), optional :: m
    state(1) = x(1, 1)
  end subroutine pick_it

  subroutine fillv(x)
    real(r8), intent(out) :: x(2)
    x = 0.0_r8
  end subroutine fillv

  subroutine fillm(x)
    real(r8), intent(out) :: x(2, 2)
    x = 0.0_r8
  end subroutine fillm

  function pick_norm(m, x) result(r)
    integer, intent(in) :: m
    real(r8), intent(in) :: x(m)
    real(r8) :: r
    r = x(1)
  end function pick_norm

  subroutine tailv(x)
    real(r8), intent(inout) :: x(*)
    x(1) = 0.0_r8
  end subroutine tailv

  subroutine tailm(x)
    real(r8), intent(inout) :: x(2, *)
    x(1, 1) = 0.0_r8
  end subroutine tailm

  function tail_norm(x) result(r)
    real(r8), intent(in) :: x(*)
    real(r8) :: r
    r = x(1)
  end function tail_norm

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

  subroutine io_edges(u, ok, x, name)
    integer, intent(in) :: u
    logical, intent(out) :: ok
    real(r8), intent(out) :: x
    character(len=*), intent(in) :: name
    integer :: ios, u2
    character(len=20) :: enc
    open(newunit=u2, file=name, status='old')
    rewind(u)
    backspace(u)
    inquire(unit=u, opened=ok)
    read(u, *, iostat=ios) x
    print *, x
    close(u)
    inquire(unit=u, encoding=enc)
    error stop 'nothing to do'
  end subroutine io_edges
  subroutine pseudorank(a, n, tau, k, kp1)
    real(r8), intent(in) :: a(n, n)
    integer, intent(in) :: n
    real(r8), intent(in) :: tau
    integer, intent(out) :: k, kp1
    integer :: j, i, m
    do j = 1, n
      if (abs(a(j, j)) <= tau) exit
    end do
    k = j - 1
    kp1 = j
    do i = 1, n
      kp1 = kp1 + i
    end do
    m = n
    do i = 1, m, 2
      m = m - 1
    end do
    k = k + i
  end subroutine pseudorank

  function bump(x, cnt) result(y)
    real(r8), intent(in) :: x
    integer, intent(inout) :: cnt
    real(r8) :: y
    cnt = cnt + 1
    y = x * 2.0_r8
  end function bump

  subroutine search(x, cnt, alpha, t)
    real(r8), intent(in) :: x
    integer, intent(inout) :: cnt
    real(r8), intent(out) :: alpha, t
    alpha = bump(x, cnt)
    t = 1.0_r8 + bump(x, cnt)
  end subroutine search

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

  function tail_sum(n, x) result(y)
    integer, intent(in) :: n
    real(r8), intent(in) :: x(*)
    real(r8) :: y
    y = sum(x(1:n))
  end function tail_sum
end module sibling_mod
"""

CALLER = """\
module caller_mod
  use precision_mod, only: r8 => wp_r8
  use sibling_mod, only: cscale, rise, tail_sum
  implicit none
contains
  subroutine drive(a, s)
    real(r8), intent(inout) :: a(10)
    real(r8), intent(inout) :: s
    call cscale(a, s)
    s = rise(s)
    s = tail_sum(2, a(3))
  end subroutine drive
end module caller_mod
"""


CALLBACK = """\
module callback_mod
  implicit none
  abstract interface
    subroutine func(n, x, fvec, iflag)
      implicit none
      integer, intent(in) :: n
      real, intent(in) :: x(n)
      real, intent(out) :: fvec(n)
      integer, intent(inout) :: iflag
    end subroutine func
    real function score(v)
      implicit none
      real, intent(in) :: v
    end function score
  end interface
contains
  subroutine sweep(fcn, n, x, work, iflag)
    procedure(func) :: fcn
    integer, intent(in) :: n
    real, intent(inout) :: x(n)
    real, intent(inout) :: work(n)
    integer, intent(inout) :: iflag
    call fcn(n, x, work, iflag)
  end subroutine sweep

  subroutine untyped(fcn, n, x)
    external :: fcn
    integer, intent(in) :: n
    real, intent(inout) :: x(n)
    call fcn(n, x)
  end subroutine untyped
end module callback_mod
"""


@pytest.fixture(scope="module")
def sources(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("emit")
    paths = {}
    for name, text in (
        ("emit_mod", SOURCE),
        ("sibling_mod", COMPANION),
        ("caller_mod", CALLER),
        ("callback_mod", CALLBACK),
    ):
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
    poison: bool = False,
    poison_integers: bool = False,
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
        poison_undefined=poison,
        poison_integers=poison_integers,
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
    """And is the constructor rather than an element of it.

    This asserted the nested spelling, ``np.array([[...]])``, until the shape
    it produces was noticed: Fortran's ``(/ (2*n, n=1,3) /)`` is three
    elements and that is one element holding three, ``(1, 3)`` against
    ``(3,)``. Every subsequent index into the result is off by a dimension,
    which numpy broadcasts rather than refuses.
    """
    statements, nodes = build(sources["emit_mod"], "sections")
    assert statements.render(nodes[3], 1) == [
        "    ks[...] = np.array([(2 * n) for n in range(1, I_3 + 1)])"
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


def test_a_do_index_read_after_the_loop_gets_its_completion_value(
    sources: dict[str, Path],
) -> None:
    """Fortran leaves the index one past the last iteration when the loop
    runs out (and at its start when it never runs); Python leaves it at the
    last iteration. hfti's ``do j=1,ldiag; if (...) exit; end do; k=j-1``
    reads that value as the pseudorank, so a loop whose index is read
    afterwards gets an ``else`` that sets it -- an EXIT skips it, as the
    index keeps its value there on both sides.

    The completion is spelled with the one unified ``(low) + trips * step``
    form (``trips`` never negative), whatever the step: a unit step is
    ``increment`` 1, so this reads ``(1) + max(0, ((n) - (1) + (1)) // (1))
    * (1)`` -- the same value as ``max(1, n + 1)`` and the same shape as the
    stepped loops below (see the ``do i = 1, m, 2`` case)."""
    statements, nodes = build(sources["emit_mod"], "pseudorank")
    first = statements.render(pick(nodes, f03.Block_Nonlabel_Do_Construct), 1)
    assert first[0] == "    for j in range(1, n + 1):"
    assert first[-2:] == [
        "    else:",
        "        j = (1) + max(0, ((n) - (1) + (1)) // (1)) * (1)",
    ]


def test_a_do_index_redefined_before_any_read_needs_no_completion_value(
    sources: dict[str, Path],
) -> None:
    """The next ``do i`` redefines ``i`` before anything reads it, so the
    loop renders as it always did."""
    statements, nodes = build(sources["emit_mod"], "pseudorank")
    second = statements.render(pick(nodes, f03.Block_Nonlabel_Do_Construct, 1), 1)
    assert second == ["    for i in range(1, n + 1):", "        kp1 = (kp1 + i)"]


def test_a_do_whose_body_writes_a_bound_holds_the_bounds_it_started_with(
    sources: dict[str, Path],
) -> None:
    """Fortran evaluates the bounds once, at entry. The body writes ``m``,
    which the upper bound names, so the completion value ``k = k + i`` reads
    would be wrong recomputed from ``m`` afterwards: the bounds are held in
    temporaries and both the range and the completion read those."""
    statements, nodes = build(sources["emit_mod"], "pseudorank")
    third = statements.render(pick(nodes, f03.Block_Nonlabel_Do_Construct, 2), 1)
    assert third[:4] == [
        "    _dolo_i = 1",
        "    _dohi_i = m",
        "    _dost_i = 2",
        "    for i in range(_dolo_i, _dohi_i + 1, _dost_i):",
    ]
    assert third[-2:] == [
        "    else:",
        "        i = (_dolo_i) + max(0, ((_dohi_i) - (_dolo_i) + (_dost_i)) // (_dost_i))"
        " * (_dost_i)",
    ]


def test_a_division_in_a_declared_bound_is_integer_division(sources: dict[str, Path]) -> None:
    """``(n+1)*(n+2)/2`` -- the packed triangle SLSQP hands ``slsqpb`` -- is
    an integer expression in Fortran; rendered with Python's ``/`` it was a
    float, and the slice it sized refused it."""
    statements, _ = build(sources["emit_mod"], "pseudorank")
    assert statements.expressions.bound("(n+1)*(n+2)/2") == "_f_int_div((n + 1) * (n + 2), 2)"
    assert statements.expressions.bound("n+1") == "n+1"


def test_a_function_hands_its_inout_dummies_back_beside_its_result(
    sources: dict[str, Path],
) -> None:
    """SLSQP's ``linmin`` drives a line search through ``mode`` and eighteen
    INOUT scalars; a translation returning the result alone kept them at
    zero on every call. The function returns ``(result, *outputs)`` and a
    whole-statement reference unpacks the tuple the way a CALL's is."""
    statements, nodes = build(sources["emit_mod"], "bump")
    assert statements.returned_value() == "y, cnt"
    statements, nodes = build(sources["emit_mod"], "search")
    assert statements.render(nodes[0], 1) == ["    alpha, cnt = bump(x, cnt)"]


def test_a_function_with_inout_dummies_inside_an_expression_is_refused(
    sources: dict[str, Path],
) -> None:
    """``1 + bump(x, cnt)`` has nowhere to put ``cnt``; refused by name
    rather than rendered as an expression that drops the write."""
    statements, nodes = build(sources["emit_mod"], "search")
    with pytest.raises(REFUSED, match="bump has OUT/INOUT dummy argument"):
        statements.render(nodes[1], 1)


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
    assert statements.render(nodes[0], 1) == ["    buf = np.zeros((n,), dtype=np.float64)"]
    assert statements.render(nodes[1], 1) == ["    idx = np.zeros((n,), dtype=np.int32)"]


def test_an_allocate_is_undefined_memory_too_and_is_poisoned_with_the_rest(
    sources: dict[str, Path],
) -> None:
    """``allocate(buf(n))`` leaves ``buf`` undefined exactly as a local
    automatic array is, and the tool this arm came from poisons by patching
    the one allocation helper -- which reaches this site along with the
    prologue's. Covering the prologue alone would report a clean run for a
    defect here.
    """
    statements, nodes = build(sources["emit_mod"], "alloc", poison=True)
    assert statements.render(nodes[0], 1) == ["    buf = np.full((n,), np.nan, dtype=np.float64)"]
    assert statements.render(nodes[1], 1) == ["    idx = np.zeros((n,), dtype=np.int32)"]

    statements, nodes = build(sources["emit_mod"], "alloc", poison=True, poison_integers=True)
    assert statements.render(nodes[1], 1) == [
        f"    idx = np.full((n,), {INT_SENTINEL}, dtype=np.int32)"
    ]


def test_an_allocated_lower_bound_shifts_later_subscripts(sources: dict[str, Path]) -> None:
    """``allocate(off(0:n))`` re-bases the array; ``off(i)`` afterwards must
    shift by 0, not by the 1 its declaration would suggest."""
    statements, nodes = build(sources["emit_mod"], "alloc")
    assert statements.render(nodes[2], 1) == [
        "    off = np.zeros(((n) - (0) + 1,), dtype=np.float64)"
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
    rank-1 actual to a rank-2 formal refills it in column-major order --
    the first ``n*j`` cells of it, which is all the dummy spans: SLSQP's
    ``nnls(w, n1, n1, m, ...)`` hands ``a(mda, n)`` the head of a longer
    workspace, and reshaping the whole of ``w`` raised on the size."""
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 8), 1) == ["    vec2(w[:, j - 1])"]
    assert statements.render(pick(nodes, f03.Call_Stmt, 9), 1) == [
        "    consume(n, j, np.reshape(flat[:(n) * (j)], (n, j,), order='F'))"
    ]


def test_an_element_for_an_assumed_size_dummy_is_the_tail_of_the_actual(
    sources: dict[str, Path],
) -> None:
    """``x(*)`` spans the caller's storage from the element to the end of the
    array, and only the caller knows how far that is: ``a(n)`` is ``a[n-1:]``,
    a view, so what the callee writes in place is in the caller's array and
    the copy-out onto the same view changes nothing. Rendering the element
    alone -- what an unbounded dummy used to get -- handed the callee one
    number to subscript, and an OUT dummy's writes landed on ``None``."""
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 14), 1) == [
        "    _f_copy_out(a[(n - 1):], np.ravel(tailv(a[(n - 1):]), order='F'))"
    ]
    assignments = [n for n in nodes if isinstance(n, f03.Assignment_Stmt)]
    assert statements.render(assignments[-2], 1) == ["    s = tail_norm(a[(n - 1):])"]


def test_an_element_for_a_rank_2_assumed_size_dummy_folds_the_tail(
    sources: dict[str, Path],
) -> None:
    """``x(2, *)`` handed ``a(n)``: the tail from the element on, folded onto
    the leading extent with the last axis taking the whole columns left --
    the runtime's ``_f_seq_tail`` -- and written back through
    ``_f_seq_tail_out`` onto the same storage. It used to be refused for
    having no extent to reshape to."""
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 15), 1) == [
        "    _f_seq_tail_out(a, (n - 1), tailm(_f_seq_tail(a, (n - 1), 2)))"
    ]


def test_an_element_of_a_matrix_for_an_assumed_size_dummy_is_its_column_major_tail(
    sources: dict[str, Path],
) -> None:
    """SLSQP's ``dcopy(n, a(i, 1), la, ...)`` and ``h12(..., c(i, 1), lc,
    ...)``: an element of a rank-2 actual for ``x(*)`` or ``x(2, *)`` is the
    storage from that element to the end in column-major order. There is no
    slice of a rank-2 array that spells it, so the runtime hands the callee
    ``_f_seq_tail`` -- a view of a Fortran-contiguous actual -- and hands
    the callee's array back whole to ``_f_seq_tail_out``, which writes it
    onto the storage unless it is that view already; the whole matrix to
    ``x(*)`` is the same thing from position 0. Every such call was refused
    as "only a view when both are rank-1", which deferred every block that
    recovers a matrix row."""
    statements, nodes = build(sources["emit_mod"], "calls")
    start = "(2 - 1) + (j - 1) * 1 * np.size(m2, 0)"
    assert statements.render(pick(nodes, f03.Call_Stmt, 16), 1) == [
        f"    _f_seq_tail_out(m2, {start}, tailv(_f_seq_tail(m2, {start})))"
    ]
    assert statements.render(pick(nodes, f03.Call_Stmt, 17), 1) == [
        f"    _f_seq_tail_out(m2, {start}, tailm(_f_seq_tail(m2, {start}, 2)))"
    ]
    assert statements.render(pick(nodes, f03.Call_Stmt, 18), 1) == [
        "    _f_seq_tail_out(m2, 0, tailv(_f_seq_tail(m2, 0)))"
    ]
    assignments = [n for n in nodes if isinstance(n, f03.Assignment_Stmt)]
    assert statements.render(assignments[-1], 1) == [f"    s = tail_norm(_f_seq_tail(m2, {start}))"]


def test_a_reshape_reads_the_callee_s_bound_in_whatever_case_it_was_written(
    sources: dict[str, Path],
) -> None:
    """``x(Lda, k)`` names the dummy ``lda``: Fortran has one name there, not
    two. Missing the capital rendered the *callee's* parameter in the
    *caller's* scope -- a name the caller does not have -- instead of the
    actual it passed."""
    statements, nodes = build(sources["emit_mod"], "calls")
    assert statements.render(pick(nodes, f03.Call_Stmt, 10), 1) == [
        "    spread_it(2, j, np.reshape(flat[:(2) * (j)], (2, j,), order='F'))"
    ]


def test_a_dummy_dimension_this_call_never_bound_is_refused(
    sources: dict[str, Path],
) -> None:
    """``x(m, 2)`` with ``m`` an optional the call leaves out: there is no
    actual to reshape to, and the caller's own ``m`` -- if it has one -- is a
    different variable. A refusal, not a guess."""
    statements, nodes = build(sources["emit_mod"], "calls")
    with pytest.raises(REFUSED):
        statements.render(pick(nodes, f03.Call_Stmt, 11), 1)


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


def test_a_companion_function_reference_binds_its_actuals_by_formal(
    sources: dict[str, Path],
) -> None:
    """Sequence association reaches a sibling's function too: ``ddot(n,
    w(i4), 1, w(iff), 1)`` into a translated BLAS is bound to ``dx(*)`` and
    ``dy(*)`` the way a call to a procedure of this file is, so the callee
    gets the tail of the array and not two scalars to subscript."""
    sibling = interface.extract(sources["sibling_mod"], kind_assumptions=KINDS)
    remotes = {s["name"]: Remote("_sib", s["name"]) for s in sibling["subprograms"]}
    statements, nodes = build(
        sources["caller_mod"], "drive", companions=(sibling,), remotes=remotes
    )
    assert statements.render(nodes[2], 1) == ["    s = _sib.tail_sum(2, a[(I_3 - 1):])"]


# --- I/O and control ---------------------------------------------------------


def test_writes_split_on_where_the_records_go(sources: dict[str, Path]) -> None:
    """Three destinations, three translations. ``write(*, ...)`` is a log and
    stays the stub it has always been; an INTERNAL write assigns a character
    variable; a write to a unit an OPEN here connected to a file puts records
    in that file, which for a subprogram whose only product is the file is the
    whole translation (see the ADVANCE= test below)."""
    statements, nodes = build(sources["emit_mod"], "io")
    assert statements.render(nodes[1], 1) == ["    pass  # write(*,...) log — no dataflow"]
    assert statements.render(nodes[2], 1) == ["    line = _f_list_write(s, a[0])"]
    with pytest.raises(REFUSED):
        statements.render(nodes[3], 1)  # iostat= is control flow, not logging


def test_a_read_with_pos_seeks_before_it_reads(sources: dict[str, Path]) -> None:
    """POS= is stream access: where in the file the values start, counted in
    bytes from one. Refusing it deferred the one statement that says where a
    program's header ended and its binary payload began."""
    statements, nodes = build(sources["emit_mod"], "io")
    assert statements.render(nodes[4], 1) == [
        "    _, ccode = _f_read(u, None, [('str', None, 1)], pos=(offset - 1))"
    ]


def test_a_non_advancing_write_carries_advance_to_the_runtime(
    sources: dict[str, Path],
) -> None:
    """ADVANCE='no' says the record does not end here, which is a property of
    the file the statement writes -- so it is carried to ``_f_write`` rather
    than refused. An *internal* write has one record and nowhere to put it,
    so ADVANCE= is refused there instead."""
    statements, nodes = build(sources["emit_mod"], "io")
    assert statements.render(nodes[5], 1) == ["    _f_write(u, '(a1)', [ccode], advance='no')"]
    with pytest.raises(REFUSED):
        statements.render(nodes[6], 1)


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


def test_a_case_value_range_is_a_closed_interval(sources: dict[str, Path]) -> None:
    """Was ``test_a_case_value_range_is_refused``, and before that
    ``..._slips_past_the_refusal``: the pipeline looked at the wrong node,
    so ``case (1:2)`` came out as two equality tests -- right there by
    luck, wrong for any wider range. Reading the value list item by item
    made the refusal fire on slsqp's ``bvls_wrapper``; a range is a closed
    interval on the selector, and either end may be open. The spelling is
    the pipeline's #44 fix (840c3f2): each comparison parenthesized."""
    statements, nodes = build(sources["emit_mod"], "switch")
    lines = statements.render(pick(nodes, f03.Case_Construct, 1), 1)
    assert lines[0].startswith("    if ((1 <= n) and (n <= ")


def test_an_open_ended_case_range_tests_one_side(tmp_path: Path) -> None:
    from recast.fortran import constants, interface
    from recast.fortran._parse import f03, parse, walk
    from recast.transform.numpy.subprograms import Subprograms
    from recast.transform.profiles import PROFILES

    source = Path(tmp_path) / "rng.f90"
    source.write_text(
        "module rng_mod\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine pick(n, v)\n"
        "    integer, intent(in) :: n\n"
        "    real(8), intent(out) :: v\n"
        "    select case (n)\n"
        "    case (:0, 7:)\n"
        "      v = 2d0\n"
        "    case default\n"
        "      v = 0d0\n"
        "    end select\n"
        "  end subroutine pick\n"
        "end module rng_mod\n"
    )
    assembler = Subprograms(
        record=interface.extract(source),
        constants=constants.extract(source),
        profile=PROFILES["gfortran"],
    )
    node = next(iter(walk(parse(source), f03.Subroutine_Subprogram)))
    lines, _ = assembler.render(node, "pick")
    branch = next(line for line in lines if line.lstrip().startswith("if "))
    assert "((n <= 0)) or (" in branch and "<= n))" in branch


def test_a_character_case_range_is_refused(tmp_path: Path) -> None:
    """``case ('a':'m')`` stays a queue, as the pipeline keeps it (840c3f2):
    Python's collation is not Fortran's."""
    from recast.fortran import constants, interface
    from recast.fortran._parse import f03, parse, walk
    from recast.transform.numpy.subprograms import Subprograms
    from recast.transform.profiles import PROFILES

    source = Path(tmp_path) / "chr.f90"
    source.write_text(
        "module chr_mod\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine pick(c, v)\n"
        "    character(len=1), intent(in) :: c\n"
        "    real(8), intent(out) :: v\n"
        "    select case (c)\n"
        "    case ('a':'m')\n"
        "      v = 2d0\n"
        "    case default\n"
        "      v = 0d0\n"
        "    end select\n"
        "  end subroutine pick\n"
        "end module chr_mod\n"
    )
    assembler = Subprograms(
        record=interface.extract(source),
        constants=constants.extract(source),
        profile=PROFILES["gfortran"],
    )
    node = next(iter(walk(parse(source), f03.Subroutine_Subprogram)))
    lines, _ = assembler.render(node, "pick")
    queued = next(line for line in lines if "AGENT_QUEUE" in line)
    assert "case value range (character)" in queued


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


def test_file_positioning_moves_the_position_a_later_read_reads(
    sources: dict[str, Path],
) -> None:
    """REWIND and BACKSPACE write no variable, but the position they move is
    what the next READ reads from -- a stub left the loop that rewinds and
    re-reads a file reading the same records twice -- and the unit they name
    is a read the gate compares."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    assert statements.render(pick(nodes, f03.Rewind_Stmt), 1) == ["    _f_rewind(u)"]
    assert statements.render(pick(nodes, f03.Backspace_Stmt), 1) == ["    _f_backspace(u)"]


def test_the_connection_statements_carry_their_writes(sources: dict[str, Path]) -> None:
    """``newunit=`` is where OPEN puts the unit it allocated, and a stub left
    it at whatever it held; CLOSE names a unit and nothing else."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    assert statements.render(pick(nodes, f03.Open_Stmt), 1) == [
        "    _, u2 = _f_open(None, name, status='old')"
    ]
    assert statements.render(pick(nodes, f03.Close_Stmt), 1) == ["    _f_close(u)"]


def test_inquire_assigns_every_specifier_it_can_answer(sources: dict[str, Path]) -> None:
    """Where the pipeline this was migrated from renders INQUIRE as ``pass``:
    ``opened=ok`` writes ``ok``, and a ``pass`` leaves it at whatever it held
    while the read/write gate is told nothing happened. A specifier the
    runtime cannot answer is still refused, by name."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    inquires = [n for n in nodes if isinstance(n, f03.Inquire_Stmt)]
    assert statements.render(inquires[0], 1) == ["    ok = _f_inquire(u, None, 'opened')"]
    with pytest.raises(REFUSED, match="ENCODING="):
        statements.render(inquires[1], 1)


def test_read_unpacks_its_item_list_and_its_iostat(sources: dict[str, Path]) -> None:
    """A READ writes every item in its list, so the translation is the
    assignment those writes make -- with IOSTAT= in front of them, because a
    statement that asks for the status does not abort on a bad record."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    assert statements.render(pick(nodes, f03.Read_Stmt), 1) == [
        "    ios, x = _f_read(u, None, [('float64', None, None)], strict=False)"
    ]


def test_print_reads_its_item_list(sources: dict[str, Path]) -> None:
    """A ``pass`` told the read/write gate that a statement reading ``x`` read
    nothing."""
    statements, nodes = build(sources["emit_mod"], "io_edges")
    assert statements.render(pick(nodes, f03.Print_Stmt), 1) == ["    _f_print(None, x)"]


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


def test_a_negative_case_value_keeps_its_sign(tmp_path):
    """``case (0, -1)``: the minus is a node above the literal, and walking
    the leaves under the selector read it as ``1`` -- so CLM-ml's
    FluxProfileSolution took the well-mixed branch under the implicit
    solver's setting, and aborted."""
    from pathlib import Path

    from recast.fortran import constants, interface
    from recast.fortran._parse import f03, parse, walk
    from recast.transform.numpy.subprograms import Subprograms
    from recast.transform.profiles import PROFILES

    source = Path(tmp_path) / "sw.f90"
    source.write_text(
        "module sw_mod\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine pick(k, x)\n"
        "    integer, intent(in) :: k\n"
        "    real(8), intent(out) :: x\n"
        "    select case (k)\n"
        "    case (0, -1)\n"
        "      x = 1.0d0\n"
        "    case (1)\n"
        "      x = 2.0d0\n"
        "    case default\n"
        "      x = 0.0d0\n"
        "    end select\n"
        "  end subroutine pick\n"
        "end module sw_mod\n"
    )
    assembler = Subprograms(
        record=interface.extract(source),
        constants=constants.extract(source),
        profile=PROFILES["gfortran"],
    )
    node = next(
        sub
        for sub in walk(parse(source), f03.Subroutine_Subprogram)
        if str(walk(sub, f03.Subroutine_Stmt)[0].children[1]).lower() == "pick"
    )
    lines, _ = assembler.render(node, "pick")
    branch = next(line for line in lines if line.strip().startswith("if"))
    assert "-1" in branch.replace("- 1", "-1") or "(-1)" in branch, branch
    assert "== 1)" not in branch or "(k == -1)" in branch.replace("- 1", "-1"), branch


def test_a_function_interface_body_is_read_as_a_function(tmp_path):
    """The body of ``interface / function f(x) ... end function`` is a
    ``Function_Body``, not a ``Function_Subprogram``; read as a subroutine
    it had no ``Subroutine_Stmt`` to index, and every module that declares
    one -- fftpack, fortran-utils, a submodule's parent -- stopped at the
    frontend with an IndexError."""
    from pathlib import Path

    from recast.fortran import interface

    source = Path(tmp_path) / "fn.f90"
    source.write_text(
        "module fn_mod\n"
        "  implicit none\n"
        "  interface\n"
        "    function fdum(x) result(y)\n"
        "    real(8), intent(in) :: x\n"
        "    real(8) :: y\n"
        "    end function fdum\n"
        "  end interface\n"
        "end module fn_mod\n"
    )
    record = interface.extract(source)
    assert record["interfaces"]["fdum"]["result"] == "y"
    assert [a["name"] for a in record["interfaces"]["fdum"]["args"]] == ["x"]


def test_a_call_through_a_procedure_dummy_binds_to_its_interface(tmp_path):
    """``external :: func`` and ``call func(x, val)`` in a solver, with the
    interface block above saying ``val`` is OUT: the dummy is the callable
    the caller passed, and the interface says what comes back (CLM-ml's
    hybrid/zbrent/bisection, which every stomatal and Obukhov solve calls)."""
    from pathlib import Path

    from recast.fortran import constants, interface
    from recast.fortran._parse import f03, parse, walk
    from recast.transform.numpy.subprograms import Subprograms
    from recast.transform.profiles import PROFILES

    source = Path(tmp_path) / "solve.f90"
    source.write_text(
        "module solve_mod\n"
        "  implicit none\n"
        "  interface\n"
        "    subroutine func (x, val)\n"
        "    real(8), intent(in) :: x\n"
        "    real(8), intent(out) :: val\n"
        "    end subroutine func\n"
        "  end interface\n"
        "contains\n"
        "  subroutine once(func, x, y)\n"
        "    external :: func\n"
        "    real(8), intent(in) :: x\n"
        "    real(8), intent(out) :: y\n"
        "    call func(x, y)\n"
        "  end subroutine once\n"
        "end module solve_mod\n"
    )
    record = interface.extract(source)
    assert "func" in record["interfaces"]
    assembler = Subprograms(
        record=record, constants=constants.extract(source), profile=PROFILES["gfortran"]
    )
    node = next(
        sub
        for sub in walk(parse(source), f03.Subroutine_Subprogram)
        if str(walk(sub, f03.Subroutine_Stmt)[0].children[1]).lower() == "once"
    )
    lines, _ = assembler.render(node, "once")
    assert any(line.strip() == "y = func(x)" for line in lines), lines


# --- dummy procedures --------------------------------------------------------


def test_a_call_through_a_dummy_procedure_binds_by_its_interface(
    sources: dict[str, Path],
) -> None:
    """``procedure(func) :: fcn`` says what calling ``fcn`` means, so the
    call is bound like any other -- IN arguments in, OUT arguments back out --
    and spelled with the argument's own name, which is the parameter the
    translated subprogram already takes."""
    statements, nodes = build(sources["callback_mod"], "sweep")
    assert statements.render(pick(nodes, f03.Call_Stmt), 1) == [
        "    _out = fcn(n, x, iflag)",
        "    _f_copy_out(work, _out[0])",
        "    iflag = _out[1]",
    ]


def test_a_dummy_procedure_with_no_interface_still_refuses(
    sources: dict[str, Path],
) -> None:
    """``external :: fcn`` names no argument list, so there is nothing to bind
    against; guessing is how an OUT argument silently becomes an IN one."""
    statements, nodes = build(sources["callback_mod"], "untyped")
    with pytest.raises(REFUSED, match="call to external subroutine 'fcn'"):
        statements.render(pick(nodes, f03.Call_Stmt), 1)


def test_a_stubbed_call_names_the_reads_it_dropped(sources: dict[str, Path]) -> None:
    """The stub is a ``pass``; the names the call's arguments read are gone
    from the target and are handed to the read/write protocol so the gate
    does not count the source's reads of them against it."""
    statements, nodes = build(sources["emit_mod"], "calls", stubs={"outfld": "pass"})
    statements.dropped_reads.clear()
    statements.render(pick(nodes, f03.Call_Stmt, 5), 1)
    dropped = statements.dropped_reads
    assert dropped and all(n == n.lower() for n in dropped)
    assert "outfld" not in dropped
