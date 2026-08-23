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
from recast.fortran._parse import f03, parse, walk
from recast.fortran.semantics import for_subprogram
from recast.transform.numpy.expressions import Expressions, Remote
from recast.transform.numpy.names import for_subprogram as names_for
from recast.transform.numpy.statements import REFUSED, Statements
from recast.transform.profiles import PROFILES

SOURCE = """\
module emit_mod
  use shr_kind_mod, only: r8 => shr_kind_r8
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
end module emit_mod
"""

COMPANION = """\
module sibling_mod
  use shr_kind_mod, only: r8 => shr_kind_r8
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
  use shr_kind_mod, only: r8 => shr_kind_r8
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
) -> tuple[Statements, list[Any]]:
    """A ``Statements`` for one subprogram, plus its executable nodes."""
    record = interface.extract(src)
    semantics = for_subprogram(record, name, companions=companions)
    names = names_for(semantics, constants.extract(src))
    expressions = Expressions(
        semantics,
        names,
        PROFILES["ifx"],
        externals=externals or {},
        remotes=remotes or {},
        stubs=function_stubs or {},
    )
    statements = Statements(
        semantics, names, expressions, externals=externals or {}, stubs=stubs or {}
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


def test_a_masked_elsewhere_is_refused(sources: dict[str, Path]) -> None:
    """ELSEWHERE with its own mask condition composes three masks; emitting
    only the negation would assign through the wrong one."""
    statements, nodes = build(sources["emit_mod"], "masked")
    with pytest.raises(REFUSED):
        statements.render(pick(nodes, f03.Where_Construct, 1), 1)


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
    sibling = interface.extract(sources["sibling_mod"])
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


def test_a_single_line_if_refuses_a_multi_line_action(sources: dict[str, Path]) -> None:
    statements, nodes = build(sources["emit_mod"], "switch")
    single = pick(nodes, f03.If_Stmt)
    assert statements.render(single, 1) == ["    if (s > 0.0):", "        s = 0.0"]
    with pytest.raises(REFUSED):
        statements.render(pick(nodes, f03.If_Stmt, 1), 1)


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
    """``hist_fld_active(name_out)`` parses as a plain reference, and refuses
    even though the stub table has an answer -- the pipeline hands that shape
    to a human, and a fabricated ``False`` here once turned the surrounding
    construct into ``if False:``, emitted, dead, and silent about it. The
    same call over a character literal parses as a structure constructor,
    which is the one place the pipeline consults its table, so it stubs."""
    statements, nodes = build(
        sources["emit_mod"], "framework", function_stubs={"hist_fld_active": "False"}
    )
    with pytest.raises(REFUSED):
        statements.render(nodes[0], 1)
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
