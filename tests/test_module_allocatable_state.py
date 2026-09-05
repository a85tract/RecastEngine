"""A module allocatable the run sizes is run state with a shape.

``pftvarcon``'s ``vcmax_np1(:)`` is allocated ``(0:mxpft)`` in the module's
own init routine, and ``mxpft`` is a module variable the run sets. The
plan used to refuse it (``extent [{'lb': '1', 'ub': None}] not
resolvable``); now the ALLOCATE is the shape, ``mxpft`` is one more
recorded scalar, the extent is spelled by its flat name, the Fortran
adapter allocates over the original bounds before assigning, and the
recorder takes the extent from ``size()`` at run time.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recast.fortran.flatten import FlatConventions, plans_for
from recast.fortran.frontend import FortranFrontend
from recast.oracle.flat import fortran_adapter
from recast.oracle.record import recorder_module

pytest.importorskip("fparser")

TYPES = """\
module types_mod
  implicit none
  type :: canopy_type
     real(8), pointer :: gs(:)
  end type canopy_type
contains
  subroutine Init(this, begp, endp)
    class(canopy_type) :: this
    integer, intent(in) :: begp, endp
    allocate(this%gs(begp:endp))
  end subroutine Init
end module types_mod
"""

PARAMS = """\
module params_mod
  implicit none
  integer :: mxpft = 50
  real(8), allocatable :: vcmax_np1(:)
  real(8), allocatable :: table(:,:)
contains
  subroutine read_params()
    allocate( vcmax_np1          (0:mxpft) )
    allocate( table(0:mxpft, 2) )
    vcmax_np1(:) = 1.0d0
    table(:,:) = 2.0d0
  end subroutine read_params
end module params_mod
"""

PHYSICS = """\
module physics_mod
  use types_mod, only: canopy_type
  use params_mod, only: vcmax_np1, table
  implicit none
  private
  public :: Scale
contains
  subroutine Scale(num, filter, itype, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    integer, intent(in) :: itype(:)
    type(canopy_type), intent(inout) :: inst
    integer :: f, p
    do f = 1, num
       p = filter(f)
       inst%gs(p) = inst%gs(p) * vcmax_np1(itype(p)) + table(itype(p), 2)
    end do
  end subroutine Scale
end module physics_mod
"""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "types_mod.f90").write_text(TYPES)
    (tmp_path / "params_mod.f90").write_text(PARAMS)
    (tmp_path / "physics_mod.f90").write_text(PHYSICS)
    return tmp_path


def _plan(tree: Path) -> Any:
    frontend = FortranFrontend(buffer_out_arrays="all")
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:physics_mod")
    facts = frontend.analyze(unit, tree)
    plans = plans_for(facts, tree, FlatConventions())
    return next(p for p in plans if p.subprogram["name"].lower() == "scale")


def test_the_allocate_is_the_shape_and_its_extent_a_recorded_scalar(tree: Path) -> None:
    plan = _plan(tree)
    assert plan.unsupported == []
    by_name = {s.name: s for s in plan.states}
    assert set(by_name) == {"mxpft", "vcmax_np1", "table"}
    assert by_name["mxpft"].extents == []
    assert by_name["vcmax_np1"].extents == ["((params_mod__mxpft) - (0) + 1)"]
    assert by_name["vcmax_np1"].bounds == [("0", "params_mod__mxpft")]
    assert by_name["table"].extents == ["((params_mod__mxpft) - (0) + 1)", "2"]
    assert by_name["table"].bounds == [("0", "params_mod__mxpft"), ("1", "2")]


def test_the_plan_round_trips_its_bounds(tree: Path) -> None:
    from recast.fortran.flatten import FlatPlan

    plan = _plan(tree)
    again = FlatPlan.from_dict(plan.to_dict())
    assert [s.bounds for s in again.states] == [s.bounds for s in plan.states]


def test_the_fortran_adapter_allocates_over_the_original_bounds(tree: Path) -> None:
    plan = _plan(tree)
    text = fortran_adapter("physics_mod", [plan], [])
    assert "allocate(vcmax_np1(0:params_mod__mxpft))" in text
    assert "allocate(table(0:params_mod__mxpft, 1:2))" in text
    assert text.index("allocate(vcmax_np1(") < text.index("vcmax_np1 = params_mod__vcmax_np1")
    # the scalar the extent names is declared before the array it sizes
    assert text.index("params_mod__mxpft\n") < text.index("params_mod__vcmax_np1(")


def test_the_recorder_takes_a_run_time_extent_from_size(tree: Path) -> None:
    plan = _plan(tree)
    text = recorder_module("physics_mod", [plan])
    assert "size(vcmax_np1, 1)" in text
    assert "'(51)'" not in text


def test_a_reader_subscripts_the_companions_allocatable_over_its_lower_bound(tree: Path) -> None:
    """``vcmax_np1(itype)`` in a module that use-imports the array shifted by
    one under the reader's ``(:)`` declaration; the companion's
    ``allocate (vcmax_np1 (0:mxpft))`` is the bound, and the subscript
    shifts by zero."""
    from recast.transform.numpy.tree import TreeConventions, TreeTranslation

    frontend = FortranFrontend(buffer_out_arrays="all", flatten=True)
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:physics_mod")
    candidate = TreeTranslation(TreeConventions()).apply(
        unit, frontend.analyze(unit, tree), {"root": str(tree)}
    )
    physics = candidate.files[Path("physics_mod_numpy.py")].decode()
    assert "vcmax_np1[(itype[p - 1]) - (0)]" in physics
    assert "table[(itype[p - 1]) - (0), 1]" in physics


def test_the_recorder_guards_an_array_the_run_may_not_have_allocated(tree: Path) -> None:
    """A pointer component the configuration never allocated (ELM's CN
    state under SP) is a fault to read; the probe tests it and records
    poison of the planned shape instead, so the replay has a value and any
    use of it shows. A declared-shape array needs no test."""
    plan = _plan(tree)
    text = recorder_module("physics_mod", [plan])
    assert "use, intrinsic :: ieee_arithmetic" in text
    assert "if (associated(inst%gs)) then" in text
    assert "spread(nanv, 1, (np_))" in text
    assert "if (allocated(vcmax_np1)) then" in text
    assert "spread(nanv, 1, (((mxpft) - (0) + 1)))" in text
    assert text.count("flush (u_scale)") >= 4


POINTED = """\
module pointed_mod
  use types_mod, only: canopy_type
  implicit none
  private
  public :: Fill
contains
  subroutine Fill(num, filter, phase, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    character(len=3), intent(in) :: phase
    type(canopy_type), intent(inout) :: inst
    real(8), pointer :: view(:)
    integer :: f, p
    if (phase == 'sun') then
       view => inst%gs
    else
       view => inst%ncan_r
    end if
    do f = 1, num
       p = filter(f)
       view(p) = 2.0d0 * inst%gs(p)
    end do
  end subroutine Fill
end module pointed_mod
"""


def test_a_write_through_a_pointer_alias_writes_every_target_it_may_have(tmp_path: Path) -> None:
    """``psn_z => photosyns_vars%psnsun_z_patch`` under sun, the shade array
    under shade, then ``psn_z(p,iv) = ...``: the routine's own outputs. An
    associate alias was followed; a pointer assignment was not, and the
    plan had them read-only, so the gate never compared them."""
    types = TYPES.replace(
        "     real(8), pointer :: gs(:)\n",
        "     real(8), pointer :: gs(:)\n     real(8), pointer :: ncan_r(:)\n",
    ).replace(
        "    allocate(this%gs(begp:endp))\n",
        "    allocate(this%gs(begp:endp))\n    allocate(this%ncan_r(begp:endp))\n",
    )
    (tmp_path / "types_mod.f90").write_text(types)
    (tmp_path / "pointed_mod.f90").write_text(POINTED)
    frontend = FortranFrontend(buffer_out_arrays="all")
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:pointed_mod")
    facts = frontend.analyze(unit, tmp_path)
    plan = next(p for p in plans_for(facts, tmp_path, FlatConventions()) if p.subprogram["name"].lower() == "fill")
    inst = next(o for o in plan.objects if o.name == "inst")
    written = {c.name for c in inst.components if c.written}
    assert written == {"gs", "ncan_r"}, {c.name: c.written for c in inst.components}
