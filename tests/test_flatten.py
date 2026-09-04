"""Derived-type interfaces flattened: the plan, both adapters, the recorder.

A three-file tree: a type module (a derived type with pointer components
allocated over the driver's range and a fixed layer count), a state module
(a variable the run sets), and a physics module that takes the object,
aliases its components in an ``associate`` and writes one of them.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from recast.fortran.flatten import FlatConventions, FlatPlan, plans_for, plans_from_facts, signature
from recast.fortran.frontend import FortranFrontend
from recast.fortran.tree import integer_parameters, module_sources, named_extents, use_imports
from recast.oracle.flat import fortran_adapter, unspellable
from recast.oracle.record import RECORDER_MODULE, probe_tree, recorder_module
from recast.transform.numpy.flat import python_adapter
from recast.transform.numpy.standins import stand_ins

pytest.importorskip("fparser")

TYPES = """\
module types_mod
  implicit none
  integer, parameter :: nlev = 3
  type :: canopy_type
     real(8), pointer :: tleaf(:,:)
     real(8), pointer :: gs(:)
     integer, pointer :: ncan(:)
  contains
     procedure :: Init
  end type canopy_type
contains
  subroutine Init(this, begp, endp)
    class(canopy_type) :: this
    integer, intent(in) :: begp, endp
    allocate(this%tleaf(begp:endp, 1:nlev))
    allocate(this%gs(begp:endp))
    allocate(this%ncan(begp:endp))
  end subroutine Init
end module types_mod
"""

STATE = """\
module state_mod
  implicit none
  real(8) :: scale = 1.0d0
end module state_mod
"""

PHYSICS = """\
module physics_mod
  use types_mod, only: canopy_type, nlev
  use state_mod, only: scale
  implicit none
  private
  public :: Warm, Reset
contains
  subroutine Warm(num, filter, dt, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    real(8), intent(in) :: dt
    type(canopy_type), intent(inout) :: inst
    integer :: f, p, ic
    associate (tleaf => inst%tleaf, gs => inst%gs, ncan => inst%ncan)
    do f = 1, num
       p = filter(f)
       do ic = 1, ncan(p)
          tleaf(p,ic) = tleaf(p,ic) + dt * gs(p) * scale
       end do
    end do
    end associate
  end subroutine Warm

  subroutine Reset(profile, inst)
    real(8), intent(in) :: profile(nlev)
    type(canopy_type), intent(out) :: inst
    inst%gs(:) = profile(1)
  end subroutine Reset
end module physics_mod
"""

DRIVER = """\
module driver_mod
  use types_mod, only: canopy_type
  use physics_mod, only: Warm
  implicit none
contains
  subroutine step(inst, n, filt)
    type(canopy_type), intent(inout) :: inst
    integer, intent(in) :: n, filt(:)
    call Warm(n, filt, &
              0.5d0, inst)   ! a continued call
  end subroutine step
end module driver_mod
"""


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    (tmp_path / "types_mod.f90").write_text(TYPES)
    (tmp_path / "state_mod.f90").write_text(STATE)
    (tmp_path / "physics_mod.f90").write_text(PHYSICS)
    (tmp_path / "driver_mod.f90").write_text(DRIVER)
    return tmp_path


def _facts(tree: Path, **options: Any) -> Any:
    frontend = FortranFrontend(**options)
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:physics_mod")
    return frontend.analyze(unit, tree)


CONVENTIONS = FlatConventions(constant_modules=frozenset({"types_mod"}))


# --- the tree ----------------------------------------------------------------


def test_tree_questions(tree: Path) -> None:
    assert [p.name for p in module_sources(tree, frozenset({"types_mod"}))] == ["types_mod.f90"]
    facts = _facts(tree)
    assert named_extents(facts.interface["subprograms"]) == ["nlev"]
    assert integer_parameters(["nlev", "nothing"], tree, frozenset({"types_mod"})) == {"nlev": 3}
    assert use_imports(facts.interface, frozenset({"types_mod"})) == ["canopy_type", "nlev"]


# --- the plan ----------------------------------------------------------------


def test_plan_names_touched_components_and_run_state(tree: Path) -> None:
    facts = _facts(tree)
    plans = plans_for(facts, tree, CONVENTIONS)
    warm = next(p for p in plans if p.subprogram["name"] == "warm")
    assert warm.usable, warm.unsupported
    (inst,) = warm.objects
    assert inst.kind == "dummy" and inst.type_module == "types_mod"
    by_name = {c.name: c for c in inst.components}
    assert set(by_name) == {"tleaf", "gs", "ncan"}
    assert by_name["tleaf"].written and not by_name["gs"].written
    assert by_name["tleaf"].bounds == [("1", "np_"), ("1", "3")]
    assert by_name["tleaf"].extents == ["np_", "3"]
    assert by_name["gs"].pointer
    # The module variable the body reads is an input of the adapter.
    assert [(s.module, s.name, s.extents) for s in warm.states] == [("state_mod", "scale", [])]
    names = [a["name"] for a in warm.flat_args]
    assert names == [
        "num",
        "filter",
        "dt",
        "np_",
        "inst__gs",
        "inst__ncan",
        "inst__tleaf",
        "state_mod__scale",
    ]
    filt = next(a for a in warm.flat_args if a["name"] == "filter")
    # ``filter(:)`` pairs with ``num_filter`` when there is one; here it is
    # sized over the driver's range.
    assert filt["dims"] == [{"lb": "1", "ub": "np_"}]
    assert next(a for a in warm.flat_args if a["name"] == "inst__tleaf")["intent"] == "INOUT"


def test_plan_conventions_are_the_callers(tree: Path) -> None:
    facts = _facts(tree)
    plans = plans_for(
        facts, tree, FlatConventions(constant_modules=frozenset({"types_mod"}), patch_count="ncol_")
    )
    warm = next(p for p in plans if p.subprogram["name"] == "warm")
    assert "ncol_" in [a["name"] for a in warm.flat_args]
    tleaf = next(c for c in warm.objects[0].components if c.name == "tleaf")
    assert tleaf.extents == ["ncol_", "3"]


def test_plan_survives_the_facts_round_trip(tree: Path) -> None:
    facts = _facts(tree)
    plan = next(p for p in plans_for(facts, tree, CONVENTIONS) if p.subprogram["name"] == "warm")
    again = FlatPlan.from_dict(plan.to_dict())
    assert again.flat_args == plan.flat_args
    assert signature(again) == signature(plan)
    assert again.objects[0].components[0].bounds == plan.objects[0].components[0].bounds


def test_frontend_stores_plans_and_dim_parameters(tree: Path) -> None:
    facts = _facts(tree, constant_modules=["types_mod"], flatten=True)
    assert facts.extra["dim_parameters"] == {"nlev": 3}
    stored = plans_from_facts(facts)
    assert [p.subprogram["name"] for p in stored] == ["warm", "reset"]
    assert "flat_plans" not in _facts(tree).extra


def test_frontend_can_keep_an_intent_out_object(tree: Path) -> None:
    plain = _facts(tree)
    reset = next(s for s in plain.interface["subprograms"] if s["name"] == "reset")
    assert next(a for a in reset["args"] if a["name"] == "inst")["intent"] == "OUT"
    corrected = _facts(tree, derived_intent_out_as_inout=True)
    reset = next(s for s in corrected.interface["subprograms"] if s["name"] == "reset")
    assert next(a for a in reset["args"] if a["name"] == "inst")["intent"] == "INOUT"
    assert corrected.provenance["derived_intent_out_as_inout"] == ["physics_mod/reset/inst"]


# --- the adapters ------------------------------------------------------------


def _warm(tree: Path) -> FlatPlan:
    facts = _facts(tree, constant_modules=["types_mod"], flatten=True)
    return next(p for p in plans_from_facts(facts) if p.subprogram["name"] == "warm")


def test_fortran_adapter_allocates_copies_calls_and_copies_back(tree: Path) -> None:
    text = fortran_adapter("physics_mod", [_warm(tree)], ["warm"])
    assert "module physics_mod_flat" in text
    assert "use types_mod, only: canopy_type" in text
    assert "use state_mod, only: scale" in text
    assert (
        "subroutine warm_flat(num, filter, dt, np_, inst__gs, inst__ncan, inst__tleaf, "
        "state_mod__scale)" in text
    )
    assert "allocate(inst%tleaf(1:np_, 1:3))" in text
    assert "inst%gs = inst__gs" in text
    assert "scale = state_mod__scale" in text
    assert "call warm(num=num, filter=filter, dt=dt, inst=inst)" in text
    assert "inst__tleaf = inst%tleaf" in text
    assert "inst__gs = inst%gs" not in text  # read only: nothing to copy back


def test_python_adapter_builds_the_object_and_returns_what_was_written(tree: Path) -> None:
    text = python_adapter([_warm(tree)])
    assert (
        "def warm_flat(num, filter, dt, np_, inst__gs, inst__ncan, inst__tleaf, state_mod__scale):"
        in text
    )
    assert "inst = _Record(gs=inst__gs, ncan=inst__ncan, tleaf=inst__tleaf)" in text
    assert "_state_mod.scale = state_mod__scale" in text
    assert "_out = warm(num=num, filter=filter, dt=dt, inst=inst)" in text
    assert "inst__tleaf = inst.tleaf" in text
    assert "return inst__tleaf" in text
    assert "_SIGNATURES.update({" in text and "'warm_flat'" in text


def test_unspellable_names_the_reason() -> None:
    assert unspellable({"args": [{"name": "x", "dtype": "float64"}]}) is None
    assert unspellable({"args": [{"name": "f", "dtype": "PROCEDURE"}]}) == "f: procedure dummy"
    assert (
        unspellable({"args": [{"name": "o", "dtype": "UNKNOWN(TYPE(t))"}]}) == "o: UNKNOWN(TYPE(t))"
    )


# --- the recorder ------------------------------------------------------------


def test_recorder_probes_the_original_signature(tree: Path) -> None:
    text = recorder_module("physics_mod", [_warm(tree)], calls=5)
    assert f"module {RECORDER_MODULE}" in text
    assert "integer, parameter :: max_calls = 5" in text
    assert "subroutine rec_warm(phase, num, filter, dt, inst)" in text
    assert "type(canopy_type) :: inst" in text
    assert "np_ = size(inst%gs, 1)" in text
    assert "'# PROBE physics_mod.warm_flat: call='" in text
    assert (
        "call rec_r1(u_warm, 'INPUT', 'inst__tleaf', trim(dims), "
        "reshape(inst%tleaf, (/size(inst%tleaf)/)))" in text
    )
    assert "call rec_r1(u_warm, 'INPUT', 'state_mod__scale', '', (/scale/))" in text
    assert "call rec_r1(u_warm, 'OUTPUT', 'inst__tleaf'" in text
    assert "'OUTPUT', 'inst__gs'" not in text


def test_recorder_window_records_only_the_steps_it_names(tree: Path) -> None:
    """A recording of the first calls is a recording of the model's start;
    the window makes one of a later state without the calls before it."""
    text = recorder_module(
        "physics_mod", [_warm(tree)], calls=5, window=("clock_mod", "step", 673, 720)
    )
    assert "use clock_mod, only: step" in text
    body = text[text.index("subroutine rec_warm(") :]
    guard = body.index("if (step < 673 .or. step > 720) return")
    assert guard < body.index("if (phase == 0) then")  # both phases, before the count
    assert "window" not in recorder_module("physics_mod", [_warm(tree)], calls=5)


def test_probe_tree_brackets_a_continued_call(tree: Path, tmp_path: Path) -> None:
    out = tmp_path / "probed"
    sites = probe_tree(tree, out, {"physics_mod": [_warm(tree)]})
    assert sites == {"warm": 1}
    probed = (out / "driver_mod.f90").read_text()
    assert f"  use {RECORDER_MODULE}\n" in probed
    lines = probed.splitlines()
    at = next(i for i, ln in enumerate(lines) if "call rec_warm(0, n, filt, 0.5d0, inst)" in ln)
    assert "call Warm(n, filt, &" in lines[at + 1]
    assert "call rec_warm(1, n, filt, 0.5d0, inst)" in lines[at + 3]
    # The tree the frontend read is untouched.
    assert RECORDER_MODULE not in (tree / "driver_mod.f90").read_text()


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_generated_fortran_compiles(tree: Path, tmp_path: Path) -> None:
    plan = _warm(tree)
    build = tmp_path / "build"
    build.mkdir()
    (build / "physics_mod_flat.f90").write_text(fortran_adapter("physics_mod", [plan], ["warm"]))
    (build / f"{RECORDER_MODULE}.f90").write_text(recorder_module("physics_mod", [plan]))
    for name in ("types_mod", "state_mod", "physics_mod", "physics_mod_flat", RECORDER_MODULE):
        src = build / f"{name}.f90" if (build / f"{name}.f90").exists() else tree / f"{name}.f90"
        subprocess.run(
            [
                "gfortran",
                "-c",
                "-J",
                str(build),
                "-I",
                str(build),
                "-o",
                str(build / f"{name}.o"),
                str(src),
            ],
            check=True,
            cwd=build,
        )


# --- stand-ins ---------------------------------------------------------------


def test_stand_ins_resolve_the_trees_initializers_and_answer_the_framework(tree: Path) -> None:
    emitted = "import state_mod_numpy as _state_mod\nimport abortutils_numpy as _abortutils\n"
    files, report = stand_ins(
        emitted,
        tree,
        set(),
        modules=frozenset({"state_mod"}),
        framework={"abortutils": "def endrun(msg=''):\n    raise RuntimeError(msg)\n"},
    )
    assert {p.name for p in files} == {"state_mod_numpy.py", "abortutils_numpy.py"}
    state = files[Path("state_mod_numpy.py")].decode()
    assert "SCALE = " in state and "scale = SCALE" in state
    assert report["state_mod"]["resolved"] == ["scale"]
    assert "def endrun" in files[Path("abortutils_numpy.py")].decode()
    assert report["abortutils"]["framework"] is True
    again, _ = stand_ins(emitted, tree, {"state_mod_numpy.py"}, modules=frozenset({"state_mod"}))
    assert {p.name for p in again} == {"abortutils_numpy.py"}


def test_a_bundled_companion_resolves_its_own_use_constants(tree: Path) -> None:
    """The companion's ``<module>_use_constants.py`` carries the names *it*
    use-imports, not the caller's table handed down."""
    from recast.transform.numpy.tree import TreeConventions, TreeTranslation

    (tree / "consts_mod.f90").write_text(
        "module consts_mod\n  implicit none\n  real(8), parameter :: tfrz = 273.15d0\n"
        "  integer, parameter :: nlev2 = 3\nend module consts_mod\n"
    )
    (tree / "helper_mod.f90").write_text(
        "module helper_mod\n  use consts_mod, only: tfrz\n  implicit none\ncontains\n"
        "  function celsius(t) result(c)\n    real(8), intent(in) :: t\n    real(8) :: c\n"
        "    c = t - tfrz\n  end function celsius\nend module helper_mod\n"
    )
    (tree / "caller_mod.f90").write_text(
        "module caller_mod\n  use consts_mod, only: nlev2\n  use helper_mod, only: celsius\n"
        "  implicit none\ncontains\n  subroutine run(t, c)\n    real(8), intent(in) :: t(nlev2)\n"
        "    real(8), intent(out) :: c(nlev2)\n    integer :: i\n    do i = 1, nlev2\n"
        "      c(i) = celsius(t(i))\n    end do\n  end subroutine run\nend module caller_mod\n"
    )
    frontend = FortranFrontend(constant_modules=["consts_mod"], stub_modules=["consts_mod"])
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:caller_mod")
    conventions = TreeConventions(
        constant_modules=frozenset({"consts_mod"}), stub_modules=frozenset({"consts_mod"})
    )
    candidate = TreeTranslation(conventions).apply(
        unit, frontend.analyze(unit, tree), {"root": str(tree)}
    )
    assert "helper_mod" in candidate.notes["tree"]["bundled"]
    helper = candidate.files[Path("helper_mod_use_constants.py")].decode()
    assert "TFRZ" in helper and "NLEV2" not in helper
    caller = candidate.files[Path("caller_mod_use_constants.py")].decode()
    assert "NLEV2" in caller


def test_state_read_behind_a_companion_of_a_companion_reaches_the_plan(tree: Path) -> None:
    """``driver -> physics.Warm -> ops.amplify`` where ``ops_mod`` uses a
    state module the driver never does: the callee closure is the unit's
    dataflow, so ``gain`` must be an input of the driver's plan. One level
    of companions is not enough -- the orchestrator case (``GetObu`` into
    ``hybrid``, ``aH12`` read in the callback) is exactly this shape."""
    (tree / "state2_mod.f90").write_text(
        "module state2_mod\n  implicit none\n  real(8) :: gain\nend module state2_mod\n"
    )
    (tree / "ops_mod.f90").write_text(
        "module ops_mod\n  use state2_mod, only: gain\n"
        "  use types_mod, only: canopy_type\n  implicit none\ncontains\n"
        "  subroutine amplify(inst)\n    type(canopy_type), intent(inout) :: inst\n"
        "    inst%gs = inst%gs * gain\n  end subroutine amplify\nend module ops_mod\n"
    )
    (tree / "physics_mod.f90").write_text(
        PHYSICS.replace(
            "  use state_mod, only: scale\n",
            "  use state_mod, only: scale\n  use ops_mod, only: amplify\n",
        ).replace(
            "    end associate\n  end subroutine Warm\n",
            "    end associate\n    call amplify(inst)\n  end subroutine Warm\n",
        )
    )
    frontend = FortranFrontend(constant_modules=["types_mod"], flatten=True)
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:driver_mod")
    facts = frontend.analyze(unit, tree)
    step = next(p for p in plans_from_facts(facts) if p.subprogram["name"] == "step")
    names = {(s.module, s.name) for s in step.states}
    assert ("state2_mod", "gain") in names
    assert ("state_mod", "scale") in names


def test_a_component_written_through_a_companion_call_is_written(tree: Path) -> None:
    """``call scale_in_place(inst%gs)`` into a sibling module writes the
    component; the plan must return it, or a functional lowering loses the
    write the NumPy adapter's in-place arrays hide."""
    (tree / "ops_mod.f90").write_text(
        "module ops_mod\n  implicit none\ncontains\n"
        "  subroutine scale_in_place(x, f)\n    real(8), intent(inout) :: x(:)\n"
        "    real(8), intent(in) :: f\n    x = x * f\n  end subroutine scale_in_place\n"
        "  subroutine fill(y, v)\n    real(8), intent(out) :: y(:)\n"
        "    real(8), intent(in) :: v\n    y = v\n  end subroutine fill\nend module ops_mod\n"
    )
    (tree / "physics_mod.f90").write_text(
        PHYSICS.replace(
            "  use state_mod, only: scale\n",
            "  use state_mod, only: scale\n  use ops_mod, only: scale_in_place, fill\n",
        ).replace(
            "    end associate\n  end subroutine Warm\n",
            "    call scale_in_place(gs(1:num), dt)\n"
            "    end associate\n"
            "    call fill(inst%ncan, 1.0d0)\n  end subroutine Warm\n",
        )
    )
    facts = _facts(tree, constant_modules=["types_mod"], flatten=True)
    warm = next(p for p in plans_from_facts(facts) if p.subprogram["name"] == "warm")
    written = {c.name for c in warm.objects[0].components if c.written}
    assert written == {"tleaf", "gs", "ncan"}


def test_the_adapter_declares_a_lower_bound_and_calls_through_a_generic() -> None:
    """A private specific of a public generic is reached through the generic
    (``public_via``), and an axis declared ``-2:2`` keeps its five rows."""
    from recast.oracle.flat import _axis, _declare

    assert _axis({"lb": "-2", "ub": "2"}) == "-2:2"
    assert _axis({"lb": "1", "ub": "ndim"}) == "ndim"
    assert _axis({"lb": None, "ub": None}) == ":"
    declared = _declare(
        {
            "name": "lhs",
            "dtype": "float64",
            "intent": "INOUT",
            "dims": [{"lb": "-2", "ub": "2"}, {"lb": "1", "ub": "ngrdcol"}],
        }
    )
    assert declared == "    real(8), intent(inout) :: lhs(-2:2, ngrdcol)"

    subprogram = {
        "name": "solve_one",
        "public": True,
        "public_via": "solve",
        "kind": "subroutine",
        "args": [
            {"name": "n", "dtype": "int32", "intent": "IN", "optional": False},
            {
                "name": "x",
                "dtype": "float64",
                "intent": "INOUT",
                "optional": False,
                "dims": [{"lb": "1", "ub": "n"}],
            },
        ],
    }
    plan = FlatPlan(subprogram=subprogram, objects=[])
    text = fortran_adapter(
        "solve_mod", [plan], ["solve_one", "solve_many"], {"solve_many": "solve"}
    )
    assert "use solve_mod, only: solve\n" in text  # both specifics, one generic
    assert "call solve(n=n, x=x)" in text
    assert "solve_one(" not in text.replace("subroutine solve_one_flat(", "")


# --- a CLUBB-shaped object: many components in one ALLOCATE, sized by itself

GRID = """\
module grid_class
  implicit none
  private
  public :: grid, setup_grid
  integer, parameter :: t_above = 1, t_below = 2
  type grid
    integer :: nzm, nzt
    real(8), allocatable, dimension(:,:) :: zm, zt
    real(8), allocatable, dimension(:,:,:) :: weights_zt2zm
    real(8) :: grid_dir
  end type grid
contains
  subroutine setup_grid( ngrdcol, nzmax, gr )
    integer, intent(in) :: ngrdcol, nzmax
    type(grid), intent(inout) :: gr
    integer :: ierr
    gr%nzm = nzmax
    gr%nzt = nzmax - 1
    allocate( gr%zm(ngrdcol,gr%nzm), gr%zt(ngrdcol,gr%nzt), & ! two at once
              gr%weights_zt2zm(ngrdcol,gr%nzm,t_above:t_below), &
              stat=ierr )
    gr%grid_dir = 1.0d0
  end subroutine setup_grid
end module grid_class
"""

COLUMN = """\
module column_mod
  use grid_class, only: grid
  implicit none
  private
  public :: ddz
contains
  subroutine ddz( nzm, ngrdcol, gr, x, dxdz )
    integer, intent(in) :: nzm, ngrdcol
    type(grid), intent(in) :: gr
    real(8), intent(in), dimension(ngrdcol, nzm) :: x
    real(8), intent(out), dimension(ngrdcol, nzm) :: dxdz
    integer :: i, k
    do k = 1, nzm
      do i = 1, ngrdcol
        dxdz(i,k) = gr%grid_dir * x(i,k) * gr%zm(i,k) * gr%weights_zt2zm(i,k,1)
      end do
    end do
  end subroutine ddz
end module column_mod
"""

CLUBB_CONVENTIONS = FlatConventions(patch_count="ngrdcol", bounds_pattern=r"^ngrdcol$")


def test_an_object_allocated_many_at_once_and_sized_by_itself(tmp_path: Path) -> None:
    """CLUBB's grid: one ALLOCATE over every component, the object named by
    the setup routine's dummy rather than ``this``, an axis sized by another
    component of the same object (``gr%nzm``) and one by the module's
    private parameters (``t_above:t_below``). The plan carries ``nzm`` as an
    input the body never reads, spells the extents by it, and declares the
    driver's extent once, because ``ngrdcol`` is a dummy already."""
    (tmp_path / "grid_class.f90").write_text(GRID)
    (tmp_path / "column_mod.f90").write_text(COLUMN)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:column_mod")
    facts = frontend.analyze(unit, tmp_path)
    (plan,) = plans_for(facts, tmp_path, CLUBB_CONVENTIONS)
    assert plan.usable, plan.unsupported
    (gr,) = plan.objects
    by_name = {c.name: c for c in gr.components}
    assert set(by_name) == {"grid_dir", "zm", "weights_zt2zm", "nzm"}
    assert by_name["nzm"].written is False
    assert by_name["zm"].extents == ["ngrdcol", "gr__nzm"]
    assert by_name["weights_zt2zm"].extents == ["ngrdcol", "gr__nzm", "2"]
    assert by_name["weights_zt2zm"].bounds[2] == ("1", "2")
    names = [a["name"] for a in plan.flat_args]
    assert names.count("ngrdcol") == 1
    assert names.index("gr__nzm") < names.index("gr__zm")
    text = fortran_adapter("column_mod", [plan], [])
    assert "real(8), intent(in) :: gr__zm(ngrdcol, gr__nzm)" in text
    assert "allocate(gr%zm(1:ngrdcol, 1:gr__nzm))" in text


COEFS = """\
module coefs_mod
  implicit none
  private
  public :: coefs_type, init_coefs
  type coefs_type
    real(8), allocatable, dimension(:,:) :: coef
  end type coefs_type
contains
  subroutine init_coefs( ngrdcol, nz, c )
    integer, intent(in) :: ngrdcol, nz
    type(coefs_type), intent(out) :: c
    allocate( c%coef(1:ngrdcol,1:nz) )
    c%coef = 0.0d0
  end subroutine init_coefs
end module coefs_mod
"""

USES_COEFS = """\
module solver_mod
  use coefs_mod, only: coefs_type
  implicit none
  private
  public :: apply
contains
  subroutine apply( nzt, ngrdcol, c, x )
    integer, intent(in) :: nzt, ngrdcol
    type(coefs_type), intent(in) :: c
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    x = x * c%coef(:, 1:nzt)
  end subroutine apply
end module solver_mod
"""


def test_an_extent_the_plan_cannot_spell_becomes_an_argument(tmp_path: Path) -> None:
    """``coef`` is allocated ``(1:ngrdcol, 1:nz)`` by the initializer's dummy
    ``nz``; the planned subprogram takes ``nzt``, not ``nz``. The extent is
    the run's own: the plan makes it an integer argument, the recorder writes
    it from ``size()``, and the adapters declare the component by it."""
    from recast.oracle.record import recorder_module

    (tmp_path / "coefs_mod.f90").write_text(COEFS)
    (tmp_path / "solver_mod.f90").write_text(USES_COEFS)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:solver_mod")
    facts = frontend.analyze(unit, tmp_path)
    (plan,) = plans_for(facts, tmp_path, CLUBB_CONVENTIONS)
    assert plan.usable, plan.unsupported
    assert plan.extent_args == {"c__coef_n2": ["c", "coef", 2]}
    (coef,) = plan.objects[0].components
    assert coef.extents == ["ngrdcol", "c__coef_n2"]
    names = [a["name"] for a in plan.flat_args]
    assert names.index("c__coef_n2") < names.index("c__coef")
    adapter = fortran_adapter("solver_mod", [plan], [])
    assert "integer, intent(in) :: c__coef_n2" in adapter
    assert "real(8), intent(in) :: c__coef(ngrdcol, c__coef_n2)" in adapter
    recorder = recorder_module("solver_mod", [plan])
    assert "'# c__coef_n2 = ', merge(size(c%coef, 2), 0, allocated(c%coef))" in recorder
    # ngrdcol is a dummy of the probe already: declared once, not assigned.
    probe = recorder[recorder.index("subroutine rec_apply(") :]
    assert probe.count("integer :: ngrdcol") == 0
    assert "ngrdcol = " not in probe.split("phase == 0")[0]
    again = FlatPlan.from_dict(plan.to_dict())
    assert again.extent_args == plan.extent_args


def test_a_call_continued_with_trailing_comments_is_probed_whole(tmp_path: Path) -> None:
    """CLUBB continues its calls as ``a, b, & ! In`` on every line: the
    comment after the ampersand left a blank the joiner did not strip, and
    the probe carried ``&`` into its argument list."""
    from recast.oracle.record import probe_tree

    (tmp_path / "coefs_mod.f90").write_text(COEFS)
    (tmp_path / "solver_mod.f90").write_text(USES_COEFS)
    (tmp_path / "driver_mod.f90").write_text(
        """\
module driver_mod
  use coefs_mod, only: coefs_type
  use solver_mod, only: apply
  implicit none
contains
  subroutine step( nzt, ngrdcol, c, x )
    integer, intent(in) :: nzt, ngrdcol
    type(coefs_type), intent(in) :: c
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    call apply( nzt, ngrdcol, & ! In
                c, &        ! In
                x )         ! In/out
  end subroutine step
end module driver_mod
"""
    )
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:solver_mod")
    facts = frontend.analyze(unit, tmp_path)
    plans = plans_for(facts, tmp_path, CLUBB_CONVENTIONS)
    sites = probe_tree(tmp_path, tmp_path / "probed", {"solver_mod": plans})
    assert sites == {"apply": 1}
    probed = (tmp_path / "probed" / "driver_mod.f90").read_text()
    assert "call rec_apply(0, nzt, ngrdcol, c, x)" in probed
    assert "&" not in probed.split("call rec_apply(0")[1].split("\n")[0]


def test_the_recorder_guards_a_component_the_run_may_not_allocate(tmp_path: Path) -> None:
    """CLUBB allocates its scalar-tracer coefficients only when sclr_dim > 0;
    reshape of an unallocated component faulted the recording run."""
    from recast.oracle.record import recorder_module

    (tmp_path / "coefs_mod.f90").write_text(COEFS)
    (tmp_path / "solver_mod.f90").write_text(USES_COEFS)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:solver_mod")
    facts = frontend.analyze(unit, tmp_path)
    (plan,) = plans_for(facts, tmp_path, CLUBB_CONVENTIONS)
    recorder = recorder_module("solver_mod", [plan])
    assert "if (allocated(c%coef)) then" in recorder
    assert "'# INPUT: c__coef(0,0)'" in recorder
    assert "merge(size(c%coef, 2), 0, allocated(c%coef))" in recorder


def test_the_recorder_writes_the_plain_out_dummies_too(tmp_path: Path) -> None:
    """CLUBB's advance_* hand back ``wp2``, ``wp3``... as INOUT dummies beside
    what they write into their objects. Recorded only through the objects,
    the replay found no value for the required outputs (mono_flux_limiter's
    ``low_lev_effect``)."""
    from recast.oracle.record import recorder_module

    (tmp_path / "coefs_mod.f90").write_text(COEFS)
    (tmp_path / "solver_mod.f90").write_text(USES_COEFS)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:solver_mod")
    facts = frontend.analyze(unit, tmp_path)
    (plan,) = plans_for(facts, tmp_path, CLUBB_CONVENTIONS)
    recorder = recorder_module("solver_mod", [plan])
    assert "call rec_r1(u_apply, 'INPUT', 'x', trim(dims), reshape(x, (/size(x)/)))" in recorder
    assert "call rec_r1(u_apply, 'OUTPUT', 'x', trim(dims), reshape(x, (/size(x)/)))" in recorder
