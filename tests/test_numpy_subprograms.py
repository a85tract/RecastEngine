"""Tests for the subprogram assembly layer.

The differential says assembly matches the pipeline over 276 subprograms and
18,520 emitted lines. What it cannot say: the corpus pins none of the
prologue's rarer shapes (optional outputs, donor-borrowed allocation, the
parameter initializer forms), and it never exercises a patch, because the
differential runs with an empty patch table. Those are here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import constants, interface
from recast.fortran._parse import f03, parse, walk
from recast.transform.numpy.statements import INT_SENTINEL
from recast.transform.numpy.subprograms import Subprograms
from recast.transform.profiles import PROFILES

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


SOURCE = """\
module asm_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  real(r8) :: counter

  type pack_t
    real(r8) :: v
  end type pack_t

  type wide_t
    real(r8) :: rows(nowhere)
  end type wide_t

contains

  subroutine fill_box(box)
    type(pack_t), intent(out) :: box
    box%v = 1.0_r8
  end subroutine fill_box

  subroutine fill_wide(w)
    type(wide_t), intent(out) :: w
    w%rows = 0.0_r8
  end subroutine fill_wide

  subroutine work(n, a, b, opt, flags, out1)
    integer, intent(in) :: n
    real(r8), intent(in) :: a(:)
    real(r8), intent(inout) :: b
    real(r8), intent(out), optional :: opt
    logical, intent(in), optional :: flags
    real(r8), intent(out) :: out1(:)
    real(r8), parameter :: half = 0.5_r8
    real(r8), parameter :: gains(2) = (/ 1.0_r8, 2.0_r8 /)
    character(len=2), parameter :: tags(2) = (/ 'ab', 'cd' /)
    integer, parameter :: three = 3
    real(r8), parameter :: twice_half = 1._r8/half
    integer, parameter :: mask = z'FF'
    logical, parameter :: yes = .true.
    real(r8) :: scr(n)
    integer :: tally(n)
    real(r8), allocatable :: dyn(:)
    type(pack_t) :: box
    integer :: i
    counter = 0.0_r8
    do i = 1, n
      b = b + a(i)
    end do
    call missing_sub(b)
    out1 = a * half
  end subroutine work

  function total(n) result(t)
    integer, intent(in) :: n
    real(r8) :: t
    t = 0.0_r8
    return
  end function total

  function pick(n) result(t)
    integer, intent(in) :: n
    real(r8) :: t
    if (n > 0) then
      t = 1.0_r8
      return
    end if
    t = 0.0_r8
  end function pick

  subroutine blend(x, y, z)
    real(r8), intent(in) :: x, y
    real(r8), intent(out) :: z
    z = exp(x) + x**y
  end subroutine blend

  subroutine escape(s)
    real(r8), intent(inout) :: s
    if (s > 1.0_r8) go to 50
    s = s + 1.0_r8
50  continue
    s = s * 2.0_r8
  end subroutine escape
  subroutine climb(k)
    integer, intent(inout) :: k
 10 continue
    if (k > 12) then
       k = k - 12
       go to 10
    end if
    k = k + 1
  end subroutine climb

end module asm_mod
"""


@pytest.fixture(scope="module")
def source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = tmp_path_factory.mktemp("asm") / "asm_mod.f90"
    path.write_text(SOURCE)
    return path


def build(
    source: Path,
    patches: dict[str, Any] | None = None,
    profile: str = "ifx",
    intrinsics: dict[str, Any] | None = None,
    poison: bool = False,
    poison_integers: bool = False,
) -> Subprograms:
    return Subprograms(
        record=interface.extract(source, kind_assumptions=KINDS),
        constants=constants.extract(source),
        profile=PROFILES[profile],
        patches=patches or {},
        intrinsics=intrinsics or {},
        poison_undefined=poison,
        poison_integers=poison_integers,
    )


def node_of(source: Path, name: str) -> Any:
    return next(
        sub
        for sub in walk(parse(source), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == name
    )


# --- the signature -----------------------------------------------------------


def test_the_signature_reorders_by_intent(source: Path) -> None:
    """An optional OUT is not a parameter but a ``want_`` sentinel; an
    optional IN becomes a keyword; a plain OUT vanishes from the def line
    entirely, because the callee owns its buffer and returns it."""
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert lines[0] == "def work(n, a, b, want_opt=False, flags=None):"


def test_the_return_tuple_carries_every_out_intent(source: Path) -> None:
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert lines[-2] == "    return b, opt, out1"
    assert lines[-1] == ""


def test_an_explicit_trailing_return_is_not_doubled(source: Path) -> None:
    lines, _ = build(source).render(node_of(source, "total"), "total")
    returns = [line for line in lines if line.strip().startswith("return")]
    assert returns == ["    return t"]


def test_a_return_nested_in_a_branch_does_not_stand_in_for_the_final_one(source: Path) -> None:
    """Only a function-level return may. The branch that skips the nested one
    otherwise falls off the end and returns None (the pipeline's T45)."""
    lines, _ = build(source).render(node_of(source, "pick"), "pick")
    returns = [line for line in lines if line.strip().startswith("return")]
    assert returns == ["        return t", "    return t"]


# --- the compiler profile ----------------------------------------------------


def test_by_default_the_transcendentals_are_the_system_libm(source: Path) -> None:
    lines, _ = build(source).render(node_of(source, "blend"), "blend")
    body = "\n".join(lines)
    assert "math.exp(x)" in body
    assert "(x ** y)" in body


def test_an_override_table_replaces_them(source: Path) -> None:
    """A reference binary linked against a maths library that is not the
    system one computes different numbers, and a translation held to it has
    to call the same library. Which one is a fact about the build, so it
    arrives as configuration and the package that knows the build ships it.
    ``**`` rides along, being lowered the same way."""
    overrides = {"scalar": {"exp": "other_libm.exp", "**": "other_libm.pow"}}
    lines, _ = build(source, intrinsics=overrides).render(node_of(source, "blend"), "blend")
    body = "\n".join(lines)
    assert "other_libm.exp(x)" in body
    assert "other_libm.pow(x, y)" in body
    assert "math.exp(" not in body


# --- the prologue ------------------------------------------------------------


def test_module_state_written_becomes_global(source: Path) -> None:
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert "    global counter" in lines


def test_out_arguments_are_allocated_or_zeroed(source: Path) -> None:
    """An assumed-shape OUT borrows the shape of a same-rank assumed-shape
    IN argument -- Fortran took the extent from the actual, and the donor is
    the only place that extent still exists."""
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert "    out1 = np.zeros(np.shape(a), dtype=np.float64)" in lines
    assert "    opt = 0.0  # optional OUT: may not be assigned" in lines


def test_parameter_initializers_render_by_form(source: Path) -> None:
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert "    half = np.float64('0.5')" in lines
    assert "    gains = np.array([1.0, 2.0])" in lines
    assert "    tags = np.array(['ab', 'cd'])" in lines
    assert "    three = 3" in lines
    assert "    yes = True" in lines


def test_locals_are_determinized(source: Path) -> None:
    """Fortran locals are stack garbage until assigned; each gets the
    initialization that makes a read-before-write reproducible instead of
    undefined. An automatic array allocates off its declared extent, an
    allocatable waits as None, a derived type gets its factory."""
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert "    scr = np.zeros((n,), dtype=np.float64)" in lines
    assert "    dyn = None" in lines
    assert "    box = _make_pack_t()" in lines
    assert "    i = 0" in lines


def test_poisoning_makes_a_read_before_write_visible(source: Path) -> None:
    """The other half of the test above, and the reason it is not enough.

    Determinizing makes an unwritten read reproducible; it does not make it
    *findable*. The zero fill is reproducible but says nothing, so the answer
    looks stable and the defect survives review. Poisoned, every float Fortran left
    undefined starts as NaN, so the read propagates into the outputs the gate
    compares and ``differential.bitexact`` counts it as ``nan_mismatch``.
    """
    lines, _ = build(source, poison=True).render(node_of(source, "work"), "work")
    assert "    scr = np.full((n,), np.nan, dtype=np.float64)" in lines
    # Unchanged: an allocatable is not undefined memory, it is unallocated,
    # and a scalar is outside what this covers.
    assert "    dyn = None" in lines
    assert "    i = 0" in lines


def test_the_integer_arm_is_a_second_switch(source: Path) -> None:
    """Off even when the float arm is on, because it is answered differently:
    nothing propagates from an integer to a NaN scan, so its detector is an A/B
    diff against the unpoisoned run rather than the gate already running."""
    plain, _ = build(source, poison=True).render(node_of(source, "work"), "work")
    both, _ = build(source, poison=True, poison_integers=True).render(
        node_of(source, "work"), "work"
    )
    assert "    tally = np.zeros((n,), dtype=np.int32)" in plain
    assert f"    tally = np.full((n,), {INT_SENTINEL}, dtype=np.int32)" in both


def test_a_function_result_is_preinitialized(source: Path) -> None:
    lines, _ = build(source).render(node_of(source, "total"), "total")
    assert "    t = 0.0" in lines


# --- blocks ------------------------------------------------------------------


def test_a_refused_block_defers_and_says_why(source: Path) -> None:
    lines, report = build(source).render(node_of(source, "work"), "work")
    deferred = [entry for entry in report if entry["status"] == "agent_queue"]
    assert len(deferred) == 1
    assert deferred[0]["block"] == "B003"
    assert "missing_sub" in deferred[0]["reason"]
    raising = next(line for line in lines if "NotImplementedError" in line)
    assert raising.endswith("# B003")


def test_a_patch_replaces_its_block_verbatim(source: Path) -> None:
    patched = build(source, patches={"work/B003": {"reason": "audited", "python": ["b = 99"]}})
    lines, report = patched.render(node_of(source, "work"), "work")
    assert "    b = 99" in lines
    assert not any("NotImplementedError" in line for line in lines)
    entry = next(e for e in report if e["block"] == "B003")
    assert entry["status"] == "agent_patched"
    assert entry["reason"] == "audited"


def test_a_top_level_forward_goto_becomes_a_region(source: Path) -> None:
    """The jump crosses top-level blocks, which no loop break can express;
    the blocks up to the label wrap in try/except and the label block is
    consumed by the wrapper, keeping its marker for output re-scans."""
    lines, report = build(source).render(node_of(source, "escape"), "escape")
    assert "    try:  # forward-goto region (label 50)" in lines
    assert "            raise _FGoto('50')  # goto 50" in lines
    assert "    except _FGoto as _g:" in lines
    assert "    pass  # label block consumed by region" in lines
    assert all(entry["status"] == "mechanical" for entry in report)


def test_a_block_inside_a_region_carries_its_marker_at_the_region_indent(source: Path) -> None:
    """Every line of a block inside a goto region is one level deeper, its
    marker comment included. The refusing path worked this out and the
    accepting one did not, so a region's blocks sat under markers at the
    outer indent -- the emitted Python was right and the map to it was not."""
    lines, _ = build(source).render(node_of(source, "escape"), "escape")
    opened = lines.index("    try:  # forward-goto region (label 50)")
    closed = lines.index("    except _FGoto as _g:")
    inside = [line for line in lines[opened + 1 : closed] if line.lstrip().startswith("# B")]
    assert inside, "no block marker inside the region"
    assert all(line.startswith("        # B") for line in inside), inside


def test_an_intent_out_derived_dummy_is_a_fresh_object_at_entry(source: Path) -> None:
    """Fortran says the callee sees an INTENT(OUT) argument undefined, and
    the return convention makes the callee its owner. Emitting nothing left
    the body's first ``box%v = ...`` running against whatever the caller
    passed -- or against a name never bound."""
    lines, _ = build(source).render(node_of(source, "fill_box"), "fill_box")
    assert "    box = _make_pack_t()" in lines


def test_a_derived_dummy_whose_component_extent_is_unresolvable_is_queued(
    source: Path,
) -> None:
    """``rows(nowhere)`` names nothing reachable at module scope, so the
    factory would leave the component None and every read of it would see an
    absent array. Queue it rather than guess a size -- the rule the pipeline
    settled on."""
    lines, _ = build(source).render(node_of(source, "fill_wide"), "fill_wide")
    body = "\n".join(lines)
    assert "dims not statically resolvable" in body
    assert "raise NotImplementedError(" in body
    assert "_make_wide_t()" not in body


def test_a_local_parameter_reference_keeps_its_case(source: Path) -> None:
    """A module constant is spelled upper case in the emitted source; a
    reference to one of this subprogram's own parameters is not, because that
    one was emitted as a local assignment under its own name. Uppercasing the
    whole initializer text bound it to a module constant that does not exist:
    ``1._r8/half`` came out ``1./HALF`` beside the ``half`` it meant."""
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert "    twice_half = 1. / half" in lines


def test_a_boz_local_parameter_is_its_integer_value(source: Path) -> None:
    """The pipeline takes a BOZ here and this did not, so ``mask`` fell to the
    token pass and came out as the bare text upper-cased."""
    lines, _ = build(source).render(node_of(source, "work"), "work")
    assert "    mask = 255" in lines


KEYWORD_RESULT = """\
module kw_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
contains
  function latent(t) result(lambda)
    real(r8), intent(in) :: t
    real(r8) :: lambda
    if (t > 273.15_r8) then
      lambda = 2.5e6_r8
    else
      lambda = 2.8e6_r8
    end if
  end function latent
end module kw_mod
"""


def test_a_result_named_after_a_python_keyword_is_renamed_in_its_initializer(
    tmp_path: Path,
) -> None:
    """``result(lambda)`` is legal Fortran and ``lambda = 0.0`` is not Python.
    The uses and the return were already spelled ``lambda_``; the UB-guard
    initializer emitted the source name and the file did not parse
    (CLM-ml's ``MLWaterVaporMod/LatVap``)."""
    path = tmp_path / "kw_mod.f90"
    path.write_text(KEYWORD_RESULT)
    lines, _ = build(path).render(node_of(path, "latent"), "latent")
    assert "    lambda_ = 0.0" in lines
    assert not any(line.strip().startswith("lambda =") for line in lines)
    compile("\n".join(lines), "<latent>", "exec")


REBASED_COMPONENT = """\
module con_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  integer, parameter :: mxpft = 3
  type con_t
    real(r8), allocatable :: slatop(:)
  end type con_t
  type(con_t), public :: con
contains
  subroutine init(this)
    class(con_t) :: this
    allocate (this%slatop(0:mxpft))
  end subroutine init

  subroutine pick(i, inst, v)
    integer, intent(in) :: i
    type(con_t), intent(in) :: inst
    real(r8), intent(out) :: v
    v = inst%slatop(i) + con%slatop(i)
  end subroutine pick
end module con_mod
"""


def test_a_component_subscript_shifts_by_its_allocated_lower_bound(tmp_path: Path) -> None:
    path = tmp_path / "con_mod.f90"
    path.write_text(REBASED_COMPONENT)
    lines, _ = build(path).render(node_of(path, "pick"), "pick")
    body = next(line for line in lines if "slatop" in line and "v =" in line)
    assert "inst.slatop[i]" in body or "inst.slatop[(i) - (0)]" in body, body
    assert "- 1" not in body


def test_an_associate_alias_inherits_its_selector_lower_bound(tmp_path: Path) -> None:
    """``associate (slatop => con%slatop)`` then ``slatop(i)``: the alias is
    a plain name in the body, and it has to shift the way the component
    does. CLM-ml writes every physics routine this way."""
    source = REBASED_COMPONENT.replace(
        "    v = inst%slatop(i) + con%slatop(i)\n",
        "    associate (s => con%slatop)\n    v = s(i)\n    end associate\n",
    )
    path = tmp_path / "con_mod.f90"
    path.write_text(source)
    lines, _ = build(path).render(node_of(path, "pick"), "pick")
    body = next(line for line in lines if "v =" in line)
    assert "- 1" not in body, body


def test_a_local_sized_by_a_derived_type_component_renders_the_attribute(
    tmp_path: Path,
) -> None:
    """``real(r8) :: albd(bounds%begp:bounds%endp)``: CLM sizes locals off the
    bounds object, and the bound tokenizer refused the ``%``."""
    source = REBASED_COMPONENT.replace(
        "  subroutine pick(i, inst, v)\n",
        "  subroutine sized(b, v)\n    type(con_t), intent(in) :: b\n"
        "    real(r8), intent(out) :: v\n    real(r8) :: work(b%mxpft)\n"
        "    work = 0._r8\n    v = work(1)\n  end subroutine sized\n\n"
        "  subroutine pick(i, inst, v)\n",
    ).replace(
        "    real(r8), allocatable :: slatop(:)\n",
        "    real(r8), allocatable :: slatop(:)\n    integer :: mxpft\n",
    )
    path = tmp_path / "con_mod.f90"
    path.write_text(source)
    lines, _ = build(path).render(node_of(path, "sized"), "sized")
    assert any("b.mxpft" in line for line in lines), lines


def test_a_local_with_a_declared_initializer_starts_from_it(tmp_path: Path) -> None:
    """``real(r8) :: minlwp = -2._r8`` came out ``minlwp = 0.0`` -- the
    UB-guard zero over the value the declaration gives (CLM-ml's
    SoilResistance, where every layer's evaporation then vanished)."""
    source = REBASED_COMPONENT.replace(
        "    real(r8), intent(out) :: v\n    v = inst%slatop(i) + con%slatop(i)\n",
        "    real(r8), intent(out) :: v\n    real(r8) :: floor_ = -2._r8\n"
        "    v = inst%slatop(i) + floor_\n",
    )
    assert "floor_ = -2._r8" in source
    path = tmp_path / "con_mod.f90"
    path.write_text(source)
    lines, _ = build(path).render(node_of(path, "pick"), "pick")
    assert any(line.strip().startswith("floor_ = ") and "-2" in line for line in lines), lines
    assert not any(line.strip() == "floor_ = 0.0" for line in lines)


def test_a_top_level_backward_goto_becomes_a_loop_region(source: Path) -> None:
    """``10 continue ... if (cond) ... go to 10`` at the subprogram's top
    level is a loop: the span from the label to the last such goto runs
    under ``while True``, the goto restarts it through ``_FGoto``, and
    falling off the end breaks."""
    lines, report = build(source).render(node_of(source, "climb"), "climb")
    assert "    while True:  # backward-goto region (label 10)" in lines
    assert any(line.lstrip() == "raise _FGoto('10')  # goto 10" for line in lines)
    assert "            break  # natural exit" in lines
    assert "        except _FGoto as _g:" in lines
    assert all(entry["status"] == "mechanical" for entry in report)
    opened = lines.index("    while True:  # backward-goto region (label 10)")
    closed = lines.index("        except _FGoto as _g:")
    inside = [line for line in lines[opened + 1 : closed] if line.lstrip().startswith("# B")]
    assert inside and all(line.startswith("            # B") for line in inside), inside


CHARS = """\
module chars_mod
  implicit none
  character(len=4), parameter :: choice_evhc = 'maxi'
  character(len=*), parameter :: letters = choice_evhc // 'x'
  character(len=*), parameter :: mixed = 'a' // nothere
contains
  function pick(x) result(y)
    real(8), intent(in) :: x
    real(8) :: y
    character(len=6), parameter :: choice = 'maxi'
    character(len=*), parameter :: tag = choice_evhc // repeat('x', 2)
    character(len=*), parameter :: odd = 'q' // nothere
    y = x
    if (choice == 'maxi') y = x * 2.0d0
  end function pick
end module chars_mod
"""


def test_a_local_character_parameter_references_the_folded_constant(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """#47: the prologue binds a character parameter to the constants
    module's folded, length-fitted value instead of re-rendering it (the
    token pass would upper-case 'maxi' and pass // through); one the
    frontend could not fold is queued, not emitted as Python's //."""
    path = tmp_path_factory.mktemp("chars") / "chars_mod.f90"
    path.write_text(CHARS)
    record = constants.extract(path)
    module = {p["name"]: p for p in record["module_parameters"]}
    assert (module["choice_evhc"]["kind"], module["choice_evhc"]["payload"]) == ("str", "maxi")
    assert module["letters"]["payload"] == "maxix"
    assert module["mixed"]["kind"] == "skip"
    local = {p["name"]: p for p in record["local_parameters"]}
    assert local["choice"]["payload"] == "maxi  "
    assert local["tag"]["payload"] == "maxixx"
    lines, _ = build(path).render(node_of(path, "pick"), "pick")
    assert "    choice = PICK__CHOICE" in lines
    assert "    tag = PICK__TAG" in lines
    assert any(
        "odd = None  # AGENT_QUEUE" in line and "character expression" in line for line in lines
    )
