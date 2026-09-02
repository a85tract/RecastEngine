"""Language-level facts the Fortran analysis must report.

Separate from ``test_fortran_frontend.py``, which tests the ``Frontend``
contract. These test what the analysis *says about Fortran*, on the smallest
source that exercises each rule.

Every case here was found by diffing this package against the pipeline it was
migrated from, run over ~500KB of real CAM source. That corpus lives in a
domain repository and cannot ship with the engine -- so the behaviours it
surfaced are pinned here on synthetic source instead, which is where a language
fact belongs anyway. Without this file, the coverage would leave with the
corpus.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import chunk, constants, expr, interface, use
from recast.fortran._parse import f03, parse, walk

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


STATE = """\
module state_mod
  implicit none
  real, allocatable :: table(:)
  integer :: counter
  real :: threshold
contains
  subroutine setup(n)
    integer, intent(in) :: n
    allocate(table(n))
    counter = 0
  end subroutine setup

  subroutine teardown
    deallocate(table)
  end subroutine teardown

  subroutine bump
    if (counter > threshold) counter = counter + 1
  end subroutine bump
end module state_mod
"""

SHAPES = """\
module shapes_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  integer, parameter :: rk = selected_real_kind(12)
  integer, parameter :: sk = selected_real_kind(6)

  type grid_t
    real(r8), allocatable :: lat(:)
    real(r8), pointer :: lon(:)
    real(r8) :: dx
  end type grid_t

  interface swap
    module procedure swap_int, swap_real
  end interface swap
contains
  function profile(n, a, b) result(p)
    integer, intent(in) :: n
    real(r8), intent(in) :: a(n)
    real(r8), intent(in) :: b(:)
    real(r8), allocatable :: c(:)
    real(r8) :: p(n)
    real(mystery) :: scratch
    p = a + b
  end function profile

  subroutine swap_int(i, j)
    integer, intent(inout) :: i, j
  end subroutine swap_int

  subroutine swap_real(x, y)
    real(r8), intent(inout) :: x, y
  end subroutine swap_real
end module shapes_mod
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text)
    return path


# --- module state effects ----------------------------------------------------


def test_deallocate_is_a_write_not_a_read(tmp_path: Path) -> None:
    """``deallocate(table)`` sets the allocation status to unallocated, which a
    translation spells ``table = None``. The pipeline this was migrated from
    counted it as a read, which put the one statement that destroys module
    state on the wrong side of every read/write gate."""
    record = interface.extract(_write(tmp_path, "state.f90", STATE), kind_assumptions=KINDS)
    subs = {s["name"]: s for s in record["subprograms"]}
    assert subs["teardown"]["module_state_written"] == ["table"]
    assert subs["teardown"]["module_state_read"] == []


def test_module_state_and_parameters_are_separated(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "state.f90", STATE), kind_assumptions=KINDS)
    assert {s["name"] for s in record["module_state"]} == {"table", "counter", "threshold"}
    assert record["module_parameters"] == []


def test_a_read_in_a_condition_counts(tmp_path: Path) -> None:
    """Reads outside assignments are still reads. A kernel whose behaviour
    depends on module state it never assigns still cannot be called in
    isolation."""
    record = interface.extract(_write(tmp_path, "state.f90", STATE), kind_assumptions=KINDS)
    bump = {s["name"]: s for s in record["subprograms"]}["bump"]
    assert "threshold" in bump["module_state_read"]
    assert bump["module_state_written"] == ["counter"]


# --- shapes, kinds, derived types --------------------------------------------


def test_explicit_and_assumed_bounds_are_distinguished(tmp_path: Path) -> None:
    """``ub`` of ``None`` means the bound has to be recovered at call time.
    Collapsing the two makes an assumed-shape dummy look allocatable-free and a
    generated wrapper allocate the wrong thing."""
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES), kind_assumptions=KINDS)
    args = {a["name"]: a for a in record["subprograms"][0]["args"]}
    assert args["a"]["dims"] == [{"lb": "1", "ub": "n"}]
    assert args["b"]["dims"] == [{"lb": "1", "ub": None}]


def test_an_array_valued_function_reports_its_result_shape(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES), kind_assumptions=KINDS)
    profile = record["subprograms"][0]
    assert profile["result"] == "p"
    assert profile["result_dtype"] == "float64"
    assert profile["result_dims"] == [{"lb": "1", "ub": "n"}]


def test_derived_type_components_report_allocatable_and_pointer(tmp_path: Path) -> None:
    """Both change what a translation has to emit before first use, and neither
    is recoverable from the dtype."""
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES), kind_assumptions=KINDS)
    grid = record["types"]["grid_t"]
    assert (grid["lat"]["allocatable"], grid["lat"]["pointer"]) == (True, False)
    assert (grid["lon"]["allocatable"], grid["lon"]["pointer"]) == (False, True)
    assert (grid["dx"]["allocatable"], grid["dx"]["pointer"]) == (False, False)


def test_a_component_carries_a_shape_spelled_on_the_dimension_attribute(tmp_path: Path) -> None:
    """``real, dimension(4) :: edge`` says what ``real :: edge(4)`` says, and
    reading only the second reported the component as a scalar -- not a
    refusal but a wrong answer, which every reader of the record inherits.
    Where an entity carries its own shape it wins, because Fortran lets
    ``dimension(4) :: a, b(7)`` give the two different ones."""
    src = _write(
        tmp_path,
        "comps.f90",
        """\
module comp_mod
  implicit none
  type box_t
    real, dimension(4) :: viaattr
    real :: viaentity(4)
    real, dimension(4) :: shared, own(7)
    real, dimension(:), allocatable :: dyn
  end type box_t
end module comp_mod
""",
    )
    box = interface.extract(src)["types"]["box_t"]
    four = [{"lb": "1", "ub": "4"}]
    assert box["viaattr"]["dims"] == four
    assert box["viaentity"]["dims"] == four
    assert box["shared"]["dims"] == four
    assert box["own"]["dims"] == [{"lb": "1", "ub": "7"}]
    assert box["dyn"]["dims"] == [{"lb": "1", "ub": None}], "a deferred shape is still rank 1"


def test_kinds_resolve_from_both_a_use_rename_and_selected_real_kind(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES), kind_assumptions=KINDS)
    assert record["kind_map"]["r8"] == "float64", "use precision_mod, only: r8 => wp_r8"
    assert record["kind_map"]["rk"] == "float64", "selected_real_kind(12)"
    assert record["kind_map"]["sk"] == "float32", "selected_real_kind(6)"


def test_an_unresolvable_kind_is_named_not_defaulted(tmp_path: Path) -> None:
    """A silently wrong precision is the one error this stage can cause that no
    downstream type check catches -- it surfaces much later as a tolerance
    failure nobody can attribute."""
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES), kind_assumptions=KINDS)
    locals_ = {loc["name"]: loc for loc in record["subprograms"][0]["locals"]}
    assert locals_["scratch"]["dtype"] == "UNKNOWN_REAL_KIND(mystery)"


def test_kind_assumptions_supply_what_the_tree_does_not_contain(tmp_path: Path) -> None:
    """The same knob as intent overrides, for the same reason: the fact is real,
    the source does not state it, and the frontend must not invent it."""
    src = _write(tmp_path, "shapes.f90", SHAPES.replace("r8 => wp_r8", "r8 => other_r8"))
    assert interface.extract(src, kind_assumptions=KINDS)["kind_map"].get("r8") is None
    assumed = interface.extract(src, kind_assumptions={"other_r8": "float64"})
    assert assumed["kind_map"]["r8"] == "float64"


SPELLINGS = """\
module spellings_mod
  use, intrinsic :: iso_fortran_env
  implicit none
  integer, parameter :: rk = real64
  integer, parameter :: wp = rk
  integer, parameter :: sp = real32
  integer, parameter :: qp = real128
  integer, parameter :: dp = kind(0.d0)
  integer, parameter :: def = kind(1.0)
  integer, parameter :: big = selected_int_kind(18)
  integer, parameter :: small = selected_int_kind(9)
contains
  subroutine spelled(a, b, c, n, m)
    real(wp), intent(in) :: a
    real(dp), intent(in) :: b
    real(qp), intent(in) :: c
    integer(big), intent(in) :: n
    integer(small), intent(in) :: m
  end subroutine spelled
end module spellings_mod
"""


def test_kinds_spelled_by_iso_fortran_env_resolve(tmp_path: Path) -> None:
    """``real64`` is how Fortran written since 2008 spells a double, and a
    frontend that only reads ``selected_real_kind`` leaves every argument of
    such a module untyped -- which reaches the f2py oracle as a dummy it
    cannot declare, and drops the subprogram out of the bit-exact gate."""
    record = interface.extract(_write(tmp_path, "spellings.f90", SPELLINGS))
    assert record["kind_map"]["rk"] == "float64"
    assert record["kind_map"]["sp"] == "float32"
    assert record["kind_map"]["dp"] == "float64", "kind(0.d0)"
    assert record["kind_map"]["def"] == "float32", "kind(1.0) is default real"


def test_a_kind_named_after_another_kind_resolves(tmp_path: Path) -> None:
    """``wp = rk`` where ``rk = real64``. Both are module parameters, and the
    alias is often declared before the thing it aliases, so one pass down the
    declarations is not enough."""
    record = interface.extract(_write(tmp_path, "spellings.f90", SPELLINGS))
    assert record["kind_map"]["wp"] == "float64"
    args = {a["name"]: a for a in record["subprograms"][0]["args"]}
    assert args["a"]["dtype"] == "float64"


def test_real128_stays_unresolved(tmp_path: Path) -> None:
    """numpy has no portable 128-bit float. Reporting float64 for one would be
    exactly the silent precision loss the UNKNOWN marker exists to prevent."""
    record = interface.extract(_write(tmp_path, "spellings.f90", SPELLINGS))
    assert "qp" not in record["kind_map"]
    args = {a["name"]: a for a in record["subprograms"][0]["args"]}
    assert args["c"]["dtype"] == "UNKNOWN_REAL_KIND(qp)"


def test_integer_kinds_are_read_by_width_not_assumed_default(tmp_path: Path) -> None:
    """``selected_int_kind(18)`` does not fit in 32 bits. An unresolvable
    integer kind still reports the default -- that is where the pipeline
    stood -- but a resolvable one is not thrown away."""
    record = interface.extract(_write(tmp_path, "spellings.f90", SPELLINGS))
    args = {a["name"]: a for a in record["subprograms"][0]["args"]}
    assert args["n"]["dtype"] == "int64"
    assert args["m"]["dtype"] == "int32"


def test_a_real_does_not_take_its_kind_from_an_integer_kind_parameter(tmp_path: Path) -> None:
    """One kind map holds both widths now. Reading it without the base type
    would turn ``real(big)`` into an int64 argument."""
    record = interface.extract(_write(tmp_path, "spellings.f90", SPELLINGS))
    assert interface.dtype_of("REAL", "big", record["kind_map"]) == "UNKNOWN_REAL_KIND(big)"


def test_an_assumed_size_dummy_has_the_rank_it_was_declared_with(tmp_path: Path) -> None:
    """``dx(*)`` is one dimension. fparser hangs two ``None`` children off the
    node, and a walk that descends into them reports rank 2 -- which the
    read/write check then refuses as a sequence-association rank mismatch, and
    the f2py wrapper declares as ``dx(None, None)``, which is not Fortran."""
    src = _write(
        tmp_path,
        "blas.f90",
        """\
module blas_mod
  implicit none
contains
  subroutine dscal(n, da, dx, work)
    integer, intent(in) :: n
    real, intent(in) :: da
    real, intent(inout) :: dx(*)
    real, intent(inout) :: work(3, *)
  end subroutine dscal
end module blas_mod
""",
    )
    args = {a["name"]: a for a in interface.extract(src)["subprograms"][0]["args"]}
    # The assumed-size dimension carries a marker: its ``ub`` of None is
    # otherwise indistinguishable from an assumed-shape ``(:)``, and only
    # assumed-size means the caller owns storage the callee cannot size.
    assert args["dx"]["dims"] == [{"lb": "1", "ub": None, "assumed_size": True}]
    assert args["work"]["dims"] == [
        {"lb": "1", "ub": "3"},
        {"lb": "1", "ub": None, "assumed_size": True},
    ]


def test_generic_interfaces_map_to_their_specific_procedures(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES), kind_assumptions=KINDS)
    assert record["generics"] == {"swap": ["swap_int", "swap_real"]}


# --- the zero-literal rule ---------------------------------------------------

LITERALS = """\
module lit_mod
  implicit none
  integer, parameter :: pcols = 16
contains
  subroutine work(x, n)
    real, intent(inout) :: x(:)
    integer, intent(in) :: n
    real :: buffer(pcols, 5)
    integer :: i
    do i = 1, n
      x(i) = x(i) * 2 + 0.5
      if (x(i) > 273.15) x(i) = 1.0e12
      x(i) = x(i) + 273.15_r8
    end do
  end subroutine work
end module lit_mod
"""


def test_structural_literals_are_left_alone(tmp_path: Path) -> None:
    """``do i = 1, n`` and ``* 2`` are structure, not physics. Hoisting them
    would bury the constants that matter in noise."""
    got = constants.extract(_write(tmp_path, "lit.f90", LITERALS))
    values = {e["value"] for e in got["hoisted_literals"].values()}
    assert not {"1", "2", "0.5"} & values


def test_physical_literals_are_hoisted_to_deterministic_names(tmp_path: Path) -> None:
    """Deterministic because both sides of a differential check have to be able
    to point at the same constant by the same name."""
    got = constants.extract(_write(tmp_path, "lit.f90", LITERALS))
    assert got["hoisted_literals"]["F32_273P15"]["value"] == "273.15"
    assert got["hoisted_literals"]["F32_1P0E12"]["value"] == "1.0e12"
    assert got["literal_map"]["work"]["273.15"] == "F32_273P15"


def test_a_default_kind_literal_is_a_different_constant_from_a_suffixed_one(
    tmp_path: Path,
) -> None:
    """``273.15`` is default REAL -- single precision -- and the value the
    program sees is that single promoted, 273.1499938964844. ``273.15_r8``
    is 273.15. Same digits, different numbers, so different names."""
    got = constants.extract(_write(tmp_path, "lit.f90", LITERALS))
    assert got["hoisted_literals"]["F32_273P15"]["value"] == "273.15"
    assert got["hoisted_literals"]["F_273P15"]["value"] == "273.15"
    assert got["literal_map"]["work"]["273.15_r8"] == "F_273P15"


def test_declaration_bounds_are_hoisted_too(tmp_path: Path) -> None:
    """The prologue that allocates ``buffer(pcols, 5)`` has to name the 5."""
    got = constants.extract(_write(tmp_path, "lit.f90", LITERALS))
    assert "I_5" in got["hoisted_literals"]
    assert any(loc.endswith(":decl") for loc in got["hoisted_literals"]["I_5"]["locations"])


def test_an_unclassifiable_initializer_refuses(tmp_path: Path) -> None:
    """An initializer over a name nothing defines is reported as unresolved
    rather than approximated."""
    src = """\
module refuse_mod
  implicit none
  real, parameter :: derived = mystery_constant * 2.0
end module refuse_mod
"""
    rec = constants.extract(_write(tmp_path, "refuse.f90", src))["module_parameters"][0]
    assert rec["kind"] == "skip"
    assert "mystery_constant" in rec["payload"], "the refusal names what it could not resolve"

    # Telling it the name comes from a sibling translation turns the refusal
    # into a resolvable expression, without the frontend having gone looking
    # for a generated file to import.
    known = constants.extract(
        _write(tmp_path, "refuse.f90", src), extern_names={"mystery_constant"}
    )["module_parameters"][0]
    assert known["kind"] == "expr"
    assert {"t": "ref", "v": "mystery_constant"} in known["payload"]


# --- one expression, two languages -------------------------------------------

CONSTS_A = """\
module phys_a
  implicit none
  real, parameter :: pi = 3.14159265358979
  real, parameter :: two_pi = 2.0 * pi
end module phys_a
"""

CONSTS_B = """\
module phys_b
  implicit none
  real, parameter :: r_earth = 6.371e6
  real, parameter :: circumference = two_pi * r_earth
end module phys_b
"""


def test_one_tree_renders_the_same_arithmetic_in_two_languages(tmp_path: Path) -> None:
    """The reason ``Expr`` exists. The ancestor of this code printed the same
    initializer twice, once as Fortran and once as Python, so that a stand-in
    module and a translated constants file would agree; two independently
    written printers is exactly how they stop agreeing. A fold over one tree
    cannot drift, because grouping is decided in ``render`` and not in either
    callback."""
    _write(tmp_path, "a.f90", CONSTS_A)
    _write(tmp_path, "b.f90", CONSTS_B)
    resolved = use.resolve(["circumference"], [tmp_path / "a.f90", tmp_path / "b.f90"])
    tree = {r["name"]: r["expr"] for r in resolved}["two_pi"]

    fortran = expr.render(tree, real=lambda t: f"{t}_r8", integer=str, name=str)
    python = expr.render(tree, real=str, integer=str, name=lambda n: f"C.{n}")
    assert fortran == "(2.0_r8 * pi)"
    assert python == "(2.0 * C.pi)"


def test_use_resolution_orders_dependencies_first(tmp_path: Path) -> None:
    """Safe to emit or evaluate top to bottom, across files."""
    _write(tmp_path, "a.f90", CONSTS_A)
    _write(tmp_path, "b.f90", CONSTS_B)
    resolved = use.resolve(["circumference"], [tmp_path / "a.f90", tmp_path / "b.f90"])
    order = [r["name"] for r in resolved]
    assert order.index("pi") < order.index("two_pi") < order.index("circumference")
    assert order.index("r_earth") < order.index("circumference")


def test_a_transitively_pulled_constant_is_marked_as_such(tmp_path: Path) -> None:
    _write(tmp_path, "a.f90", CONSTS_A)
    _write(tmp_path, "b.f90", CONSTS_B)
    resolved = use.resolve(["circumference"], [tmp_path / "a.f90", tmp_path / "b.f90"])
    got = {r["name"]: r for r in resolved}
    assert got["circumference"]["requested"] is True
    assert got["pi"]["requested"] is False


def test_a_missing_constant_fails_rather_than_vanishing(tmp_path: Path) -> None:
    """A physical constant that silently becomes undefined downstream costs far
    more to diagnose than a failure here that names it."""
    _write(tmp_path, "a.f90", CONSTS_A)
    with pytest.raises(use.UnresolvedConstant, match="gravit"):
        use.resolve(["gravit"], [tmp_path / "a.f90"])


def test_an_initializer_too_rich_to_model_refuses(tmp_path: Path) -> None:
    """These are sums, products and powers over literals. A function call is
    not approximated, it is declined."""
    src = """\
module rich_mod
  implicit none
  real, parameter :: weird = sqrt(2.0)
end module rich_mod
"""
    _write(tmp_path, "rich.f90", src)
    with pytest.raises(expr.UnsupportedExpression):
        use.resolve(["weird"], [tmp_path / "rich.f90"])


# --- block boundaries as shared vocabulary -----------------------------------


def test_chunking_gives_one_id_per_statement_or_whole_construct(tmp_path: Path) -> None:
    """Every later stage -- translation, instrumentation, read/write sets --
    addresses code by these ids, so a construct must be one block and not one
    block per statement inside it."""
    src = _write(tmp_path, "lit.f90", LITERALS)
    sub = walk(parse(src), f03.Subroutine_Subprogram)[0]
    blocks = chunk.blocks_of(sub, LITERALS.splitlines())
    assert [b["id"] for b in blocks] == ["B001"]
    assert blocks[0]["type"] == "Block_Nonlabel_Do_Construct"
    assert blocks[0]["known_type"] is True
    assert "end do" in blocks[0]["fortran"].lower(), "the whole construct, not its first line"


# --- local parameters, both spellings ----------------------------------------

LOCAL_PARAMS = """\
module local_mod
  implicit none
contains
  subroutine work(x)
    real, intent(inout) :: x
    real, parameter :: alpha = 0.61
    real :: beta
    parameter (beta = 9.80616)
    x = x * alpha + beta
  end subroutine work
end module local_mod
"""


def test_local_parameters_are_found_in_both_declaration_forms(tmp_path: Path) -> None:
    """F77's separate ``parameter (beta = ...)`` statement is still in the
    sources being modernized, and a constant missed there reappears as a bare
    magic number in the translation."""
    record = interface.extract(_write(tmp_path, "local.f90", LOCAL_PARAMS), kind_assumptions=KINDS)
    params = {p["name"]: p for p in record["subprograms"][0]["local_parameters"]}
    assert set(params) == {"alpha", "beta"}
    assert params["beta"]["init_expr"] == "9.80616"
    assert "beta" not in {loc["name"] for loc in record["subprograms"][0]["locals"]}, (
        "a named constant is not a local variable"
    )


def test_local_parameters_are_namespaced_by_subprogram(tmp_path: Path) -> None:
    """Two subprograms may each define ``alpha``. The emitted name has to keep
    them apart without the operator arranging it."""
    got = constants.extract(_write(tmp_path, "local.f90", LOCAL_PARAMS))
    consts = {p["name"]: p for p in got["local_parameters"]}
    assert consts["alpha"]["const"] == "WORK__ALPHA"
    assert consts["alpha"]["form"] == "declaration"
    assert consts["beta"]["form"] == "parameter_stmt"


# --- the three shapes a Fortran file comes in --------------------------------


def test_a_main_program_borrows_its_own_name(tmp_path: Path) -> None:
    src = "program driver\n  implicit none\n  call go()\nend program driver\n"
    record = interface.extract(_write(tmp_path, "driver.f90", src), kind_assumptions=KINDS)
    assert record["module"] == "driver"


def test_a_file_of_bare_subprograms_borrows_the_file_stem(tmp_path: Path) -> None:
    """There is no module name to use, but a Unit still needs a stable uid."""
    src = "subroutine loose(x)\n  real :: x\n  x = x + 1.0\nend subroutine loose\n"
    record = interface.extract(_write(tmp_path, "helpers.f90", src), kind_assumptions=KINDS)
    assert record["module"] == "helpers"
    assert [s["name"] for s in record["subprograms"]] == ["loose"]


# --- per-block read and write sets -------------------------------------------

RW = """\
module rw_mod
  implicit none
  real, allocatable :: pool(:)
contains
  subroutine fill(out, n, flag)
    real, intent(out) :: out(:)
    integer, intent(in) :: n
    logical, intent(in) :: flag
    integer :: i
    real :: acc
    acc = 0.0
    do i = 1, n
      acc = acc + real(pool(i), kind(acc))
      if (flag) out(i) = acc
    end do
    do while (acc > 1.0)
      acc = acc * 0.5
    end do
    allocate(pool(n))
    deallocate(pool)
    call sink(acc, out)
  end subroutine fill

  subroutine sink(x, y)
    real, intent(in) :: x
    real, intent(out) :: y(:)
    y = x
  end subroutine sink
end module rw_mod
"""


def _blocks(tmp_path: Path, sub_name: str = "fill") -> dict[str, dict[str, Any]]:
    from recast.fortran import rwset

    src = _write(tmp_path, "rw.f90", RW)
    record = interface.extract(src, kind_assumptions=KINDS)
    node = next(
        s
        for s in walk(parse(src), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(s, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == sub_name
    )
    scope = rwset.scope_for(record, sub_name)
    return {b["id"]: b for b in rwset.block_rwsets(node, scope)}


def test_a_counted_loop_writes_its_counter_and_reads_its_bounds(tmp_path: Path) -> None:
    loop = _blocks(tmp_path)["B002"]
    assert "i" in loop["writes"]
    assert "n" in loop["reads"]


def test_a_do_while_reads_its_condition_and_writes_no_counter(tmp_path: Path) -> None:
    """The pipeline this was migrated from assumed every DO was counted and
    raised a TypeError on the other two forms -- which crashed the read/write
    analysis outright on four of CAM's thirty translated modules."""
    loop = _blocks(tmp_path)["B003"]
    assert "acc" in loop["reads"]
    assert loop["writes"] == ["acc"], "the body's assignment, not a loop counter"


def test_a_kind_argument_is_not_a_read(tmp_path: Path) -> None:
    """``real(x, kind(acc))`` reads ``x``. Counting the kind would fire on
    nearly every line of CESM physics."""
    loop = _blocks(tmp_path)["B002"]
    assert "pool" in loop["reads"]
    assert "kind" not in loop["reads"]


def test_allocate_and_deallocate_are_writes(tmp_path: Path) -> None:
    blocks = _blocks(tmp_path)
    assert blocks["B004"]["writes"] == ["pool"] and "n" in blocks["B004"]["reads"]
    assert blocks["B005"] == {"id": "B005", "reads": [], "writes": ["pool"]}


def test_a_call_splits_its_arguments_by_declared_intent(tmp_path: Path) -> None:
    call = _blocks(tmp_path)["B006"]
    # ``y(:)`` is an intent(out) the callee cannot size, so it is the
    # caller's buffer (#36): passed in as well as returned, which the
    # emitter renders as a call argument and an unpack target -- so this
    # side records the read as well as the write (#38).
    assert call["reads"] == ["acc", "out"], "intent(in), and the buffer OUT"
    assert call["writes"] == ["out"], "intent(out)"


DERIVED = """\
module derived_mod
  implicit none
  type bundle_t
    real, allocatable :: q(:)
  end type bundle_t
contains
  subroutine drive(b, n)
    type(bundle_t), intent(inout) :: b
    integer, intent(in) :: n
    call refill(b % q, n)
    b % q(n) = 0.0
  end subroutine drive

  subroutine refill(slot, n)
    real, intent(out) :: slot(:)
    integer, intent(in) :: n
    slot = real(n)
  end subroutine refill
end module derived_mod
"""


def test_a_component_name_is_read_on_the_out_argument_path_only(tmp_path: Path) -> None:
    """``b % q`` writes ``b``. On an assignment, ``q`` is an attribute and not
    a symbol; passed to an intent(out) dummy, the pipeline this came from
    counts it as a read as well.

    The two disagree, and the disagreement is preserved. Resolving it would be
    a change to answers a bit-exact gate has been run against, and the two
    sites in CAM where it shows are both in modules with no translation to
    check the tidier answer against.
    """
    from recast.fortran import rwset

    src = _write(tmp_path, "derived.f90", DERIVED)
    record = interface.extract(src, kind_assumptions=KINDS)
    node = next(
        s
        for s in walk(parse(src), f03.Subroutine_Subprogram)
        if str(walk(s, f03.Subroutine_Stmt)[0].children[1]).lower() == "drive"
    )
    blocks = {b["id"]: b for b in rwset.block_rwsets(node, rwset.scope_for(record, "drive"))}
    # ``slot(:)`` is a caller-buffer OUT (#36), so ``b`` is read as well as
    # written on the call (#38); ``q`` is the pipeline's component read.
    assert blocks["B001"] == {"id": "B001", "reads": ["b", "n", "q"], "writes": ["b"]}, (
        "out-argument"
    )
    assert blocks["B002"] == {"id": "B002", "reads": ["n"], "writes": ["b"]}, "assignment"


def test_a_local_shadows_an_intrinsic_of_the_same_name(tmp_path: Path) -> None:
    """Fortran lets a variable be called ``sum``. Treating it as the intrinsic
    loses a real dataflow edge."""
    from recast.fortran import rwset

    src = _write(
        tmp_path,
        "shadow.f90",
        "module s_mod\ncontains\n"
        "  subroutine go(out)\n"
        "    real, intent(out) :: out\n"
        "    real :: sum\n"
        "    sum = 1.0\n"
        "    out = sum\n"
        "  end subroutine go\n"
        "end module s_mod\n",
    )
    record = interface.extract(src, kind_assumptions=KINDS)
    node = walk(parse(src), f03.Subroutine_Subprogram)[0]
    blocks = rwset.block_rwsets(node, rwset.scope_for(record, "go"))
    assert blocks[1] == {"id": "B002", "reads": ["sum"], "writes": ["out"]}


def test_the_intrinsic_table_carries_names_not_translations(tmp_path: Path) -> None:
    """The read/write analysis asked "is this an intrinsic" and never once what
    it maps to, so only the names are a Fortran fact. Keeping the mapping out
    is what lets this run without the 2,883-line emitter."""
    from recast.fortran import intrinsics

    assert {"sqrt", "sum", "present"} <= intrinsics.ALL
    assert all(isinstance(n, str) and n.islower() for n in intrinsics.ALL)


CONSTRUCTS = """\
module cons_mod
  implicit none
  real, pointer :: hook(:) => null()
  interface scale_it
    module procedure scale_scalar, scale_vector
  end interface scale_it
contains
  subroutine drive(v, m, mode, label, target_arr)
    real, intent(inout) :: v(:)
    logical, intent(in) :: m(:)
    integer, intent(in) :: mode
    character(len=32), intent(out) :: label
    real, intent(in), target :: target_arr(:)
    real :: s
    s = 1.0
    where (m)
      v = v * 2.0
    elsewhere
      v = 0.0
    end where
    select case (mode)
    case (1)
      s = 2.0
    case default
      s = 3.0
    end select
    call scale_it(v, s)
    call scale_it(s, s)
    hook => target_arr
    nullify(hook)
    write(label, *) s
    call endrun(label)
  end subroutine drive

  subroutine scale_scalar(x, f)
    real, intent(inout) :: x
    real, intent(in) :: f
    x = x * f
  end subroutine scale_scalar

  subroutine scale_vector(x, f)
    real, intent(inout) :: x(:)
    real, intent(in) :: f
    x = x * f
  end subroutine scale_vector
end module cons_mod
"""


def _construct_blocks(tmp_path: Path, **kw: Any) -> dict[str, dict[str, Any]]:
    from recast.fortran import rwset

    src = _write(tmp_path, "cons.f90", CONSTRUCTS)
    record = interface.extract(src, kind_assumptions=KINDS)
    node = next(
        s
        for s in walk(parse(src), f03.Subroutine_Subprogram)
        if str(walk(s, f03.Subroutine_Stmt)[0].children[1]).lower() == "drive"
    )
    scope = rwset.scope_for(record, "drive", **kw)
    return {b["id"]: b for b in rwset.block_rwsets(node, scope)}


def test_a_where_construct_reads_its_mask(tmp_path: Path) -> None:
    block = _construct_blocks(tmp_path)["B002"]
    assert "m" in block["reads"] and block["writes"] == ["v"]


def test_a_select_case_reads_the_selector_and_the_case_values(tmp_path: Path) -> None:
    block = _construct_blocks(tmp_path)["B003"]
    assert "mode" in block["reads"] and block["writes"] == ["s"]


def test_a_generic_call_dispatches_on_argument_rank(tmp_path: Path) -> None:
    """Scalar and vector overloads share their arities, so rank is the only
    discriminator -- and picking the wrong one swaps which argument is written.
    Here both overloads write their first argument, so the check is that
    dispatch resolved at all rather than falling through to the unknown-callee
    path, which would have read both arguments and written neither."""
    blocks = _construct_blocks(tmp_path)
    assert blocks["B004"] == {"id": "B004", "reads": ["s", "v"], "writes": ["v"]}
    assert blocks["B005"] == {"id": "B005", "reads": ["s"], "writes": ["s"]}


def test_pointer_association_and_nullify_are_writes(tmp_path: Path) -> None:
    blocks = _construct_blocks(tmp_path)
    assert blocks["B006"] == {"id": "B006", "reads": ["target_arr"], "writes": ["hook"]}
    assert blocks["B007"] == {"id": "B007", "reads": [], "writes": ["hook"]}


def test_an_internal_write_writes_its_unit(tmp_path: Path) -> None:
    """``write(label, *) s`` fills a character variable; ``write(6, *) s`` goes
    to a log and carries no dataflow at all."""
    block = _construct_blocks(tmp_path)["B008"]
    assert block == {"id": "B008", "reads": ["s"], "writes": ["label"]}


def test_an_unknown_callee_reads_its_arguments_and_writes_none(tmp_path: Path) -> None:
    """The safe default. Over-reporting a read costs a block a review."""
    block = _construct_blocks(tmp_path)["B009"]
    assert block == {"id": "B009", "reads": ["label"], "writes": []}


def test_an_externals_entry_says_which_arguments_a_procedure_writes(tmp_path: Path) -> None:
    """Without it the analysis has to assume a procedure it cannot see writes
    nothing, which is wrong in whichever direction it actually behaves."""
    blocks = _construct_blocks(tmp_path, externals={"endrun": {"out_positions": [0]}})
    assert blocks["B009"] == {"id": "B009", "reads": [], "writes": ["label"]}


# --- source-side rwset: shadowing and companion knowledge --------------------

SHADOWED = """\
module shadow_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
contains
  subroutine interpolate(gamhat, n)
    real(r8), intent(out) :: gamhat(n)
    integer, intent(in) :: n
    real(r8) :: gamma(4)
    integer :: i
    do i = 1, n
      gamhat(i) = gamma(i)
    end do
    call qsat_water(gamhat(1), gamhat(2), gamma(1), gamma(2))
    gamhat(1) = wv_sat_svp_water(gamma(1))
  end subroutine interpolate
end module shadow_mod
"""


def test_a_declared_array_shadows_an_intrinsic_name(tmp_path: Path) -> None:
    """zm_conv declares ``gamma(pcols,pver)``; reading ``gamma(i,k)`` is
    dataflow, not a call to GAMMA. The pipeline's source half skipped it as
    an intrinsic while its target half counted it -- one of the places the
    pipeline disagreed with itself, and here a corpus block showed the
    difference, so the tidier answer wins on both halves."""
    from recast.fortran import rwset

    record = interface.extract(_write(tmp_path, "shadow.f90", SHADOWED), kind_assumptions=KINDS)
    scope = rwset.scope_for(record, "interpolate")
    blocks = {b["id"]: b for b in rwset.block_rwsets(_sub_node(tmp_path, "interpolate"), scope)}
    assert "gamma" in blocks["B001"]["reads"]


def test_companion_externals_carry_the_siblings_intents(tmp_path: Path) -> None:
    """A call into an already-translated sibling resolves outside this file.
    Without its intents, every actual is conservatively a read; with them,
    the out positions are writes and a function reference is a call rather
    than a read -- the fact the pipeline's --companions flag carried."""
    from recast.fortran import rwset

    record = interface.extract(_write(tmp_path, "shadow.f90", SHADOWED), kind_assumptions=KINDS)
    node = _sub_node(tmp_path, "interpolate")

    blind = rwset.scope_for(record, "interpolate")
    blocks = {b["id"]: b for b in rwset.block_rwsets(node, blind)}
    assert "gamhat" in blocks["B002"]["reads"]  # unknown callee: reads only
    assert "wv_sat_svp_water" in blocks["B003"]["reads"]

    informed = rwset.scope_for(
        record,
        "interpolate",
        externals={
            "qsat_water": {"kind": "subroutine", "out_positions": [2, 3]},
            "wv_sat_svp_water": {"kind": "function", "out_positions": []},
        },
    )
    informed_blocks = {b["id"]: b for b in rwset.block_rwsets(node, informed)}
    assert "gamma" in informed_blocks["B002"]["writes"]
    assert "wv_sat_svp_water" not in informed_blocks["B003"]["reads"]


def test_companion_externals_derive_from_the_siblings_record(tmp_path: Path) -> None:
    sibling = """\
module sib_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
contains
  subroutine qsat_water(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    es = t
    qs = p
  end subroutine qsat_water
end module sib_mod
"""
    record = interface.extract(_write(tmp_path, "sib.f90", sibling), kind_assumptions=KINDS)
    table = interface.companion_externals(record)
    assert table["qsat_water"] == {"kind": "subroutine", "out_positions": [2, 3]}


def _sub_node(tmp_path: Path, name: str):
    tree = parse(tmp_path / "shadow.f90")
    return next(
        sub
        for sub in walk(tree, (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(sub, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == name
    )


def test_selected_int_kind_is_a_value_not_a_skip() -> None:
    """The source can compare against a kind value, so it is evaluated the
    way gfortran does: the smallest kind holding 10**N."""
    from recast.fortran.constants import classify_init

    assert classify_init("selected_int_kind(4)", set()) == ("int", 2)
    assert classify_init("selected_int_kind(6)", set()) == ("int", 4)
    assert classify_init("selected_int_kind(18)", set()) == ("int", 8)
    assert classify_init("kind(1.0d0)", set())[0] == "skip"
    assert classify_init("'GREGORIAN'", set()) == ("str", "GREGORIAN")
    assert classify_init("'isn''t'", set()) == ("str", "isn't")
    assert classify_init('"a b"', set()) == ("str", "a b")


INTERNALS = """\
module host_mod
  implicit none
  real :: shared(3)
contains
  subroutine outer(n, x)
    integer, intent(in) :: n
    real, intent(inout) :: x
    real :: scale
    integer :: k
    scale = 2.0
    call bump()
    x = f(x)
  contains
    subroutine bump()
      x = x * scale + n
    end subroutine bump
    function f(v) result(r)
      real, intent(in) :: v
      real :: r
      r = v + scale
    end function f
  end subroutine outer

  subroutine other(y)
    real, intent(inout) :: y
    y = f(y)
  contains
    function f(v) result(r)
      real, intent(in) :: v
      real :: r
      r = -v
    end function f
  end subroutine other

end module host_mod
"""


def _internals(tmp_path: Path) -> dict:
    from recast.fortran import interface

    src = tmp_path / "host_mod.f90"
    src.write_text(INTERNALS)
    return interface.extract(src, kind_assumptions=KINDS)


def test_an_internal_procedure_records_its_host_and_the_host_variables_it_touches(
    tmp_path: Path,
) -> None:
    from recast.fortran.interface import subprogram_key

    record = _internals(tmp_path)
    by_key = {subprogram_key(s): s for s in record["subprograms"]}
    bump = by_key["outer/bump"]
    assert bump["host"] == "outer"
    assert bump["host_vars"] == ["n", "scale", "x"]  # the host's dummies and locals it uses
    assert "host" not in by_key["outer"]


def test_two_internal_procedures_of_one_name_get_distinct_emitted_names(tmp_path: Path) -> None:
    from recast.fortran.interface import emit_name, subprogram_key

    record = _internals(tmp_path)
    by_key = {subprogram_key(s): s for s in record["subprograms"]}
    assert emit_name(by_key["outer/f"]) == "outer__f"
    assert emit_name(by_key["other/f"]) == "other__f"
    assert emit_name(by_key["outer/bump"]) == "bump"  # unique, so the pipeline's flat name


# --- constant initializers the pipeline learned to read after P2 --------------

CONSTANT_FORMS = """\
module const_forms
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: pi = 3.14159_r8
  real(r8), parameter :: unset = huge(1.0_r8)
  real(r8), parameter :: fuzz = epsilon(1.0_r8)
  real(r8), parameter :: least = tiny(1.0_r8)
  real(r8), parameter :: diag = sqrt(2.0_r8) * pi
  real(r8), parameter :: promoted = real(2, r8)
  integer, parameter :: mask = z'ff'
  real(r8), parameter :: super(3) = (/0.02_r8, 0.05_r8, 0.1_r8/)
  character(len=6), parameter :: names(2) = (/'SSLT01', 'SSLT02'/)
  real(r8), parameter :: modern(3) = [1.0_r8, 2.0_r8, 3.0_r8]
  real(r8), parameter :: derived(2) = (/pi, pi/)
contains
  subroutine work(x)
    real(r8), intent(inout) :: x
    x = x + (-2.5_r8)
  end subroutine work
end module const_forms
"""


def _forms(tmp_path: Path) -> dict:
    from recast.fortran import constants

    return constants.extract(_write(tmp_path, "const_forms.f90", CONSTANT_FORMS))


def _param(record: dict, name: str) -> dict:
    return next(p for p in record["module_parameters"] if p["name"] == name)


def test_an_intrinsic_call_in_a_constant_expression_is_carried_not_folded(
    tmp_path: Path,
) -> None:
    """Folding here would fold at a different precision than the compiler
    did. The record says which intrinsic it is; the target evaluates it."""
    record = _forms(tmp_path)
    kind, payload = _param(record, "diag")["kind"], _param(record, "diag")["payload"]
    assert kind == "expr"
    assert payload[0] == {"t": "call", "v": "sqrt", "args": [[{"t": "real", "v": "2.0"}]]}
    assert payload[-1] == {"t": "ref", "v": "pi"}


def test_a_type_inquiry_keeps_its_name(tmp_path: Path) -> None:
    record = _forms(tmp_path)
    for name, intrinsic in (("unset", "huge"), ("fuzz", "epsilon"), ("least", "tiny")):
        payload = _param(record, name)["payload"]
        assert payload == [{"t": "call", "v": intrinsic, "args": [[{"t": "real", "v": "1.0"}]]}]


def test_a_trailing_kind_argument_is_not_a_value(tmp_path: Path) -> None:
    """``real(2, r8)`` says what precision to evaluate in, which the target's
    float64 already is; passing it on would be an argument too many."""
    payload = _param(_forms(tmp_path), "promoted")["payload"]
    assert payload == [{"t": "call", "v": "real", "args": [[{"t": "int", "v": "2"}]]}]


def test_a_bare_boz_literal_has_a_value(tmp_path: Path) -> None:
    assert _param(_forms(tmp_path), "mask")["payload"] == 255


def test_an_array_constructor_in_either_spelling_is_read(tmp_path: Path) -> None:
    """Both spellings carry their values through, by two different routes,
    and which route a parameter takes is the pipeline's rule rather than a
    choice: the declaration path takes ``[...]`` and spells the elements as
    kind-stripped source text, and ``(/.../)`` goes to the classifier, which
    spells each element ``np.float64('...')``. Emitting one where the other
    is expected is a difference a byte-for-byte differential reports."""
    record = _forms(tmp_path)

    modern = _param(record, "modern")
    assert modern["kind"] == "array", "[...] takes the declaration route"
    assert "1.0" in modern["payload"]

    older = _param(record, "super")
    assert older["kind"] == "expr", "(/.../) takes the classifier route"
    assert older["payload"] == [
        {
            "t": "spelled",
            "v": "np.array([np.float64('0.02'), np.float64('0.05'), np.float64('0.1')])",
        }
    ]

    names = _param(record, "names")
    assert names["kind"] == "expr"
    assert "SSLT01" in names["payload"][0]["v"]
    assert "dtype=object" in names["payload"][0]["v"], "character elements"


def test_a_constructor_over_names_is_skipped_rather_than_approximated(
    tmp_path: Path,
) -> None:
    """Its elements are not literals, so the kind-suffix strip that makes the
    literal case exact does not apply."""
    assert _param(_forms(tmp_path), "derived")["kind"] == "skip"


def test_a_signed_literal_node_is_collected(tmp_path: Path) -> None:
    """fparser reads a sign as part of the literal in some positions -- a DATA
    value, a complex component -- and as a unary minus in others. A walk that
    knew only the unsigned node left the first kind with no name to emit."""
    from recast.fortran._parse import f03, parser
    from recast.fortran.constants import literals_with_lines

    parser()
    found = literals_with_lines(f03.Signed_Real_Literal_Constant("-2.5_r8"))
    assert [(text, is_real) for text, is_real, _ in found] == [("-2.5_r8", True)]


def test_an_internal_procedure_keeps_its_hoisted_literals(tmp_path: Path) -> None:
    """The unit's Facts name subprograms host-qualified; the constants record
    keys on the plain name, because a hoisted constant is named from its
    digits and two procedures writing 0.25 share it. Narrowing one against
    the other left every literal inside an internal procedure with no name to
    emit -- 79 refusals across the public corpus."""
    from recast.fortran.frontend import FortranFrontend
    from recast.model import Unit

    source = """\
module inner_lit
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine outer(x)
    real(r8), intent(inout) :: x
    call inner()
  contains
    subroutine inner()
      x = x * 0.25_r8
    end subroutine inner
  end subroutine outer
end module inner_lit
"""
    root = tmp_path / "tree"
    root.mkdir()
    (root / "inner_lit.f90").write_text(source)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(root) if u.uid == "fortran:inner_lit")
    facts = frontend.analyze(unit, root)
    assert facts.constants["literal_map"]["inner"]["0.25_r8"] == "F_0P25"
    assert isinstance(unit, Unit)


def test_a_write_only_f77_dummy_is_given_the_intent_its_use_shows(tmp_path: Path) -> None:
    """A dummy declared without INTENT gets the intent its use shows.

    Assigned and never read is intent(out); assigned and also read is
    intent(inout), because Fortran passes by reference and the update is the
    caller's to see. Without either attribute the return convention drops the
    dummy from the signature and from the return, and the value is lost.
    """
    from recast.fortran import interface

    source = """\
module f77ish
  implicit none
contains
  subroutine rates(x, made, used, buf, n, onward)
    real :: x, made, used, buf(4), onward
    integer :: n
    made = x * 2.0
    used = x + used
    buf(1) = x
    n = 3
    call elsewhere(onward)
  end subroutine rates
end module f77ish
"""
    record = interface.extract(_write(tmp_path, "f77ish.f90", source), kind_assumptions=KINDS)
    intents = {a["name"]: a["intent"] for a in record["subprograms"][0]["args"]}
    assert intents["made"] == "OUT"  # assigned, never read
    assert intents["n"] == "OUT"
    assert intents["used"] == "INOUT"  # read on its own right-hand side
    assert intents["x"] == "UNKNOWN"  # only read
    assert intents["buf"] == "UNKNOWN"  # an array mutates through its buffer
    assert intents["onward"] == "UNKNOWN"  # only passed on; its fate is the callee's


def test_a_module_allocatable_records_the_lower_bound_its_allocate_gave_it(
    tmp_path: Path,
) -> None:
    """``allocate(x(0:n))`` sets a bound the declaration does not carry, and
    the allocate is in the init routine while the references are everywhere
    else. Without the module-wide record each of those gets the blanket
    one-based shift and lands a slot off."""
    from recast.fortran import interface
    from recast.fortran.interface import CONFLICTING_BOUNDS

    source = """\
module allocs
  implicit none
  integer, parameter :: nmax = 8
  real, allocatable :: grid(:), agree(:), clash(:), local_bound(:)
contains
  subroutine init(n)
    integer, intent(in) :: n
    integer :: helper
    helper = n
    allocate(grid(0:nmax))
    allocate(agree(0:nmax))
    allocate(clash(0:nmax))
    allocate(local_bound(helper:nmax))
  end subroutine init

  subroutine again()
    allocate(agree(0:nmax))
    allocate(clash(2:nmax))
  end subroutine again
end module allocs
"""
    bounds = interface.extract(_write(tmp_path, "allocs.f90", source))["module_allocate_bounds"]
    assert [d["lb"] for d in bounds["grid"]] == ["0"]
    assert [d["lb"] for d in bounds["agree"]] == ["0"], "two allocates that agree are one answer"
    assert bounds["clash"] == CONFLICTING_BOUNDS, "0 and 2 cannot both be the shift"
    assert bounds["local_bound"] == CONFLICTING_BOUNDS, "a local of the allocating routine"


DATA_CONSTS = """\
module datac_mod
  implicit none
contains
  subroutine seeded(out)
    real, intent(out) :: out(3)
    real :: table(3)
    integer :: flag
    data table /1.5, 2.5, 3.5/
    data flag /7/
    out = table * flag
  end subroutine seeded
end module datac_mod
"""


def test_a_data_statement_s_values_are_hoisted_like_any_other_literal(tmp_path: Path) -> None:
    """DATA lives in the specification part, so the sweep over the execution
    part never sees its values. Without a name here the translation of the
    DATA statement has nothing to emit and refuses -- on statements the
    pipeline translates."""
    src = _write(tmp_path, "datac.f90", DATA_CONSTS)
    record = constants.extract(src)
    where = {
        name: entry["locations"]
        for name, entry in record["hoisted_literals"].items()
        if any(loc.endswith(":data") for loc in entry["locations"])
    }
    assert where, "no literal was attributed to a DATA statement"
    assert record["literal_map"]["seeded"]["1.5"] in where
    assert record["literal_map"]["seeded"]["7"] in where


def test_a_data_statement_writes_its_objects_and_reads_its_values(tmp_path: Path) -> None:
    """The general statement rule sees names in expression positions, so it
    reported a DATA object as a read and nothing as written -- the exact
    inverse of what the emitted assignments do, and the read/write gate
    compares against exactly this."""
    from recast.fortran._parse import parse
    from recast.fortran.rwset import block_rwsets, scope_for

    src = _write(tmp_path, "datac.f90", DATA_CONSTS)
    record = interface.extract(src)
    subprogram = next(
        s
        for s in walk(parse(src), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(s, f03.Subroutine_Stmt)[0].children[1]).lower() == "seeded"
    )
    blocks = {b["id"]: b for b in block_rwsets(subprogram, scope_for(record, "seeded"))}
    assert blocks["D001"] == {"id": "D001", "reads": [], "writes": ["table"]}
    assert blocks["D002"] == {"id": "D002", "reads": [], "writes": ["flag"]}


ARRAY_PARAMS = """\
module arrp_mod
  implicit none
contains
  subroutine sides(n)
    integer, intent(in) :: n
    integer, dimension(4), parameter :: imin = (/1, 0, 1, 1/)
    integer, dimension(4), parameter :: shift = (/-1, -1, 0, 1/)
    integer, dimension(4), parameter :: imax = (/nc, nc, nc, nc + 1/)
    real, dimension(2), parameter :: gains = (/1.5_r8, 2.5_r8/)
  end subroutine sides
end module arrp_mod
"""


def test_an_array_parameter_shaped_on_the_attribute_still_gets_a_value(tmp_path: Path) -> None:
    """``integer, dimension(4), parameter :: imin = (/1,0,1,1/)`` carries its
    shape on the attribute, so the declaration path -- which looks for a shape
    on the entity -- passed it by, and it was then skipped as an "array
    constructor over more than literals" when every element is one."""
    record = constants.extract(_write(tmp_path, "arrp.f90", ARRAY_PARAMS))
    by_name = {p["name"]: p for p in record["local_parameters"]}
    assert by_name["imin"]["kind"] == "expr"
    assert by_name["imin"]["payload"] == [{"t": "spelled", "v": "np.array([1, 0, 1, 1])"}]


def test_a_negative_element_survives_fparser_s_spacing(tmp_path: Path) -> None:
    """fparser writes ``-1`` as ``- 1``. The spacing is the parser's, not the
    source's, and an element is a single literal or it is not one at all."""
    record = constants.extract(_write(tmp_path, "arrp.f90", ARRAY_PARAMS))
    by_name = {p["name"]: p for p in record["local_parameters"]}
    assert by_name["shift"]["payload"] == [{"t": "spelled", "v": "np.array([-1, -1, 0, 1])"}]


def test_a_constructor_over_names_says_which_name(tmp_path: Path) -> None:
    """``(/nc, nc, nc, nc + 1/)`` cannot be evaluated here, and the reason has
    to name ``nc`` the way every other unevaluable expression does -- "more
    than literals" said only that something was wrong."""
    record = constants.extract(_write(tmp_path, "arrp.f90", ARRAY_PARAMS))
    by_name = {p["name"]: p for p in record["local_parameters"]}
    assert by_name["imax"]["kind"] == "skip"
    assert "unknown name 'nc'" in by_name["imax"]["payload"]


def test_a_call_to_a_host_associated_procedure_is_a_call_not_a_read(tmp_path: Path) -> None:
    """An internal procedure is filed under ``host/name``; a call site spells
    the bare name.

    ``scope_for`` keyed its subprogram table by ``subprogram_key``, which is
    right for choosing which subprogram to analyse and wrong for the only
    question the table is asked afterwards -- "is this name a call, or an
    array being indexed". ``norm(n, a)`` therefore looked like an array
    element read, and the translation counts it as a call, so the two sides
    disagreed on every block that calls a host-associated procedure. Keyed by
    the bare name, as the pipeline this was migrated from does.
    """
    source = """\
module hosted
implicit none
contains
subroutine outer(n, a, out)
  integer, intent(in) :: n
  real, intent(in) :: a(n)
  real, intent(out) :: out
  out = norm(n, a)
contains
  function norm(m, v) result(r)
    integer, intent(in) :: m
    real, intent(in) :: v(m)
    real :: r
    r = sum(v(1:m))
  end function norm
end subroutine outer
end module hosted
"""
    from recast.fortran import rwset

    src = _write(tmp_path, "hosted.f90", source)
    record = interface.extract(src)
    scope = rwset.scope_for(record, "outer")
    assert "norm" in scope.subprograms, "the bare name is the spelling a call site uses"
    node = next(
        s
        for s in walk(parse(src), (f03.Subroutine_Subprogram,))
        if str(walk(s, (f03.Subroutine_Stmt,))[0].children[1]).lower() == "outer"
    )
    reads = set().union(*(set(b["reads"]) for b in rwset.block_rwsets(node, scope)))
    assert "norm" not in reads, "a call is not a read of its own name"
    assert {"n", "a"} <= reads


ASSOCIATE = """\
module assoc_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  type inst_t
    real(r8), pointer :: dpai(:,:)
    real(r8), pointer :: cp(:,:)
  end type inst_t
contains
  subroutine heat(n, filter, inst)
    integer, intent(in) :: n
    integer, intent(in) :: filter(:)
    type(inst_t), intent(inout) :: inst
    integer :: fp, p
    real(r8) :: w
    associate ( &
    dpai => inst%dpai , &
    cp   => inst%cp     &
    )
    do fp = 1, n
       p = filter(fp)
       if (dpai(p,1) > 0._r8) then
          w = dpai(p,1) * 2._r8
          cp(p,1) = w
       end if
    end do
    end associate
  end subroutine heat
end module assoc_mod
"""


def test_an_associate_binds_its_aliases_and_analyses_its_body(tmp_path: Path) -> None:
    """``associate (a => x%c)`` fell through to the conservative fallback,
    which reported every name in the whole construct as a read and none as a
    write -- so every block of a model whose physics is written this way
    (CLM-ml: all of it) failed the static gate. The emitter spells the
    construct as alias assignments followed by the body, and the sets here
    now say the same."""
    from recast.fortran import rwset

    src = _write(tmp_path, "assoc.f90", ASSOCIATE)
    record = interface.extract(src, kind_assumptions=KINDS)
    node = next(
        s
        for s in walk(parse(src), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(s, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == "heat"
    )
    blocks = {b["id"]: b for b in rwset.block_rwsets(node, rwset.scope_for(record, "heat"))}
    (block,) = blocks.values()
    assert {"dpai", "cp", "fp", "p", "w"} <= set(block["writes"])
    assert {"inst", "n", "filter", "dpai", "w"} <= set(block["reads"])
    assert "cp" not in block["reads"]


REBASED_COMPONENT = """\
module con_mod
  use precision_mod, only: r8 => wp_r8
  implicit none
  integer, parameter :: mxpft = 3
  type con_t
    real(r8), allocatable :: slatop(:)
    real(r8), allocatable :: plain(:)
    real(r8), pointer :: tbi(:,:)
  end type con_t
  type(con_t), public :: con
contains
  subroutine init(this, begp, endp)
    class(con_t) :: this
    integer, intent(in) :: begp, endp
    allocate (this%slatop(0:mxpft))
    allocate (this%plain(mxpft))
    allocate (this%tbi(begp:endp, 0:mxpft))
  end subroutine init
end module con_mod
"""


def test_a_component_allocated_from_zero_carries_its_lower_bound(tmp_path: Path) -> None:
    """``allocate (this%slatop(0:mxpft))`` in a type's init routine sets a
    lower bound the ``(:)`` declaration does not carry; without it every
    ``pftcon%slatop(itype)`` in the model was shifted by one and read the
    wrong plant functional type (caught bit-exact on CLM-ml's
    LeafHeatCapacity)."""
    src = _write(tmp_path, "con.f90", REBASED_COMPONENT)
    record = interface.extract(src, kind_assumptions=KINDS)
    comps = record["types"]["con_t"]
    assert comps["slatop"]["allocated_dims"] == [{"lb": "0", "ub": "mxpft"}]
    assert "allocated_dims" not in comps["plain"]
    # Per axis: ``begp`` is a local nobody else can see, so that axis keeps
    # the unit origin; the ``0:`` beside it is still recorded (CLM-ml's
    # ``tbi_profile(begp:endp, 0:nlevmlcan)`` was read one layer off).
    assert comps["tbi"]["allocated_dims"] == [{"lb": "1", "ub": "endp"}, {"lb": "0", "ub": "mxpft"}]


def test_a_subscript_reads_the_names_in_its_declared_lower_bound(tmp_path: Path) -> None:
    """``rho(bounds%begp:bounds%endp)`` then ``rho(p)``: the address is
    ``p - bounds%begp``, which the translation spells out and the source
    computes silently; both sides read ``bounds`` (CLM-ml's SolarRadiation)."""
    from recast.fortran import rwset

    source = REBASED_COMPONENT.replace(
        "  subroutine init(this, begp, endp)\n",
        "  subroutine use_it(b, p, v)\n    type(con_t), intent(in) :: b\n"
        "    integer, intent(in) :: p\n    real(r8), intent(out) :: v\n"
        "    real(r8) :: rho(b%mxpft:2*b%mxpft)\n    rho(p) = 1._r8\n    v = rho(p)\n"
        "  end subroutine use_it\n\n  subroutine init(this, begp, endp)\n",
    ).replace(
        "    real(r8), allocatable :: slatop(:)\n",
        "    real(r8), allocatable :: slatop(:)\n    integer :: mxpft\n",
    )
    src = _write(tmp_path, "lb.f90", source)
    record = interface.extract(src, kind_assumptions=KINDS)
    node = next(
        s
        for s in walk(parse(src), (f03.Subroutine_Subprogram, f03.Function_Subprogram))
        if str(walk(s, (f03.Subroutine_Stmt, f03.Function_Stmt))[0].children[1]).lower() == "use_it"
    )
    blocks = rwset.block_rwsets(node, rwset.scope_for(record, "use_it"))
    assert all("b" in b["reads"] for b in blocks), blocks


def test_an_interface_bodys_dummies_are_not_module_state(tmp_path):
    """``interface / subroutine func(x, val)`` declares the interface's
    dummies, not module variables; a subprogram with its own dummy ``x``
    reads no module state through it."""
    from recast.fortran import interface

    src = tmp_path / "solver_mod.f90"
    src.write_text(
        "module solver_mod\n"
        "  implicit none\n"
        "  real(8) :: shared = 1.0d0\n"
        "  interface\n"
        "    subroutine func(x, val)\n"
        "      real(8), intent(in) :: x\n"
        "      real(8), intent(out) :: val\n"
        "    end subroutine func\n"
        "  end interface\n"
        "contains\n"
        "  function twice(x) result(y)\n"
        "    real(8), intent(in) :: x\n"
        "    real(8) :: y\n"
        "    y = 2.0d0 * x * shared\n"
        "  end function twice\n"
        "end module solver_mod\n"
    )
    record = interface.extract(src)
    assert [m["name"] for m in record["module_state"]] == ["shared"]
    (twice,) = record["subprograms"]
    assert twice["module_state_read"] == ["shared"]


def test_a_malformed_end_statement_does_not_exit_the_process(tmp_path: Path) -> None:
    """fparser's reader logs ``expected <subroutine-name> is X but got Y``
    and then calls ``sys.exit(1)``, which ended the discovery of a whole
    tree on one stub file (E3SM's external_models/emi/.../clm_varctl.F90).
    The reader is told not to exit: the line is ignored, as its own message
    says, and the parse answers for itself."""
    source = tmp_path / "bad_end.f90"
    source.write_text(
        "module bad_end\n"
        "  implicit none\n"
        "  logical :: flag = .false.\n"
        "contains\n"
        "  subroutine set_flag(flag_in)\n"
        "    logical, intent(in) :: flag_in\n"
        "    flag = flag_in\n"
        "  end subroutine set_other_flag\n"
        "end module bad_end\n"
    )
    try:
        parse(source)
    except SystemExit:  # pragma: no cover - the defect
        pytest.fail("parse called sys.exit on a malformed end statement")
    except Exception:  # a syntax error is an acceptable answer
        pass


def test_a_format_statement_does_not_stop_the_read_write_analysis(tmp_path: Path) -> None:
    """``FORMAT(I4.4, '-', I2.2)``: fparser hangs the *class*
    ``Int_Literal_Constant`` under the ``Data_Edit_Desc``, and walking it as
    a node raised ``TypeError: 'property' object is not iterable`` out of
    ``expr_reads`` -- the whole module's analysis (ELM's histFileMod) with it."""
    from recast.fortran.rwset import block_rwsets, scope_for

    source = tmp_path / "fmt.f90"
    source.write_text(
        "module fmt_mod\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine stamp(yr, mon, day, text)\n"
        "    integer, intent(in) :: yr, mon, day\n"
        "    character(len=10), intent(out) :: text\n"
        "    write(text, 10) yr, mon, day\n"
        "10  format(I4.4, '-', I2.2, '-', I2.2)\n"
        "  end subroutine stamp\n"
        "end module fmt_mod\n"
    )
    record = interface.extract(source)
    node = next(iter(walk(parse(source), f03.Subroutine_Subprogram)))
    blocks = block_rwsets(node, scope_for(record, "stamp"))
    assert any("text" in b["writes"] for b in blocks)


def test_an_internal_functions_result_is_not_host_associated(tmp_path: Path) -> None:
    """``get_tolerance(b) result(tol)`` inside a host that declares its own
    ``tol``: the result variable shadows the host's, so it is not
    host-associated -- reported as such, it was passed as a trailing actual
    the Fortran does not have (translator #43). A host variable the function
    really reads stays reported."""
    source = tmp_path / "shadow.f90"
    source.write_text(
        "module shadow\n"
        "  implicit none\n"
        "contains\n"
        "  subroutine bracket(b, outv)\n"
        "    real(8), intent(in) :: b\n"
        "    real(8), intent(out) :: outv\n"
        "    real(8) :: tol, scale\n"
        "    scale = 2.0d0\n"
        "    tol = get_tolerance(b)\n"
        "    outv = tol\n"
        "  contains\n"
        "    pure function get_tolerance(b) result(tol)\n"
        "      real(8), intent(in) :: b\n"
        "      real(8) :: tol\n"
        "      tol = scale * abs(b)\n"
        "    end function get_tolerance\n"
        "  end subroutine bracket\n"
        "end module shadow\n"
    )
    record = interface.extract(source)
    inner = next(s for s in record["subprograms"] if s["name"] == "get_tolerance")
    assert inner["host"] == "bracket"
    assert inner.get("host_vars") == ["scale"]


def test_a_character_expression_stays_a_skip_once_literals_are_values() -> None:
    """The trap behind the character-literal rule: once ``ascii_lowercase`` is a
    known name, ``ascii_lowercase // accented_lowercase`` would reach the
    token route, which has no rule for ``//`` and emits ``A / / B``. numfor's
    ``strings.f90`` is the regression: one shared constants module, twenty
    units importing it."""
    from recast.fortran.constants import classify_init

    known = {"ascii_lowercase", "accented_lowercase"}
    assert classify_init("'abc'", known) == ("str", "abc")
    assert classify_init("'it''s'", known) == ("str", "it's")
    kind, why = classify_init("ascii_lowercase // accented_lowercase", known)
    assert kind == "skip" and "character expression" in why
    assert classify_init("' ' // achar(9)", known)[0] == "skip"
    assert classify_init("new_line('a')", known)[0] == "skip"
