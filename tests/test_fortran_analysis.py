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

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran import chunk, constants, expr, interface, use
from recast.fortran._parse import f03, parse, walk

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
  use shr_kind_mod, only: r8 => shr_kind_r8
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
    record = interface.extract(_write(tmp_path, "state.f90", STATE))
    subs = {s["name"]: s for s in record["subprograms"]}
    assert subs["teardown"]["module_state_written"] == ["table"]
    assert subs["teardown"]["module_state_read"] == []


def test_module_state_and_parameters_are_separated(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "state.f90", STATE))
    assert {s["name"] for s in record["module_state"]} == {"table", "counter", "threshold"}
    assert record["module_parameters"] == []


def test_a_read_in_a_condition_counts(tmp_path: Path) -> None:
    """Reads outside assignments are still reads. A kernel whose behaviour
    depends on module state it never assigns still cannot be called in
    isolation."""
    record = interface.extract(_write(tmp_path, "state.f90", STATE))
    bump = {s["name"]: s for s in record["subprograms"]}["bump"]
    assert "threshold" in bump["module_state_read"]
    assert bump["module_state_written"] == ["counter"]


# --- shapes, kinds, derived types --------------------------------------------


def test_explicit_and_assumed_bounds_are_distinguished(tmp_path: Path) -> None:
    """``ub`` of ``None`` means the bound has to be recovered at call time.
    Collapsing the two makes an assumed-shape dummy look allocatable-free and a
    generated wrapper allocate the wrong thing."""
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES))
    args = {a["name"]: a for a in record["subprograms"][0]["args"]}
    assert args["a"]["dims"] == [{"lb": "1", "ub": "n"}]
    assert args["b"]["dims"] == [{"lb": "1", "ub": None}]


def test_an_array_valued_function_reports_its_result_shape(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES))
    profile = record["subprograms"][0]
    assert profile["result"] == "p"
    assert profile["result_dtype"] == "float64"
    assert profile["result_dims"] == [{"lb": "1", "ub": "n"}]


def test_derived_type_components_report_allocatable_and_pointer(tmp_path: Path) -> None:
    """Both change what a translation has to emit before first use, and neither
    is recoverable from the dtype."""
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES))
    grid = record["types"]["grid_t"]
    assert (grid["lat"]["allocatable"], grid["lat"]["pointer"]) == (True, False)
    assert (grid["lon"]["allocatable"], grid["lon"]["pointer"]) == (False, True)
    assert (grid["dx"]["allocatable"], grid["dx"]["pointer"]) == (False, False)


def test_kinds_resolve_from_both_a_use_rename_and_selected_real_kind(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES))
    assert record["kind_map"]["r8"] == "float64", "use shr_kind_mod, only: r8 => shr_kind_r8"
    assert record["kind_map"]["rk"] == "float64", "selected_real_kind(12)"
    assert record["kind_map"]["sk"] == "float32", "selected_real_kind(6)"


def test_an_unresolvable_kind_is_named_not_defaulted(tmp_path: Path) -> None:
    """A silently wrong precision is the one error this stage can cause that no
    downstream type check catches -- it surfaces much later as a tolerance
    failure nobody can attribute."""
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES))
    locals_ = {loc["name"]: loc for loc in record["subprograms"][0]["locals"]}
    assert locals_["scratch"]["dtype"] == "UNKNOWN_REAL_KIND(mystery)"


def test_kind_assumptions_supply_what_the_tree_does_not_contain(tmp_path: Path) -> None:
    """The same knob as intent overrides, for the same reason: the fact is real,
    the source does not state it, and the frontend must not invent it."""
    src = _write(tmp_path, "shapes.f90", SHAPES.replace("r8 => shr_kind_r8", "r8 => other_r8"))
    assert interface.extract(src)["kind_map"].get("r8") is None
    assumed = interface.extract(src, kind_assumptions={"other_r8": "float64"})
    assert assumed["kind_map"]["r8"] == "float64"


def test_generic_interfaces_map_to_their_specific_procedures(tmp_path: Path) -> None:
    record = interface.extract(_write(tmp_path, "shapes.f90", SHAPES))
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
    assert got["hoisted_literals"]["F_273P15"]["value"] == "273.15"
    assert got["hoisted_literals"]["F_1P0E12"]["value"] == "1.0e12"
    assert got["literal_map"]["work"]["273.15"] == "F_273P15"


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
    record = interface.extract(_write(tmp_path, "local.f90", LOCAL_PARAMS))
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
    record = interface.extract(_write(tmp_path, "driver.f90", src))
    assert record["module"] == "driver"


def test_a_file_of_bare_subprograms_borrows_the_file_stem(tmp_path: Path) -> None:
    """There is no module name to use, but a Unit still needs a stable uid."""
    src = "subroutine loose(x)\n  real :: x\n  x = x + 1.0\nend subroutine loose\n"
    record = interface.extract(_write(tmp_path, "helpers.f90", src))
    assert record["module"] == "helpers"
    assert [s["name"] for s in record["subprograms"]] == ["loose"]
