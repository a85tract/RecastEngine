"""Tests for symbol resolution.

The scoping cases are the ones that matter. Fortran lets a subprogram declare
a local with the same name as a module constant, and a translation that binds
the local to the constant does not merely read the wrong value -- on assignment
it writes through to something the rest of the program shares.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.fortran import constants as fortran_constants
from recast.fortran import interface, semantics
from recast.fortran._parse import f03, parser
from recast.transform.numpy import names
from recast.transform.rules import NoRule

KINDS = {"wp_r8": "float64", "wp_r4": "float32", "wp_i8": "int64"}
"""What the fixtures' own precision module would have said, supplied the way
the frontend documents: a kind the tree use-imports from a file it does not
contain."""


SOURCE = """\
module phys_mod
  implicit none
  real, parameter :: pi = 3.14159
  real, parameter :: tmin = 173.0
  real :: cached
  real :: premib
contains
  subroutine shadowing(tmin, n)
    real, intent(in) :: tmin
    integer, intent(in) :: n
    real, parameter :: local_p = 0.61
    real :: scratch
    integer :: lambda
    scratch = tmin + pi + local_p + cached + n + lambda + premib
    if (scratch > 273.15) scratch = 6.371e6
  end subroutine shadowing
end module phys_mod
"""


def _records(tmp_path: Path):
    parser()
    src = tmp_path / "phys.f90"
    src.write_text(SOURCE)
    return interface.extract(src, kind_assumptions=KINDS), fortran_constants.extract(src)


@pytest.fixture
def table(tmp_path: Path) -> names.Names:
    iface, consts = _records(tmp_path)
    sem = semantics.for_subprogram(iface, "shadowing")
    return names.for_subprogram(
        sem,
        consts,
        use_parameters={"gravit": "GRAVIT"},
        companion_globals={"premib": "_cf2.premib", "rhminl": "_cf2.rhminl"},
        use_bindings={"wp_r8": "_precision_mod.wp_r8"},
    )


# --- scoping -----------------------------------------------------------------


def test_a_dummy_shadows_a_module_parameter_of_the_same_name(table) -> None:
    """``tmin`` is both a module constant and this routine's argument. Binding
    it to the constant would read the wrong value, and assigning to it would
    write through to a constant every other routine reads."""
    assert table.symbol("tmin") == "tmin"
    assert table.module_parameters["tmin"] == "TMIN", "the constant still exists"


def test_a_module_parameter_not_shadowed_becomes_its_constant(table) -> None:
    assert table.symbol("pi") == "PI"


def test_a_local_parameter_keeps_its_own_name(table) -> None:
    assert table.symbol("local_p") == "local_p"


def test_module_state_is_not_renamed(table) -> None:
    """State is a variable, not a constant. Upper-casing it would suggest
    otherwise to everyone reading the output."""
    assert table.symbol("cached") == "cached"


def test_a_use_imported_constant_resolves_through_its_table(table) -> None:
    assert table.symbol("gravit") == "GRAVIT"


def test_a_name_that_collides_with_a_keyword_is_renamed(table) -> None:
    assert table.symbol("lambda") == "lambda_"


def test_resolution_is_case_insensitive(table) -> None:
    assert table.symbol("PI") == table.symbol("pi") == "PI"


# --- literals ----------------------------------------------------------------


def test_a_whitelisted_literal_is_written_out(table) -> None:
    assert table.literal(f03.Int_Literal_Constant("2")) == "2"
    assert table.literal(f03.Real_Literal_Constant("0.5")) == "0.5"


def test_a_hoisted_literal_resolves_to_its_constant(table) -> None:
    assert table.literal(f03.Real_Literal_Constant("273.15")) == "F32_273P15"


def test_a_kind_suffix_does_not_hide_a_whitelisted_value(table) -> None:
    assert table.literal(f03.Real_Literal_Constant("1.0_r8")) == "1.0"


def test_an_unhoisted_literal_refuses(table) -> None:
    """Emitting the bare number would put a magic number back into code the
    zero-literal rule had just cleaned, and the gate that would notice runs
    much later than this does."""
    with pytest.raises(NoRule, match="never hoisted"):
        table.literal(f03.Real_Literal_Constant("2.71828"))


def test_only_this_subprogram_s_hoisted_names_are_visible(tmp_path: Path) -> None:
    """A value hoisted in one routine must not leak into another that never
    mentioned it: the constant is named after where it was found."""
    iface, _ = _records(tmp_path)
    sem = semantics.for_subprogram(iface, "shadowing")
    elsewhere = names.for_subprogram(
        sem, {"literal_map": {"other_routine": {"273.15": "F_273P15"}}, "local_parameters": []}
    )
    with pytest.raises(NoRule):
        elsewhere.literal(f03.Real_Literal_Constant("273.15"))


# --- what the gate needs back ------------------------------------------------


def test_the_protocol_table_maps_emitted_constants_back(table) -> None:
    """The read/write cross-check has to undo the constant renames, and this
    is the only record of them. Collision renames are absent on purpose: those
    are reversible from the target language's own rules."""
    mapping = table.as_protocol_table()
    assert mapping["PI"] == "pi"
    assert mapping["GRAVIT"] == "gravit"
    assert mapping["SHADOWING__LOCAL_P"] == "local_p"
    assert "lambda_" not in mapping


# --- what USE brings in ------------------------------------------------------


def test_host_module_state_shadows_a_companion_global_of_the_same_name(table) -> None:
    """The module declares ``premib`` itself; a companion that also exports
    one is not what the Fortran reads."""
    assert table.symbol("premib") == "premib"
    assert table.symbol("rhminl") == "_cf2.rhminl"


def test_a_use_binding_catches_what_no_table_covered(table) -> None:
    assert table.symbol("wp_r8") == "_precision_mod.wp_r8"


def test_use_statements_bind_the_uncovered_and_rebind_a_rename() -> None:
    """``r8 => wp_r8`` from a module that is not a companion must not
    resolve to another companion's ``r8`` that registered first; a plain
    uncovered name binds to the module's own alias; a USE of a module that is
    not a companion is reported so the header can import a stand-in."""
    record = {
        "module_parameters": [{"name": "own"}],
        "module_state": [{"name": "cached"}],
        "use_statements": [
            "USE precision_mod, ONLY: r8 => wp_r8, i8 => wp_i8",
            "USE wv_sat_methods, ONLY: qsat_water, own, cached",
            "USE ppgrid",
        ],
    }
    companion_globals = {"r8": "_wv.r8"}
    bindings, stubs, intrinsic = names.bind_use_statements(
        record, {"_wv"}, {"qsat_water"}, companion_globals
    )
    assert not intrinsic  # none of these is a module the standard provides
    assert companion_globals["r8"] == "_precision_mod.wp_r8"
    assert bindings == {"i8": "_precision_mod.wp_i8"}
    # A companion is recognised by its alias minus the underscore, as the
    # pipeline does it, so ``_wv`` is the companion "wv" and a USE of
    # wv_sat_methods by its full name still gets a stand-in alias.
    assert stubs == {
        "precision_mod": "_precision_mod",
        "wv_sat_methods": "_wv_sat_methods",
        "ppgrid": "_ppgrid",
    }
