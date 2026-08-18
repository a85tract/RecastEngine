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

SOURCE = """\
module phys_mod
  implicit none
  real, parameter :: pi = 3.14159
  real, parameter :: tmin = 173.0
  real :: cached
contains
  subroutine shadowing(tmin, n)
    real, intent(in) :: tmin
    integer, intent(in) :: n
    real, parameter :: local_p = 0.61
    real :: scratch
    integer :: lambda
    scratch = tmin + pi + local_p + cached + n + lambda
    if (scratch > 273.15) scratch = 6.371e6
  end subroutine shadowing
end module phys_mod
"""


def _records(tmp_path: Path):
    parser()
    src = tmp_path / "phys.f90"
    src.write_text(SOURCE)
    return interface.extract(src), fortran_constants.extract(src)


@pytest.fixture
def table(tmp_path: Path) -> names.Names:
    iface, consts = _records(tmp_path)
    sem = semantics.for_subprogram(iface, "shadowing")
    return names.for_subprogram(sem, consts, use_parameters={"gravit": "GRAVIT"})


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
    assert table.literal(f03.Real_Literal_Constant("273.15")) == "F_273P15"


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
