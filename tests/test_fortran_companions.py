"""Which sibling modules a unit depends on, resolved from its own USE statements.

The engine used to be told its companions by the operator, which is workable
for a tree somebody has already mapped and useless for a library nobody here
wrote: every call into a sibling module refused. The frontend answers it now,
and the cases that matter are the ones where a USE is *not* a companion --
an intrinsic module, a framework a domain package stubs, a module the tree
does not contain, and a sibling that does not parse.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")

from recast.fortran.frontend import FortranFrontend

KINDS = """\
module mykinds
  implicit none
  integer, parameter :: wp = kind(1.0d0)
end module mykinds
"""

HELPER = """\
module helper
  use mykinds, only: wp
  implicit none
contains
  function twice(x) result(y)
    real(wp), intent(in) :: x
    real(wp) :: y
    y = 2.0_wp * x
  end function twice

  subroutine bump(x)
    real(wp), intent(inout) :: x
    x = x + 1.0_wp
  end subroutine bump
end module helper
"""

FRAMEWORK = """\
module cam_history
  implicit none
contains
  subroutine outfld(name, field)
    character(len=*), intent(in) :: name
    real, intent(in) :: field
  end subroutine outfld
end module cam_history
"""

BROKEN = """\
module halfparsed
  implicit none(type, external)
end module halfparsed
"""

MAIN = """\
module main_mod
  use, intrinsic :: iso_fortran_env, only: real64
  use mykinds, only: wp
  use helper, only: dbl => twice, bump
  use cam_history, only: outfld
  use halfparsed
  use nowhere_at_all
  implicit none
contains
  subroutine go(x)
    real(wp), intent(inout) :: x
    x = dbl(x)
    call bump(x)
  end subroutine go
end module main_mod
"""


@pytest.fixture(scope="module")
def tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tree")
    for name, text in (
        ("mykinds.f90", KINDS),
        ("helper.f90", HELPER),
        ("cam_history.f90", FRAMEWORK),
        ("halfparsed.f90", BROKEN),
        ("main_mod.f90", MAIN),
    ):
        (root / name).write_text(text)
    return root


def _facts(root: Path, **config: object):
    frontend = FortranFrontend(**config)  # type: ignore[arg-type]
    unit = next(u for u in frontend.discover(root) if u.uid == "fortran:main_mod")
    return frontend.analyze(unit, root)


def test_a_use_of_a_module_the_tree_defines_is_a_companion(tree: Path) -> None:
    found = {c["module"]: c for c in _facts(tree).provenance["companions"]}
    assert found["helper"]["source"] == "helper.f90"
    assert found["helper"]["record"]["subprograms"][0]["name"] == "twice"
    assert found["mykinds"]["constants"]["module_parameters"][0]["name"] == "wp"


def test_a_use_rename_is_carried_so_the_call_can_be_resolved(tree: Path) -> None:
    found = {c["module"]: c for c in _facts(tree).provenance["companions"]}
    assert found["helper"]["renames"] == {"dbl": "twice"}


def test_an_intrinsic_module_is_not_looked_for_in_the_tree(tree: Path) -> None:
    """The pipeline this came from treats iso_fortran_env as a companion,
    which is one of the defects its author catalogued."""
    assert "iso_fortran_env" not in {c["module"] for c in _facts(tree).provenance["companions"]}


def test_a_module_the_tree_does_not_contain_is_not_a_companion(tree: Path) -> None:
    assert "nowhere_at_all" not in {c["module"] for c in _facts(tree).provenance["companions"]}


def test_a_stubbed_framework_module_is_left_to_its_stub(tree: Path) -> None:
    """In the tree, and still not a companion: what cam_history does is call a
    framework the port does not carry. Which modules those are is the domain
    package's to say."""
    plain = {c["module"] for c in _facts(tree).provenance["companions"]}
    assert "cam_history" in plain
    stubbed = _facts(tree, stub_modules=["cam_history"]).provenance
    assert "cam_history" not in {c["module"] for c in stubbed["companions"]}
    assert stubbed["stub_modules"] == ["cam_history"]


def test_a_sibling_that_does_not_parse_is_recorded_rather_than_raised(tree: Path) -> None:
    """It drops out of the companion set -- its calls then refuse like any
    unresolved call -- and the reason is kept instead of a stack trace."""
    provenance = _facts(tree).provenance
    assert "halfparsed" not in {c["module"] for c in provenance["companions"]}
    unresolved = {c["module"]: c for c in provenance["companions_unresolved"]}
    assert unresolved["halfparsed"]["source"] == "halfparsed.f90"
    assert "line 2" in unresolved["halfparsed"]["reason"] or unresolved["halfparsed"]["reason"]


def _translated(tree: Path, config: dict[str, object]) -> tuple[str, list[str]]:
    from recast.model import Unit
    from recast.transform.numpy.translate import NumpyTranslation

    unit = Unit(uid="fortran:main_mod", kind="module", sources=(Path("main_mod.f90"),))
    candidate = NumpyTranslation().apply(unit, _facts(tree), {"root": str(tree), **config})
    return candidate.files[Path("main_mod_numpy.py")].decode(), list(candidate.deferred)


def test_resolved_companions_reach_the_translation(tree: Path) -> None:
    """The call into the sibling is emitted through its alias, under the
    sibling's own name rather than the local one the rename gave it."""
    text, deferred = _translated(tree, {})
    assert "import helper_numpy as _helper" in text
    assert "_helper.bump(" in text
    assert not [d for d in deferred if "bump" in d]


def test_the_operator_s_list_still_wins(tree: Path) -> None:
    """A tree somebody mapped by hand stays mapped by hand: resolution is a
    default, not an override. With the companions declared empty the call has
    nothing to resolve against and refuses, which is the pre-resolver
    behaviour."""
    _, deferred = _translated(tree, {"companions": []})
    assert [d for d in deferred if "bump" in d]
