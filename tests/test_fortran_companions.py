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


def test_a_use_only_list_is_carried_as_the_names_this_module_sees(tree: Path) -> None:
    """``use helper, only: dbl => twice, bump`` lets ``dbl`` and ``bump`` in
    and nothing else, spelled as this module sees them; a bare use carries
    ``None``, meaning everything the module exports."""
    found = {c["module"]: c for c in _facts(tree).provenance["companions"]}
    assert found["helper"]["only"] == ["bump", "dbl"]
    assert all("only" in c for c in found.values())


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


def test_a_kind_a_sibling_defines_resolves_without_being_assumed(tree: Path) -> None:
    """``mykinds`` declares ``wp`` and ``main_mod`` uses it. Reading only the
    second file leaves every ``real(wp)`` argument UNKNOWN, which the f2py
    oracle cannot declare and the translation cannot type -- while the answer
    is one file over, in a tree the frontend has already indexed."""
    facts = _facts(tree)
    args = {a["name"]: a for a in facts.interface["subprograms"][0]["args"]}
    assert args["x"]["dtype"] == "float64"


def test_a_kind_read_from_the_tree_is_recorded_apart_from_one_assumed(tree: Path) -> None:
    """Provenance has to say which of the two a dtype rests on. An assumption
    is somebody's claim and a sibling is evidence, and a record that spelled
    them the same way could not be checked by anyone reading it later."""
    provenance = _facts(tree).provenance
    assert provenance["kind_assumptions"] == {}
    assert provenance["kind_sources"]["wp"] == {
        "dtype": "float64",
        "module": "mykinds",
        "source": "mykinds.f90",
    }


def test_the_operator_s_kind_still_overrides_the_tree(tree: Path) -> None:
    """Resolution is a default here too. An override nothing can outvote is
    not an override, and the operator is the one who knows when a tree's own
    kinds module is not the one the production build compiles against."""
    facts = _facts(tree, kind_assumptions={"wp": "float32"})
    args = {a["name"]: a for a in facts.interface["subprograms"][0]["args"]}
    assert args["x"]["dtype"] == "float32"


REEXPORT = """\
module allofit
  use mykinds
  use helper
end module allofit
"""

VIA_REEXPORT = """\
module downstream
  use allofit, only: wp, twice
  implicit none
contains
  subroutine go(x)
    real(wp), intent(inout) :: x
    x = twice(x)
  end subroutine go
end module downstream
"""


@pytest.fixture(scope="module")
def reexport_tree(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("reexport")
    for name, text in (
        ("mykinds.f90", KINDS),
        ("helper.f90", HELPER),
        ("allofit.f90", REEXPORT),
        ("downstream.f90", VIA_REEXPORT),
    ):
        (root / name).write_text(text)
    return root


def _downstream(root: Path):
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(root) if u.uid == "fortran:downstream")
    return frontend.analyze(unit, root)


def test_an_only_list_is_followed_through_a_re_export_module(reexport_tree: Path) -> None:
    """``module allofit`` is nothing but ``use mykinds; use helper``. An
    ``only`` list against it names entities it does not declare, and stopping
    at the first module leaves them unresolved -- the call refuses as external
    and the kind reports UNKNOWN -- while the files that define them sit in
    the same tree. The walk continues exactly when something asked for is not
    there, so a module that answers for itself still costs one lookup."""
    facts = _downstream(reexport_tree)
    assert {c["module"] for c in facts.provenance["companions"]} == {
        "allofit",
        "mykinds",
        "helper",
    }
    assert facts.provenance["kind_sources"]["wp"]["module"] == "mykinds"
    args = {a["name"]: a for a in facts.interface["subprograms"][0]["args"]}
    assert args["x"]["dtype"] == "float64"


DIRECT = """\
module direct
  use helper, only: twice
  implicit none
contains
  function thrice(x) result(y)
    real, intent(in) :: x
    real :: y
    y = 3.0 * twice(x)
  end function thrice
end module direct
"""


def test_a_module_that_answers_for_its_only_list_is_not_walked_past(reexport_tree: Path) -> None:
    """The other half of the rule. ``direct`` uses ``helper, only: twice``
    and ``helper`` declares ``twice``, so ``helper``'s own ``use mykinds``
    stays its business. Without the check the walk would make every ``only``
    list as broad as a bare ``use``, which is the one thing an only-list is
    written to prevent."""
    (reexport_tree / "direct.f90").write_text(DIRECT)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(reexport_tree) if u.uid == "fortran:direct")
    facts = frontend.analyze(unit, reexport_tree)
    assert {c["module"] for c in facts.provenance["companions"]} == {"helper"}


ELSEWHERE = """\
module needs_outside
  use mykinds, only: wp
  use elsewhere_mod, only: reduce_it, tally
  implicit none
contains
  subroutine go(a, b, n, x)
    real(wp), intent(in) :: a, b
    integer, intent(in) :: n
    real(wp), intent(inout) :: x
    x = reduce_it(a, b)
    x = x + tally(3)
    x = x + reduce_it(1:n)
  end subroutine go
end module needs_outside
"""


def test_a_use_imported_function_is_called_not_subscripted(tmp_path: Path) -> None:
    """``elsewhere_mod`` is not in the tree, so its names reach the emitter
    only as USE bindings. Falling through to the subscript rule emitted
    ``_elsewhere_mod.reduce_it[a - 1, b - 1]`` -- not a refusal but runnable,
    wrong code, with the zero-based shift applied to what are arguments. The
    pipeline calls through the binding first and subscripts only after that,
    which is the order this now follows."""
    from recast.transform.numpy.translate import NumpyTranslation

    (tmp_path / "mykinds.f90").write_text(KINDS)
    (tmp_path / "needs_outside.f90").write_text(ELSEWHERE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:needs_outside")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": str(tmp_path)})
    text = candidate.files[Path("needs_outside_numpy.py")].decode()
    assert "_elsewhere_mod.reduce_it(a, b)" in text
    assert "_elsewhere_mod.tally(I_3)" in text, "the actual is a hoisted literal"
    assert "reduce_it[" not in text and "tally[" not in text
    assert candidate.deferred == []


def test_a_range_in_a_value_position_becomes_a_slice(tmp_path: Path) -> None:
    """``reduce_it(1:n)`` where ``reduce_it`` came from a module the tree does
    not contain: nothing here can say what the range indexes, so the only
    reading left is Fortran's own -- one-based start, inclusive upper edge --
    handed over as a ``slice``. Two rules had to agree for this to come out:
    the range needed a spelling, and an argument whose rank cannot be
    analysed had to stop being a refusal."""
    from recast.transform.numpy.translate import NumpyTranslation

    (tmp_path / "mykinds.f90").write_text(KINDS)
    (tmp_path / "needs_outside.f90").write_text(ELSEWHERE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:needs_outside")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": str(tmp_path)})
    text = candidate.files[Path("needs_outside_numpy.py")].decode()
    assert "_elsewhere_mod.reduce_it(slice((1) - 1, n))" in text, text
    assert candidate.deferred == []


ORIGIN = """\
module based_mod
  use mykinds, only: wp
  use elsewhere_mod, only: lo
  implicit none
contains
  subroutine fill(v, n)
    integer, intent(in) :: n
    real(wp), intent(inout) :: v(1-lo:n)
    v(1-lo:n) = 0.0_wp
  end subroutine fill
end module based_mod
"""


def test_a_slice_origin_is_spelled_through_the_use_bindings(tmp_path: Path) -> None:
    """A declared lower bound is Fortran source text, and a name in it means
    whatever this module's USE statements bound it to. Emitting the origin raw
    put a bare ``lo`` beside the ``_elsewhere_mod.lo`` that the very same
    subscript already spelled correctly -- a NameError, not a matter of
    style."""
    from recast.transform.numpy.translate import NumpyTranslation

    (tmp_path / "mykinds.f90").write_text(KINDS)
    (tmp_path / "based_mod.f90").write_text(ORIGIN)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:based_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = NumpyTranslation().apply(unit, facts, {"root": str(tmp_path)})
    text = candidate.files[Path("based_mod_numpy.py")].decode()
    body = next(line for line in text.splitlines() if line.strip().startswith("v["))
    assert "_elsewhere_mod.lo" in body, body
    assert "(1 - lo)" not in body, body
