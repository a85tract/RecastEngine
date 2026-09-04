"""``translate`` with ``target: tree``: a unit that ``use``s a sibling.

Two plain modules in one folder, one calling into the other, is the smallest
tree there is, and the shipped recipe could not verify it: ``translate.numpy``
emits the import of the sibling's translation and carries only its own
files, and the differential gate stages only what the candidate carries. The
tree transform bundles the sibling, and this test is what holds the recipe to
offering it.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytest

from recast.recipes import BUILTIN
from recast.run import RunStatus, run_recipe

SIBLING = """\
module satvap
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  function esat(t) result(es)
    real(r8), intent(in) :: t
    real(r8) :: es, tc
    tc = t - 273.15_r8
    es = 6.112_r8 * exp(17.67_r8 * tc / (tc + 243.5_r8))
  end function esat
end module satvap
"""

USER = """\
module dewpoint
  use satvap, only: esat, r8
  implicit none
contains
  function dewpt(t, rh) result(td)
    real(r8), intent(in) :: t, rh
    real(r8) :: td, e, a
    e = rh * esat(t)
    a = log(e / 6.112_r8)
    td = 243.5_r8 * a / (17.67_r8 - a) + 273.15_r8
  end function dewpt
end module dewpoint
"""

# ``log`` of a negative number raises in Python and returns NaN in Fortran,
# which is a fact about sampling rather than about bundling; the ranges keep
# the argument positive so that the only question asked is the import.
RANGES = {"differential.bitexact": {"ranges": {"t": [220.0, 320.0], "rh": [0.05, 1.0]}}}


def test_translate_accepts_the_tree_target() -> None:
    translate = BUILTIN["translate"]()
    assert translate.validate({"target": "tree"}) == []
    transforms = [s.plugin for s in translate.stages({"target": "tree"}) if s.kind == "transform"]
    assert transforms == ["translate.tree"]
    assert translate.validate({"target": "forest"}) != []


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "pair"
    root.mkdir()
    (root / "satvap.f90").write_text(SIBLING)
    (root / "dewpoint.f90").write_text(USER)
    return root


def _differential(run: object, uid: str) -> tuple[str, str]:
    for unit_run in run.units:  # type: ignore[attr-defined]
        if unit_run.unit.uid == uid:
            for outcome in unit_run.outcomes:
                if outcome.plugin == "differential.bitexact":
                    return outcome.status, outcome.detail
    raise AssertionError(f"{uid} has no differential outcome")


@pytest.mark.skipif(shutil.which("gfortran") is None, reason="needs gfortran")
def test_a_unit_that_uses_a_sibling_is_gated_under_the_tree_target(tmp_path: Path) -> None:
    root = _tree(tmp_path)
    translate = BUILTIN["translate"]()

    plain = run_recipe(
        translate,
        root,
        {"target": "numpy", "output": str(tmp_path / "out-numpy"), "stages": RANGES},
    )
    status, detail = _differential(plain, "fortran:dewpoint")
    assert status == "failed" and "does not import" in detail
    assert _differential(plain, "fortran:satvap")[0] == "ok"

    tree = run_recipe(
        translate,
        root,
        {"target": "tree", "output": str(tmp_path / "out-tree"), "stages": RANGES},
    )
    assert tree.status is RunStatus.PASSED, [
        (o.plugin, o.status, o.detail) for u in tree.units for o in u.outcomes
    ]
    assert _differential(tree, "fortran:dewpoint")[0] == "ok"


# --- what the tree transform refuses ----------------------------------------
#
# Bundling is not best-effort. A companion that is not translated, a companion
# whose source the root does not hold, a use-constant the tree does not
# initialize: each used to be a note on the candidate and a translation that
# imported, and each is now the unit's failure, naming what was missing.

BASE = """\
module base
  implicit none
  real(8), parameter :: tzero = 273.0d0
end module base
"""

# ``tfrz`` reaches outside the constants modules for ``tzero``, so resolving
# it against those modules alone finds no initializer.
CONSTS = """\
module consts
  use base, only: tzero
  implicit none
  real(8), parameter :: tfrz = tzero + 0.15d0
end module consts
"""

SIBLING_WITH_CONSTANT = """\
module satvap
  use consts, only: tfrz
  implicit none
contains
  function esat(t) result(es)
    real(8), intent(in) :: t
    real(8) :: es, tc
    tc = t - tfrz
    es = 6.112d0 * exp(17.67d0 * tc / (tc + 243.5d0))
  end function esat
end module satvap
"""

USER_OF_SIBLING = """\
module dewpoint
  use satvap, only: esat
  implicit none
contains
  function dewpt(t, rh) result(td)
    real(8), intent(in) :: t, rh
    real(8) :: td, e, a
    e = rh * esat(t)
    a = log(e / 6.112d0)
    td = 243.5d0 * a / (17.67d0 - a) + 273.15d0
  end function dewpt
end module dewpoint
"""

USER_OF_CONSTANT = """\
module dewpoint
  use consts, only: tfrz
  implicit none
contains
  function celsius(t) result(c)
    real(8), intent(in) :: t
    real(8) :: c
    c = t - tfrz
  end function celsius
end module dewpoint
"""


def _tree_transform(root: Path, user: str, sibling: str | None = None) -> tuple[Any, Any, Any]:
    """The unit ``dewpoint`` analysed under a frontend that treats ``consts``
    as a constants module, and a tree transform under the same conventions."""
    pytest.importorskip("fparser")
    pytest.importorskip("numpy")
    from recast.fortran.frontend import FortranFrontend
    from recast.transform.numpy.tree import TreeConventions, TreeTranslation

    (root / "base.f90").write_text(BASE)
    (root / "consts.f90").write_text(CONSTS)
    if sibling is not None:
        (root / "satvap.f90").write_text(sibling)
    (root / "dewpoint.f90").write_text(user)
    frontend = FortranFrontend(constant_modules=["consts"], stub_modules=["consts"])
    unit = next(u for u in frontend.discover(root) if u.uid == "fortran:dewpoint")
    conventions = TreeConventions(
        constant_modules=frozenset({"consts"}), stub_modules=frozenset({"consts"})
    )
    return unit, frontend.analyze(unit, root), TreeTranslation(conventions)


def test_a_companion_that_does_not_translate_fails_the_unit(tmp_path: Path) -> None:
    """``dewpoint`` calls ``satvap``, whose own use-constant does not resolve.
    The failure is ``dewpoint``'s, names ``satvap``, and carries the reason."""
    from recast.errors import ConfigError
    from recast.fortran.use import UnresolvedConstant

    unit, facts, transform = _tree_transform(tmp_path, USER_OF_SIBLING, SIBLING_WITH_CONSTANT)
    assert [c["module"] for c in facts.provenance["companions"]] == ["satvap"]
    with pytest.raises(
        ConfigError, match="companion 'satvap' of 'dewpoint' did not translate"
    ) as caught:
        transform.apply(unit, facts, {"root": str(tmp_path)})
    assert isinstance(caught.value.__cause__, UnresolvedConstant)
    assert "'tfrz'" in str(caught.value) and "'tzero'" in str(caught.value)


def test_a_companion_the_root_does_not_hold_fails_the_unit(tmp_path: Path) -> None:
    """The frontend resolved ``satvap`` in the tree; by the time the unit is
    translated its file is gone. Silently carrying no translation of it
    would leave the call to reach a stand-in with no such function."""
    from recast.errors import ConfigError

    unit, facts, transform = _tree_transform(tmp_path, USER_OF_SIBLING, SIBLING)
    (tmp_path / "satvap.f90").unlink()
    with pytest.raises(
        ConfigError, match="companion 'satvap' of 'dewpoint' has no unit under"
    ) as caught:
        transform.apply(unit, facts, {"root": str(tmp_path)})
    assert str(tmp_path) in str(caught.value)


def test_a_use_constant_the_tree_does_not_initialize_fails_the_unit(tmp_path: Path) -> None:
    """A name use-imported from a constants module and resolved against
    those modules alone: ``tfrz`` depends on ``tzero``, which none of them
    initializes. The unit fails, naming the constant and the missing one."""
    from recast.fortran.use import UnresolvedConstant

    unit, facts, transform = _tree_transform(tmp_path, USER_OF_CONSTANT)
    with pytest.raises(UnresolvedConstant, match="fortran:dewpoint use-imports 'tfrz'") as caught:
        transform.apply(unit, facts, {"root": str(tmp_path)})
    assert "no initializer for 'tzero'" in str(caught.value)
    assert isinstance(caught.value.__cause__, UnresolvedConstant)
