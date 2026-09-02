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
