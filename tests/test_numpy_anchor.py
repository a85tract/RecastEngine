"""Tests for ``numpy-anchor``, the reference a port is judged against.

The point of this oracle is a chain rather than a looser claim: the NumPy
translation is bit-exact against the Fortran, and the port is ULP-bounded
against the NumPy. What these tests hold is the parts of that chain the oracle
is responsible for -- that it re-derives the reference rather than borrowing
the candidate's, that its key moves when the reference does, and that it says
plainly when the first link is unevidenced.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.errors import OracleUnavailable
from recast.executors.local import LocalExecutor
from recast.fortran.frontend import FortranFrontend
from recast.model import Facts, Unit
from recast.oracle.numpy_anchor import NumpyAnchorOracle

SOURCE = """\
module anchor_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine scale(n, x, y)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: x(n)
    real(r8), intent(out) :: y(n)
    integer :: i
    do i = 1, n
      y(i) = 2.0_r8 * x(i)
    end do
  end subroutine scale
end module anchor_demo
"""


@pytest.fixture
def subject(tmp_path: Path) -> tuple[Unit, Facts, dict[str, str]]:
    root = tmp_path / "src"
    root.mkdir()
    (root / "anchor_demo.f90").write_text(SOURCE)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(root) if u.uid == "fortran:anchor_demo")
    return unit, frontend.analyze(unit, root), {"root": str(root)}


def test_the_key_moves_with_anything_that_moves_the_reference(subject: tuple) -> None:
    """Deliberately over-inclusive. Naming the individual keys the NumPy
    transform reads would duplicate another plugin's config surface and fall
    out of step with it silently, in the direction that serves stale
    references."""
    unit, facts, config = subject
    oracle = NumpyAnchorOracle()
    base = oracle.key(unit, facts, config)

    assert oracle.key(unit, facts, dict(config)) == base
    assert oracle.key(unit, facts, {**config, "profile": "ifx"}) != base
    assert oracle.key(unit, facts, {**config, "function_stubs": {"outfld": "pass"}}) != base
    moved = dataclasses.replace(facts, provenance={**facts.provenance, "digest": "0" * 64})
    assert oracle.key(unit, moved, config) != base


def test_paths_are_not_part_of_a_references_identity(subject: tuple) -> None:
    """Two machines verifying the same claim should agree about the key, and a
    fresh temporary directory should not invalidate a cache."""
    unit, facts, config = subject
    oracle = NumpyAnchorOracle()
    assert oracle.key(unit, facts, {**config, "root": "/somewhere/else"}) == oracle.key(
        unit, facts, config
    )


def test_a_config_it_cannot_serialize_moves_the_key_by_presence(subject: tuple) -> None:
    """A behaviour hook changes the reference; its address does not, and
    folding an address in would break the cache on every run."""
    unit, facts, config = subject
    oracle = NumpyAnchorOracle()
    base = oracle.key(unit, facts, config)
    first = oracle.key(unit, facts, {**config, "deferred_handler": lambda site: None})
    second = oracle.key(unit, facts, {**config, "deferred_handler": lambda site: None})
    assert first != base
    assert first == second


def test_it_materializes_a_callable_reference(subject: tuple, tmp_path: Path) -> None:
    unit, facts, config = subject
    ref = NumpyAnchorOracle().materialize(unit, facts, tmp_path / "ws", LocalExecutor(), config)
    handle = ref.handle
    assert callable(getattr(handle["module"], "scale", None))
    assert handle["wrappers"] == {"scale": "scale"}


def test_it_declares_how_it_spells_and_returns(subject: tuple, tmp_path: Path) -> None:
    """The differential harness was written against f2py and assumed three of
    its conventions. A reference that differs has to say so rather than be
    guessed at, which is what these two keys are for."""
    unit, facts, config = subject
    handle = (
        NumpyAnchorOracle()
        .materialize(unit, facts, tmp_path / "ws", LocalExecutor(), config)
        .handle
    )
    assert handle["arg_naming"] == "pysafe"
    assert handle["return_convention"] == "emitted"
    assert handle["device"] == "cpu"


def test_an_unevidenced_anchor_says_so_rather_than_implying_otherwise(
    subject: tuple, tmp_path: Path
) -> None:
    """This oracle cannot check the first link of the chain. Absent is a
    legitimate answer and an informative one; silence would not be."""
    unit, facts, config = subject
    oracle = NumpyAnchorOracle()
    plain = oracle.materialize(unit, facts, tmp_path / "a", LocalExecutor(), config)
    assert plain.handle["anchor_evidence"] is None

    cited = oracle.materialize(
        unit, facts, tmp_path / "b", LocalExecutor(), {**config, "anchor_evidence": "file:///m"}
    )
    assert cited.handle["anchor_evidence"] == "file:///m"


def test_it_records_what_it_derived(subject: tuple, tmp_path: Path) -> None:
    """The reference is re-derived from the Unit and Facts, never read off the
    Candidate -- an Oracle that saw the artifact would stop being independent
    of it -- so the record says which transform produced it and what came out."""
    unit, facts, config = subject
    handle = (
        NumpyAnchorOracle()
        .materialize(unit, facts, tmp_path / "ws", LocalExecutor(), config)
        .handle
    )
    assert handle["anchor_transform"] == "recast.translate.fortran-to-numpy"
    assert len(handle["anchor_digest"]) == 64
    assert handle["anchor_deferred"] == []


def test_a_unit_with_no_translation_has_no_anchor(tmp_path: Path) -> None:
    """Fails closed, and as a RecastError so the runner records the stage
    rather than losing the whole run."""
    oracle = NumpyAnchorOracle()
    unit = Unit(uid="fortran:nothing", kind="module")
    with pytest.raises(OracleUnavailable, match="no validated NumPy module"):
        oracle.materialize(unit, Facts(unit=unit.uid), tmp_path, LocalExecutor(), {})


def test_release_is_idempotent(subject: tuple, tmp_path: Path) -> None:
    unit, facts, config = subject
    oracle = NumpyAnchorOracle()
    ref = oracle.materialize(unit, facts, tmp_path / "ws", LocalExecutor(), config)
    oracle.release(ref)
    oracle.release(ref)
