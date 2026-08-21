"""The port recipe end to end: Fortran in, a ULP-bounded verdict out.

The chain this proves is the reason the port side anchors on NumPy rather than
on Fortran. XLA cannot reproduce libm bit for bit, so a direct comparison could
only ever be a tolerance; the NumPy translation of the same unit *can* be
bit-exact against the Fortran, and the translate spine holds it there. Anchoring
here turns one loose claim into two tight ones joined.

Gated on JAX because this is the half that runs. Everything about *emitting* the
port is tested without it in ``test_jax_transform.py``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")
pytest.importorskip("jax", reason="needs recast-engine[jax]")

from recast.model import Confidence
from recast.recipes import BUILTIN
from recast.run import run_recipe

SOURCE = """\
module port_spine
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine settle(n, rho, dz, w)
    integer,  intent(in)    :: n
    real(r8), intent(in)    :: rho(n), dz(n)
    real(r8), intent(inout) :: w(n)
    integer :: i
    do i = 1, n
      w(i) = w(i) + rho(i) * dz(i)
    end do
  end subroutine settle

  function column_mass(n, rho, dz) result(total)
    integer,  intent(in) :: n
    real(r8), intent(in) :: rho(n), dz(n)
    real(r8) :: total
    integer :: i
    total = 0.0_r8
    do i = 1, n
      total = total + rho(i) * dz(i)
    end do
  end function column_mass
end module port_spine
"""

CONFIG = {
    "units": ["fortran:port_spine"],
    "stages": {
        "differential.tolerance": {
            "trials": 3,
            "dims": {"n": 8},
            "ranges": {"rho": [0.1, 2.0], "dz": [10.0, 100.0], "w": [-5.0, 5.0]},
        }
    },
}


@pytest.fixture
def verdict(tmp_path: Path):
    (tmp_path / "port_spine.f90").write_text(SOURCE)
    run = run_recipe(BUILTIN["port"](), tmp_path, {**CONFIG, "workspace": tmp_path / "ws"})
    assert len(run.units) == 1
    return run, run.units[0]


def test_the_port_spine_reaches_a_bounded_verdict(verdict: tuple) -> None:
    run, unit_run = verdict
    assert run.passed, [(o.plugin, o.status, o.detail) for o in unit_run.outcomes]
    walked = [(o.kind, o.plugin, o.status) for o in unit_run.outcomes]
    assert walked[:4] == [
        ("frontend", "fortran", "ok"),
        ("transform", "port.jax", "ok"),
        ("oracle", "numpy-anchor", "ok"),
        ("verifier", "differential.tolerance", "ok"),
    ]


def test_the_verdict_is_a_bound_and_not_a_tolerance(verdict: tuple) -> None:
    """On a kernel with no transcendentals there is nothing for XLA to lower
    differently, so the tail stays inside the bound too and the gate says the
    stronger of the two things it can say."""
    _, unit_run = verdict
    awarded = unit_run.verdicts[0]
    assert awarded.confidence in (Confidence.ULP_BOUNDED, Confidence.BIT_EXACT), awarded.detail
    assert awarded.metrics["max_ulp"] <= awarded.metrics["ulp_gate"]


def test_both_sides_say_which_device_they_ran_on(verdict: tuple) -> None:
    """A ULP bound between a GPU and a CPU is a different claim, and the
    record has to make a same-device comparison legible as one."""
    _, unit_run = verdict
    metrics = unit_run.verdicts[0].metrics
    assert metrics["candidate_device"].startswith(("cpu", "gpu", "tpu"))
    assert metrics["reference_device"] == "cpu"


def test_the_reference_is_the_anchor_and_the_run_is_recorded(verdict: tuple) -> None:
    _, unit_run = verdict
    assert unit_run.oracle.oracle == "numpy-anchor"
    assert unit_run.oracle.handle["anchor_transform"] == "recast.translate.fortran-to-numpy"
    assert unit_run.evidence, "a Candidate without Evidence is a draft"
