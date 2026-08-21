"""Tests for ``port.jax``: two halves of a port inside one Transform.

No JAX installed and none needed. Everything here is about what the Transform
*emits* -- which files, which subprograms became kernels, what landed in
``deferred`` -- and emission is pure AST work. Running the emitted module is
the gate's job, at the ULP tier, and needs an accelerator this suite does not
assume.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("fparser", reason="needs recast-engine[fortran]")
pytest.importorskip("numpy", reason="needs recast-engine[translate]")

from recast.fortran.frontend import FortranFrontend
from recast.model import Candidate, Facts, Unit
from recast.transform.jax.translate import KernelToJax

PORTABLE = """\
module port_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine settle(n, rho, dz, mass)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: rho(n), dz(n)
    real(r8), intent(out) :: mass(n)
    integer :: i
    do i = 1, n
      mass(i) = rho(i) * dz(i)
    end do
  end subroutine settle
end module port_demo
"""

# A module-state write makes a subprogram ineligible for a kernel, by the same
# eligibility rule the Numba backend uses. It is host-delegated, not deferred:
# the emitted module still calls it, through the NumPy anchor.
DELEGATES = """\
module port_state
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8) :: cached
contains
  subroutine remember(x)
    real(r8), intent(in) :: x
    cached = x
  end subroutine remember

  subroutine scale(x, y)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: y
    y = 2.0_r8 * x
  end subroutine scale
end module port_state
"""


def subject(tmp_path: Path, source: str, module: str) -> tuple[Unit, Facts, dict[str, str]]:
    root = tmp_path / module
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{module}.f90").write_text(source)
    frontend = FortranFrontend()
    unit = next(u for u in frontend.discover(root) if u.uid == f"fortran:{module}")
    return unit, frontend.analyze(unit, root), {"root": str(root)}


def port(tmp_path: Path, source: str, module: str) -> Candidate:
    unit, facts, config = subject(tmp_path, source, module)
    return KernelToJax().apply(unit, facts, config)


def test_one_apply_produces_both_halves(tmp_path: Path) -> None:
    """The NumPy anchor is not scaffolding left behind; the emitted JAX module
    imports it and calls it for everything it could not lower."""
    candidate = port(tmp_path, PORTABLE, "port_demo")
    assert sorted(str(p) for p in candidate.files) == [
        "port_demo_constants.py",
        "port_demo_jax.py",
        "port_demo_jax_runtime.py",
        "port_demo_numpy.py",
    ]
    emitted = candidate.files[Path("port_demo_jax.py")].decode()
    assert "import port_demo_numpy as _host" in emitted
    assert "from port_demo_jax_runtime import *" in emitted


def test_what_it_could_lower_is_recorded_as_a_kernel(tmp_path: Path) -> None:
    candidate = port(tmp_path, PORTABLE, "port_demo")
    assert candidate.notes["jax"]["kernels"] == ["settle"]
    assert candidate.notes["jax"]["delegated"] == {}
    assert candidate.notes["jax"]["anchor"] == "port_demo_numpy.py"


def test_host_delegation_is_not_a_deferral(tmp_path: Path) -> None:
    """A deferred site raises at run time and belongs to the agent queue. A
    delegated one runs, on the slower path -- and putting it in ``deferred``
    would make the differential gate skip the subprograms most likely to be
    right, since the gate excludes deferred ones from comparison.
    """
    candidate = port(tmp_path, DELEGATES, "port_state")
    delegated = candidate.notes["jax"]["delegated"]
    assert "remember" in delegated
    assert "module-state write" in delegated["remember"]
    assert candidate.deferred == [], "nothing was deferred; one thing was delegated"


def test_the_anchor_still_carries_its_own_deferrals(tmp_path: Path) -> None:
    """``deferred`` is the anchor's, so a site the NumPy backend refused is
    still visible as refused after the port."""
    refused = PORTABLE.replace(
        "      mass(i) = rho(i) * dz(i)",
        "      associate (v => rho(i) * dz(i))\n        mass(i) = v\n      end associate",
    )
    candidate = port(tmp_path, refused, "port_demo")
    assert any("Associate" in entry for entry in candidate.deferred)


def test_the_emitted_artifact_is_reproducible(tmp_path: Path) -> None:
    """``deterministic = True`` is a claim about bytes: no model anywhere in
    the path, both halves rule-driven."""
    unit, facts, config = subject(tmp_path, PORTABLE, "port_demo")
    transform = KernelToJax()
    assert transform.deterministic
    first = transform.apply(unit, facts, dict(config))
    second = transform.apply(unit, facts, dict(config))
    assert first.digest() == second.digest()


def test_it_is_applicable_exactly_where_its_anchor_is(tmp_path: Path) -> None:
    """A unit the NumPy backend refuses has no anchor to be faithful to."""
    unit, facts, _ = subject(tmp_path, PORTABLE, "port_demo")
    transform = KernelToJax()
    assert transform.applicable(unit, facts)
    assert not transform.applicable(unit, Facts(unit=unit.uid))
    assert not transform.applicable(Unit(uid="x", kind="frobnicator"), facts)


def test_the_runtime_is_written_beside_the_module(tmp_path: Path) -> None:
    """Self-contained: a ported kernel has to run on a node that never heard
    of this engine, so the shim ships in the Candidate rather than being
    imported from it."""
    candidate = port(tmp_path, PORTABLE, "port_demo")
    runtime = candidate.files[Path("port_demo_jax_runtime.py")].decode()
    assert "_f_min" in runtime and "jax_enable_x64" in runtime
