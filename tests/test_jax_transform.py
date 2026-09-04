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

# A character output has no flat spelling, so the subprogram is ineligible
# for a kernel. It is host-delegated, not deferred: the emitted module still
# calls it, through the NumPy anchor. (A module-state write no longer
# delegates -- the state threads through the closure and the wrapper writes
# it back on the host.)
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

  subroutine name_of(kind, label)
    integer, intent(in) :: kind
    character(len=8), intent(out) :: label
    if (kind == 1) then
      label = 'first'
    else
      label = 'other'
    end if
  end subroutine name_of

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
    assert "name_of" in delegated
    assert "str arg" in delegated["name_of"]
    assert candidate.deferred == [], "nothing was deferred; one thing was delegated"
    # The state writer is a kernel now: the write threads through the
    # closure and the host wrapper stores it back.
    assert "remember" in candidate.notes["jax"]["kernels"]
    assert "_host.cached = _res[0]" in candidate.files[Path("port_state_jax.py")].decode()


def test_the_anchor_still_carries_its_own_deferrals(tmp_path: Path) -> None:
    """``deferred`` is the anchor's, so a site the NumPy backend refused is
    still visible as refused after the port."""
    refused = PORTABLE.replace(
        "    integer :: i", "    integer :: i\n    character(len=32) :: tag"
    ).replace(
        "      mass(i) = rho(i) * dz(i)",
        "      mass(i) = rho(i) * dz(i)\n      write(tag, '(D8.2)') mass(i)",
    )
    candidate = port(tmp_path, refused, "port_demo")
    assert any("formatted internal write" in entry for entry in candidate.deferred)


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


def test_a_module_state_write_threads_through_the_closure() -> None:
    """A kernel that stores module state takes its current value through the
    closure, returns the new one, and the host wrapper writes it back; a
    caller binds the write and carries it onward -- nothing stores to a
    module under tracing."""
    import ast as _ast

    from recast.transform.jax.backend import build_module

    tree = _ast.parse(
        "def tick(step):\n"
        "    global cache\n"
        "    cache = cache + step\n"
        "    return\n"
        "\n"
        "def use_tick(x):\n"
        "    tick(x)\n"
        "    return x + cache\n"
    )
    interface = {
        "subprograms": [
            {
                "name": "tick",
                "kind": "subroutine",
                "args": [{"name": "step", "dtype": "float64", "intent": "IN", "dims": None}],
                "module_state_read": [],
                "module_state_written": ["cache"],
                "calls": [],
            },
            {
                "name": "use_tick",
                "kind": "function",
                "args": [{"name": "x", "dtype": "float64", "intent": "IN", "dims": None}],
                "module_state_read": ["cache"],
                "module_state_written": [],
                "calls": ["tick"],
            },
        ]
    }
    pieces, jitted, _delegated = build_module(interface, tree)
    assert sorted(jitted) == ["tick", "use_tick"]
    text = "\n\n".join(pieces)
    assert "_host.cache = _res[0]" in text  # tick's wrapper stores the write back
    assert "cache = _tick_k_impl(x, cache)" in text  # the caller binds the write
    assert "global" not in text


def test_a_guard_around_nothing_lowers_to_nothing() -> None:
    """``if not _skip_1: pass`` -- a region guard whose block lost its log
    lines and aborts. There is nothing to carry, and refusing would drop the
    whole kernel for a branch with no effect."""
    import ast as _ast

    from recast.transform.jax.backend import KernelLowerer

    body = _ast.parse("y = x + 1.0\nif y > 0.0:\n    pass\nz = y * 2.0\n").body
    lowered = KernelLowerer().lower_block(body, 0)
    text = _ast.unparse(_ast.fix_missing_locations(_ast.Module(body=lowered, type_ignores=[])))
    assert "lax.cond" not in text and "pass" not in text
    assert "z = y * 2.0" in text


def test_a_constant_table_is_read_through_jnp() -> None:
    """``MDAYLEAP[m - 1]`` with a traced ``m``: numpy indexing would call
    __array__ on the tracer; the kernel reads the table as a jnp array."""
    import ast as _ast

    from recast.transform.jax.backend import ExprMap

    node = _ast.parse("d = MDAYLEAP[m - 1] + x[i]").body[0]
    ExprMap().visit(node)
    text = _ast.unparse(node)
    assert "jnp.asarray(MDAYLEAP)[m - 1]" in text
    assert "x[i]" in text  # a lowercase array is a traced value already


CYCLES = """\
module cycle_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine clip_below(n, zlo, z, v, w)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: zlo
    real(r8), intent(in)  :: z(n), v(n)
    real(r8), intent(out) :: w(n)
    integer :: k
    do k = 1, n
      if ( z(k) < zlo ) then
        w(k) = 0.0_r8
        cycle
      end if
      w(k) = v(k) * 2.0_r8
    end do
  end subroutine clip_below
  subroutine first_above(n, zlo, z, kfound)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: zlo
    real(r8), intent(in)  :: z(n)
    integer,  intent(out) :: kfound
    integer :: k
    kfound = 0
    do k = 1, n
      if ( z(k) > zlo ) then
        kfound = k
        exit
      end if
    end do
  end subroutine first_above
end module cycle_demo
"""


def test_a_cycle_folds_into_the_branch_and_an_exit_is_delegated(tmp_path: Path) -> None:
    """CLUBB's interpolators: ``if ( ... ) then ... cycle end if`` in a DO
    loop. The lowering passed the ``continue`` through into a ``lax.cond``
    branch -- a SyntaxError that took the whole emitted module down. Folded
    into the branch structure it is a kernel; an EXIT has no fori_loop
    shape and is delegated to the host, not emitted."""
    import importlib
    import sys

    candidate = port(tmp_path, CYCLES, "cycle_demo")
    assert candidate.notes["jax"]["kernels"] == ["clip_below"]
    assert "first_above" in candidate.notes["jax"]["delegated"]
    emitted = candidate.files[Path("cycle_demo_jax.py")].decode()
    assert "continue" not in emitted.replace("continuation", "")
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("cycle_demo_jax")
        import numpy as np

        z = np.array([1.0, 2.0, 3.0, 4.0])
        v = np.array([1.0, 1.0, 1.0, 1.0])
        w = np.asarray(module.clip_below(4, 2.5, z, v))
        assert w.tolist() == [0.0, 0.0, 2.0, 2.0]
        assert int(module.first_above(4, 2.5, z)) == 3
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"cycle_demo{suffix}", None)


SHIMS = """\
module shim_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine norms(n, x, root, total, err)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: x(n)
    real(r8), intent(out) :: root(n), total, err(n)
    root = sqrt( x )
    total = sum( x )
    err = erf( x )
  end subroutine norms
end module shim_demo
"""


def test_the_jax_runtime_carries_sqrt_sum_and_erf(tmp_path: Path) -> None:
    """CLUBB's clipping and PDF closure: ``sqrt``, ``sum`` and ``erf`` reach
    the kernels as ``_f_sqrt``, ``_f_vsum`` and ``_f_verf``, which the NumPy
    runtime defines and the JAX one did not -- a NameError at the first
    call, on every kernel of the unit."""
    import importlib
    import sys

    candidate = port(tmp_path, SHIMS, "shim_demo")
    assert candidate.notes["jax"]["kernels"] == ["norms"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("shim_demo_jax")
        import numpy as np

        x = np.array([4.0, 9.0, -1.0])
        root, total, err = (np.asarray(v) for v in module.norms(3, x))
        assert root[:2].tolist() == [2.0, 3.0] and np.isnan(root[2])
        assert total == 12.0
        assert abs(err[0] - 0.9999999845827421) < 1e-12
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"shim_demo{suffix}", None)


EMPTY_AXIS = """\
module tracer_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine scale_tracers(n, m, f, x, y)
    integer,  intent(in)  :: n, m
    real(r8), intent(in)  :: f
    real(r8), intent(in)  :: x(n, m)
    real(r8), intent(out) :: y(n, m)
    integer :: s
    do s = 1, m
      y(:, s) = f * x(:, s)
    end do
  end subroutine scale_tracers
end module tracer_demo
"""


def test_a_loop_over_a_zero_extent_axis_runs_no_iteration(tmp_path: Path) -> None:
    """CLUBB's scalar tracers under ``sclr_dim = 0``: ``do sclr = 1, sclr_dim``
    over ``(ngrdcol, nzm, 0)`` arrays. ``fori_loop`` traced the body once
    and JAX refused the index into the size-0 axis."""
    import importlib
    import sys

    candidate = port(tmp_path, EMPTY_AXIS, "tracer_demo")
    assert candidate.notes["jax"]["kernels"] == ["scale_tracers"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("tracer_demo_jax")
        import numpy as np

        empty = np.zeros((3, 0), order="F")
        assert np.asarray(module.scale_tracers(3, 0, 2.0, empty)).shape == (3, 0)
        x = np.ones((3, 2), order="F")
        assert np.asarray(module.scale_tracers(3, 2, 2.0, x)).tolist() == [[2.0, 2.0]] * 3
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"tracer_demo{suffix}", None)
