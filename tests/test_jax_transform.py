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
    pieces, jitted, _delegated, _hosted = build_module(interface, tree)
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
  subroutine exit_index(n, zlo, z, kfound)
    integer,  intent(in)  :: n
    real(r8), intent(in)  :: zlo
    real(r8), intent(in)  :: z(n)
    integer,  intent(out) :: kfound
    integer :: k
    do k = 1, n
      if ( z(k) > zlo ) exit
    end do
    kfound = k
  end subroutine exit_index
  subroutine next_above(n, m, z, kfound)
    integer,  intent(in)  :: n, m
    real(r8), intent(in)  :: z(m)
    integer,  intent(out) :: kfound(n)
    integer :: i, k
    do i = 1, n
      do k = i, m
        if ( z(k) > 0.0_r8 ) exit
      end do
      kfound(i) = k
    end do
  end subroutine next_above
end module cycle_demo
"""


def test_a_cycle_folds_into_the_branch_and_an_exit_is_a_carried_flag(tmp_path: Path) -> None:
    """CLUBB's interpolators: ``if ( ... ) then ... cycle end if`` in a DO
    loop. The lowering passed the ``continue`` through into a ``lax.cond``
    branch -- a SyntaxError that took the whole emitted module down. Folded
    into the branch structure it is a kernel. An EXIT is a flag the loop
    carries: the trips after it do nothing, and the DO variable's value at
    the exit is what the code after the loop reads (CLUBB's window search
    and its sponge damping)."""
    import importlib
    import sys

    candidate = port(tmp_path, CYCLES, "cycle_demo")
    assert candidate.notes["jax"]["kernels"] == [
        "clip_below",
        "exit_index",
        "first_above",
        "next_above",
    ]
    assert candidate.notes["jax"]["delegated"] == {}
    emitted = candidate.files[Path("cycle_demo_jax.py")].decode()
    assert "continue" not in emitted.replace("continuation", "")
    assert "break" not in emitted
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
        assert int(module.first_above(4, 9.0, z)) == 0
        assert int(module.exit_index(4, 2.5, z)) == 3
        assert int(module.exit_index(4, 9.0, z)) == 5  # ran to completion: n + 1
        # The completion value of an inner DO whose bounds trace (the outer
        # index): the anchor's Python max over a tracer.
        zz = np.array([-1.0, 2.0, -1.0])
        assert np.asarray(module.next_above(3, 3, zz)).tolist() == [2, 2, 4]
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
    integer :: i
    root = sqrt( x )
    total = sum( x )
    do i = 1, n
      err(i) = erf( x(i) )
    end do
  end subroutine norms
end module shim_demo
"""


def test_the_jax_runtime_carries_sqrt_sum_and_erf(tmp_path: Path) -> None:
    """CLUBB's clipping and PDF closure: ``sqrt``, ``sum`` and ``erf`` reach
    the kernels as ``_f_sqrt``, ``_f_vsum`` and ``_f_verf``, which the NumPy
    runtime defines and the JAX one did not -- a NameError at the first
    call, on every kernel of the unit. A scalar ``erf`` arrives as
    ``math.erf``, which the math-to-jnp mapping cannot spell (jax.numpy has
    no erf): it goes through the same shim."""
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


FLOORED = """\
module floor_mod
  implicit none
  type coefs_type
    real(8), allocatable :: coef(:, :)
  end type coefs_type
contains
  subroutine init_coefs( nz, ngrdcol, c )
    integer, intent(in) :: nz, ngrdcol
    type(coefs_type), intent(inout) :: c
    allocate( c%coef(1:ngrdcol, 1:nz) )
    c%coef = 2.0d0
  end subroutine init_coefs
  subroutine apply( nzt, ngrdcol, c, x, floor )
    integer, intent(in) :: nzt, ngrdcol
    type(coefs_type), intent(in) :: c
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(in), optional :: floor
    x = x * c%coef(:, 1:nzt)
    if ( present( floor ) ) then
      x = max( x, floor )
    end if
  end subroutine apply
end module floor_mod
"""


def test_a_flat_kernel_spells_the_optional_dummy_the_plan_leaves_out(tmp_path: Path) -> None:
    """CLUBB's grid interpolators take an optional ``zt_min``; the plan drops
    it (the adapter calls with it absent) and the kernel inlines a body that
    tests ``zt_min is not None`` -- a NameError on every call."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "floor_mod.f90").write_text(FLOORED)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:floor_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    ported = candidate.files[Path("floor_mod_jax.py")].decode()
    assert "floor = None" in ported
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("floor_mod_jax")
        import numpy as np

        coef = np.full((2, 3), 2.0, order="F")
        x = np.ones((2, 3), order="F")
        assert "apply_flat" in candidate.notes["jax"]["kernels"]
        result = np.asarray(module.apply_flat(3, 2, x, 3, coef))
        assert result.tolist() == [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"floor_mod{suffix}", None)


PAIRS_IN_A_LOOP = """\
module pair_demo
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  elemental subroutine split(x, lo, hi)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: lo, hi
    lo = x - 1.0_r8
    hi = x + 1.0_r8
  end subroutine split
  subroutine bracket(n, nz, x, lo, hi)
    integer,  intent(in)  :: n, nz
    real(r8), intent(in)  :: x(n, nz)
    real(r8), intent(out) :: lo(n, nz), hi(n, nz)
    integer :: i
    do i = 1, n
      call split( x(i, :), lo(i, :), hi(i, :) )
    end do
  end subroutine bracket
end module pair_demo
"""


def test_a_call_result_tuple_inside_a_loop_is_the_bodys_own(tmp_path: Path) -> None:
    """The anchor spells a two-output call as ``_out = split(...)`` and
    unpacks it on the next lines. Carried through the fori_loop, ``_out``
    was read for the initial carry before any assignment
    (UnboundLocalError; CLUBB's new_hybrid_pdf_driver). A name not bound
    before the loop is the body's own."""
    import importlib
    import sys

    candidate = port(tmp_path, PAIRS_IN_A_LOOP, "pair_demo")
    assert "bracket" in candidate.notes["jax"]["kernels"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("pair_demo_jax")
        import numpy as np

        x = np.array([[1.0, 2.0], [3.0, 4.0]], order="F")
        lo, hi = (np.asarray(v) for v in module.bracket(2, 2, x))
        assert lo.tolist() == [[0.0, 1.0], [2.0, 3.0]] and hi.tolist() == [[2.0, 3.0], [4.0, 5.0]]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"pair_demo{suffix}", None)


WRITER = """\
module writer_mod
  implicit none
contains
  subroutine fill(n, x, y)
    integer,  intent(in)  :: n
    real(8),  intent(in)  :: x(n)
    real(8) :: y(n)
    y = x + 1.0d0
  end subroutine fill
end module writer_mod
"""

CALLS_WRITER = """\
module caller_mod
  use writer_mod, only: fill
  implicit none
contains
  subroutine run(n, x, y)
    integer,  intent(in)  :: n
    real(8),  intent(in)  :: x(n)
    real(8),  intent(out) :: y(n)
    call fill( n, x, y )
    y = y * 2.0d0
  end subroutine run
end module caller_mod
"""


def test_a_bare_call_binds_the_kernels_returned_buffer(tmp_path: Path) -> None:
    """CLUBB declares ``xp3_lg_2005_ansatz``'s ``xp3`` without an intent and
    the extension's frontend overrides it to OUT. The caller's anchor, seeing
    no intent at the call, calls bare: the callee writes ``y`` in place and
    returns it, and the caller ignores the return. A kernel cannot write in
    place -- its return *is* the output -- so the statement binds what
    comes back to the actual; without it advance_xp3 got back the zeros it
    passed in for every xp3."""
    import importlib
    import sys

    from recast.registry import REGISTRY
    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    def overriding_frontend(**_config: object) -> FortranFrontend:
        return FortranFrontend(
            flatten=True, buffer_out_arrays="all", intent_overrides={"fill": {"y": "OUT"}}
        )

    REGISTRY.register("frontend", "fortran-y-out", overriding_frontend, replace=True)
    (tmp_path / "writer_mod.f90").write_text(WRITER)
    (tmp_path / "caller_mod.f90").write_text(CALLS_WRITER)
    frontend = overriding_frontend()
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:caller_mod")
    facts = frontend.analyze(unit, tmp_path)
    conventions = TreeConventions(frontend="fortran-y-out")
    candidate = TreeToJax(conventions).apply(unit, facts, {"root": str(tmp_path)})
    assert "run" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    ported = candidate.files[Path("caller_mod_jax.py")].decode()
    assert "_writer_mod.fill(n, x, y)" in candidate.files[Path("caller_mod_numpy.py")].decode()
    assert "y = _writer_mod_jax._fill_k_impl(n, x, y)" in ported
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("caller_mod_jax")
        import numpy as np

        got = module.run(2, np.array([1.0, 2.0]), np.zeros(2))
        assert np.asarray(got).tolist() == [4.0, 6.0]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("caller_mod", "writer_mod")):
                sys.modules.pop(name, None)


ELEMENTAL_COMPANION = """\
module elem_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  elemental subroutine split(x, lo, hi)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: lo, hi
    lo = x - 1.0_r8
    hi = x + 1.0_r8
  end subroutine split
end module elem_mod
"""

CALLS_ELEMENTAL_COMPANION = """\
module bracket_mod
  use elem_mod, only: split
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains
  subroutine bracket(n, nz, x, lo, hi)
    integer,  intent(in)  :: n, nz
    real(r8), intent(in)  :: x(n, nz)
    real(r8), intent(out) :: lo(n, nz), hi(n, nz)
    integer :: i
    do i = 1, n
      call split( x(i, :), lo(i, :), hi(i, :) )
    end do
  end subroutine bracket
end module bracket_mod
"""


def test_an_elemental_call_of_a_companion_broadcasts_its_kernel(tmp_path: Path) -> None:
    """CLUBB's new_hybrid_pdf_driver calls new_hybrid_pdf's elemental
    ``calculate_mixture_fraction`` over column slices: the anchor's
    ``_f_ecall(_new.calculate_mixture_fraction, ...)``. Left as the host
    attribute, the vectorize traced a NumPy function; the companion's kernel
    implementation goes under it instead."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "elem_mod.f90").write_text(ELEMENTAL_COMPANION)
    (tmp_path / "bracket_mod.f90").write_text(CALLS_ELEMENTAL_COMPANION)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:bracket_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "bracket" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    ported = candidate.files[Path("bracket_mod_jax.py")].decode()
    assert "_f_ecall(_elem_mod_jax._split_k_impl, " in ported
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("bracket_mod_jax")
        import numpy as np

        x = np.array([[1.0, 2.0], [3.0, 4.0]], order="F")
        lo, hi = (np.asarray(v) for v in module.bracket(2, 2, x))
        assert lo.tolist() == [[0.0, 1.0], [2.0, 3.0]] and hi.tolist() == [[2.0, 3.0], [4.0, 5.0]]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("bracket_mod", "elem_mod")):
                sys.modules.pop(name, None)


GUARDED_BY_A_STATIC = """\
module guard_demo
  implicit none
contains
  subroutine first_tracer(n, m, x, y)
    integer,  intent(in)  :: n, m
    real(8),  intent(in)  :: x(n, m)
    real(8),  intent(out) :: y(n)
    y = 0.0d0
    if ( m > 0 ) then
      y(:) = x(:, 1)
    end if
  end subroutine first_tracer
end module guard_demo
"""


def test_a_branch_on_a_static_scalar_is_a_trace_time_if(tmp_path: Path) -> None:
    """CLUBB guards its scalar-tracer stores with ``if ( sclr_dim > 0 )``.
    Lowered to lax.cond both arms are traced, and the store into the
    zero-extent (or never allocated) array is an IndexError at trace time.
    ``sclr_dim`` is a static argument of the kernel -- a Python int under
    jit -- so the branch is a Python if."""
    import importlib
    import sys

    candidate = port(tmp_path, GUARDED_BY_A_STATIC, "guard_demo")
    assert candidate.notes["jax"]["kernels"] == ["first_tracer"]
    ported = candidate.files[Path("guard_demo_jax.py")].decode()
    assert "if _f_concrete(m > 0):" in ported and "if m > 0:" in ported
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("guard_demo_jax")
        import numpy as np

        assert np.asarray(module.first_tracer(2, 0, np.zeros((2, 0)))).tolist() == [0.0, 0.0]
        x = np.array([[1.0, 2.0], [3.0, 4.0]], order="F")
        assert np.asarray(module.first_tracer(2, 2, x)).tolist() == [1.0, 3.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"guard_demo{suffix}", None)


EARLY_RETURNS = """\
module early_mod
  implicit none
  type coefs_type
    real(8), allocatable :: coef(:, :)
  end type coefs_type
contains
  subroutine init_coefs( nz, ngrdcol, c )
    integer, intent(in) :: nz, ngrdcol
    type(coefs_type), intent(inout) :: c
    allocate( c%coef(1:ngrdcol, 1:nz) )
    c%coef = 2.0d0
  end subroutine init_coefs
  subroutine apply( nzt, ngrdcol, c, x, y, bad, worse )
    integer, intent(in) :: nzt, ngrdcol
    type(coefs_type), intent(in) :: c
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(out), dimension(ngrdcol, nzt) :: y
    logical, intent(in) :: bad, worse
    x = x * c%coef(:, 1:nzt)
    y = x
    if ( bad ) then
      if ( worse ) then
        return
      end if
      y = y + 1.0d0
      return
    end if
    y = y + 10.0d0
  end subroutine apply
end module early_mod
"""


def test_early_returns_of_the_same_tuple_fold_into_the_branches(tmp_path: Path) -> None:
    """CLUBB's advance_clubb_core returns after each solver when the error
    code says so, the outputs as they stand. The single-exit rewrite merges
    one value with a where and refused a tuple; when every early return is
    the final tuple, the continuation folds into the non-returning branches
    and the kernel keeps one exit."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "early_mod.f90").write_text(EARLY_RETURNS)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:early_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "apply_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("early_mod_jax")
        import numpy as np

        coef = np.full((1, 2), 2.0, order="F")

        def run(bad, worse):
            x = np.ones((1, 2), order="F")
            _x, y = module.apply_flat(2, 1, x, np.zeros((1, 2), order="F"), bad, worse, 2, coef)
            return np.asarray(y).tolist()

        assert run(False, False) == [[12.0, 12.0]]
        assert run(True, False) == [[3.0, 3.0]]
        assert run(True, True) == [[2.0, 2.0]]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"early_mod{suffix}", None)


RETURNS_IN_A_LOOP = """\
module loopret_mod
  implicit none
  type coefs_type
    real(8), allocatable :: coef(:, :)
  end type coefs_type
contains
  subroutine init_coefs( nz, ngrdcol, c )
    integer, intent(in) :: nz, ngrdcol
    type(coefs_type), intent(inout) :: c
    allocate( c%coef(1:ngrdcol, 1:nz) )
    c%coef = 2.0d0
  end subroutine init_coefs
  subroutine apply( nzt, ngrdcol, c, x, y, err )
    integer, intent(in) :: nzt, ngrdcol
    type(coefs_type), intent(in) :: c
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(out), dimension(ngrdcol, nzt) :: y
    integer, intent(in) :: err(ngrdcol)
    integer :: i
    real(8) :: lo, hi
    x = x * c%coef(:, 1:nzt)
    y = x
    do i = 1, ngrdcol
      if ( err(i) /= 0 ) then
        return
      end if
      y(i, :) = y(i, :) + 1.0d0
    end do
    y = y + 10.0d0
    call bracket( x(1, 1), lo, hi )
    y = y + lo + hi
  end subroutine apply
  elemental subroutine bracket( v, lo, hi )
    real(8), intent(in)  :: v
    real(8), intent(out) :: lo, hi
    lo = v - 1.0d0
    hi = v + 1.0d0
  end subroutine bracket
end module loopret_mod
"""


def test_an_early_return_inside_a_loop_becomes_a_flag(tmp_path: Path) -> None:
    """advance_clubb_core checks the error code per column inside a loop and
    returns. No branch structure holds that; the return sets a flag, every
    later statement runs under it, and the remaining iterations do nothing."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "loopret_mod.f90").write_text(RETURNS_IN_A_LOOP)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:loopret_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "apply_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("loopret_mod_jax")
        import numpy as np

        coef = np.full((2, 1), 2.0, order="F")

        def run(err):
            x = np.ones((2, 1), order="F")
            _x, y = module.apply_flat(
                1, 2, x, np.zeros((2, 1), order="F"), np.array(err, dtype=np.int32), 1, coef
            )
            return np.asarray(y).ravel().tolist()

        # x(1,1) is 2 after scaling: lo + hi = 4 on the path that reaches the
        # bracket call (an elemental with two outputs, the anchor's _out).
        assert run([0, 0]) == [17.0, 17.0]
        assert run([0, 1]) == [3.0, 2.0]  # the second column returns before its +1 and the rest
        assert run([1, 0]) == [2.0, 2.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"loopret_mod{suffix}", None)


UNPACKS_AN_OBJECT = """\
module unpack_mod
  implicit none
  type knobs_type
    real(8) :: gain = 2.0d0
  end type knobs_type
contains
  subroutine scale( nzt, ngrdcol, x, y, k )
    integer, intent(in) :: nzt, ngrdcol
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(inout), dimension(ngrdcol, nzt) :: y
    type(knobs_type), intent(inout), optional :: k
    if ( present( k ) ) k%gain = k%gain * 2.0d0
    x = x * 2.0d0
    y = x + 1.0d0
  end subroutine scale
  subroutine step( nzt, ngrdcol, k, x, y )
    integer, intent(in) :: nzt, ngrdcol
    type(knobs_type), intent(inout) :: k
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(inout), dimension(ngrdcol, nzt) :: y
    call scale( nzt, ngrdcol, x, y, k )
    y = y + k%gain
  end subroutine step
end module unpack_mod
"""


def test_unpacking_an_object_from_the_elided_call_buffer_is_already_true(tmp_path: Path) -> None:
    """advance_clubb_core's anchor unpacks what a solver hands back --
    ``stats = _out[0]``, ``pdf_params = _out[4]`` -- from the buffer the
    flat rewrite elides at the call: the flat outputs bound to the actuals,
    so the object's unpack is already true and the statement goes; an
    array's is the actual it names. The object here is an *optional* INOUT
    (calc_brunt_vaisala_freq_sqd's ``stats``): its slot is in the anchor's
    tuple all the same, and was not in the rewrite's list."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "unpack_mod.f90").write_text(UNPACKS_AN_OBJECT)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:unpack_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    assert "k = _out[2]" in candidate.files[Path("unpack_mod_numpy.py")].decode()
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("unpack_mod_jax")
        import numpy as np

        x = np.ones((1, 2), order="F")
        result = module.step_flat(2, 1, x, np.zeros((1, 2), order="F"), np.float64(2.0))
        got = [np.asarray(v).tolist() for v in result]
        # The optional object is absent in the flat world (the plan leaves
        # optional dummies out, as the adapter calls without them), so the
        # callee's ``present(k)`` branch does not run: gain stays 2, y is
        # x * 2 + 1 + 2.
        assert got == [[[2.0, 2.0]], [[5.0, 5.0]], 2.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"unpack_mod{suffix}", None)


CHECKS = """\
module checks_mod
  implicit none
contains
  subroutine complain( n, x, msg )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    character(len=*), intent(out) :: msg
    msg = "fine"
    if ( any( x < 0.0d0 ) ) msg = "negative"
  end subroutine complain
end module checks_mod
"""

GUARDED_CHECK = """\
module guarded_mod
  use checks_mod, only: complain
  implicit none
contains
  subroutine step( n, debug_level, x, y )
    integer, intent(in) :: n, debug_level
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    character(len=16) :: msg
    y = 2.0d0 * x
    if ( debug_level >= 2 ) then
      call complain( n, y, msg )
    end if
  end subroutine step
end module guarded_mod
"""


def test_a_companion_procedure_the_port_left_on_the_host_stays_under_its_guard(
    tmp_path: Path,
) -> None:
    """CLUBB's advance_clubb_core calls numerical_check's parameterization
    check under ``clubb_at_least_debug_level_api(2)``, false with statistics
    off; the check takes character arguments and its port leaves it on the
    host. Refusing the whole step for a call that never runs was the
    alternative: the call stays the host's under its trace-time guard, and
    the note names it."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "checks_mod.f90").write_text(CHECKS)
    (tmp_path / "guarded_mod.f90").write_text(GUARDED_CHECK)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:guarded_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    assert candidate.notes["jax"]["host_calls"] == {"step": ["checks_mod.complain"]}
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("guarded_mod_jax")
        import numpy as np

        assert np.asarray(module.step(2, 0, np.array([1.0, -1.0]))).tolist() == [2.0, -2.0]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("guarded_mod", "checks_mod")):
                sys.modules.pop(name, None)


OBJECT_CHECK = """\
module ocheck_mod
  implicit none
  type knobs_type
    real(8) :: gain = 2.0d0
  end type knobs_type
contains
  subroutine inspect( n, k, x, msg )
    integer, intent(in) :: n
    type(knobs_type), intent(inout) :: k
    real(8), intent(in) :: x(n)
    character(len=*), intent(out) :: msg
    msg = "fine"
    if ( any( x < k%gain ) ) msg = "small"
    k%gain = k%gain + 1.0d0
  end subroutine inspect
end module ocheck_mod
"""

GUARDED_OBJECT_CHECK = """\
module oguarded_mod
  use ocheck_mod, only: knobs_type, inspect
  implicit none
contains
  subroutine step( n, debug_level, k, x, y )
    integer, intent(in) :: n, debug_level
    type(knobs_type), intent(inout) :: k
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    character(len=16) :: msg
    y = k%gain * x
    if ( debug_level >= 2 ) then
      call inspect( n, k, y, msg )
    end if
    y = y + k%gain
  end subroutine step
end module oguarded_mod
"""


def test_a_flat_companion_the_port_left_on_the_host_stays_under_its_guard(tmp_path: Path) -> None:
    """The same, through the flat-callee path: the check takes the object
    (advance_clubb_core's parameterization check takes gr and err_info), so
    it has a plan, and the rewrite of the call into the port's flat kernel
    found none."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "ocheck_mod.f90").write_text(OBJECT_CHECK)
    (tmp_path / "oguarded_mod.f90").write_text(GUARDED_OBJECT_CHECK)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:oguarded_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    assert candidate.notes["jax"]["host_calls"] == {"step_flat": ["ocheck_mod.inspect"]}
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("oguarded_mod_jax")
        import numpy as np

        got = module.step_flat(2, 0, np.array([1.0, -1.0]), np.zeros(2), 1, np.float64(2.0))
        y, gain = (np.asarray(v).tolist() for v in got)
        assert y == [4.0, 0.0] and gain == 2.0
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("oguarded_mod", "ocheck_mod")):
                sys.modules.pop(name, None)


OBJECT_QUERY = """\
module oquery_mod
  use ocheck_mod, only: knobs_type
  use stats_query_mod, only: var_on_list
  implicit none
contains
  subroutine step( n, k, x, y )
    integer, intent(in) :: n
    type(knobs_type), intent(in) :: k
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    y = k%gain * x
    if ( var_on_list( k, "gain" ) ) then
      y = y + k%gain
    end if
  end subroutine step
end module oquery_mod
"""

STATS_QUERY = """\
module stats_query_mod
  use ocheck_mod, only: knobs_type
  implicit none
contains
  logical function var_on_list( k, name )
    type(knobs_type), intent(in) :: k
    character(len=*), intent(in) :: name
    var_on_list = .true.
  end function var_on_list
end module stats_query_mod
"""


def test_an_object_handed_whole_to_the_host_is_rebuilt_at_entry(tmp_path: Path) -> None:
    """``if ( var_on_stats_list( stats, "rsat" ) )``: the query is a
    framework stand-in's, and it takes the object whole -- which the kernel
    took apart into components. Rebuilt once at entry from them, as the
    NumPy flat wrapper does, instead of an UnboundLocalError on a name the
    kernel never bound."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "ocheck_mod.f90").write_text(OBJECT_CHECK)
    (tmp_path / "oquery_mod.f90").write_text(OBJECT_QUERY)
    (tmp_path / "stats_query_mod.f90").write_text(STATS_QUERY)
    frontend = FortranFrontend(flatten=True, stub_modules=["stats_query_mod"])
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:oquery_mod")
    facts = frontend.analyze(unit, tmp_path)
    conventions = TreeConventions(
        stub_modules=frozenset({"stats_query_mod"}),
        framework={"stats_query_mod": "def var_on_list(k, name):\n    return name == 'gain'\n"},
    )
    candidate = TreeToJax(conventions).apply(unit, facts, {"root": str(tmp_path)})
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    emitted = candidate.files[Path("oquery_mod_jax.py")].decode()
    assert "k = _host._Record(gain=k__gain)" in emitted
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("oquery_mod_jax")
        import numpy as np

        got = module.step_flat(2, np.array([1.0, -1.0]), np.zeros(2), 2, np.float64(2.0))
        assert np.asarray(got).tolist() == [4.0, 0.0]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("oquery_mod", "ocheck_mod", "stats_query_mod")):
                sys.modules.pop(name, None)


EMPTY_LOOP = """\
module emptyloop_mod
  implicit none
contains
  subroutine scale( n, x, y )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: i
    y = 2.0d0 * x
    do i = 1, n
      continue
    end do
  end subroutine scale
end module emptyloop_mod
"""


def test_a_loop_whose_body_lowered_to_nothing_is_dropped(tmp_path: Path) -> None:
    """CLUBB's clipping routines loop over the columns to sample statistics;
    with the sampling calls dropped by their stand-in, the loop's body is
    ``pass`` and there is nothing to carry. Fortran ran it for nothing."""
    import importlib
    import sys

    candidate = port(tmp_path, EMPTY_LOOP, "emptyloop_mod")
    assert candidate.notes["jax"]["kernels"] == ["scale"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("emptyloop_mod_jax")
        import numpy as np

        got = module.scale(2, np.array([1.0, 3.0]))
        assert np.asarray(got).tolist() == [2.0, 6.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"emptyloop_mod{suffix}", None)


STRIDED = """\
module strided_mod
  implicit none
contains
  subroutine running( n, dir, x, y )
    integer, intent(in) :: n, dir
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: k, lb, ub
    real(8) :: acc
    if ( dir > 0 ) then
      lb = 1
      ub = n
    else
      lb = n
      ub = 1
    end if
    acc = 0.0d0
    do k = lb, ub, dir
      acc = acc + x(k)
      y(k) = acc
    end do
  end subroutine running
end module strided_mod
"""


def test_a_loop_with_a_named_stride_runs_in_that_direction(tmp_path: Path) -> None:
    """``do k = gr%k_lb_zt, gr%k_ub_zt, gr%grid_dir_indx``: CLUBB's grid
    direction is a run-time +-1, a name in the stride slot. The trip count
    is the runtime's, the index is remapped from the trip."""
    import importlib
    import sys

    candidate = port(tmp_path, STRIDED, "strided_mod")
    assert candidate.notes["jax"]["kernels"] == ["running"], candidate.notes["jax"]
    assert "_f_trips(" in candidate.files[Path("strided_mod_jax.py")].decode()
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("strided_mod_jax")
        import numpy as np

        x = np.array([1.0, 2.0, 4.0])
        up = np.asarray(module.running(3, 1, x)).tolist()
        down = np.asarray(module.running(3, -1, x)).tolist()
        assert up == [1.0, 3.0, 7.0]
        assert down == [7.0, 6.0, 4.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"strided_mod{suffix}", None)


SWITCHED_CHECK = """\
module switched_mod
  use silent_mod, only: complain
  implicit none
contains
  subroutine step( n, l_check, x, y )
    integer, intent(in) :: n
    logical, intent(in) :: l_check
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    y = 2.0d0 * x
    if ( l_check ) then
      call complain( n, y )
    end if
  end subroutine step
end module switched_mod
"""


def test_a_logical_scalar_dummy_is_a_static_switch(tmp_path: Path) -> None:
    """CLUBB's configuration flags are logical dummies (the model's
    ``clubb_config_flags`` components). Static under jit: the branch is a
    Python if at trace time, and the check the port left on the host --
    under a flag the run has off -- is never traced."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "silent_mod.f90").write_text(SILENT_CHECK)
    (tmp_path / "switched_mod.f90").write_text(SWITCHED_CHECK)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:switched_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    emitted = candidate.files[Path("switched_mod_jax.py")].decode()
    assert "static_argnums=(0, 1)" in emitted
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("switched_mod_jax")
        import numpy as np

        got = module.step(2, False, np.array([1.0, -1.0]))
        assert np.asarray(got).tolist() == [2.0, -2.0]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("switched_mod", "silent_mod")):
                sys.modules.pop(name, None)


OMITS_AN_OBJECT = """\
module omit_mod
  implicit none
  type knobs_type
    real(8) :: gain = 2.0d0
  end type knobs_type
contains
  subroutine scale( nzt, ngrdcol, x, y, k )
    integer, intent(in) :: nzt, ngrdcol
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(inout), dimension(ngrdcol, nzt) :: y
    type(knobs_type), intent(inout), optional :: k
    if ( present( k ) ) k%gain = k%gain * 2.0d0
    x = x * 2.0d0
    y = x + 1.0d0
  end subroutine scale
  subroutine step( nzt, ngrdcol, k, x, y )
    integer, intent(in) :: nzt, ngrdcol
    type(knobs_type), intent(inout) :: k
    real(8), intent(inout), dimension(ngrdcol, nzt) :: x
    real(8), intent(inout), dimension(ngrdcol, nzt) :: y
    call scale( nzt, ngrdcol, x, y )
    y = y + k%gain
  end subroutine step
end module omit_mod
"""


def test_an_optional_object_the_caller_leaves_out_is_absent_in_the_callee(tmp_path: Path) -> None:
    """pdf_closure_driver_zm calls pdf_closure without its optional
    ``pdf_implicit_coefs_terms``. The callee's kernel still takes the
    object's components (its flat signature has them): ``None`` in each,
    ``present()`` false at trace time, and what it hands back for them
    dropped."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "omit_mod.f90").write_text(OMITS_AN_OBJECT)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:omit_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("omit_mod_jax")
        import numpy as np

        x = np.ones((1, 2), order="F")
        result = module.step_flat(2, 1, x, np.zeros((1, 2), order="F"), np.float64(2.0))
        got = [np.asarray(v).tolist() for v in result]
        # gain is read, never written, so it is not returned; the callee's
        # present(k) branch did not run: y is x * 2 + 1 + 2.
        assert got == [[[2.0, 2.0]], [[5.0, 5.0]]]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"omit_mod{suffix}", None)


DISCARDS_AN_OUTPUT = """\
module discard_mod
  implicit none
  type knobs_type
    real(8) :: gain = 2.0d0
  end type knobs_type
contains
  subroutine solve( nzt, ngrdcol, k, x, y, resid )
    integer, intent(in) :: nzt, ngrdcol
    type(knobs_type), intent(in) :: k
    real(8), intent(in), dimension(ngrdcol, nzt) :: x
    real(8), intent(inout), dimension(ngrdcol, nzt) :: y
    real(8), intent(out), dimension(ngrdcol, nzt), optional :: resid
    y = k%gain * x
    if ( present( resid ) ) resid = y - x
  end subroutine solve
  subroutine step( nzt, ngrdcol, k, x, y )
    integer, intent(in) :: nzt, ngrdcol
    type(knobs_type), intent(in) :: k
    real(8), intent(in), dimension(ngrdcol, nzt) :: x
    real(8), intent(inout), dimension(ngrdcol, nzt) :: y
    call solve( nzt, ngrdcol, k, x, y )
    y = y + 1.0d0
  end subroutine step
end module discard_mod
"""


def test_an_output_the_anchor_discards_is_dropped_from_the_elided_buffer(tmp_path: Path) -> None:
    """CLUBB's solvers return an optional residual the caller does not ask
    for: the anchor's ``_ = _out[4]`` names a slot with no actual, and the
    unpack is dropped rather than refused."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "discard_mod.f90").write_text(DISCARDS_AN_OUTPUT)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:discard_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "_ = _out[" in candidate.files[Path("discard_mod_numpy.py")].decode()
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("discard_mod_jax")
        import numpy as np

        x = np.ones((1, 2), order="F")
        got = module.step_flat(2, 1, x, np.zeros((1, 2), order="F"), np.float64(3.0))
        assert np.asarray(got).tolist() == [[4.0, 4.0]]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"discard_mod{suffix}", None)


SIZED_BY_AN_ALLOCATOR = """\
module sized_mod
  implicit none
  type coefs_type
    real(8), allocatable :: coef(:, :)
  end type coefs_type
contains
  subroutine init_coefs( ngrdcol, nz, p )
    integer, intent(in) :: ngrdcol, nz
    type(coefs_type), intent(out) :: p
    allocate( p%coef(1:ngrdcol, 1:nz) )
    p%coef = 0.0d0
  end subroutine init_coefs
  subroutine apply_coefs( ngrdcol, p, x )
    integer, intent(in) :: ngrdcol
    type(coefs_type), intent(in) :: p
    real(8), intent(inout) :: x(ngrdcol)
    integer :: i
    do i = 1, ngrdcol
      x(i) = x(i) + sum( p%coef(i, :) )
    end do
  end subroutine apply_coefs
  subroutine step( ngrdcol, nz, p, x )
    integer, intent(in) :: ngrdcol, nz
    type(coefs_type), intent(inout) :: p
    real(8), intent(inout) :: x(ngrdcol)
    p%coef(:, 1) = p%coef(:, 1) + 1.0d0
    call apply_coefs( ngrdcol, p, x )
  end subroutine step
end module sized_mod
"""


def test_a_callee_extent_argument_is_the_callers_axis(tmp_path: Path) -> None:
    """A component allocated by another routine's dummy (CLUBB's
    ``coef_wp4_implicit(1:ngrdcol, 1:nz)``) reaches a callee that has no
    dummy of that name as an extent argument of its plan. The caller
    passes that axis of its own component."""
    import importlib
    import sys

    from recast.fortran.flatten import plans_from_facts
    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "sized_mod.f90").write_text(SIZED_BY_AN_ALLOCATOR)
    frontend = FortranFrontend(flatten={"patch_count": "ngrdcol", "bounds_pattern": r"^ngrdcol$"})
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:sized_mod")
    facts = frontend.analyze(unit, tmp_path)
    plans = {p.name: p for p in plans_from_facts(facts)}
    assert plans["apply_coefs_flat"].extent_args, plans["apply_coefs_flat"]
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("sized_mod_jax")
        import numpy as np

        coef = np.zeros((1, 2), order="F")
        got = module.step_flat(1, 2, np.array([1.0]), coef)
        assert [np.asarray(v).tolist() for v in got] == [[2.0], [[1.0, 0.0]]]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"sized_mod{suffix}", None)


OWN_CHECK_UNDER_A_SWITCH = """\
module ownswitch_mod
  implicit none
contains
  subroutine complain( n, x, msg )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    character(len=*), intent(out) :: msg
    msg = "fine"
    if ( any( x < 0.0d0 ) ) msg = "negative"
  end subroutine complain
  subroutine step( n, l_check, x, y )
    integer, intent(in) :: n
    logical, intent(in) :: l_check
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    character(len=16) :: msg
    y = 2.0d0 * x
    if ( l_check ) then
      call complain( n, y, msg )
    end if
  end subroutine step
end module ownswitch_mod
"""


def test_a_subprogram_of_the_module_the_port_could_not_emit_stays_on_the_host(
    tmp_path: Path,
) -> None:
    """pdf_closure_driver calls its zm variant under a switch the run has
    off; fill_holes_vertical_api dispatches on a type. A same-module
    callee the port could not emit no longer delegates its caller and the
    callers above it: the call stays the host's (``_host.<name>``) under
    the anchor's guard, named in the notes, never traced while the guard
    holds."""
    import importlib
    import sys

    candidate = port(tmp_path, OWN_CHECK_UNDER_A_SWITCH, "ownswitch_mod")
    assert candidate.notes["jax"]["kernels"] == ["step"], candidate.notes["jax"]
    assert "complain" in candidate.notes["jax"]["delegated"]
    assert candidate.notes["jax"]["host_calls"] == {"step": ["ownswitch_mod.complain"]}
    emitted = candidate.files[Path("ownswitch_mod_jax.py")].decode()
    assert "_host.complain(" in emitted
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("ownswitch_mod_jax")
        import numpy as np

        got = module.step(2, False, np.array([1.0, -1.0]))
        assert np.asarray(got).tolist() == [2.0, -2.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"ownswitch_mod{suffix}", None)


WINDOWED = """\
module window_mod
  implicit none
contains
  subroutine fill( n, w, dir, x, y )
    integer, intent(in) :: n, w, dir
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: k, lo, hi
    y = x
    do k = 1 + w, n - w
      lo = k - dir * w
      hi = k + dir * w
      if ( any( x(lo:hi:dir) < 0.0d0 ) ) then
        y(k) = -1.0d0
      else
        y(k) = maxval( x(lo:hi:dir) )
      end if
    end do
  end subroutine fill
end module window_mod
"""


def test_a_window_with_traced_bounds_a_static_distance_apart_is_a_gather(tmp_path: Path) -> None:
    """CLUBB's sliding-window hole filler reads ``field(i, k_start:k_end:dir)``
    with ``k_start = k - dir * n`` and ``k_end = k + dir * n``: bounds that
    trace (the loop index), a distance apart that does not. The slice is a
    gather at ``lo + arange(trips) * step`` -- the mask rules, for a slice
    whose length depends on a traced bound, never see it."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "window_mod.f90").write_text(WINDOWED)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:window_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert candidate.notes["jax"]["kernels"] == ["fill"], candidate.notes["jax"]
    emitted = candidate.files[Path("window_mod_jax.py")].decode()
    assert "jnp.arange(_f_trips(0, " in emitted, emitted
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("window_mod_jax")
        import numpy as np

        x = np.array([1.0, 5.0, 2.0, -3.0, 4.0, 6.0])
        got = np.asarray(module.fill(6, 1, 1, x)).tolist()
        assert got == [1.0, 5.0, -1.0, -1.0, -1.0, 6.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"window_mod{suffix}", None)


RESHAPED_ACTUAL = """\
module reshaped_mod
  implicit none
  type knobs_type
    real(8) :: gain = 2.0d0
  end type knobs_type
contains
  subroutine solve( n, k, rhs )
    integer, intent(in) :: n
    type(knobs_type), intent(in) :: k
    real(8), intent(inout) :: rhs(n, 1)
    rhs(:, 1) = k%gain * rhs(:, 1)
  end subroutine solve
  subroutine step( n, k, x )
    integer, intent(in) :: n
    type(knobs_type), intent(in) :: k
    real(8), intent(inout) :: x(n)
    call solve( n, k, x )
    x = x + 1.0d0
  end subroutine step
end module reshaped_mod
"""


def test_an_array_passed_through_a_reshape_comes_back_in_its_own_shape(tmp_path: Path) -> None:
    """CLUBB hands a 2-d right-hand side to a solver declared over three
    axes: sequence association, which the anchor spells as
    ``np.reshape(rhs, (n, m, 1), order='F')`` for the actual. The kernel's
    output is reshaped back to the array's own shape and rebinds it -- a
    reshape is no store target."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "reshaped_mod.f90").write_text(RESHAPED_ACTUAL)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:reshaped_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "reshape(x" in candidate.files[Path("reshaped_mod_numpy.py")].decode()
    assert "step_flat" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("reshaped_mod_jax")
        import numpy as np

        got = module.step_flat(2, np.array([1.0, 3.0]), 2, np.float64(2.0))
        assert np.asarray(got).tolist() == [3.0, 7.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"reshaped_mod{suffix}", None)


EMPTY_GUARD = """\
module emptyguard_mod
  implicit none
contains
  subroutine scale( n, level, x, y )
    integer, intent(in) :: n, level
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    y = 2.0d0 * x
    if ( level > 0 ) then
      print *, "scaled", n
    end if
    if ( level > 1 ) then
      print *, "twice"
    else
      y = y + 1.0d0
    end if
  end subroutine scale
end module emptyguard_mod
"""


def test_a_static_branch_whose_arms_lowered_to_nothing_is_no_branch(tmp_path: Path) -> None:
    """``if ( clubb_at_least_debug_level( 0 ) ) then`` around a print: the
    print is dropped, the static Python if was emitted with no body -- a
    SyntaxError that took the whole emitted module down. Nothing in both
    arms is no branch; nothing in one arm is ``pass``."""
    import importlib
    import sys

    candidate = port(tmp_path, EMPTY_GUARD, "emptyguard_mod")
    assert candidate.notes["jax"]["kernels"] == ["scale"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("emptyguard_mod_jax")
        import numpy as np

        assert np.asarray(module.scale(2, 0, np.array([1.0, 3.0]))).tolist() == [3.0, 7.0]
        assert np.asarray(module.scale(2, 2, np.array([1.0, 3.0]))).tolist() == [2.0, 6.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"emptyguard_mod{suffix}", None)


PDF_KINDS = """\
module kinds_mod
  implicit none
  integer, parameter :: I_PLAIN = 1
  integer, parameter :: I_TWICE = 2
  logical, parameter :: L_QUINTIC = .false.
end module kinds_mod
"""

DISPATCHES_ON_A_KIND = """\
module dispatch_mod
  use kinds_mod, only: I_PLAIN, I_TWICE
  use silent_mod, only: complain
  implicit none
contains
  subroutine pick( n, kind, x, y )
    integer, intent(in) :: n, kind
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    if ( kind == I_TWICE ) then
      y = 2.0d0 * x
    else if ( kind == I_PLAIN ) then
      y = x
    else
      y = 0.0d0
      call complain( n, y )
    end if
    if ( .not. ( kind == I_PLAIN ) ) then
      y = y + 1.0d0
    end if
  end subroutine pick
end module dispatch_mod
"""


def test_a_dispatch_on_a_module_constant_through_its_alias_is_static(tmp_path: Path) -> None:
    """``iipdf_type == _mod.IIPDF_ADG1``: pdf_closure picks its PDF by a
    constant of model_flags, spelled through the module alias. Static, the
    way the bare upper-case spelling is, so the arm the run never takes --
    with its host call -- is never traced."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "kinds_mod.f90").write_text(PDF_KINDS)
    (tmp_path / "silent_mod.f90").write_text(SILENT_CHECK)
    (tmp_path / "dispatch_mod.f90").write_text(DISPATCHES_ON_A_KIND)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:dispatch_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "pick" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    emitted = candidate.files[Path("dispatch_mod_jax.py")].decode()
    # The dual form: a Python if when the kind is concrete (through the jit
    # wrapper it is), the lax.cond only for a traced caller.
    assert "if _f_concrete(kind == _kinds_mod.I_TWICE):" in emitted, emitted
    # ``.not. ( kind == I_PLAIN )``: the Python form's test is Python's
    # ``not``, not jnp.logical_not, which jit stages into a tracer no Python
    # if can convert (interpolation's ``.not. l_quintic_poly_interp``).
    assert "if not kind == _kinds_mod.I_PLAIN:" in emitted, emitted
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("dispatch_mod_jax")
        import numpy as np

        x = np.array([1.0, 3.0])
        assert np.asarray(module.pick(2, 2, x)).tolist() == [3.0, 7.0]
        assert np.asarray(module.pick(2, 1, x)).tolist() == [1.0, 3.0]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("dispatch_mod", "kinds_mod", "silent_mod")):
                sys.modules.pop(name, None)


NESTED_WHILE = """\
module nestedwhile_mod
  implicit none
contains
  subroutine locate( n, m, x, grid, idx )
    integer, intent(in) :: n, m
    real(8), intent(in) :: x(n), grid(m)
    integer, intent(out) :: idx(n)
    integer :: i, k
    logical :: calc_done
    idx = 0
    do i = 1, n
      if ( x(i) > 0.0d0 ) then
        k = 1
        calc_done = .false.
        do while ( .not. calc_done .and. k <= m )
          if ( grid(k) >= x(i) ) then
            idx(i) = k
            calc_done = .true.
          end if
          k = k + 1
        end do
      end if
    end do
  end subroutine locate
end module nestedwhile_mod
"""


def test_a_while_inside_a_branch_inside_a_loop_has_its_flag_before_the_branch(
    tmp_path: Path,
) -> None:
    """interpolation's lin_interp_between_grids: a DO WHILE search under an
    IF inside a DO. The while's exit flag is a carry of the enclosing cond,
    so it needs a value before the branch -- at the top of the function,
    like the goto-region flags -- not only beside its loop."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "nestedwhile_mod.f90").write_text(NESTED_WHILE)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:nestedwhile_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert candidate.notes["jax"]["kernels"] == ["locate"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("nestedwhile_mod_jax")
        import numpy as np

        x = np.array([0.5, -1.0, 2.5, 9.0])
        grid = np.array([1.0, 2.0, 3.0])
        got = np.asarray(module.locate(4, 3, x, grid)).tolist()
        assert got == [1, 0, 3, 0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"nestedwhile_mod{suffix}", None)


EXIT_UNDER_A_GUARD = """\
module guardedexit_mod
  implicit none
contains
  subroutine first_above( n, m, zlo, z, kfound )
    integer, intent(in) :: n, m
    real(8), intent(in) :: zlo(n), z(n, m)
    integer, intent(out) :: kfound(n)
    integer :: i, k
    kfound = -1
    do i = 1, n
      if ( zlo(i) > 0.0d0 ) then
        kfound(i) = 0
        do k = 1, m
          if ( z(i, k) > zlo(i) ) then
            kfound(i) = k
            exit
          end if
        end do
      end if
    end do
  end subroutine first_above
end module guardedexit_mod
"""


def test_an_exit_inside_a_branch_inside_a_loop_has_its_flag_before_the_branch(
    tmp_path: Path,
) -> None:
    """advance_helper's window search: a DO with an EXIT under an IF inside
    the column loop. The break flag and the kept index are carries of the
    enclosing cond and need a value before the branch -- at the top of the
    function, like the return flag -- not only beside their loop."""
    import importlib
    import sys

    candidate = port(tmp_path, EXIT_UNDER_A_GUARD, "guardedexit_mod")
    assert candidate.notes["jax"]["kernels"] == ["first_above"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("guardedexit_mod_jax")
        import numpy as np

        zlo = np.array([2.5, -1.0, 9.0])
        z = np.asfortranarray([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        got = np.asarray(module.first_above(3, 3, zlo, z)).tolist()
        assert got == [3, -1, 0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"guardedexit_mod{suffix}", None)


COUNTED_UNDER_SWITCHES = """\
module counted_mod
  implicit none
contains
  subroutine solve_count( n, sclr_dim, l_a, l_b, x, y )
    integer, intent(in) :: n, sclr_dim
    logical, intent(in) :: l_a, l_b
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    real(8), allocatable :: work(:, :)
    logical :: l_ab
    integer :: nrhs
    l_ab = l_a .and. l_b
    if ( sclr_dim > 0 ) then
      nrhs = 1
    else
      nrhs = 2 + sclr_dim
      if ( l_a ) then
        nrhs = nrhs + 2
        if ( l_ab ) then
          nrhs = nrhs + 2
        end if
      end if
    end if
    allocate( work(n, nrhs) )
    work = 1.0d0
    y = sum( work, dim = 2 ) * x
    deallocate( work )
  end subroutine solve_count
end module counted_mod
"""


def test_a_shape_counted_under_switches_is_a_trace_time_value(tmp_path: Path) -> None:
    """advance_xm_wpxp counts its right-hand sides under configuration
    switches and a local derived from them, then sizes its solver arrays
    by the count. The count is a trace-time value: its branches are Python
    ifs, its stores are never strengthened to jnp scalars, and
    ``jnp.zeros((n, nrhs))`` sees a Python int."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "counted_mod.f90").write_text(COUNTED_UNDER_SWITCHES)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:counted_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "solve_count" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("counted_mod_jax")
        import numpy as np

        x = np.array([1.0, 2.0])
        assert np.asarray(module.solve_count(2, 0, True, True, x)).tolist() == [6.0, 12.0]
        assert np.asarray(module.solve_count(2, 0, True, False, x)).tolist() == [4.0, 8.0]
        assert np.asarray(module.solve_count(2, 1, True, True, x)).tolist() == [1.0, 2.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"counted_mod{suffix}", None)


AFTER_A_CHECK = """\
module aftercheck_mod
  implicit none
contains
  subroutine scale_tail( n, x, y )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    integer :: k, k0
    y = x
    if ( any( x < 0.0d0 ) ) return
    k0 = n - 1
    do k = k0, n
      y(k) = 2.0d0 * x(k)
    end do
  end subroutine scale_tail
end module aftercheck_mod
"""


def test_a_store_after_a_traced_return_is_a_carried_value(tmp_path: Path) -> None:
    """mixing_length's ``start_index = gr%k_lb_zt + gr%grid_dir_indx`` after
    its error check: a trace-time expression, but under the return's guard
    -- a traced branch -- it is a carried value, and a carry of a NumPy
    int32 beside a Python int is a cond with unequal arms. Judged on the
    body as the exits leave it, it is strengthened like any carry."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "aftercheck_mod.f90").write_text(AFTER_A_CHECK)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:aftercheck_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "scale_tail" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("aftercheck_mod_jax")
        import numpy as np

        n = np.int32(3)  # the gate hands static scalars over as NumPy ints
        assert np.asarray(module.scale_tail(n, np.array([1.0, 2.0, 3.0]))).tolist() == [
            1.0,
            4.0,
            6.0,
        ]
        assert np.asarray(module.scale_tail(n, np.array([1.0, -2.0, 3.0]))).tolist() == [
            1.0,
            -2.0,
            3.0,
        ]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"aftercheck_mod{suffix}", None)


NAMED_BRANCH = """\
module named_mod
  implicit none
contains
  subroutine label( n, x, y )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    character(len=8) :: name
    integer :: k
    name = ""
    do k = 1, n
      if ( x(k) < 0.0d0 ) then
        name = "negative"
        y(k) = -x(k)
      else
        name = "positive"
        y(k) = x(k)
      end if
    end do
    if ( name == "negative" ) print *, "last was negative"
  end subroutine label
end module named_mod
"""


def test_a_character_local_is_never_a_carry(tmp_path: Path) -> None:
    """advance_xm_wpxp picks statistics names by the solve type under a
    traced branch. No JAX type carries a string; with statistics off
    nothing reads it but the dropped calls. It is a trace-time value, not
    a carry of the cond or the loop."""
    import importlib
    import sys

    candidate = port(tmp_path, NAMED_BRANCH, "named_mod")
    assert candidate.notes["jax"]["kernels"] == ["label"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("named_mod_jax")
        import numpy as np

        got = np.asarray(module.label(3, np.array([1.0, -2.0, 3.0]))).tolist()
        assert got == [1.0, 2.0, 3.0]
    finally:
        sys.path.remove(str(out))
        for suffix in ("_jax", "_numpy", "_jax_runtime", "_constants"):
            sys.modules.pop(f"named_mod{suffix}", None)


SILENT_CHECK = """\
module silent_mod
  implicit none
contains
  subroutine complain( n, x )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    if ( any( x < 0.0d0 ) ) print *, "negative", n
  end subroutine complain
end module silent_mod
"""

TRACED_GUARD_CHECK = """\
module tguarded_mod
  use silent_mod, only: complain
  implicit none
contains
  subroutine step( n, x, y )
    integer, intent(in) :: n
    real(8), intent(in) :: x(n)
    real(8), intent(out) :: y(n)
    y = 2.0d0 * x
    if ( y(1) < 0.0d0 ) then
      call complain( n, y )
    end if
  end subroutine step
end module tguarded_mod
"""


def test_a_host_only_call_under_a_traced_guard_carries_nothing(tmp_path: Path) -> None:
    """The same check under a guard on the data (``any(err_code == fatal)``):
    the branch binds nothing and nothing in it could run under a tracer, so
    the lowering carries nothing rather than refusing the kernel."""
    import importlib
    import sys

    from recast.transform.jax.tree import TreeToJax
    from recast.transform.numpy.tree import TreeConventions

    (tmp_path / "silent_mod.f90").write_text(SILENT_CHECK)
    (tmp_path / "tguarded_mod.f90").write_text(TRACED_GUARD_CHECK)
    frontend = FortranFrontend(flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:tguarded_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(TreeConventions()).apply(unit, facts, {"root": str(tmp_path)})
    assert "step" in candidate.notes["jax"]["kernels"], candidate.notes["jax"]
    out = tmp_path / "emitted"
    out.mkdir()
    for path, content in candidate.files.items():
        (out / path.name).write_bytes(content)
    sys.path.insert(0, str(out))
    try:
        module = importlib.import_module("tguarded_mod_jax")
        import numpy as np

        assert np.asarray(module.step(2, np.array([-1.0, 1.0]))).tolist() == [-2.0, 2.0]
    finally:
        sys.path.remove(str(out))
        for name in list(sys.modules):
            if name.startswith(("tguarded_mod", "silent_mod")):
                sys.modules.pop(name, None)
