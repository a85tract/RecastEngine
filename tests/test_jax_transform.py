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
