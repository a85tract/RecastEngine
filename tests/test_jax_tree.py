"""``port.tree-jax``: the flat function the plan gives the JAX backend.

No JAX needed: emission is AST work. The fixture tree is
``tests/test_flatten.py``'s -- a physics routine over an object with
pointer components, an ``associate``, and a module variable the run sets.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytest.importorskip("fparser")
pytest.importorskip("numpy")

from recast.fortran.flatten import plans_from_facts
from recast.fortran.frontend import FortranFrontend
from recast.transform.jax.tree import TreeToJax, flattened_module
from recast.transform.numpy.tree import TreeConventions, TreeTranslation
from tests.test_flatten import DRIVER, PHYSICS, STATE, TYPES

CONVENTIONS = TreeConventions(constant_modules=frozenset({"types_mod"}))


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name, text in (
        ("types_mod", TYPES),
        ("state_mod", STATE),
        ("physics_mod", PHYSICS),
        ("driver_mod", DRIVER),
    ):
        (tmp_path / f"{name}.f90").write_text(text)
    return tmp_path


def _unit_and_facts(tree: Path):
    frontend = FortranFrontend(constant_modules=["types_mod"], flatten=True)
    unit = next(u for u in frontend.discover(tree) if u.uid == "fortran:physics_mod")
    return unit, frontend.analyze(unit, tree)


def test_the_flat_function_spells_components_state_and_outputs(tree: Path) -> None:
    unit, facts = _unit_and_facts(tree)
    anchor = TreeTranslation(CONVENTIONS).apply(unit, facts, {"root": str(tree)})
    module = ast.parse(anchor.files[Path("physics_mod_numpy.py")].decode())
    flat, interface, notes = flattened_module(
        module, facts.interface, plans_from_facts(facts, gated=False)
    )
    assert notes["refused"] == {} and notes["aborts_dropped"] == {}
    # ``do ic = 1, ncan(p)``: a static trip count over the component's axis,
    # the iterations the source meant kept by a guard.
    assert "ic: 1..inst__ncan[p - 1] + 1" in notes["static_loops"]["warm_flat"]
    fns = {n.name: n for n in flat.body if isinstance(n, ast.FunctionDef)}
    warm = ast.unparse(fns["warm_flat"])
    assert warm.startswith(
        "def warm_flat(num, filter, dt, np_, inst__gs, inst__ncan, inst__tleaf, state_mod__scale):"
    )
    # The associate aliases are gone; the components and the module variable
    # are their flat names; the written component is what comes back.
    assert "inst.tleaf" not in warm and "_state_mod.scale" not in warm
    assert (
        "inst__tleaf[p - 1, ic - 1] = "
        "inst__tleaf[p - 1, ic - 1] + dt * inst__gs[p - 1] * state_mod__scale" in warm
    )
    assert warm.rstrip().endswith("return inst__tleaf")
    entry = next(s for s in interface["subprograms"] if s["name"] == "warm_flat")
    assert entry["module_state_read"] == [] and entry["module_state_written"] == []
    assert all("UNKNOWN(TYPE" not in str(a["dtype"]) for a in entry["args"])


def test_the_port_emits_the_flat_function_as_a_kernel(tree: Path) -> None:
    unit, facts = _unit_and_facts(tree)
    candidate = TreeToJax(CONVENTIONS).apply(unit, facts, {"root": str(tree)})
    ported = candidate.files[Path("physics_mod_jax.py")].decode()
    assert "def _warm_flat_k_impl(" in ported
    assert "lax.fori_loop" in ported and ".at[p - 1, ic - 1].set(" in ported
    assert "_JAX_KERNELS = ['reset_flat', 'warm_flat']" in ported
    # The originals take the object and stay host-delegated; the flat
    # signatures reach the ported module's table for the gate.
    assert "warm = _host.warm" in ported
    assert candidate.notes["jax"]["delegated"]["warm"] == "[elig] derived type"
    assert "_SIGNATURES.update({" in ported and "'warm_flat': {'kind': 'subroutine'" in ported
    assert Path("physics_mod_numpy.py") in candidate.files
    assert Path("physics_mod_jax_runtime.py") in candidate.files


def test_a_bundled_companion_the_root_does_not_hold_fails_the_port(tree: Path) -> None:
    """The anchor says it bundled ``ghost``; no unit under the root defines
    it. A port that carried on would spell kernels that call into a module
    it never ported."""
    from recast.errors import ConfigError
    from recast.model import Candidate

    unit, facts = _unit_and_facts(tree)
    anchor = Candidate(unit=unit.uid, transform="test", notes={"tree": {"bundled": ["ghost"]}})
    with pytest.raises(ConfigError, match="companion 'ghost' of 'fortran:physics_mod'"):
        TreeToJax(CONVENTIONS)._port_companions(anchor, facts, {"root": str(tree)})


def test_early_returns_merge_into_one_exit() -> None:
    from recast.transform.jax.tree import _single_exit

    fn = ast.parse(
        "def f(x):\n"
        "    if x == 0.0:\n"
        "        root = 1.0\n"
        "        return root\n"
        "    y = x * 2.0\n"
        "    return y\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    body = _single_exit(fn.body)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])))
    assert "_ret = False" in text and "_r = root" in text and "_ret = True" in text
    assert "if not _ret:\n    y = x * 2.0" in text
    assert text.rstrip().endswith("return jnp.where(_ret, _r, y)")


def test_while_loops_become_lax_while_or_a_counted_for() -> None:
    from recast.transform.jax.tree import _WhileLoops

    forever = ast.parse(
        "while True:\n    n = n + 1\n    x = x * 0.5\n    if x < tol:\n        break\n    y = x\n"
    ).body
    lowered = _WhileLoops().visit_block(forever)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[])))
    assert "_done_1 = False" in text and "lax.while_loop(_wcond_1, _wbody_1" in text
    assert "jnp.logical_not(_done_1)" in text
    counted = ast.parse("while abs(b - a) > err and n <= nmax:\n    n = n + 1\n    a = b\n").body
    lowered = _WhileLoops({"nmax": 50}).visit_block(counted)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[])))
    assert text.startswith("for _w1 in range(0, 51):\n    if abs(b - a) > err and n <= nmax:")


def test_single_exit_refuses_a_tuple_return() -> None:
    import pytest

    from recast.transform.jax.tree import NotFlat, _single_exit

    fn = ast.parse(
        "def f(x):\n"
        "    if x == 0.0:\n"
        "        return (1.0, 2.0)\n"
        "    y = x * 2.0\n"
        "    return (y, x)\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    with pytest.raises(NotFlat, match="several outputs"):
        _single_exit(fn.body)


def test_a_subroutines_early_returns_of_its_output_names_merge_on_the_flag() -> None:
    """A subroutine's translation returns the same tuple of output names at
    every ``return`` (ELM's ``hybrid``: ``if (f0 == 0) return`` ahead of
    the loop). Nothing to select through ``jnp.where``: the guarded
    statements after the flag leave the outputs as they were."""
    from recast.transform.jax.tree import _single_exit

    fn = ast.parse(
        "def hybrid(x0, f0):\n"
        "    gs = 0.0\n"
        "    if f0 == 0.0:\n"
        "        return (x0, gs)\n"
        "    gs = x0 * 2.0\n"
        "    return (x0, gs)\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    body = _single_exit(fn.body)
    text = "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in body)
    assert "_ret = False" in text and "_ret = True" in text
    assert "_r =" not in text and "where" not in text
    assert text.rstrip().endswith("return (x0, gs)")
    assert "if not _ret:\n    gs = x0 * 2.0" in text


def test_counted_while_needs_the_counter_advanced_every_pass() -> None:
    from recast.transform.jax.tree import _WhileLoops

    # The counter steps under a branch: the trip count is not nmax + 1, so
    # the loop stays a lax.while_loop.
    branched = ast.parse(
        "while abs(b - a) > err and n <= nmax:\n    if a > 0.0:\n        n = n + 1\n    a = b\n"
    ).body
    lowered = _WhileLoops({"nmax": 50}).visit_block(branched)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[])))
    assert "lax.while_loop" in text and "range(0, 51)" not in text
    # ``n += 1`` at the top level counts too.
    aug = ast.parse("while n < nmax:\n    n += 2\n    a = b\n").body
    lowered = _WhileLoops({"nmax": 50}).visit_block(aug)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[])))
    assert text.startswith("for _w1 in range(0, 51):")


def test_a_backward_goto_region_lowers_as_a_while_loop() -> None:
    from recast.transform.jax.tree import _WhileLoops

    region = ast.parse(
        "while True:\n"
        "    try:\n"
        "        if isleap(mcyear):\n"
        "            dpm = mdayleap[mcmnth - 1]\n"
        "        else:\n"
        "            dpm = mday[mcmnth - 1]\n"
        "        if mcday > dpm:\n"
        "            mcday = mcday - dpm\n"
        "            mcmnth = mcmnth + 1\n"
        "            raise _FGoto('10')\n"
        "        break\n"
        "    except _FGoto as _g:\n"
        "        if _g.args[0] != '10':\n"
        "            raise\n"
        "        pass\n"
    ).body
    lowered = _WhileLoops().visit_block(region)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[])))
    assert "lax.while_loop" in text
    assert "_FGoto" not in text and "raise" not in text
    assert "_restart_1 = True" in text


def test_two_spellings_of_one_dynamic_bound_compare_equal() -> None:
    from recast.transform.jax.tree import _fold

    assert ast.unparse(_fold(ast.parse("(0) - (0)", mode="eval").body)) == "0"
    assert (
        ast.unparse(_fold(ast.parse("ncan[p - 1] - 0 + 1", mode="eval").body)) == "ncan[p - 1] + 1"
    )
    assert ast.unparse(_fold(ast.parse("n + 0", mode="eval").body)) == "n"


def test_a_forward_goto_region_becomes_a_skip_flag() -> None:
    from recast.transform.jax.tree import _WhileLoops

    region = ast.parse(
        "try:\n"
        "    a = 1.0\n"
        "    raise _FGoto('100')\n"
        "    b = dead_write()\n"
        "except _FGoto as _g:\n"
        "    if _g.args[0] != '100':\n"
        "        raise\n"
        "    pass\n"
    ).body
    lowered = _WhileLoops().visit_block(region)
    text = ast.unparse(ast.fix_missing_locations(ast.Module(body=lowered, type_ignores=[])))
    assert "_FGoto" not in text and "try" not in text
    assert "_skip_1 = True" in text
    assert "if not _skip_1:" in text and "dead_write" in text


SOLVE = """\
module solve_mod
  use types_mod, only: canopy_type
  implicit none
  private
  public :: Drive
contains
  subroutine Solve(x0, root, iter, inst)
    real(8), intent(in) :: x0
    real(8), intent(out) :: root
    integer, intent(out) :: iter
    type(canopy_type), intent(inout) :: inst
    iter = 0
    root = x0
    if (x0 == 0.0d0) return
    iter = 1
    root = x0 * inst%gs(1)
    inst%gs(1) = root
  end subroutine Solve

  subroutine Drive(num, filter, dt, inst)
    integer, intent(in) :: num
    integer, intent(in) :: filter(:)
    real(8), intent(in) :: dt
    type(canopy_type), intent(inout) :: inst
    integer :: f, p, it
    real(8) :: r
    do f = 1, num
       p = filter(f)
       call Solve(dt, r, it, inst)
       inst%gs(p) = inst%gs(p) + r
    end do
  end subroutine Drive
end module solve_mod
"""


def test_a_callee_with_an_out_scalar_before_its_object_is_rewritten_to_its_kernel(
    tmp_path: Path,
) -> None:
    """ELM's ``hybrid(..., gs_mol, iter, atm2lnd_vars, photosyns_vars)``: the
    anchor's call omits the OUT scalars, and counting them against the
    callee's dummies shifted the objects (``object photosyns_vars not
    passed``). The callee's own early ``return`` of its output names merges
    on the flag alone. Both are lowered now, the caller calling the
    callee's kernel."""
    (tmp_path / "types_mod.f90").write_text(TYPES)
    (tmp_path / "solve_mod.f90").write_text(SOLVE)
    frontend = FortranFrontend(constant_modules=["types_mod"], flatten=True)
    unit = next(u for u in frontend.discover(tmp_path) if u.uid == "fortran:solve_mod")
    facts = frontend.analyze(unit, tmp_path)
    candidate = TreeToJax(CONVENTIONS).apply(unit, facts, {"root": str(tmp_path)})
    ported = candidate.files[Path("solve_mod_jax.py")].decode()
    assert candidate.notes["jax"]["flat_refused"] == {}, candidate.notes["jax"]["flat_refused"]
    assert "_JAX_KERNELS = ['drive_flat', 'solve_flat']" in ported
    drive = next(
        n
        for n in ast.walk(ast.parse(ported))
        if isinstance(n, ast.FunctionDef) and n.name == "_drive_flat_k_impl"
    )
    calls = {
        ast.unparse(n.func) for n in ast.walk(drive) if isinstance(n, ast.Call)
    }
    assert "_solve_flat_k_impl" in calls, calls


def test_a_pointer_local_pointed_in_a_branch_is_seeded_with_its_first_target() -> None:
    """The anchor starts a pointer local as ``None`` and points it under
    ``phase == 'sun'`` or ``'sha'``; a ``lax.cond`` cannot carry ``None``
    against an array. The kernel starts it as the first arm's target."""
    from recast.transform.jax.tree import _seed_pointer_locals

    fn = ast.parse(
        "def f(phase, inst):\n"
        "    par_z = None\n"
        "    n = None\n"
        "    if phase == 'sun':\n"
        "        par_z = inst.parsun\n"
        "    elif phase == 'sha':\n"
        "        par_z = inst.parsha\n"
        "    n = par_z.shape[0]\n"
        "    return par_z\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    assert _seed_pointer_locals(fn.body) == ["par_z"]
    assert ast.unparse(fn.body[0]) == "par_z = inst.parsun"
    assert ast.unparse(fn.body[1]) == "n = None"  # a computed value is not a seed


def test_an_integer_literal_into_a_real_local_is_a_float_in_the_kernel() -> None:
    """``nscaler = 1`` where ``nscaler`` is real(r8): Fortran converts; the
    backend typed the bare constant int32 and a ``lax.cond`` arm disagreed
    with the float64 one beside it (ELM's Photosynthesis, the
    nu_com_leaf_physiology branch)."""
    import copy

    from recast.transform.jax.tree import _Rewrite, _Spelling, _guard_inits

    fn = ast.parse(
        "def f(flag, x):\n"
        "    nscaler = 0.0\n"
        "    k = 0\n"
        "    if flag:\n"
        "        nscaler = 1\n"
        "        k = 2\n"
        "    return nscaler * x + k\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    rewrite = _Rewrite(None, _Spelling(None, {}), {}, {})
    rewrite.inits = _guard_inits(fn)
    body = [rewrite.visit(copy.deepcopy(s)) for s in fn.body]
    text = "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in body if s is not None)
    assert "nscaler = jnp.float64(1)" in text
    assert "k = 2" in text  # an int into an int local is the backend's int32


def test_a_while_true_capped_by_a_counter_break_is_a_fixed_count_for() -> None:
    """ELM's ``hybrid``: ``while True: iter = iter + 1; ...; if
    converged: break; ...; if iter > itmax: break``. A ``lax.while_loop``
    has no reverse-mode rule; the cap makes it a fixed count of passes
    under the done flag, which lowers to a scan reverse mode can transpose."""
    from recast.transform.jax.tree import _WhileLoops

    fn = ast.parse(
        "def hybrid(x, f):\n"
        "    iter = 0\n"
        "    itmax = 40\n"
        "    while True:\n"
        "        iter = iter + 1\n"
        "        x = x - f * x\n"
        "        if abs(f * x) < 1e-6:\n"
        "            break\n"
        "        if iter > itmax:\n"
        "            x = 0.0\n"
        "            break\n"
        "    return x\n"
    ).body[0]
    assert isinstance(fn, ast.FunctionDef)
    source = ast.unparse(fn)
    lowered = _WhileLoops({"iter": 0, "itmax": 40}, frozenset({"iter", "itmax"})).visit_block(
        fn.body
    )
    text = "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in lowered)
    assert "while" not in text
    assert "_done_1 = False" in text and "for _w1 in range(0, 41):" in text
    assert "if not _done_1:" in text and "_done_1 = True" in text
    # The same loop after an early return: ``while not _ret`` (the
    # single-exit flag) is still capped, the flag joining the guard.
    again = ast.parse(source.replace("while True:", "while not _ret:")).body[0]
    assert isinstance(again, ast.FunctionDef)
    lowered = _WhileLoops({"iter": 0, "itmax": 40}, frozenset({"iter", "itmax"})).visit_block(
        again.body
    )
    text = "\n".join(ast.unparse(ast.fix_missing_locations(s)) for s in lowered)
    assert "while" not in text and "for _w1 in range(0, 41):" in text
    assert "if not _done_1 and (not _ret):" in text
