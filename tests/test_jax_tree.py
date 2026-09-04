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
