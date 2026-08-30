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
    assert notes == {"refused": {}, "aborts_dropped": {}, "companions": []}
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
