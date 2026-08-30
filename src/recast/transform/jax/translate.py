"""``port.jax``: a physics kernel retargeted to JAX, in one pass.

The port recipe names one Transform and this is it, but the work has two
halves. The JAX backend transforms the *Python* the NumPy backend emitted --
it never re-parses Fortran, because the thing it is faithful to is the
validated ``<module>_numpy.py``, whose emission grammar is stable enough to
pattern-match and whose numbers have already been held to a bit-exact gate.
So the two halves run inside one ``apply``: the NumPy translation first, the
JAX lowering second, one Candidate out. That is composition inside a
Transform, which is the only place the engine has for it -- a Unit has one
Candidate, so two transform stages would have replaced rather than composed.

The Candidate carries both modules and the anchor is not incidental baggage.
Every subprogram the backend could not lower is *host-delegated*: the emitted
JAX module imports the NumPy one and calls it for those, so a port with two
kernels and forty delegations is a working module, not a stub. Which is also
why host-delegation does not go into ``Candidate.deferred``. A deferred site
raises at run time and is the agent queue's; a delegated one runs, on the
slower path, and putting it in ``deferred`` would make the differential gate
skip exactly the subprograms most likely to be right.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from recast.errors import ConfigError
from recast.model import Candidate, Facts, Unit
from recast.plugins.transform import Transform

__all__ = ["KernelToJax", "factory"]


class KernelToJax(Transform):
    """NumPy first, JAX second, one Candidate."""

    name = "recast.port.kernel-to-jax"
    requires = ("interface", "constants", "effects")
    """The NumPy translation's requirements, because it runs inside this one."""

    deterministic = True
    """No model anywhere in the path: both halves are rule-driven AST work."""

    def __init__(self) -> None:
        from recast.transform.numpy.translate import NumpyTranslation

        self._anchor = NumpyTranslation()

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        # Whatever the anchor can translate, this can attempt to lower. A unit
        # the NumPy backend refuses has no anchor to be faithful to.
        return self._anchor.applicable(unit, facts)

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        from recast.transform.jax.backend import HEADER, build_module, emit_runtime

        anchor = self._anchor.apply(unit, facts, config)
        module = facts.interface["module"]
        constants_stem = config.get("constants_stem", f"{module}_constants")
        runtime_stem = config.get("jax_runtime_stem", f"{module}_jax_runtime")

        source = anchor.files[Path(f"{module}_numpy.py")].decode()
        tree = ast.parse(source)
        pieces, jitted, delegated = build_module(facts.interface, tree)
        emitted = (
            HEADER.format(module=module, constants=constants_stem, runtime=runtime_stem)
            + _signatures_of(tree)
            + "\n\n\n".join(pieces)
        )

        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={
                **anchor.files,
                Path(f"{runtime_stem}.py"): emit_runtime().encode(),
                Path(f"{module}_jax.py"): (emitted + "\n").encode(),
            },
            # The anchor's, and only the anchor's: a site the NumPy backend
            # deferred has no translation at all, while one this backend
            # delegated has a working one that is merely not accelerated.
            deferred=list(anchor.deferred),
            notes={
                **anchor.notes,
                "jax": {
                    "anchor": f"{module}_numpy.py",
                    "kernels": sorted(jitted),
                    "delegated": dict(sorted(delegated.items())),
                    "runtime": f"{runtime_stem}.py",
                },
            },
        )


def _signatures_of(anchor: ast.Module) -> str:
    """Carry the anchor's ``_SIGNATURES`` table into the ported module.

    Added by this Transform rather than by the backend, which is why
    ``tools/jax_diff.py`` stays green: what it holds to the byte is what
    ``build_module`` emits, and this is not that. The script the backend came
    from had no need for the table because its comparisons were hand-written
    tests that already knew the signatures. The engine's differential gate
    generates its inputs from the table instead, so a module without one
    cannot be judged -- and a ported artifact that cannot describe its own
    interface is worse than inconvenient, it is unverifiable.

    Lifted from the anchor rather than re-rendered, so the two modules cannot
    drift into disagreeing about the interface they share.
    """
    table = None
    updates: list[str] = []
    for node in anchor.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "_SIGNATURES" for target in node.targets
        ):
            table = ast.unparse(node)
        elif (
            # ``_SIGNATURES.update({...})``: the flat adapters' entries, added
            # after the table by ``recast.transform.numpy.flat``.
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "update"
            and isinstance(node.value.func.value, ast.Name)
            and node.value.func.value.id == "_SIGNATURES"
        ):
            updates.append(ast.unparse(node))
    if table is not None:
        return "\n".join([table, *updates]) + "\n\n\n"
    raise ConfigError(
        "the NumPy anchor carries no _SIGNATURES table, so the ported module "
        "would have nothing for a differential gate to generate inputs from"
    )


def factory(**_config: Any) -> KernelToJax:
    """The entry-point hook. Configuration arrives per-``apply``, like the
    anchor's, because the tables are per-module and an instance is per-run."""
    return KernelToJax()
