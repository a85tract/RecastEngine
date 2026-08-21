"""``numpy-anchor``: the validated NumPy translation, as the reference to port against.

A JAX port is not compared against Fortran directly, and the reason is not
convenience. XLA cannot reproduce libm bit for bit, so a Fortran comparison
could only ever be a tolerance; but the NumPy translation of the same unit
*can* be bit-exact against Fortran, and is -- that is what the ``translate``
recipe's ``f2py-golden`` and ``differential.bitexact`` establish. Anchoring the
port on it turns one loose comparison into a chain: NumPy is bit-exact against
the Fortran, JAX is ULP-bounded against the NumPy.

**The chain is only as good as its first link, and this oracle cannot check
that link.** It materializes the anchor; whether that anchor ever passed a
bit-exact gate is a separate run's evidence, and the honest thing is to say so
rather than imply it. So the reference records what it is -- the transform that
produced it and the digest of what it produced -- and an operator who has the
translate evidence points at it with ``config["anchor_evidence"]``, which is
carried into the Verdict. A port whose anchor was never gated is a port
verified against an unverified thing, and the record should let a reader see
that at a glance.

It never sees the Candidate. It re-derives the anchor from the same Unit and
Facts the Transform was given, through the registered NumPy transform, which is
what keeps it an independent reference rather than a mirror of the artifact
under judgement.

Cheap: no compiler, no build, nothing submitted to the executor. That is what
lets the cache key be deliberately over-inclusive -- see ``key``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

from recast.errors import ConfigError, OracleUnavailable
from recast.model import Facts, OracleRef, Unit
from recast.plugins.executor import Executor
from recast.plugins.oracle import Oracle
from recast.registry import REGISTRY

__all__ = ["NumpyAnchorOracle", "factory"]

ANCHOR_TRANSFORM = "translate.numpy"

_NOT_THE_REFERENCE = frozenset({"root", "workspace", "store_root"})
"""Config keys that locate things rather than describe them.

A path is not part of a reference's identity -- the source's content digest
already covers what the path points at -- and folding one in would give two
machines different keys for the same reference, and a fresh temporary
directory a different key on every run.
"""


class NumpyAnchorOracle(Oracle):
    """The NumPy translation of the same unit, importable and callable."""

    name = "numpy-anchor"
    cost = "cheap"

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        """Fold the source and, deliberately, the whole rest of the config.

        Naming the individual keys the NumPy transform reads would duplicate
        another plugin's config surface here, and get out of step with it the
        first time that surface grows -- silently, in the direction that
        matters. Over-inclusive keying costs a rebuild of something that takes
        milliseconds; under-inclusive keying serves a stale reference and every
        Verdict behind it is a comparison against the wrong thing.
        """
        digest = hashlib.sha256()
        digest.update(ANCHOR_TRANSFORM.encode())
        digest.update(str(facts.provenance.get("digest")).encode())
        digest.update(_stable(config).encode())
        module = facts.interface.get("module", unit.uid)
        return f"numpy-anchor:{module}:{digest.hexdigest()[:16]}"

    def materialize(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> OracleRef:
        transform = REGISTRY.get("transform", ANCHOR_TRANSFORM)()
        if not transform.applicable(unit, facts):
            raise OracleUnavailable(
                f"{ANCHOR_TRANSFORM!r} cannot translate {unit.uid}, so there is no "
                "validated NumPy module to anchor a port on"
            )
        anchor = transform.apply(unit, facts, config)

        module_name = facts.interface.get("module")
        if not module_name:
            raise ConfigError(f"{unit.uid} has no module name to build an anchor for")
        staged = workspace / f"anchor-{self.key(unit, facts, config).rsplit(':', 1)[-1]}"
        staged.mkdir(parents=True, exist_ok=True)
        for path, content in anchor.files.items():
            target = staged / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)

        module = _import_from(staged, f"{module_name}_numpy")
        subprograms = [s["name"] for s in facts.interface["subprograms"] if s.get("public", True)]
        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=self.key(unit, facts, config),
            handle={
                "module": module,
                # Identity, not ``w_`` wrappers: both sides of this comparison
                # were emitted by the same backend and spell a name the same
                # way. The f2py oracle needs wrappers because Fortran does not
                # expose a module procedure to C without one.
                "wrappers": {name: name for name in subprograms},
                # And the arguments are spelled the emitted way rather than
                # lowercased, for the same reason.
                "arg_naming": "pysafe",
                # And it returns every out-intent argument in declaration
                # order, the way the emitter writes them, rather than f2py's
                # split between returned ``intent(out)`` and mutated ``inout``.
                "return_convention": "emitted",
                # NumPy is CPU-only, so this is a fact rather than a guess, and
                # recording it is what makes a same-device comparison legible
                # as one -- next to a candidate that says ``gpu:0``, it is the
                # difference between a ULP bound and a cross-device ULP bound.
                "device": "cpu",
                "anchor_digest": anchor.digest(),
                "anchor_transform": anchor.transform,
                "anchor_deferred": list(anchor.deferred),
                # What, if anything, says this anchor was ever gated. Absent is
                # a legitimate answer and an informative one.
                "anchor_evidence": config.get("anchor_evidence"),
            },
            cost=self.cost,
        )


def _stable(config: dict[str, Any]) -> str:
    """A canonical string for the part of a config that describes the reference.

    Values that will not serialize -- a behaviour hook, say -- are folded in by
    key rather than by value: their identity is an address, which would change
    the key on every run and make the cache useless, while their presence
    genuinely does change the reference.
    """
    described = {k: v for k, v in sorted(config.items()) if k not in _NOT_THE_REFERENCE}
    parts = []
    for name, value in described.items():
        try:
            parts.append(f"{name}={json.dumps(value, sort_keys=True)}")
        except TypeError:
            parts.append(f"{name}=<present>")
    return ";".join(parts)


def _import_from(directory: Path, module_name: str) -> Any:
    """Import one staged module, with its siblings importable beside it."""
    path = directory / f"{module_name}.py"
    if not path.is_file():
        raise OracleUnavailable(f"the anchor produced no {path.name}")
    sys.path.insert(0, str(directory))
    try:
        for cached in list(sys.modules):
            if cached == module_name or cached.endswith("_constants"):
                del sys.modules[cached]
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:  # pragma: no cover - defensive
            raise OracleUnavailable(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    except OracleUnavailable:
        raise
    except Exception as error:
        raise OracleUnavailable(f"the anchor does not import: {error}") from error
    finally:
        sys.path.remove(str(directory))
    return module


def factory(**_config: Any) -> NumpyAnchorOracle:
    return NumpyAnchorOracle()
