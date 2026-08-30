"""The untouched Python/NumPy module as an independent executable oracle."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from recast.errors import ConfigError, OracleUnavailable
from recast.model import Facts, OracleRef, Unit
from recast.plugins.executor import Executor
from recast.plugins.oracle import Oracle

__all__ = ["PythonSourceOracle", "factory"]


class PythonSourceOracle(Oracle):
    """Identify untouched source for an isolated verifier-side execution."""

    name = "python-source"
    cost = "cheap"

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        del config
        digest = str(facts.provenance.get("digest", ""))
        source = str(facts.provenance.get("source", ""))
        if not digest or not source:
            raise ConfigError(f"{unit.uid} has no Python source provenance")
        value = hashlib.sha256(f"python-source-v2\0{source}\0{digest}".encode()).hexdigest()
        return f"python-source:{value[:24]}"

    def materialize(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> OracleRef:
        del workspace, executor
        root = Path(config.get("root", ".")).resolve()
        relative = Path(str(facts.provenance.get("source", "")))
        source = (root / relative).resolve()
        try:
            source.relative_to(root)
        except ValueError as error:
            raise OracleUnavailable(
                f"python oracle source escapes project root: {relative}"
            ) from error
        if not source.is_file():
            raise OracleUnavailable(f"python oracle source is missing: {relative.as_posix()}")
        exports = tuple(str(name) for name in facts.interface.get("exports", ()))
        if not exports or any(not name.isidentifier() for name in exports):
            raise OracleUnavailable("python oracle has no valid callable exports")
        module_name = str(
            facts.interface.get("module") or relative.with_suffix("").as_posix()
        ).replace("/", ".")
        if any(not part.isidentifier() for part in module_name.split(".")):
            raise OracleUnavailable(f"python oracle module name is invalid: {module_name!r}")
        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=self.key(unit, facts, config),
            # Keep this handle inert.  Importing project code here would let an
            # oracle poison sys.modules (including JAX/Numba) before the
            # verifier establishes an independent backend identity.
            handle={
                "root": str(root),
                "source": relative.as_posix(),
                "module_name": module_name,
                "functions": exports,
            },
            cost=self.cost,
        )


def factory(**_config: Any) -> PythonSourceOracle:
    return PythonSourceOracle()
