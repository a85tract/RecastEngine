"""Oracle: the reference behaviour a Candidate is judged against."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from recast.model import Facts, OracleRef, Unit


class Oracle(ABC):
    """Materialize something that exhibits the *original* behaviour.

    The three oracles in use today are deliberately different in cost and in
    strength, and the engine treats that difference as first-class:

    ``f2py-golden``   compile the untouched Fortran, call it directly  (build)
    ``dump-replay``   replay inputs/outputs captured from a real run   (cheap)
    ``pinned-run``    a full model run at fixed revision and rank count (batch)

    An Oracle never sees the Candidate. If it did, it would stop being an
    independent reference.
    """

    name: str
    cost: str = "build"
    """``cheap`` | ``build`` | ``batch``. Schedulers use this to order work."""

    @abstractmethod
    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        """Cache key. Two calls with the same key must be behaviourally identical.

        Must fold in everything that can move the reference: source hash,
        compiler and its version, optimization flags, and rank count. Getting
        this wrong silently invalidates every downstream Verdict.
        """

    @abstractmethod
    def materialize(
        self, unit: Unit, facts: Facts, workspace: Path, config: dict[str, Any]
    ) -> OracleRef:
        """Build or fetch the reference. May be expensive; will be cached."""

    def release(self, ref: OracleRef) -> None:
        """Optional teardown for oracles holding processes or scratch space."""
        return None
