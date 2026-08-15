"""Transform: the unit of modernization work.

A Transform is *any* mechanical change to a Unit -- language translation,
architectural refactoring, accelerator porting, or a targeted repair. All three
of SciRecast's current workloads are Transforms over the same spine:

    recast.translate.fortran-to-numpy   rule-driven Fortran -> NumPy
    recast.refactor.carve-control-plane codegen adapters + ordered source patches
    recast.port.kernel-to-jax           kernel retarget to an accelerator

A Transform must never decide whether its own output is correct. That is the
Verifier's job, and keeping the two apart is what makes the gate meaningful.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from recast.model import Candidate, Facts, Unit


class Transform(ABC):
    """Produce a Candidate from a Unit plus its Facts."""

    name: str
    """Dotted, namespaced, stable. Appears in every Evidence record."""

    requires: tuple[str, ...] = ()
    """``Facts`` fields that must be non-empty, e.g. ``("interface", "effects")``.

    The engine checks these before calling ``apply`` and reports a missing
    prerequisite as a configuration error, not a transform failure.
    """

    deterministic: bool = True
    """False if this Transform consults an ``AgentProvider``.

    Non-deterministic transforms must still be reproducible: record the model,
    prompt digest, and sampling parameters in ``Candidate.notes``.
    """

    @abstractmethod
    def applicable(self, unit: Unit, facts: Facts) -> bool:
        """Cheap pre-filter. Return False rather than raising on a bad match."""

    @abstractmethod
    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        """Do the work.

        Anything that cannot be handled mechanically goes into
        ``Candidate.deferred`` rather than raising. A partial Candidate with a
        populated deferred list is a normal, useful result -- it is what the
        agent layer consumes next.
        """
