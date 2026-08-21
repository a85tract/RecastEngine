"""Transform: the unit of modernization work.

A Transform is *any* mechanical change to a Unit -- language translation,
architectural refactoring, accelerator porting, or a targeted repair. All three
of SciRecast's current workloads are Transforms over the same spine:

    recast.translate.fortran-to-numpy   rule-driven Fortran -> NumPy
    recast.refactor.carve-control-plane codegen adapters + ordered source patches
    recast.port.kernel-to-jax           kernel retarget to an accelerator

A Transform must never decide whether its own output is correct. That is the
Verifier's job, and keeping the two apart is what makes the gate meaningful.

A Transform may be rule-driven (``deterministic = True``) or consult an
``AgentProvider`` (``deterministic = False``). Both produce only a ``Candidate``,
never a ``Verdict`` -- an LLM's confidence in its own output is worth nothing at
the gate.

The two are placements of the same slot, and they compose *inside* one
Transform rather than across two stages: the rules run, whatever they refused
is attempted through the ``AgentProvider``, and one Candidate comes out. A
recipe declares exactly one transform stage and the runner refuses more,
because a Unit has one Candidate -- a second stage's would replace the first's
along with the ``deferred`` list that was the whole point of running them in
that order. See ``docs/architecture.md`` for why the agentic placement makes
the wall between Transform and Verifier more load-bearing, not less.
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
    """True for a rule-driven Transform; False if ``apply`` consults an
    ``AgentProvider``.

    A ``False`` Transform cannot satisfy ``Candidate.digest()`` equality across
    runs -- an LLM does not reproduce its bytes, even at temperature zero across
    provider versions. Its reproducibility contract is therefore by *provenance*,
    not by digest: it records the model, prompt digest, sampling parameters, and
    which sites it filled in ``Candidate.notes``, so its Evidence replays to a
    *valid* artifact rather than to the same bytes. The plan stays reproducible
    -- the stage list does not change -- only the artifact does.

    An agentic Transform is only safe under a gating Verifier that awards
    ``BIT_EXACT`` or an explicit tolerance: it emits plausible output for the
    cases the rules refuse, and only execution against the Oracle catches a
    plausible wrong answer. ``conformance/`` makes this a rule.
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
