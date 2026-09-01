"""Recipe: a named composition of the other plugins.

A Recipe is how a workload gets a name. It does not contain logic -- it declares
which plugin fills each slot, and the engine drives the spine. That is the
mechanism by which "RecastEngine does translation" became "RecastEngine does
translation, refactoring, GPU porting, and security testing" without the core
changing shape.

The four shipped recipes and where they came from:

    translate      Fortran -> NumPy/Numba/CUDA, gated on f2py golden
    refactor-todo  carve a Python control plane into a Fortran monolith,
                   gated on a pinned full-model run
    port           retarget a kernel to an accelerator, gated on captured dumps
    audit          cyber gate only: scan -> adjudicate -> Sec-Track
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Stage:
    """One slot in a Recipe, bound to a registered plugin by name."""

    kind: str
    """``frontend`` | ``transform`` | ``oracle`` | ``verifier`` | ``scanner``
    | ``adjudicator`` | ``executor`` | ``store``.

    Most kinds are steps the engine walks in order. An ``executor`` stage is not
    a step: it declares the executor the run's Oracles, Verifiers, Scanners and
    Adjudicators receive as an argument, and so it comes first. A recipe that
    declares any of those four has to declare one.
    """

    plugin: str
    """Name of a registered plugin. A recipe that has an opinion about *which*
    one reads it from config rather than hardcoding it -- ``pbs-<site>`` is
    site knowledge, and the four shipped recipes have to stay publishable."""

    config: dict[str, Any] = field(default_factory=dict)

    optional: bool = False
    """If True, a missing plugin downgrades the run instead of failing it.

    Use for enrichment (extra scanners, extra backends). Never for a gate: an
    optional Verifier is not a gate, it is a suggestion.
    """

    gate: bool = False
    """If True, a failing Verdict stops this Unit from proceeding.

    On a ``scanner`` stage it means something slightly different: the stage
    fails the Unit when it finds anything at or above the scanner's
    ``blocks_on``, but does not stop the walk, so the other scanners still
    run and the operator gets the whole list. That is ``hpc-devsecops``'s
    gate -- a check that found something is the verdict, with no adjudication
    in between -- and the shipped ``audit`` recipe is built on it.

    Stops it -- there is no retry. A Verdict never flows back into a Transform,
    and no stage re-runs because a later one failed. For a ``deterministic``
    Transform a re-run is a no-op by construction; for an agentic one, feeding
    the gate's own numbers back to the thing being gated turns the Oracle into a
    fitness function and overfits the Candidate to the cases the gate happens to
    sample. Iteration belongs in ``Candidate.deferred``, in a later stage, or
    out of band in the rules -- never in a loop around the gate.
    """


class Recipe(ABC):
    """A workload definition."""

    name: str
    summary: str = ""

    @abstractmethod
    def stages(self, config: dict[str, Any]) -> list[Stage]:
        """Return the ordered stages for this run.

        May branch on config -- ``port`` picks a JAX or a Numba transform from
        the requested target -- but must return the same stages for the same
        config, so a run can be replayed from its Evidence.
        """

    def validate(self, config: dict[str, Any]) -> list[str]:
        """Return human-readable problems with this config. Empty means usable.

        Checked before any work starts, so a missing oracle or an unreachable
        scheduler is reported in a second rather than three hours in.
        """
        return []
