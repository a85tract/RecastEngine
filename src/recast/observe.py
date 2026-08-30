"""Read-only events emitted while a recipe is running.

The run engine deliberately performs one immutable attempt.  An outer control
plane may want to display that attempt, persist it, or decide what to run next,
but none of those concerns belongs in a Transform or Verifier.  ``RunObserver``
is the narrow boundary between the two: the engine reports facts after making
its decisions and never asks the observer for one.

Events contain identifiers and outcomes, not source, configuration, Facts,
Candidate notes, or Finding bodies.  They nevertheless default to embargoed:
an exception or stage reason can reveal something about code which has not
completed disclosure review.  A consumer may publish a projection only after
applying its own disclosure policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol

from recast.model import Access

__all__ = ["RunEvent", "RunEventAction", "RunEventEntity", "RunObserver"]


class RunEventEntity(StrEnum):
    """The thing whose lifecycle changed."""

    RUN = "run"
    UNIT = "unit"
    STAGE = "stage"
    CANDIDATE = "candidate"
    VERDICT = "verdict"
    EVIDENCE = "evidence"


class RunEventAction(StrEnum):
    """Lifecycle edge reported by the engine."""

    STARTED = "started"
    FINISHED = "finished"


@dataclass(frozen=True, slots=True)
class RunEvent:
    """One immutable, JSON-safe lifecycle event.

    ``sequence`` is monotonic within ``run_id`` and is the authoritative order;
    ``emitted_at`` is presentation metadata and must not be used for ordering.
    A stage is identified by its position in the Recipe as well as its kind and
    plugin, so declaring the same plugin twice is still unambiguous.  Candidate,
    Verdict, and Evidence events use that stage position plus ``unit_id`` to
    correlate their start and finish edges; a produced Candidate's content
    address and a stored Evidence URI are only known on their finish edges.

    ``status`` is ``running`` on start.  Finish statuses use the runner's own
    vocabulary (``ok``, ``failed``, ``skipped``, ``incomplete``), plus
    ``passed`` for a completed run/unit and ``aborted`` when an exception left
    no normal outcome.  ``reason_code`` is stable and intended for automation;
    ``reason`` is an operator-facing explanation.
    """

    schema: int = field(default=1, init=False)
    run_id: str
    sequence: int
    emitted_at: str
    entity: RunEventEntity
    action: RunEventAction
    recipe: str
    status: str
    reason_code: str
    reason: str = ""
    unit_id: str | None = None
    stage_index: int | None = None
    stage_kind: str | None = None
    stage_plugin: str | None = None
    candidate_digest: str | None = None
    evidence_uri: str | None = None
    evidence_index: int | None = None
    verifier: str | None = None
    confidence: str | None = None
    access: Access = Access.EMBARGOED

    @property
    def event_id(self) -> str:
        """Stable identity for idempotent observer storage."""
        return f"{self.run_id}:{self.sequence}"

    def to_record(self) -> dict[str, Any]:
        """Return a JSON-serializable record without empty optional fields."""
        record: dict[str, Any] = {
            "schema": self.schema,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "sequence": self.sequence,
            "emitted_at": self.emitted_at,
            "entity": self.entity.value,
            "action": self.action.value,
            "recipe": self.recipe,
            "status": self.status,
            "reason_code": self.reason_code,
            "reason": self.reason,
            "access": self.access.value,
        }
        optional = {
            "unit_id": self.unit_id,
            "stage_index": self.stage_index,
            "stage_kind": self.stage_kind,
            "stage_plugin": self.stage_plugin,
            "candidate_digest": self.candidate_digest,
            "evidence_uri": self.evidence_uri,
            "evidence_index": self.evidence_index,
            "verifier": self.verifier,
            "confidence": self.confidence,
        }
        record.update({name: value for name, value in optional.items() if value is not None})
        return record


class RunObserver(Protocol):
    """A synchronous sink for run events.

    Delivery is ordered and at-most-once from the engine.  Returning normally
    acknowledges an event.  Raising aborts the run rather than allowing an
    audit trail with a silent hole; durable retry and replay belong in the
    caller's control plane.
    """

    def observe(self, event: RunEvent) -> None:
        """Accept one event before the engine proceeds to the next action."""
        ...
