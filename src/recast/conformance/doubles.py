"""Stand-ins the conformance checks drive the engine with.

Some rules are about the *runner* rather than about any plugin: that a failed
gate does not drive a retry, that a Verdict never flows back into a Transform,
that a refused job surfaces rather than being routed around. Checking those
needs plugins whose behaviour is known in advance -- a transform that counts
its calls, a gate that always fails, an executor that refuses everything.

They ship here rather than inside the suite because an out-of-tree author needs
the same ones: substituting a refusing executor is how you find out whether your
Verifier reaches for ``subprocess`` when the executor says no.

Each double registers under a ``conformance.`` name. Nothing here is exported
as an entry point -- these are constructed, or registered into a private
``Registry``, by whoever is doing the checking.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from recast.model import Candidate, Confidence, Evidence, Facts, OracleRef, Unit, Verdict
from recast.plugins.executor import Executor, Job, JobResult
from recast.plugins.frontend import Frontend
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.store import EvidenceStore
from recast.plugins.transform import Transform
from recast.plugins.verifier import Verifier

__all__ = [
    "CountingTransform",
    "FailingVerifier",
    "GateFailsRecipe",
    "RecordingEvidenceStore",
    "RefusingExecutor",
    "StubFrontend",
    "TwoFrontendsRecipe",
]


class StubFrontend(Frontend):
    """Two units, no source tree, no parsing. Deterministic by construction."""

    name = "conformance.frontend"
    uids: ClassVar[tuple[str, ...]] = ("conformance:unit-a", "conformance:unit-b")

    def __init__(self, **_config: Any) -> None:
        pass

    @classmethod
    def claiming(cls, name: str, *uids: str) -> type[StubFrontend]:
        """Another stub, under its own name, finding its own units.

        For the checks about a recipe declaring more than one frontend, where
        what matters is that the two are different plugins finding different
        things -- a second language in the same tree.
        """
        return type(name, (cls,), {"name": name, "uids": tuple(uids)})

    def discover(self, root: Path) -> Iterable[Unit]:
        return [Unit(uid=uid, kind="module") for uid in self.uids]

    def analyze(self, unit: Unit, root: Path) -> Facts:
        return Facts(unit=unit.uid, interface={"module": unit.uid})


class CountingTransform(Transform):
    """Records how many times it was asked to produce a Candidate, per unit.

    The count is on the class because the runner constructs a fresh Transform
    for each stage walk -- which is itself the reason a retry would be invisible
    to an instance counter.
    """

    name = "conformance.counting"
    deterministic = True
    calls: ClassVar[dict[str, int]] = {}

    @classmethod
    def reset(cls) -> None:
        cls.calls.clear()

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return True

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        type(self).calls[unit.uid] = type(self).calls.get(unit.uid, 0) + 1
        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={Path("candidate.txt"): unit.uid.encode()},
        )


class FailingVerifier(Verifier):
    """A gate that always fails, and reports numbers while doing it.

    ``provides`` is the strongest confidence it could ever award, not what it
    awards here: a gate that fails is still a hard gate, and a recipe built
    from this double has to satisfy the same rule about agentic transforms as
    a real one.
    """

    name = "conformance.failing"
    provides = Confidence.BIT_EXACT

    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        return Verdict(
            unit=unit.uid,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=Confidence.FAILED,
            metrics={"compared": 0, "reason": "conformance double"},
            detail="this gate always fails, on purpose",
        )


class RefusingExecutor(Executor):
    """Refuses every job. Hand this to a plugin to find out what it does next.

    A Verifier that returns ``FAILED`` has honoured the contract. One that
    returns anything else, or that succeeds, went around the executor.
    """

    name = "conformance.refusing"
    supports_batch = False

    def submit(self, job: Job) -> str:
        raise RuntimeError(
            f"conformance: this executor refuses every job; {job.label or 'unlabelled'} was not run"
        )

    def wait(self, handle: str, timeout_s: float | None = None) -> JobResult:
        raise RuntimeError(
            f"conformance: nothing was ever submitted, so {handle!r} cannot be waited on"
        )


@dataclass
class RecordingEvidenceStore(EvidenceStore):
    """Keeps what it was handed, in memory, on the class.

    Class-level for the same reason as ``CountingTransform``: the runner builds
    a store per stage walk, and what the checks ask about is the whole run.
    """

    root: Path
    name: str = "conformance.recording"
    written: ClassVar[list[Evidence]] = []

    @classmethod
    def reset(cls) -> None:
        cls.written.clear()

    def put(self, evidence: Evidence) -> str:
        type(self).written.append(evidence)
        return f"conformance:{len(type(self).written)}"

    def get(self, uri: str) -> Evidence:
        index = int(uri.partition(":")[2])
        return type(self).written[index - 1]

    def query(self, **selectors: Any) -> Iterable[Evidence]:
        return list(type(self).written)


class TwoFrontendsRecipe(Recipe):
    """A run whose units come from two independent frontends.

    Nothing here fails; the question is only whether both frontends' units are
    walked, and whether each is analyzed by the one that found it.
    """

    name = "conformance.two-frontends"
    summary = "A recipe with two frontends. Used to check the runner, not a plugin."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", config.get("executor", "local")),
            *[Stage("frontend", name) for name in config["frontends"]],
            Stage("transform", CountingTransform.name),
        ]


class GateFailsRecipe(Recipe):
    """A whole run whose gate fails: transform, hard gate, store.

    Ordinary in every respect except the outcome, which is what makes it able
    to answer whether a failure causes anything to run twice.
    """

    name = "conformance.gate-fails"
    summary = "A recipe whose gate always fails. Used to check the runner, not a plugin."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", StubFrontend.name)),
            Stage("transform", CountingTransform.name),
            Stage("verifier", FailingVerifier.name, gate=True),
            Stage("store", RecordingEvidenceStore.name),
        ]
