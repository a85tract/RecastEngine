"""Verifier: the gate. Nothing ships without one."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from recast.model import Candidate, Confidence, OracleRef, Unit, Verdict


class Verifier(ABC):
    """Compare a Candidate against an OracleRef and state how sure you are.

    The Verifier owns *both* sides of the comparison, because in practice each
    strategy knows how to invoke both: a differential driver imports the
    translated module and calls the f2py truth module; a dump comparator feeds
    captured arrays to both; the full-model gate submits two runs and diffs the
    history files.

    A Verifier must fail closed. If it cannot run the comparison -- build
    failed, scheduler rejected the job, oracle unavailable -- the Verdict is
    ``FAILED``, never ``SAMPLED``.
    """

    name: str

    provides: Confidence = Confidence.SAMPLED
    """The strongest confidence this Verifier can ever award."""

    @abstractmethod
    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        config: dict[str, Any],
    ) -> Verdict:
        """Run the comparison.

        Populate ``Verdict.metrics`` with the numbers, not just the conclusion.
        ``{"max_ulp": 0, "bit_exact": 512, "total_points": 512}`` is reviewable;
        ``{"ok": true}`` is not.
        """


class StaticVerifier(Verifier):
    """A Verifier that needs no Oracle -- lint, read/write-set cross-check, CPG audit.

    These cannot award more than ``SAMPLED`` on their own, but they run in
    milliseconds and catch whole classes of transform bugs before anything is
    compiled.
    """

    provides = Confidence.SAMPLED

    @abstractmethod
    def check(
        self, unit: Unit, candidate: Candidate, workspace: Path, config: dict[str, Any]
    ) -> Verdict: ...

    def verify(
        self,
        unit: Unit,
        candidate: Candidate,
        oracle: OracleRef,
        workspace: Path,
        config: dict[str, Any],
    ) -> Verdict:
        return self.check(unit, candidate, workspace, config)
