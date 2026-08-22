"""Scanner: the cyber half of CC-Test.

Scanners answer a different question from Verifiers. A Verifier asks "does this
port compute what the original computed"; a Scanner asks "is this code, ported
or not, defective in a way an attacker can use". They share the Unit and Facts
that a Frontend produced, and nothing else.

The cyber gate already in production covers four families, and all four are
domain-independent -- they run against any git repository, which is exactly why
they belong in the open-source engine:

    secret         credentials committed to history
    composition    SBOM -> CVE -> VEX triage
    audit          LLM-driven source audit (memory safety, injection)
    dynamic        sanitizer builds (ASan) and fuzz harnesses

Findings default to ``Access.EMBARGOED``. Lowering that is a coordinated
disclosure decision made by a human, never by a scanner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recast.model import Facts, Finding, Unit


class Scanner(ABC):
    """Find defects. Never fix them, never judge equivalence."""

    name: str

    family: str = "audit"
    """``secret`` | ``composition`` | ``audit`` | ``dynamic``."""

    needs_build: bool = False
    """True for sanitizer and fuzz scanners, which need a compiled artifact."""

    @abstractmethod
    def scan(
        self, unit: Unit, facts: Facts, workspace: Path, config: dict[str, Any]
    ) -> Iterable[Finding]:
        """Yield findings for one Unit.

        Yield ``Disclosure.PLAUSIBLE`` freely -- precision is the adjudicator's
        job, not the scanner's. Suppressing an uncertain finding here loses it
        permanently; emitting it costs one adjudication pass.

        Raise ``ScannerUnavailable`` when the scan could not run: the tool is
        not on PATH, the API refused, the build this scanner needs is not
        there. **Do not return an empty iterable for that.** An empty iterable
        means "I ran and found nothing", the run is entitled to report a clean
        scan on the strength of it, and a security gate that says clean when it
        means untested is worse than one that says nothing. The runner marks
        such a stage ``incomplete``, which is neither a pass nor a failure and
        does not become either by omission.
        """


class Adjudicator(ABC):
    """Promote or kill a Finding after independent verification.

    Sec-Track's discovery loops run scan -> adversarially verify -> dedupe ->
    reclassify, and the verify step is where most of the value is: loop-2 turned
    108 raw findings into 43 confirmed and 59 downgraded. This ABC is that step.
    """

    name: str

    @abstractmethod
    def adjudicate(self, finding: Finding, workspace: Path, config: dict[str, Any]) -> Finding:
        """Return the finding with ``disclosure``, ``severity``, and
        ``exploitability`` revised, and the reasoning recorded in ``evidence``.

        Must be prepared to return ``Disclosure.REFUTED``. An adjudicator that
        never refutes anything is not adding information.

        Raise ``ScannerUnavailable`` on the same terms a Scanner does, and it
        matters more here: an adjudicator is usually the recipe's gate, so
        "the gate could not run" is the one incompleteness that must never read
        as a pass. Returning the finding unchanged would say it was examined
        and left ``PLAUSIBLE``, which is a claim about the finding rather than
        about the adjudicator.
        """
