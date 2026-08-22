"""Adjudicator: promote or kill a Finding after independent verification.

Its own module, beside its own kind. It shipped inside ``scanner.py`` for a
while, and every author who read ``registry.KINDS`` -- which lists
``adjudicator`` -- or the entry-point group ``recast.adjudicators`` then tried
the obvious import and got an ``ImportError`` for their trouble. A kind whose
ABC lives under another kind's name is guessable, but only after the guess that
should have worked fails.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from recast.model import Finding
from recast.plugins.executor import Executor


class Adjudicator(ABC):
    """Promote or kill a Finding after independent verification.

    Sec-Track's discovery loops run scan -> adversarially verify -> dedupe ->
    reclassify, and the verify step is where most of the value is: loop-2 turned
    108 raw findings into 43 confirmed and 59 downgraded. This ABC is that step.
    """

    name: str

    tool: str | None = None
    """As ``Scanner.tool``: the binary this runs, so ``recast plan`` can ask."""

    @abstractmethod
    def adjudicate(
        self, finding: Finding, workspace: Path, executor: Executor, config: dict[str, Any]
    ) -> Finding:
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

        Reproducing a finding means running something; do it through
        ``executor``.
        """
