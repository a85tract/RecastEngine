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

What a scanner is *of*
----------------------

Two of the four families above are not about a Unit at all. gitleaks' value is
in history -- a credential deleted in a later commit is still in the pack, and
no file in a working tree shows it. syft and grype describe the dependency
state of a whole repository, and that state is one fact, not one per module.
Handing such a scanner a Unit and calling it once per Unit is N identical
scans, the same findings attributed N times, and a runtime that grows with a
number unrelated to the work; handing it the Unit's files instead is a
materially weaker check than the tool exists to perform. The first in-tree
scanner did the second of those for a day, and that day is why ``subject``
exists.

A scanner declares ``subject``. ``"unit"`` is walked once per Unit with that
Unit's Facts. ``"repository"`` is walked once per run, against a Unit the
runner synthesizes for the tree -- ``kind="repository"``, ``sources=()``,
empty Facts -- so that adjudication, storage and the run's status work the
same way for both and a repository finding is a ``Finding`` like any other.

Where a scanner runs
--------------------

It receives an ``Executor``, the same one Oracles and Verifiers receive, and
for the same reason: nothing that leaves the process may use ``subprocess``
directly. The ``secret`` scanner ran gitleaks through ``subprocess`` for one
day because the contract gave it nothing else, and that was the contract's
defect rather than the scanner's. A recipe that declares a scanner declares an
executor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any, Literal

from recast.model import Facts, Finding, Unit
from recast.plugins.executor import Executor

Subject = Literal["unit", "repository"]


class Scanner(ABC):
    """Find defects. Never fix them, never judge equivalence."""

    name: str

    family: str = "audit"
    """``secret`` | ``composition`` | ``audit`` | ``dynamic``."""

    subject: Subject = "unit"
    """What one call examines. See the module docstring.

    ``"unit"``: once per Unit, with its Facts. ``"repository"``: once per run,
    with a synthesized Unit for the whole tree and empty Facts. A scanner of
    history is a repository scanner -- history is a property of the tree, not
    of any file in it.
    """

    tool: str | tuple[str, ...] | None = None
    """The external binary this scanner runs, when it runs one -- or several.

    Declared so the engine can ask before the run whether it is there:
    ``recast plan`` reports a missing tool beside the stage, which is the cheap
    check that should fail in a second rather than two stages in. The operator
    may point at a different binary through ``config[tool]`` -- so a scanner
    with ``tool = "gitleaks"`` reads ``config.get("gitleaks", "gitleaks")``, and
    one with ``tool = ("syft", "grype")`` reads each. ``None`` for a scanner
    that wraps nothing.
    """

    needs_build: bool = False
    """True for sanitizer and fuzz scanners, which need a compiled artifact."""

    @abstractmethod
    def scan(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Iterable[Finding]:
        """Yield findings for ``unit`` -- a real one, or the repository.

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

        Run the tool through ``executor``. ``config["root"]`` is the tree.
        """
