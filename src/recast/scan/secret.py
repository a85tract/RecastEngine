"""``secret``: gitleaks, as a Scanner.

The detection is gitleaks'. What this module does is decide *what to point it
at*, and that decision is where the contract and the tool disagree -- see the
note in ``SecretScanner.scan``.

The shape is ``hpc-devsecops``'s ``tools/devsecops-local.sh``, by Chien-Wei
Huang: invoke gitleaks with ``--exit-code 0`` and a SARIF report path, read the
report back, count a tool that is not installed as a check that did not
complete. Written here as Python against the Scanner contract; the shell was
not ported.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recast import sarif
from recast.errors import ScannerUnavailable
from recast.model import Facts, Finding, Unit
from recast.plugins.scanner import Scanner


class SecretScanner(Scanner):
    """Credentials committed to a repository, found by gitleaks.

    ``family = "secret"``, matching the four families the engine's Scanner
    docstring names -- which are CC-Test's four checks.
    """

    name = "secret"
    family = "secret"
    needs_build = False

    def scan(
        self, unit: Unit, facts: Facts, workspace: Path, config: dict[str, Any]
    ) -> Iterable[Finding]:
        """Scan this Unit's source files.

        And that is the problem, recorded here rather than smoothed over.

        gitleaks' whole value is that it reads **history**: a credential
        removed in a later commit is still in the pack, and a working-tree
        scan cannot see it. ``hpc-devsecops`` runs it as ``gitleaks git <repo>
        --log-opts=<range>``, over a range of commits, or over a patch on
        stdin. Neither is a thing a Unit describes.

        A Unit is "an addressable piece of software under modernization" with
        ``sources`` -- files in a working tree. Scanning those is the only
        reading of ``scan(unit, ...)`` available, and it is a materially
        weaker check than the one the tool exists to perform. It is what this
        does, because the alternative is to ignore the argument the contract
        passes and scan the repository once per Unit, which is worse: N
        identical scans, the same findings attributed N times, and a runtime
        that grows with a number that has nothing to do with the work.

        The contract question is therefore real and not cosmetic: a Scanner
        whose subject is a repository and a history has no way to say so.
        """
        gitleaks = shutil.which(config.get("gitleaks", "gitleaks"))
        if gitleaks is None:
            # This used to `return []`, with a note saying there was no way to
            # report "I could not run" -- an empty iterable being
            # indistinguishable from a clean scan. Both halves of that excuse
            # are gone: the engine walks scanner stages, and ScannerUnavailable
            # is in the contract. The stage is now `incomplete`, the run cannot
            # report `passed` on it, and an operator who knows gitleaks is not
            # on this machine says so by name in `allow_incomplete`.
            raise ScannerUnavailable(
                f"gitleaks is not on PATH (looked for {config.get('gitleaks', 'gitleaks')!r}); "
                "no secret scan was performed"
            )

        root = Path(config.get("root", workspace)).resolve()
        targets = [root / source for source in unit.sources]
        targets = [t for t in targets if t.exists()]
        if not targets:
            return []

        findings: list[Finding] = []
        with tempfile.TemporaryDirectory() as tmp:
            for target in targets:
                report = Path(tmp) / f"{target.name}.sarif"
                # Directly, not through an Executor: the Scanner contract
                # passes none. The engine's rule is that nothing leaving the
                # process bypasses the seam, and a scanner that shells out has
                # no way to honour it -- recorded under P5 finding 6, whose
                # subject this is (a scanner's relationship to where and on
                # what it runs). ``needs_build`` scanners will force the answer.
                subprocess.run(  # noqa: S603
                    [
                        gitleaks,
                        "dir",
                        str(target),
                        "--report-format",
                        "sarif",
                        "--report-path",
                        str(report),
                        "--exit-code",
                        "0",
                        "--no-banner",
                    ],
                    capture_output=True,
                    check=False,
                )
                # Unconditionally, even when the report is absent: a report
                # that was never written means gitleaks died, and
                # `sarif.load` says so instead of returning an empty list.
                # Skipping the call when the file is missing is how a crash
                # used to arrive looking like a clean scan.
                findings.extend(
                    sarif.findings_from(
                        report,
                        unit=unit.uid,
                        scanner=self.name,
                        tool="gitleaks",
                        cwe="CWE-798",
                        exploitability="credential-disclosure",
                        default_path=str(target),
                    )
                )
        return findings


def factory() -> SecretScanner:
    return SecretScanner()
