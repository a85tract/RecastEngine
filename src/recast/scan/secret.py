"""``secret``: gitleaks, as a Scanner.

The detection is gitleaks'. What this module does is decide *what to point it
at*, and the answer is the repository: gitleaks' value is in history, where a
credential deleted in a later commit is still in the pack, and no file in a
working tree shows that. So ``subject = "repository"`` and the run invokes
this once, against the tree, rather than once per Unit against that Unit's
files -- which is what it did for a day, and was a materially weaker check
than the tool exists to perform.

The shape is ``hpc-devsecops``'s ``tools/devsecops-local.sh``, by Chien-Wei
Huang: ``gitleaks git <repo>`` over history, ``--exit-code 0`` so findings are
read from the SARIF report rather than inferred from the exit status, a tool
that is not installed counted as a check that did not complete. Written here
as Python against the Scanner contract; the shell was not ported.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recast import sarif
from recast.errors import ScannerUnavailable
from recast.model import Facts, Finding, Unit
from recast.plugins.executor import Executor, Job
from recast.plugins.scanner import Scanner


class SecretScanner(Scanner):
    """Credentials committed to a repository, found by gitleaks."""

    name = "secret"
    family = "secret"
    subject = "repository"
    tool = "gitleaks"
    needs_build = False

    def scan(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Iterable[Finding]:
        binary = config.get("gitleaks", "gitleaks")
        if shutil.which(binary) is None:
            raise ScannerUnavailable(
                f"gitleaks is not on PATH (looked for {binary!r}); no secret scan was performed"
            )
        root = Path(config.get("root", workspace)).resolve()
        mode = _mode(root)

        # The report goes to a temporary directory and not to the workspace.
        # gitleaks' SARIF quotes the matched secret, and the workspace is under
        # the project root -- inside the checkout, which is the one place the
        # findings store refuses to write for exactly this reason.
        with tempfile.TemporaryDirectory() as tmp:
            report = Path(tmp) / "gitleaks.sarif"
            job = Job(
                argv=[
                    binary,
                    mode,
                    str(root),
                    "--report-format",
                    "sarif",
                    "--report-path",
                    str(report),
                    "--exit-code",
                    "0",
                    "--no-banner",
                ],
                cwd=root,
                label="gitleaks",
            )
            result = executor.wait(executor.submit(job))
            if not report.exists():
                # No report means gitleaks did not get as far as writing one.
                # ``sarif.load`` would say so too; this says it with the exit
                # status and stderr, which is what the operator needs to see.
                raise ScannerUnavailable(
                    f"gitleaks exited {result.returncode} without writing a report"
                    + (f": {result.stderr.strip()}" if result.stderr.strip() else "")
                )
            return sarif.findings_from(
                report,
                unit=unit.uid,
                scanner=self.name,
                tool="gitleaks",
                cwe="CWE-798",
                exploitability="credential-disclosure",
                default_path=str(root),
            )


def _mode(root: Path) -> str:
    """``git`` when ``root`` is itself a repository, else ``dir``.

    ``git`` is the scan worth having. ``dir`` is the fallback for a tree that
    is not a repository -- an export, an example directory -- where there is no
    history to read. A subdirectory *of* a repository gets ``dir`` too, on
    purpose: ``gitleaks git`` on it would find the enclosing repository and
    scan the whole of that history, which is not what pointing the engine at a
    subtree asked for.
    """
    return "git" if (root / ".git").exists() else "dir"


def factory() -> SecretScanner:
    return SecretScanner()
