"""``composition``: syft -> grype -> VEX, as a Scanner.

The dependency state of a repository, and the known vulnerabilities in it.
``hpc-devsecops`` (Chien-Wei Huang) runs this as three steps and says in a
comment why it is scoped the way it is -- "dependency analysis intentionally
describes the resulting repository state, rather than only the patch" -- which
is the reason this is a ``repository`` scanner and not a per-Unit one. The
steps, kept as they are there:

    syft scan dir:<root> -o spdx-json=<sbom>
    [ -f <root>/.vex/openvex.json ] && --vex <that>
    grype sbom:<sbom> --add-cpes-if-none [--vex ...] -o json

and from grype's JSON the ``Critical`` and ``High`` matches are what it counts.
Both become Findings here, at ``PLAUSIBLE``; ``hpc-devsecops`` blocks on
Critical alone and reports High as a number, and the equivalent of that
decision in this engine is the adjudicator's, not the scanner's.

A match names a dependency, so ``Finding.upstream`` is set -- the defect is
the dependency's, and the disclosure it needs is to them. The finding is still
embargoed by default, because the fact that *this* project ships the
vulnerable version is a fact about this project.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from recast.errors import ScannerUnavailable
from recast.model import Access, Disclosure, Facts, Finding, Severity, Unit
from recast.plugins.executor import Executor, Job
from recast.plugins.scanner import Scanner

_SEVERITY = {
    "Critical": Severity.CRITICAL,
    "High": Severity.HIGH,
    "Medium": Severity.MEDIUM,
    "Low": Severity.LOW,
    "Negligible": Severity.INFO,
}

_COUNTED = ("Critical", "High")
"""What hpc-devsecops counts. Everything below is in grype's output and not in
its summary line, and not here either."""


class CompositionScanner(Scanner):
    """Known vulnerabilities in what the repository depends on."""

    name = "composition"
    family = "composition"
    subject = "repository"
    tool = ("syft", "grype")
    needs_build = False
    blocks_on = Severity.CRITICAL
    """hpc-devsecops blocks on Critical alone; High is counted and reported."""

    def scan(
        self,
        unit: Unit,
        facts: Facts,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Iterable[Finding]:
        syft = config.get("syft", "syft")
        grype = config.get("grype", "grype")
        absent = [b for b in (syft, grype) if shutil.which(b) is None]
        if absent:
            raise ScannerUnavailable(
                f"{' and '.join(absent)} not on PATH; no composition scan was performed"
            )
        root = Path(config.get("root", workspace)).resolve()

        with tempfile.TemporaryDirectory() as tmp:
            sbom = Path(tmp) / "sbom.spdx.json"
            made = executor.wait(
                executor.submit(
                    Job(
                        argv=[syft, "scan", f"dir:{root}", "-o", f"spdx-json={sbom}", "-q"],
                        cwd=root,
                        label="syft",
                    )
                )
            )
            # The same two conditions hpc-devsecops puts on syft: it exited
            # clean, and the SBOM it wrote is not empty. Either failing is an
            # incomplete scan, not a repository with no dependencies.
            if not made.ok or not sbom.exists() or sbom.stat().st_size == 0:
                raise ScannerUnavailable(
                    f"syft exited {made.returncode} without producing an SBOM"
                    + _stderr(made.stderr)
                )

            argv = [grype, f"sbom:{sbom}", "--add-cpes-if-none"]
            vex = root / ".vex" / "openvex.json"
            if vex.is_file():
                argv += ["--vex", str(vex)]
            argv += ["-o", "json"]
            matched = executor.wait(executor.submit(Job(argv=argv, cwd=root, label="grype")))
            if not matched.ok:
                raise ScannerUnavailable(
                    f"grype exited {matched.returncode}" + _stderr(matched.stderr)
                )
            try:
                matches = json.loads(matched.stdout).get("matches", [])
            except (ValueError, AttributeError) as error:
                raise ScannerUnavailable(
                    f"grype's output is not the JSON it was asked for ({error}); "
                    "unparseable output is not a clean scan"
                ) from error

        return [_finding(m, unit.uid) for m in matches if _severity_name(m) in _COUNTED]


def _severity_name(match: dict[str, Any]) -> str:
    return str((match.get("vulnerability") or {}).get("severity") or "")


def _finding(match: dict[str, Any], unit: str) -> Finding:
    vulnerability = match.get("vulnerability") or {}
    artifact = match.get("artifact") or {}
    vuln_id = vulnerability.get("id") or "unknown"
    name = artifact.get("name") or "unknown"
    version = artifact.get("version") or ""
    locations = artifact.get("locations") or []
    fix = (vulnerability.get("fix") or {}).get("state") or "unknown"
    return Finding(
        uid=f"composition:{unit}:{vuln_id}:{name}@{version}",
        unit=unit,
        scanner="composition",
        title=f"{vuln_id} in {name} {version}".rstrip(),
        cwe=None,
        severity=_SEVERITY.get(_severity_name(match), Severity.INFO),
        disclosure=Disclosure.PLAUSIBLE,
        access=Access.EMBARGOED,
        location={"path": locations[0].get("path")} if locations else {},
        exploitability="unknown",
        evidence={
            "tool": "grype",
            "vulnerability": vuln_id,
            "artifact": name,
            "version": version,
            "fix": fix,
            "data_source": vulnerability.get("dataSource"),
        },
        upstream=name,
    )


def _stderr(text: str) -> str:
    text = text.strip()
    return f": {text}" if text else ""


def factory() -> CompositionScanner:
    return CompositionScanner()
