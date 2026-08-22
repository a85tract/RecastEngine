"""SARIF in, ``Finding`` out.

SARIF is what security tools already speak -- gitleaks, the LLM source audit,
and most static analyzers emit it -- so a Scanner that wraps one of them spends
its interesting lines deciding *what to point the tool at*, and its boring ones
translating the report. The boring half is identical for every such scanner and
is not domain knowledge about anything, which is why it belongs here rather
than being rewritten once per plugin.

Two of the rules below are not conveniences and are the reason this is a module
rather than a snippet. Both come from the gate this engine's ``audit`` recipe is
modelled on -- ``hpc-devsecops``, running on Derecho, whose ``SECURITY.md``
states the contract: "Missing tools, malformed SARIF/JSON, unavailable
credentials, and scanner execution failures return 2; security findings return
1; only completed clean checks return 0."

    A report that will not parse is ``ScannerUnavailable``, never an empty list.
    A report whose invocation says it failed is ``ScannerUnavailable`` too.

Both are the same mistake this engine keeps finding: unparseable output and a
clean scan are different facts, and a converter that answers ``[]`` to both
hands the run a clean bill of health it never earned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from recast.errors import ScannerUnavailable
from recast.model import Access, Disclosure, Finding, Severity

__all__ = ["findings_from", "load"]

_LEVEL = {
    "error": Severity.HIGH,
    "warning": Severity.MEDIUM,
    "note": Severity.LOW,
    "none": Severity.INFO,
}
"""SARIF's four levels. A result with no level means ``warning`` per the spec,
and an unknown one is read as ``warning`` too rather than as ``info`` -- the
safe end, since a scanner inventing a level has not told us it is harmless."""


def load(report: Path | str, *, scanner: str) -> dict[str, Any]:
    """Read a SARIF log, or say the scan did not happen.

    ``scanner`` names the plugin in the error, because by the time this raises
    the traceback is several frames from anything that identifies it.
    """
    path = Path(report)
    try:
        document = json.loads(path.read_text())
    except OSError as error:
        raise ScannerUnavailable(
            f"{scanner}: no SARIF report at {path} ({error.strerror or error}); "
            "the scan produced nothing to read"
        ) from error
    except ValueError as error:
        raise ScannerUnavailable(
            f"{scanner}: the SARIF report at {path} is not valid JSON ({error}); "
            "unparseable output is not a clean scan"
        ) from error
    if not isinstance(document, dict) or not isinstance(document.get("runs"), list):
        raise ScannerUnavailable(
            f"{scanner}: the report at {path} has no SARIF 'runs' array; "
            "unparseable output is not a clean scan"
        )
    return document


def findings_from(
    report: Path | str | dict[str, Any],
    *,
    unit: str,
    scanner: str,
    tool: str = "",
    cwe: str | None = None,
    exploitability: str = "unknown",
    default_path: str = "",
) -> list[Finding]:
    """Every result in a SARIF log, as Findings at the safe end.

    ``disclosure`` is ``PLAUSIBLE`` and ``access`` is ``EMBARGOED`` for all of
    them, and neither is a parameter. A scanner opts *down* through an
    Adjudicator, never up by omission, and a converter that accepted an access
    class would be the omission.

    Raises ``ScannerUnavailable`` if the report will not parse or if it records
    a failed invocation -- see the module docstring.
    """
    document = report if isinstance(report, dict) else load(report, scanner=scanner)
    out: list[Finding] = []
    for run in document.get("runs", []):
        _require_the_invocation_succeeded(run, scanner)
        for index, result in enumerate(run.get("results", [])):
            rule = result.get("ruleId") or "rule"
            out.append(
                Finding(
                    uid=f"{scanner}:{unit}:{rule}:{index}",
                    unit=unit,
                    scanner=scanner,
                    title=(result.get("message") or {}).get("text") or f"{rule} matched",
                    cwe=cwe,
                    severity=_LEVEL.get(result.get("level", "warning"), Severity.MEDIUM),
                    disclosure=Disclosure.PLAUSIBLE,
                    access=Access.EMBARGOED,
                    location=_location(result, default_path),
                    exploitability=exploitability,
                    evidence={"rule": rule, "tool": tool or scanner},
                )
            )
    return out


def _require_the_invocation_succeeded(run: dict[str, Any], scanner: str) -> None:
    """SARIF carries its own "did this actually work" flag. Read it.

    ``invocations[].executionSuccessful`` is how a tool reports that it ran and
    failed, and a tool that failed usually also emits zero results -- so
    ignoring this field is how "crashed" arrives looking like "clean". Absent
    is not False: plenty of tools omit invocations entirely, and treating
    silence as failure would make every one of them permanently unavailable.
    """
    for invocation in run.get("invocations") or []:
        if invocation.get("executionSuccessful") is False:
            reason = (invocation.get("exitCodeDescription") or "").strip()
            raise ScannerUnavailable(
                f"{scanner}: the SARIF report records a failed invocation"
                + (f" ({reason})" if reason else "")
                + "; results from a failed run are not a clean scan"
            )


def _location(result: dict[str, Any], default_path: str) -> dict[str, Any]:
    physical = (result.get("locations") or [{}])[0].get("physicalLocation") or {}
    if not physical:
        return {"path": default_path} if default_path else {}
    region = physical.get("region") or {}
    location = {
        "path": (physical.get("artifactLocation") or {}).get("uri") or default_path,
        "line": region.get("startLine"),
    }
    return {key: value for key, value in location.items() if value not in (None, "")}
