"""SARIF to Finding, and the two cases that are not conveniences.

Both are the same mistake in different clothes: a report that will not parse
and a report from a tool that crashed are not clean scans, and a converter that
answers ``[]`` to either hands the run a clean bill of health nobody earned.
The rule is `hpc-devsecops`'s, whose SECURITY.md puts malformed SARIF in the
same exit class as a missing tool.

The fake tool is driven through PATH rather than by monkeypatching the parser,
so the argv, the subprocess and the file the tool writes are all in the run.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from recast.conformance.fake_tool import fake_tool, on_path
from recast.errors import ScannerUnavailable
from recast.model import Access, Disclosure, Severity
from recast.sarif import findings_from, load


def _log(**result: Any) -> dict[str, Any]:
    return {"runs": [{"results": [{"ruleId": "r1", "message": {"text": "found"}, **result}]}]}


# --- the translation ---------------------------------------------------------


def test_a_result_becomes_a_finding_at_the_safe_end() -> None:
    """``disclosure`` and ``access`` are not parameters. A scanner opts down
    through an Adjudicator, never up by omission."""
    (finding,) = findings_from(_log(), unit="u:a", scanner="secret", tool="gitleaks")
    assert finding.unit == "u:a"
    assert finding.scanner == "secret"
    assert finding.disclosure is Disclosure.PLAUSIBLE
    assert finding.access is Access.EMBARGOED
    assert finding.evidence == {"rule": "r1", "tool": "gitleaks"}


def test_the_uid_separates_two_results_of_one_rule() -> None:
    """Two AWS keys in one file are two findings, not one seen twice."""
    log = {"runs": [{"results": [{"ruleId": "aws"}, {"ruleId": "aws"}]}]}
    uids = [f.uid for f in findings_from(log, unit="u:a", scanner="secret")]
    assert uids == ["secret:u:a:aws:0", "secret:u:a:aws:1"]


@pytest.mark.parametrize(
    ("level", "severity"),
    [
        ("error", Severity.HIGH),
        ("warning", Severity.MEDIUM),
        ("note", Severity.LOW),
        ("none", Severity.INFO),
    ],
)
def test_sarif_levels_map_to_severities(level: str, severity: Severity) -> None:
    (finding,) = findings_from(_log(level=level), unit="u:a", scanner="s")
    assert finding.severity is severity


def test_a_missing_or_unknown_level_reads_as_warning() -> None:
    """SARIF's own default, and the safe end: a tool inventing a level has not
    told us the result is harmless."""
    (absent,) = findings_from(_log(), unit="u:a", scanner="s")
    (invented,) = findings_from(_log(level="catastrophe"), unit="u:a", scanner="s")
    assert absent.severity is invented.severity is Severity.MEDIUM


def test_the_location_carries_the_path_and_line() -> None:
    log = _log(
        locations=[
            {
                "physicalLocation": {
                    "artifactLocation": {"uri": "a/b.f90"},
                    "region": {"startLine": 7},
                }
            }
        ]
    )
    (finding,) = findings_from(log, unit="u:a", scanner="s")
    assert finding.location == {"path": "a/b.f90", "line": 7}


def test_a_result_with_no_location_falls_back_rather_than_inventing() -> None:
    (finding,) = findings_from(_log(), unit="u:a", scanner="s", default_path="whole-repo")
    assert finding.location == {"path": "whole-repo"}
    (bare,) = findings_from(_log(), unit="u:a", scanner="s")
    assert bare.location == {}


# --- the two that are not conveniences ---------------------------------------


def test_unparseable_output_is_unavailable_not_clean(tmp_path: Path) -> None:
    report = tmp_path / "gitleaks.sarif"
    report.write_text("{not json at all")
    with pytest.raises(ScannerUnavailable, match="not valid JSON"):
        load(report, scanner="secret")


def test_a_report_that_was_never_written_is_unavailable_not_clean(tmp_path: Path) -> None:
    """The tool died before writing. An empty findings list here says the
    repository is clean, on the strength of a file that does not exist."""
    with pytest.raises(ScannerUnavailable, match="no SARIF report"):
        load(tmp_path / "never-written.sarif", scanner="secret")


def test_valid_json_that_is_not_sarif_is_unavailable(tmp_path: Path) -> None:
    report = tmp_path / "r.sarif"
    report.write_text(json.dumps({"error": "rate limited"}))
    with pytest.raises(ScannerUnavailable, match="no SARIF 'runs' array"):
        load(report, scanner="audit.llm")


def test_a_failed_invocation_is_unavailable_even_with_zero_results() -> None:
    """SARIF carries its own "did this work" flag, and a tool that crashed
    usually also emits no results -- so ignoring the flag is precisely how
    "crashed" arrives looking like "clean"."""
    log = {"runs": [{"invocations": [{"executionSuccessful": False}], "results": []}]}
    with pytest.raises(ScannerUnavailable, match="failed invocation"):
        findings_from(log, unit="u:a", scanner="audit.llm")


def test_a_failed_invocation_says_why_when_the_tool_said() -> None:
    log = {
        "runs": [
            {
                "invocations": [
                    {"executionSuccessful": False, "exitCodeDescription": "no API key"}
                ],
                "results": [],
            }
        ]
    }
    with pytest.raises(ScannerUnavailable, match="no API key"):
        findings_from(log, unit="u:a", scanner="audit.llm")


def test_an_absent_invocation_is_not_a_failed_one() -> None:
    """Plenty of tools omit invocations. Reading silence as failure would make
    every one of them permanently unavailable."""
    assert findings_from(_log(), unit="u:a", scanner="s")


def test_a_successful_invocation_with_no_results_is_a_clean_scan() -> None:
    """The one case that *should* come back empty, so the rules above cannot be
    read as "always raise on an empty report"."""
    log = {"runs": [{"invocations": [{"executionSuccessful": True}], "results": []}]}
    assert findings_from(log, unit="u:a", scanner="s") == []


# --- driven through a fake tool on PATH --------------------------------------


def _run_fake(name: str, report: Path) -> int:
    """Invoke the fake the way a Scanner would: by name, found on PATH."""
    return subprocess.run(
        [name, "dir", ".", "--report-path", str(report)], capture_output=True
    ).returncode


def test_a_fake_tool_writes_where_it_was_told_and_the_report_converts(tmp_path: Path) -> None:
    bin_dir, report = tmp_path / "bin", tmp_path / "out.sarif"
    bin_dir.mkdir()
    fake_tool(bin_dir, "gitleaks", sarif=_log(level="error"))
    with on_path(bin_dir):
        assert _run_fake("gitleaks", report) == 0
    (finding,) = findings_from(report, unit="u:a", scanner="secret", tool="gitleaks")
    assert finding.severity is Severity.HIGH


def test_a_fake_tool_can_emit_the_case_a_document_cannot_express(tmp_path: Path) -> None:
    """Output that is not JSON at all. This is why the fake takes a string."""
    bin_dir, report = tmp_path / "bin", tmp_path / "out.sarif"
    bin_dir.mkdir()
    fake_tool(bin_dir, "gitleaks", payload="panic: runtime error\n", exit_code=2)
    with on_path(bin_dir):
        assert _run_fake("gitleaks", report) == 2
    with pytest.raises(ScannerUnavailable):
        load(report, scanner="secret")


def test_on_path_only_is_how_a_missing_tool_is_checked(tmp_path: Path) -> None:
    """The absence is the fixture: an empty directory as the whole PATH."""
    import shutil

    empty = tmp_path / "empty"
    empty.mkdir()
    with on_path(empty, only=True):
        assert shutil.which("gitleaks") is None
