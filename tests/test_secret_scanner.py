"""The in-tree gitleaks wrapper, driven through a fake gitleaks on PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

from recast.conformance.fake_tool import fake_tool, on_path
from recast.errors import ScannerUnavailable
from recast.executors.local import factory as local_executor
from recast.model import Facts, Unit
from recast.scan.secret import SecretScanner, _mode

_ONE = {"runs": [{"results": [{"ruleId": "aws-access-key", "level": "error"}]}]}


def _tree() -> tuple[Unit, Facts]:
    unit = Unit(uid="repository:x", kind="repository")
    return unit, Facts(unit=unit.uid)


def test_it_is_a_repository_scanner_with_a_declared_tool() -> None:
    assert SecretScanner.subject == "repository"
    assert SecretScanner.tool == "gitleaks"


def test_git_mode_for_a_repository_and_dir_mode_otherwise(tmp_path: Path) -> None:
    assert _mode(tmp_path) == "dir"
    (tmp_path / ".git").mkdir()
    assert _mode(tmp_path) == "git"


def test_a_subdirectory_of_a_repository_gets_dir_mode(tmp_path: Path) -> None:
    """``gitleaks git`` on it would scan the enclosing repository's whole
    history, which is not what pointing the engine at a subtree asked for."""
    (tmp_path / ".git").mkdir()
    (tmp_path / "sub").mkdir()
    assert _mode(tmp_path / "sub") == "dir"


def test_findings_come_back_through_the_executor(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool(bin_dir, "gitleaks", sarif=_ONE)
    unit, facts = _tree()
    with on_path(bin_dir):
        found = list(
            SecretScanner().scan(unit, facts, tmp_path, local_executor(), {"root": tmp_path})
        )
    (finding,) = found
    assert finding.unit == "repository:x"
    assert finding.cwe == "CWE-798"
    assert finding.evidence["tool"] == "gitleaks"


def test_a_tool_that_dies_without_a_report_is_unavailable_with_its_stderr(
    tmp_path: Path,
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool(bin_dir, "gitleaks", exit_code=3, stderr="fatal: not a git repository")
    unit, facts = _tree()
    with (
        on_path(bin_dir),
        pytest.raises(ScannerUnavailable, match=r"exited 3 .* not a git repository"),
    ):
        list(SecretScanner().scan(unit, facts, tmp_path, local_executor(), {"root": tmp_path}))


# --- a revision range ------------------------------------------------------------


def _argv_after(tmp_path: Path, config: dict) -> list[str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "argv"
    fake_tool(bin_dir, "gitleaks", sarif={"runs": [{"results": []}]}, record_argv=record)
    unit, facts = _tree()
    with on_path(bin_dir):
        list(
            SecretScanner().scan(
                unit, facts, tmp_path, local_executor(), {"root": tmp_path, **config}
            )
        )
    return record.read_text().split("\n")


def test_a_range_scopes_the_history_scan(tmp_path: Path) -> None:
    """hpc-devsecops's --range, and the mode its pre-push hook uses."""
    (tmp_path / ".git").mkdir()
    argv = _argv_after(tmp_path, {"range": "abc123..def456"})
    assert argv[:2] == ["git", str(tmp_path.resolve())]
    assert argv[argv.index("--log-opts") + 1] == "abc123..def456"


def test_no_range_scans_the_whole_history(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    argv = _argv_after(tmp_path, {})
    assert "--log-opts" not in argv


def test_a_range_on_a_tree_that_is_not_a_repository_is_unavailable(tmp_path: Path) -> None:
    """There is no history to scope. Silently scanning the directory instead
    would report on something other than what was asked about."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool(bin_dir, "gitleaks", sarif={"runs": [{"results": []}]})
    unit, facts = _tree()
    with on_path(bin_dir), pytest.raises(ScannerUnavailable, match="not a git repository"):
        list(
            SecretScanner().scan(
                unit, facts, tmp_path, local_executor(), {"root": tmp_path, "range": "a..b"}
            )
        )
