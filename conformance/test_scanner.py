"""Scanner: what a wrapped tool's silence is allowed to mean.

``scan`` returns an iterable, so "the tool is not installed", "the tool crashed"
and "the repository is clean" are all the same value coming out. Only the
scanner can tell them apart, and only by raising -- which is why the contract
has ``ScannerUnavailable`` and why these checks exist. A security gate that
reports untested as clean is worse than one that reports nothing.

The tool is faked on PATH rather than the plugin being stubbed out, so the
plugin's argv, its subprocess handling and its report parsing all stay in the
run. Faking the plugin would replace the code being checked.

A case that declares no ``tool`` skips these by name. That is the intended
reading -- unexercised, not passed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recast.conformance.fake_tool import fake_tool, on_path
from recast.errors import ScannerUnavailable
from recast.executors.local import factory as local_executor
from recast.model import Access, Disclosure, Facts, Unit

_CLEAN = {"runs": [{"results": []}]}
_ONE = {"runs": [{"results": [{"ruleId": "conformance", "message": {"text": "found"}}]}]}


@pytest.fixture
def subject(tmp_path: Path) -> tuple[Unit, Facts]:
    """A Unit with a real file under it.

    Not a bare uid: a Scanner that wraps a tool has to be given something to
    point the tool at, and one handed a Unit with no ``sources`` is entitled to
    return early without invoking anything -- which would make every check
    below pass without running the tool it is here to fake.
    """
    (tmp_path / "src").mkdir(exist_ok=True)
    source = Path("src") / "subject.f90"
    (tmp_path / source).write_text("      subroutine s()\n      end subroutine\n")
    unit = Unit(uid="conformance:unit", kind="module", sources=(source,))
    return unit, Facts(unit=unit.uid)


def _scanner(case: Any) -> Any:
    return case.build() if case.build is not None else _from_registry(case.name)


def _from_registry(name: str) -> Any:
    from recast.registry import REGISTRY

    return REGISTRY.get("scanner", name)()


def _tools(case: Any) -> tuple[str, ...]:
    """The case's declaration, else the plugin's own ``tool``; always a tuple."""
    declared = case.tool if case.tool is not None else getattr(_scanner(case), "tool", None)
    if not declared:
        return ()
    return (declared,) if isinstance(declared, str) else tuple(declared)


def _fakes(case: Any, bin_dir: Path, behaviour: str) -> None:
    """Populate ``bin_dir`` with fakes of the case's tools behaving as asked.

    A case whose tools do not write SARIF to a report path supplies ``fakes``;
    otherwise every declared tool gets a SARIF-speaking fake.
    """
    if case.fakes is not None:
        case.fakes(bin_dir, behaviour)
        return
    for tool in _tools(case):
        if behaviour == "garbage":
            fake_tool(bin_dir, tool, payload="panic: runtime error\n", exit_code=2)
        else:
            fake_tool(bin_dir, tool, sarif=_ONE if behaviour == "one" else _CLEAN)


def _scan(case: Any, subject: tuple[Unit, Facts], workspace: Path) -> list[Any]:
    unit, facts = subject
    config = {"root": workspace, **dict(case.config)}
    return list(_scanner(case).scan(unit, facts, workspace, local_executor(), config))


def _skip_unless_tooled(case: Any) -> None:
    if not _tools(case):
        pytest.skip(f"{case.name} declares no external tool")


def test_a_tool_that_is_not_installed_is_not_a_clean_scan(
    scanner_case: Any, subject: tuple[Unit, Facts], tmp_path: Path
) -> None:
    """The absence is the fixture: an empty directory as the whole PATH.

    Returning an empty iterable here tells the run the repository is clean, on
    the strength of a tool that was never invoked.
    """
    _skip_unless_tooled(scanner_case)
    empty = tmp_path / "empty-path"
    empty.mkdir()
    with on_path(empty, only=True), pytest.raises(ScannerUnavailable):
        _scan(scanner_case, subject, tmp_path)


def test_output_that_will_not_parse_is_not_a_clean_scan(
    scanner_case: Any, subject: tuple[Unit, Facts], tmp_path: Path
) -> None:
    """The tool was there, ran, and emitted something unreadable. Same rule as
    a missing tool, and the same rule ``hpc-devsecops`` states in its
    SECURITY.md: malformed SARIF is in the exit class of a missing tool, not of
    a clean check."""
    _skip_unless_tooled(scanner_case)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fakes(scanner_case, bin_dir, "garbage")
    with on_path(bin_dir), pytest.raises(ScannerUnavailable):
        _scan(scanner_case, subject, tmp_path)


def test_a_clean_report_really_is_a_clean_scan(
    scanner_case: Any, subject: tuple[Unit, Facts], tmp_path: Path
) -> None:
    """So the two checks above cannot be satisfied by raising at everything."""
    _skip_unless_tooled(scanner_case)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fakes(scanner_case, bin_dir, "clean")
    with on_path(bin_dir):
        assert _scan(scanner_case, subject, tmp_path) == []


def test_what_it_finds_arrives_at_the_safe_end(
    scanner_case: Any, subject: tuple[Unit, Facts], tmp_path: Path
) -> None:
    """A scanner opts *down* through an Adjudicator, never up by omission."""
    _skip_unless_tooled(scanner_case)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fakes(scanner_case, bin_dir, "one")
    with on_path(bin_dir):
        found = _scan(scanner_case, subject, tmp_path)
    assert found, "the fake tool reported a result and the scanner yielded none"
    for finding in found:
        assert finding.access is Access.EMBARGOED
        assert finding.disclosure is Disclosure.PLAUSIBLE
