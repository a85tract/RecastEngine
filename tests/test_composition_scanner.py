"""The in-tree syft/grype wrapper, driven through fakes on PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

from recast.conformance.fake_tool import fake_tool, on_path
from recast.errors import ScannerUnavailable
from recast.executors.local import factory as local_executor
from recast.model import Facts, Severity, Unit
from recast.scan.composition import CompositionScanner


def _match(vuln: str, severity: str, name: str = "libfoo", version: str = "1.2") -> dict:
    return {
        "vulnerability": {"id": vuln, "severity": severity, "fix": {"state": "fixed"}},
        "artifact": {"name": name, "version": version, "locations": [{"path": "/usr/lib/x"}]},
    }


def _fakes(bin_dir: Path, matches: list[dict] | None, *, syft_ok: bool = True) -> None:
    if syft_ok:
        fake_tool(bin_dir, "syft", payload='{"spdxVersion": "SPDX-2.3"}')
    else:
        fake_tool(bin_dir, "syft", payload=None, exit_code=1, stderr="syft: cannot read tree")
    fake_tool(bin_dir, "grype", sarif={"matches": matches or []}, report_flags=())


def _scan(tmp_path: Path, bin_dir: Path) -> list:
    unit = Unit(uid="repository:x", kind="repository")
    with on_path(bin_dir):
        return list(
            CompositionScanner().scan(
                unit, Facts(unit=unit.uid), tmp_path, local_executor(), {"root": tmp_path}
            )
        )


def test_it_declares_both_tools_and_the_repository_subject() -> None:
    assert CompositionScanner.tool == ("syft", "grype")
    assert CompositionScanner.subject == "repository"


def test_critical_and_high_become_findings_and_lower_does_not(tmp_path: Path) -> None:
    """What hpc-devsecops counts: its summary line says Critical and High and
    nothing else, so nothing else is a finding here either."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fakes(
        bin_dir,
        [
            _match("CVE-1", "Critical"),
            _match("CVE-2", "High", name="libbar"),
            _match("CVE-3", "Medium"),
            _match("CVE-4", "Negligible"),
        ],
    )
    found = _scan(tmp_path, bin_dir)
    assert [(f.evidence["vulnerability"], f.severity) for f in found] == [
        ("CVE-1", Severity.CRITICAL),
        ("CVE-2", Severity.HIGH),
    ]


def test_a_match_is_attributed_to_its_dependency(tmp_path: Path) -> None:
    """``upstream`` names the dependency, because the defect is theirs; the
    finding stays embargoed, because shipping it is ours."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fakes(bin_dir, [_match("CVE-1", "Critical", name="libfoo", version="1.2")])
    (finding,) = _scan(tmp_path, bin_dir)
    assert finding.upstream == "libfoo"
    assert finding.uid == "composition:repository:x:CVE-1:libfoo@1.2"
    assert finding.title == "CVE-1 in libfoo 1.2"
    assert finding.location == {"path": "/usr/lib/x"}
    assert finding.evidence["fix"] == "fixed"
    assert finding.cwe is None


def test_syft_failing_is_unavailable_not_a_tree_with_no_dependencies(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _fakes(bin_dir, [], syft_ok=False)
    with pytest.raises(ScannerUnavailable, match=r"syft exited 1 .*cannot read tree"):
        _scan(tmp_path, bin_dir)


def test_an_empty_sbom_is_unavailable_too(tmp_path: Path) -> None:
    """hpc-devsecops's own second condition on syft: ``[ -s sbom ]``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool(bin_dir, "syft", payload="")
    fake_tool(bin_dir, "grype", sarif={"matches": []}, report_flags=())
    with pytest.raises(ScannerUnavailable, match="without producing an SBOM"):
        _scan(tmp_path, bin_dir)


def test_grype_output_that_is_not_json_is_unavailable(tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool(bin_dir, "syft", payload='{"spdxVersion": "SPDX-2.3"}')
    fake_tool(bin_dir, "grype", payload="not json", report_flags=())
    with pytest.raises(ScannerUnavailable, match="not the JSON"):
        _scan(tmp_path, bin_dir)


def test_a_vex_file_is_passed_when_present(tmp_path: Path) -> None:
    """The fake grype records its argv in the report it writes, so the test can
    see whether ``--vex`` reached it. Same path hpc-devsecops checks:
    ``<root>/.vex/openvex.json``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_tool(bin_dir, "syft", payload='{"spdxVersion": "SPDX-2.3"}')
    (bin_dir / "grype").write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' \"$@\" > {tmp_path / 'grype.argv'}\n"
        "echo '{\"matches\": []}'\n"
    )
    (bin_dir / "grype").chmod(0o755)
    (tmp_path / ".vex").mkdir()
    (tmp_path / ".vex" / "openvex.json").write_text("{}")
    _scan(tmp_path, bin_dir)
    argv = (tmp_path / "grype.argv").read_text().split("\n")
    assert "--vex" in argv
    assert argv[argv.index("--vex") + 1] == str(tmp_path / ".vex" / "openvex.json")
    assert "--add-cpes-if-none" in argv
    assert argv[0].startswith("sbom:")
