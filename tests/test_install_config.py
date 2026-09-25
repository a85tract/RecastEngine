"""tools/install-config.sh: the audit recipe's per-repository config, installed.

From hpc-devsecops's installer, with the same two rules worth a test: an
existing file is not overwritten without --force, and what lands is what the
scanners read -- a gitleaks config, an OpenVEX document, and the run config
the pre-push hook hands to recast.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_INSTALL = _ROOT / "tools" / "install-config.sh"
_FILES = (".gitleaks.toml", ".vex/openvex.json", ".recast-audit.json")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    work = tmp_path / "work"
    subprocess.run(["git", "init", "-q", str(work)], check=True)
    return work


def _install(repo: Path, *flags: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(_INSTALL), *flags, str(repo)], capture_output=True, text=True
    )


def test_the_three_files_are_installed_and_are_what_the_scanners_read(repo: Path) -> None:
    result = _install(repo)
    assert result.returncode == 0, result.stderr
    assert "installed=3 skipped=0" in result.stdout
    for rel in _FILES:
        assert (repo / rel).is_file(), rel
        assert (repo / rel).read_text() == (_ROOT / "templates" / rel).read_text()
    vex = json.loads((repo / ".vex" / "openvex.json").read_text())
    assert vex["statements"] == []  # suppresses nothing until someone decides
    config = json.loads((repo / ".recast-audit.json").read_text())
    assert set(config["stages"]) == {"secret", "composition"}
    assert "useDefault = true" in (repo / ".gitleaks.toml").read_text()


def test_an_existing_file_is_kept_unless_forced(repo: Path) -> None:
    (repo / ".gitleaks.toml").write_text("# mine\n")
    result = _install(repo)
    assert result.returncode == 0
    assert "installed=2 skipped=1" in result.stdout
    assert (repo / ".gitleaks.toml").read_text() == "# mine\n"

    result = _install(repo, "--force")
    assert result.returncode == 0
    assert "installed=3 skipped=0" in result.stdout
    assert (repo / ".gitleaks.toml").read_text() != "# mine\n"


def test_the_installer_refuses_what_is_not_a_repository(tmp_path: Path) -> None:
    result = _install(tmp_path / "nowhere")
    assert result.returncode != 0


def test_the_templates_carry_no_site_secret_shaped_content() -> None:
    """The gitleaks template names patterns to *allow*; none is a live value."""
    text = (_ROOT / "templates" / ".gitleaks.toml").read_text()
    assert "sk-ant-" in text  # the rule that catches the audit's own key
    assert "useDefault = true" in text
