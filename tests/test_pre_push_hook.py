"""The pre-push hook, driven by a real ``git push`` into a local bare remote.

The hook is shell; what is being checked is the part that can go wrong in a
way no unit test of the scanner would see -- the range it hands to ``recast``
for each shape of push, and that ``recast``'s exit status actually stops the
push. ``recast`` itself is faked on PATH with ``fake_tool``, recording its
argv and exiting as told, so these run without a gitleaks and without the
engine's own plugins in the loop.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from recast.conformance.fake_tool import fake_tool, on_path

_TOOLS = Path(__file__).resolve().parents[1] / "tools"


def _git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    env = {
        **os.environ,
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@example.invalid",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@example.invalid",
    }
    return subprocess.run(
        ["git", *args], cwd=cwd, env=env, capture_output=True, text=True, check=check
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A working repository with one commit already on its bare remote."""
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-q", str(remote))
    work = tmp_path / "work"
    _git(tmp_path, "init", "-q", "-b", "main", str(work))
    (work / "a.txt").write_text("one\n")
    _git(work, "add", "a.txt")
    _git(work, "commit", "-q", "-m", "one")
    _git(work, "remote", "add", "origin", str(remote))
    _git(work, "push", "-q", "-u", "origin", "main")
    assert (
        subprocess.run(
            ["bash", str(_TOOLS / "install-hooks.sh"), str(work)], capture_output=True
        ).returncode
        == 0
    )
    return work


def _fake_recast(tmp_path: Path, exit_code: int) -> Path:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    record = tmp_path / "recast.argv"
    fake_tool(bin_dir, "recast", payload="", exit_code=exit_code, record_argv=record)
    # The hook needs git too; the fake directory is prepended, not exclusive.
    return record


def _commit(work: Path, name: str) -> str:
    (work / name).write_text(f"{name}\n")
    _git(work, "add", name)
    _git(work, "commit", "-q", "-m", name)
    return _git(work, "rev-parse", "HEAD").stdout.strip()


def test_a_push_to_an_existing_branch_scans_exactly_the_new_commits(
    repo: Path, tmp_path: Path
) -> None:
    record = _fake_recast(tmp_path, 0)
    before = _git(repo, "rev-parse", "HEAD").stdout.strip()
    after = _commit(repo, "b.txt")
    with on_path(tmp_path / "bin"):
        result = _git(repo, "push", "-q", "origin", "main", check=False)
    assert result.returncode == 0, result.stderr
    argv = record.read_text().split("\n")
    assert argv[:3] == ["run", "audit", str(repo.resolve())]
    assert argv[argv.index("--range") + 1] == f"{before}..{after}"


def test_findings_block_the_push(repo: Path, tmp_path: Path) -> None:
    _fake_recast(tmp_path, 1)
    _commit(repo, "b.txt")
    with on_path(tmp_path / "bin"):
        result = _git(repo, "push", "-q", "origin", "main", check=False)
    assert result.returncode != 0
    assert _git(repo, "rev-parse", "origin/main").stdout != _git(repo, "rev-parse", "HEAD").stdout


def test_an_incomplete_audit_blocks_the_push_too(repo: Path, tmp_path: Path) -> None:
    """Fail closed. A missing gitleaks is not a reason to let a push through;
    it is a reason nobody can say the push is clean."""
    _fake_recast(tmp_path, 2)
    _commit(repo, "b.txt")
    with on_path(tmp_path / "bin"):
        result = _git(repo, "push", "-q", "origin", "main", check=False)
    assert result.returncode != 0


def test_a_new_branch_scans_from_the_merge_base(repo: Path, tmp_path: Path) -> None:
    record = _fake_recast(tmp_path, 0)
    base = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "-b", "feature")
    tip = _commit(repo, "f.txt")
    _git(repo, "remote", "set-head", "origin", "main")
    with on_path(tmp_path / "bin"):
        result = _git(repo, "push", "-q", "origin", "feature", check=False)
    assert result.returncode == 0, result.stderr
    argv = record.read_text().split("\n")
    assert argv[argv.index("--range") + 1] == f"{base}..{tip}"


def test_a_deletion_is_not_scanned(repo: Path, tmp_path: Path) -> None:
    _fake_recast(tmp_path, 0)
    _git(repo, "checkout", "-q", "-b", "gone")
    _commit(repo, "g.txt")
    with on_path(tmp_path / "bin"):
        _git(repo, "push", "-q", "origin", "gone")
        record = _fake_recast(tmp_path, 1)  # would block if it were ever called
        record.unlink(missing_ok=True)
        result = _git(repo, "push", "-q", "origin", "--delete", "gone", check=False)
    assert result.returncode == 0, result.stderr
    assert not record.exists()


def test_a_missing_recast_blocks_rather_than_waves_through(repo: Path, tmp_path: Path) -> None:
    _commit(repo, "b.txt")
    empty = tmp_path / "empty"
    empty.mkdir()
    git_dir = Path(shutil.which("git")).parent  # type: ignore[arg-type]
    with on_path(empty, only=True):
        # git, and the bash the hook's shebang resolves through env; no recast.
        os.environ["PATH"] = os.pathsep.join([str(empty), str(git_dir), "/usr/bin", "/bin"])
        result = _git(repo, "push", "-q", "origin", "main", check=False)
    assert result.returncode != 0
    assert "not found -- blocking push" in result.stderr


def test_install_refuses_to_overwrite_a_foreign_hook(tmp_path: Path) -> None:
    work = tmp_path / "w"
    _git(tmp_path, "init", "-q", str(work))
    hooks = work / ".git" / "hooks"
    hooks.mkdir(exist_ok=True)
    (hooks / "pre-push").write_text("#!/bin/sh\nexit 0\n")
    result = subprocess.run(
        ["bash", str(_TOOLS / "install-hooks.sh"), str(work)], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
    forced = subprocess.run(
        ["bash", str(_TOOLS / "install-hooks.sh"), "--force", str(work)], capture_output=True
    )
    assert forced.returncode == 0
    assert (hooks / "pre-push").resolve() == (_TOOLS / "pre-push").resolve()
