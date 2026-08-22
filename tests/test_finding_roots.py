"""Where an embargoed finding is allowed to land.

``FindingStore.guard`` answers "may this store hold this record". These are the
other question, and the one the realistic accident turns on: may the store be
*here*. A 0700 directory inside a repository satisfies every permission check
the conformance suite makes and is still one ``git add -A`` from publishing an
unpatched vulnerability -- which is the accident the store's own docstring names
and, until this file, the one nothing checked.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from recast import WORKSPACE_DIRNAME
from recast.errors import RecastError
from recast.model import Access, Finding
from recast.run import _findings_root
from recast.store.filesystem import FilesystemFindingStore


def _finding(uid: str = "F-1") -> Finding:
    return Finding(uid=uid, unit="u", scanner="s", title="t", access=Access.EMBARGOED)


# --- the default root --------------------------------------------------------


def test_the_default_is_outside_the_project(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECAST_FINDINGS_HOME", raising=False)
    root = _findings_root(tmp_path)
    assert not root.is_relative_to(tmp_path)
    assert root.parent == Path.home() / WORKSPACE_DIRNAME / "findings"


def test_the_base_is_configurable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """For anyone who keeps embargoed material somewhere specific: an encrypted
    volume, a host that is not this one."""
    monkeypatch.setenv("RECAST_FINDINGS_HOME", str(tmp_path / "vault"))
    assert _findings_root(tmp_path).parent == tmp_path / "vault"


def test_two_clones_of_one_repository_are_two_projects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keyed by absolute path rather than by name, or the second clone's
    findings would land on the first's."""
    monkeypatch.setenv("RECAST_FINDINGS_HOME", str(tmp_path / "vault"))
    first = _findings_root(tmp_path / "a" / "engine")
    second = _findings_root(tmp_path / "b" / "engine")
    assert first != second
    assert first.name.startswith("engine-") and second.name.startswith("engine-")


def test_the_same_project_gets_the_same_directory_every_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Or yesterday's findings are not there to compare against today's."""
    monkeypatch.setenv("RECAST_FINDINGS_HOME", str(tmp_path / "vault"))
    assert _findings_root(tmp_path / "engine") == _findings_root(tmp_path / "engine")


# --- the refusal -------------------------------------------------------------


def test_a_root_inside_a_checkout_is_refused(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    with pytest.raises(RecastError, match="git working tree"):
        FilesystemFindingStore(root=tmp_path / "findings")


def test_a_root_further_down_inside_a_checkout_is_refused(tmp_path: Path) -> None:
    """The walk goes up. A findings directory six levels into a repository is
    the same accident as one at its top."""
    (tmp_path / ".git").mkdir()
    with pytest.raises(RecastError, match="git working tree"):
        FilesystemFindingStore(root=tmp_path / "a" / "b" / "c" / "findings")


def test_a_git_file_counts_as_a_checkout(tmp_path: Path) -> None:
    """``.git`` is a file rather than a directory inside a worktree or a
    submodule -- the case an ``is_dir()`` check waves through, and submodules
    are how the sibling repositories were once going to be arranged."""
    (tmp_path / ".git").write_text("gitdir: /elsewhere/.git/worktrees/wt\n")
    with pytest.raises(RecastError, match="git working tree"):
        FilesystemFindingStore(root=tmp_path / "findings")


def test_a_refusal_leaves_no_directory_behind(tmp_path: Path) -> None:
    """Checked before the directory is created. A store that refuses and still
    creates its root has put an empty, unexplained directory in someone's
    repository."""
    (tmp_path / ".git").mkdir()
    with pytest.raises(RecastError):
        FilesystemFindingStore(root=tmp_path / "findings")
    assert not (tmp_path / "findings").exists()


def test_the_refusal_says_what_to_do_instead(tmp_path: Path) -> None:
    (tmp_path / ".git").mkdir()
    with pytest.raises(RecastError, match="RECAST_FINDINGS_HOME"):
        FilesystemFindingStore(root=tmp_path / "findings")


def test_a_root_outside_any_checkout_is_accepted_and_kept_private(tmp_path: Path) -> None:
    store = FilesystemFindingStore(root=tmp_path / "vault" / "findings")
    uri = store.put(_finding())
    assert uri.startswith("file://")
    assert not (tmp_path / "vault" / "findings").stat().st_mode & 0o077
