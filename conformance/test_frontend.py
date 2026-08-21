"""Frontend: reads source, and only reads it.

Two of these are about the tree rather than the units. A Frontend is the one
plugin pointed at somebody's working copy, and the ABC invites it to cache
expensive analyses without saying where -- so "side-effect free" is worth
checking rather than assuming, and so is the narrower version of it: the engine
writes its workspace into that same tree, and a Frontend that reads its own
engine's output back has turned output into input.
"""

from __future__ import annotations

import hashlib
import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

from recast import WORKSPACE_DIRNAME
from recast.model import Unit
from recast.plugins.frontend import Frontend
from recast.registry import REGISTRY


def _shape(units: list[Unit]) -> set[tuple[str, str, str | None, tuple[str, ...]]]:
    """A Unit set, compared by what the contract says identifies one.

    Ordering is explicitly not significant -- the engine topologically sorts
    on the callgraph -- so this is a set, and comparing lists would hold the
    frontend to a promise it never made.
    """
    return {(u.uid, u.kind, u.parent, tuple(sorted(str(s) for s in u.sources))) for u in units}


def _fingerprint(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _analyzed(frontend: Frontend, unit: Unit, tree: Path) -> Any:
    """``None`` when the frontend refused. Refusing is allowed -- a file that
    does not parse is a fact about the tree, and the ABC's own reference
    implementation raises for one. What the checks below ask is what happened
    anyway: whether the tree changed, and whether the refusal is repeatable."""
    try:
        return frontend.analyze(unit, tree)
    except Exception:
        return None


@pytest.fixture
def frontend(frontend_case: Any) -> Frontend:
    missing = [m for m in frontend_case.requires if importlib.util.find_spec(m) is None]
    missing += [c for c in frontend_case.requires_commands if shutil.which(c) is None]
    if missing:
        pytest.skip(f"case {frontend_case.name!r} needs {missing}, which are not available here")
    built: Frontend = (
        frontend_case.build()
        if frontend_case.build
        else REGISTRY.get("frontend", frontend_case.name)()
    )
    return built


@pytest.fixture
def tree(frontend_case: Any, frontend: Frontend, tmp_path: Path) -> Path:
    """A copy the suite owns. Two checks write into it, and one asks whether
    the frontend did -- neither is safe against a real working copy."""
    root = tmp_path / "source"
    root.mkdir()
    frontend_case.plant_tree(root)
    return root


def test_it_discovers_what_the_case_says_it_should(
    frontend_case: Any, frontend: Frontend, tree: Path
) -> None:
    """A frontend that finds nothing satisfies every other check on this page."""
    found = {u.uid for u in frontend.discover(tree)}
    assert found, "discovery returned no units at all"
    missing = [uid for uid in frontend_case.expect_uids if uid not in found]
    assert not missing, f"expected {missing} in the discovered set; got {sorted(found)}"


def test_discovery_is_deterministic(frontend: Frontend, tree: Path) -> None:
    assert _shape(list(frontend.discover(tree))) == _shape(list(frontend.discover(tree)))


def test_a_units_sources_are_relative_to_the_root(frontend: Frontend, tree: Path) -> None:
    """``Unit.sources`` is documented relative to the project root. An absolute
    path is a machine's, and it travels into Evidence."""
    absolute = [
        f"{u.uid}: {s}" for u in frontend.discover(tree) for s in u.sources if Path(s).is_absolute()
    ]
    assert not absolute, "sources must be relative to the root:\n" + "\n".join(absolute)


def test_reading_the_tree_does_not_change_it(frontend: Frontend, tree: Path) -> None:
    """Caching an expensive analysis is encouraged; putting the cache in
    somebody's source tree is not the same thing."""
    before = _fingerprint(tree)
    for unit in frontend.discover(tree):
        _analyzed(frontend, unit, tree)
    after = _fingerprint(tree)
    assert after == before, (
        "the source tree changed while being read: "
        f"added={sorted(set(after) - set(before))} "
        f"removed={sorted(set(before) - set(after))} "
        f"modified={sorted(k for k in set(before) & set(after) if before[k] != after[k])}"
    )


def test_analysis_is_deterministic(frontend: Frontend, tree: Path) -> None:
    for unit in frontend.discover(tree):
        first = _analyzed(frontend, unit, tree)
        assert _analyzed(frontend, unit, tree) == first, f"{unit.uid}: two analyses disagree"


def test_the_engines_own_workspace_is_not_source(
    frontend_case: Any, frontend: Frontend, tree: Path
) -> None:
    """``run_recipe`` writes under ``<root>/.recast``, and an oracle leaves
    compilable sources there. Reading them back means the same tree yields a
    different unit set before and after a run, and the second run offers to
    modernize the first one's scaffolding."""
    if frontend_case.plant_workspace_artifact is None:
        pytest.skip(
            f"{frontend_case.name!r} declares no artifact it would discover, so the "
            "suite has nothing to plant -- unexercised, not passed"
        )
    before = _shape(list(frontend.discover(tree)))
    workspace = tree / WORKSPACE_DIRNAME / "translate" / "oracle-conformance"
    workspace.mkdir(parents=True)
    frontend_case.plant_workspace_artifact(workspace)
    assert _shape(list(frontend.discover(tree))) == before, (
        f"a file under {WORKSPACE_DIRNAME}/ was discovered as source; the engine's "
        "output is now its own input"
    )


def test_preprocessing_records_its_flags(
    frontend_case: Any, frontend: Frontend, tree: Path
) -> None:
    """A translation is only reproducible if the preprocessor invocation is."""
    if not frontend_case.preprocesses:
        pytest.skip(f"{frontend_case.name!r} does not override preprocess; the default is identity")

    for unit in frontend.discover(tree):
        preprocessed = frontend.preprocess(unit, tree)
        facts = frontend.analyze(preprocessed, tree)
        assert facts.provenance, (
            f"{unit.uid}: preprocess ran and Facts.provenance is empty, so nothing "
            "records which flags produced the source that was analyzed"
        )
