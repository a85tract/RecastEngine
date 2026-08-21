"""Rules about the run itself, checked by driving the engine with doubles.

These are not about any one plugin, which is why they take no case fixture and
run whatever set is selected. They are about what the runner does *around*
plugins -- and the thing most worth pinning is what it does when a gate fails,
because the tempting behaviour is the wrong one.

Feeding a gate's own numbers back to the thing being gated turns the Oracle
into a fitness function: the candidate gets fitted to the cases the gate
happens to sample, and the verdict stops being evidence about anything else.
So there is no retry, and no stage re-runs because a later one failed.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from recast.conformance.doubles import (
    CountingTransform,
    FailingVerifier,
    GateFailsRecipe,
    RecordingEvidenceStore,
    StubFrontend,
    TwoFrontendsRecipe,
)
from recast.errors import ConfigError
from recast.executors.local import factory as local_executor
from recast.model import Confidence
from recast.registry import Registry
from recast.run import RecipeRun, run_recipe


@pytest.fixture
def failed_gate_run(tmp_path: Path) -> Iterator[RecipeRun]:
    """One run of a recipe whose gate always fails, over two units."""
    CountingTransform.reset()
    RecordingEvidenceStore.reset()

    registry = Registry()
    registry.register("executor", "local", local_executor)
    registry.register("frontend", StubFrontend.name, StubFrontend)
    registry.register("transform", CountingTransform.name, CountingTransform)
    registry.register("verifier", FailingVerifier.name, FailingVerifier)
    registry.register("store", RecordingEvidenceStore.name, RecordingEvidenceStore)

    yield run_recipe(GateFailsRecipe(), tmp_path, {"workspace": tmp_path / "ws"}, registry=registry)

    CountingTransform.reset()
    RecordingEvidenceStore.reset()


def test_a_failed_gate_does_not_drive_a_retry(failed_gate_run: RecipeRun) -> None:
    assert failed_gate_run.units, "the run walked no units, so it checked nothing"
    for unit_run in failed_gate_run.units:
        calls = CountingTransform.calls.get(unit_run.unit.uid, 0)
        assert calls == 1, (
            f"{unit_run.unit.uid}: the transform ran {calls} times under a failing gate; "
            "a verdict must never flow back into the thing it judged"
        )


def test_a_failed_gate_stops_the_unit(failed_gate_run: RecipeRun) -> None:
    assert not failed_gate_run.passed
    for unit_run in failed_gate_run.units:
        assert unit_run.stopped_by == FailingVerifier.name
        assert [v.confidence for v in unit_run.verdicts] == [Confidence.FAILED]


def test_a_failed_gate_is_still_recorded(failed_gate_run: RecipeRun) -> None:
    """A gate that failed and was recorded is audit trail. One that vanished is a rumor."""
    recorded = {evidence.unit for evidence in RecordingEvidenceStore.written}
    assert recorded == {unit_run.unit.uid for unit_run in failed_gate_run.units}
    assert all(
        evidence.verdict.confidence is Confidence.FAILED
        for evidence in RecordingEvidenceStore.written
    )
    assert all(unit_run.evidence for unit_run in failed_gate_run.units), (
        "the runner recorded evidence the unit does not know about"
    )


def test_every_declared_frontend_contributes_its_units(tmp_path: Path) -> None:
    """A recipe may declare several, and they do not chain.

    Each reads the tree on its own, the Unit sets union, and each Unit is
    analyzed by the frontend that found it -- which is what walks a project
    written in more than one language in one run. Handing one frontend's Unit
    to another is how a C file gets analyzed as Fortran.
    """
    CountingTransform.reset()
    first = StubFrontend.claiming("conformance.first", "one:a", "one:b")
    second = StubFrontend.claiming("conformance.second", "two:a")

    registry = Registry()
    registry.register("executor", "local", local_executor)
    registry.register("frontend", first.name, first)
    registry.register("frontend", second.name, second)
    registry.register("transform", CountingTransform.name, CountingTransform)

    run = run_recipe(
        TwoFrontendsRecipe(),
        tmp_path,
        {"workspace": tmp_path / "ws", "frontends": [first.name, second.name]},
        registry=registry,
    )
    assert [u.unit.uid for u in run.units] == ["one:a", "one:b", "two:a"]
    owners = {u.unit.uid: [o.plugin for o in u.outcomes if o.kind == "frontend"] for u in run.units}
    assert owners == {
        "one:a": [first.name],
        "one:b": [first.name],
        "two:a": [second.name],
    }
    CountingTransform.reset()


def test_two_frontends_claiming_one_unit_is_refused(tmp_path: Path) -> None:
    """First-wins would make the run reproducible only by declaration order:
    the Unit carries one frontend's Facts and nothing records whose."""
    first = StubFrontend.claiming("conformance.first", "shared:a")
    second = StubFrontend.claiming("conformance.second", "shared:a")

    registry = Registry()
    registry.register("executor", "local", local_executor)
    registry.register("frontend", first.name, first)
    registry.register("frontend", second.name, second)
    registry.register("transform", CountingTransform.name, CountingTransform)

    with pytest.raises(ConfigError, match="both discovered 'shared:a'"):
        run_recipe(
            TwoFrontendsRecipe(),
            tmp_path,
            {"workspace": tmp_path / "ws", "frontends": [first.name, second.name]},
            registry=registry,
        )
