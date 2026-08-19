"""Tests for the run engine.

Fake plugins throughout: the runner's job is order, fail-fast, optionality,
caching and record-keeping, and none of that needs a real parser or
compiler to prove. The real chain is exercised by the examples run in the
compiler-gated suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from recast.errors import ConfigError
from recast.model import Candidate, Confidence, Facts, OracleRef, Unit, Verdict
from recast.plugins.executor import Executor, Job, JobResult
from recast.plugins.frontend import Frontend
from recast.plugins.oracle import Oracle
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.store import EvidenceStore
from recast.plugins.transform import Transform
from recast.plugins.verifier import Verifier
from recast.registry import Registry
from recast.run import run_recipe

# --- a complete fake plugin set ----------------------------------------------


class FakeExecutor(Executor):
    name = "fake-exec"

    def submit(self, job: Job) -> str:
        return "h"

    def wait(self, handle: str, timeout_s: float | None = None) -> JobResult:
        return JobResult(0, "", "")


class FakeFrontend(Frontend):
    name = "fake-frontend"

    def discover(self, root: Path) -> list[Unit]:
        return [
            Unit(uid="fake:alpha", kind="module"),
            Unit(uid="fake:alpha/sub", kind="subprogram", parent="fake:alpha"),
            Unit(uid="fake:beta", kind="module"),
        ]

    def analyze(self, unit: Unit, root: Path) -> Facts:
        return Facts(unit=unit.uid, interface={"module": unit.uid.split(":")[1]})


class FakeTransform(Transform):
    name = "fake.transform"
    requires = ("interface",)

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        return unit.kind == "module"

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        return Candidate(unit=unit.uid, transform=self.name, files={Path("out.py"): b"x = 1\n"})


class FakeOracle(Oracle):
    name = "fake-oracle"
    materialized = 0

    def key(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> str:
        return "shared-key"  # every unit shares one reference on purpose

    def materialize(self, unit, facts, workspace, executor, config) -> OracleRef:
        type(self).materialized += 1
        return OracleRef(unit=unit.uid, oracle=self.name, key="shared-key", handle={"ok": True})


class PassVerifier(Verifier):
    name = "fake.pass"
    provides = Confidence.SAMPLED

    def verify(self, unit, candidate, oracle, workspace, executor, config) -> Verdict:
        return Verdict(
            unit=unit.uid,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=Confidence.SAMPLED,
            detail=f"saw oracle {oracle.oracle}",
        )


class FailVerifier(PassVerifier):
    name = "fake.fail"

    def verify(self, unit, candidate, oracle, workspace, executor, config) -> Verdict:
        return Verdict(
            unit=unit.uid,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=Confidence.FAILED,
            detail="deliberately",
        )


class MemoryStore(EvidenceStore):
    name = "fake-store"
    written: ClassVar[list[Any]] = []

    def __init__(self, **_config: Any) -> None:
        pass

    def put(self, evidence) -> str:
        type(self).written.append(evidence)
        return f"mem://{len(type(self).written)}"

    def get(self, uri: str):
        raise NotImplementedError

    def query(self, **selectors: Any):
        raise NotImplementedError


class FakeRecipe(Recipe):
    name = "fake"
    summary = "test recipe"

    def __init__(self, stages: list[Stage]) -> None:
        self._stages = stages

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return self._stages


def _registry() -> Registry:
    registry = Registry()
    registry.register("executor", "fake-exec", FakeExecutor)
    registry.register("frontend", "fake-frontend", FakeFrontend)
    registry.register("transform", "fake.transform", FakeTransform)
    registry.register("oracle", "fake-oracle", FakeOracle)
    registry.register("verifier", "fake.pass", PassVerifier)
    registry.register("verifier", "fake.fail", FailVerifier)
    registry.register("store", "fake-store", MemoryStore)
    # Entry-point discovery would add the real plugins; a fake registry must
    # not, so mark every kind as already discovered.
    registry._loaded.update(
        {
            "executor",
            "frontend",
            "transform",
            "oracle",
            "verifier",
            "store",
            "scanner",
            "adjudicator",
            "agent",
            "recipe",
        }
    )
    return registry


def _stages(*extra: Stage) -> list[Stage]:
    return [
        Stage("executor", "fake-exec"),
        Stage("frontend", "fake-frontend"),
        Stage("transform", "fake.transform"),
        *extra,
        Stage("store", "fake-store"),
    ]


@pytest.fixture(autouse=True)
def _fresh_counters() -> None:
    FakeOracle.materialized = 0
    MemoryStore.written = []


# --- selection and order -----------------------------------------------------


def test_top_level_units_run_and_subprograms_do_not(tmp_path: Path) -> None:
    run = run_recipe(
        FakeRecipe(_stages(Stage("verifier", "fake.pass", gate=True))),
        tmp_path,
        registry=_registry(),
    )
    assert [u.unit.uid for u in run.units] == ["fake:alpha", "fake:beta"]
    assert run.passed
    assert all(u.candidate is not None for u in run.units)


def test_an_unknown_requested_unit_is_an_error_not_a_silence(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        run_recipe(FakeRecipe(_stages()), tmp_path, {"units": ["fake:nope"]}, registry=_registry())


def test_a_missing_required_plugin_fails_before_any_work(tmp_path: Path) -> None:
    stages = _stages(Stage("verifier", "not-installed", gate=True))
    with pytest.raises(ConfigError, match="not-installed"):
        run_recipe(FakeRecipe(stages), tmp_path, registry=_registry())


def test_a_missing_optional_plugin_downgrades_and_says_so(tmp_path: Path) -> None:
    stages = _stages(Stage("verifier", "not-installed", optional=True))
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry())
    assert run.passed
    outcome = next(o for u in run.units for o in u.outcomes if o.plugin == "not-installed")
    assert outcome.status == "skipped"
    assert "optional" in outcome.detail


# --- gates -------------------------------------------------------------------


def test_a_failing_gate_stops_the_unit_and_is_still_recorded(tmp_path: Path) -> None:
    stages = _stages(
        Stage("verifier", "fake.fail", gate=True),
        Stage("verifier", "fake.pass", gate=True),
    )
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry())
    assert not run.passed
    for unit_run in run.units:
        assert unit_run.stopped_by == "fake.fail"
        # The gate after the failing one never ran -- fail fast, no retry.
        assert [v.verifier for v in unit_run.verdicts] == ["fake.fail"]
        # But the store stage still did: the failing Verdict is recorded,
        # because a gate that failed and vanished is a rumor.
        assert len(unit_run.evidence) == 1


def test_a_failing_non_gate_verifier_does_not_stop_the_unit(tmp_path: Path) -> None:
    stages = _stages(
        Stage("verifier", "fake.fail"),
        Stage("verifier", "fake.pass", gate=True),
    )
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry())
    for unit_run in run.units:
        assert [v.verifier for v in unit_run.verdicts] == ["fake.fail", "fake.pass"]
        # Both verdicts, including the failed one, reach the store: a gate
        # that failed and was recorded is audit trail.
        assert len(unit_run.evidence) == 2
    assert not run.passed  # a failed outcome anywhere fails the run report


# --- the oracle cache --------------------------------------------------------


def test_units_with_one_key_share_one_materialization(tmp_path: Path) -> None:
    stages = _stages(
        Stage("oracle", "fake-oracle"),
        Stage("verifier", "fake.pass", gate=True),
    )
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry())
    assert len(run.units) == 2
    assert FakeOracle.materialized == 1
    # ...and the verifier after the oracle stage received the reference.
    for unit_run in run.units:
        assert "fake-oracle" in unit_run.verdicts[0].detail


# --- evidence ----------------------------------------------------------------


def test_evidence_carries_the_recipe_the_digest_and_the_reference(tmp_path: Path) -> None:
    stages = _stages(
        Stage("oracle", "fake-oracle"),
        Stage("verifier", "fake.pass", gate=True),
    )
    run_recipe(FakeRecipe(stages), tmp_path, registry=_registry())
    evidence = MemoryStore.written[0]
    assert evidence.recipe == "fake"
    assert evidence.executor == "fake-exec"
    assert evidence.artifact["digest"]
    assert evidence.reference == {"oracle": "fake-oracle", "key": "shared-key"}
    assert evidence.environment["engine"].startswith("recast ")
    assert evidence.meta["timestamp"]


def test_the_cli_resolves_plugin_recipes_from_the_registry() -> None:
    """A domain package's recipe attaches through the same entry-point group
    as everything else; a CLI that only knew the builtins would make that
    attachment decorative."""
    from recast.cli import _recipe
    from recast.errors import RecastError
    from recast.registry import REGISTRY

    class PluginRecipe(FakeRecipe):
        name = "plugin-made"
        summary = "from a domain package"

        def __init__(self) -> None:
            super().__init__([])

    REGISTRY.register("recipe", "plugin-made", PluginRecipe, replace=True)
    assert _recipe("plugin-made").summary == "from a domain package"
    with pytest.raises(RecastError, match="plugin-made"):
        _recipe("no-such-recipe")  # the error names what IS known


# --- the verification summary ------------------------------------------------


def test_the_summary_is_stable_across_runs(tmp_path: Path) -> None:
    """A repository commits this like a lockfile, so two runs over the same
    revisions must produce the same bytes -- otherwise every invocation
    manufactures a diff and the diffs stop meaning anything."""
    stages = _stages(
        Stage("oracle", "fake-oracle"),
        Stage("verifier", "fake.pass", gate=True),
    )
    first = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry()).summary()
    second = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry()).summary()
    assert first == second


def test_the_summary_records_confidence_oracle_and_digest(tmp_path: Path) -> None:
    stages = _stages(
        Stage("oracle", "fake-oracle"),
        Stage("verifier", "fake.pass", gate=True),
    )
    summary = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry()).summary()
    assert summary["schema"] == 1
    assert summary["recipe"] == "fake"
    unit = next(u for u in summary["units"] if u["unit"] == "fake:alpha")
    assert unit["oracle"] == {"name": "fake-oracle", "key": "shared-key"}
    assert unit["deferred"] == 0
    assert len(unit["candidate"]) == 64  # the artifact digest, not a path
    verdict = unit["verdicts"][0]
    assert (verdict["verifier"], verdict["confidence"], verdict["passed"]) == (
        "fake.pass",
        "sampled",
        True,
    )


def test_the_summary_carries_a_failed_gate_rather_than_hiding_it(tmp_path: Path) -> None:
    stages = _stages(Stage("verifier", "fake.fail", gate=True))
    summary = run_recipe(FakeRecipe(stages), tmp_path, registry=_registry()).summary()
    unit = summary["units"][0]
    assert unit["stopped_by"] == "fake.fail"
    assert unit["verdicts"][0]["confidence"] == "failed"
    assert unit["verdicts"][0]["passed"] is False


def test_the_summary_excludes_wall_clock_and_paths(tmp_path: Path) -> None:
    """Timestamps and absolute paths would change every run and on every
    machine; a committed record of them is noise, and the manifests in the
    evidence store are where the per-run provenance lives."""
    import json

    stages = _stages(Stage("verifier", "fake.pass", gate=True))
    blob = json.dumps(run_recipe(FakeRecipe(stages), tmp_path, registry=_registry()).summary())
    assert str(tmp_path) not in blob
    assert "timestamp" not in blob
