"""Tests for the run engine.

Fake plugins throughout: the runner's job is order, fail-fast, optionality,
caching and record-keeping, and none of that needs a real parser or
compiler to prove. The real chain is exercised by the examples run in the
compiler-gated suite.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from recast import WORKSPACE_DIRNAME
from recast.errors import ConfigError
from recast.model import (
    Candidate,
    Confidence,
    Disclosure,
    Facts,
    Finding,
    OracleRef,
    Severity,
    Unit,
    Verdict,
)
from recast.plugins.executor import Executor, Job, JobResult
from recast.plugins.frontend import Frontend
from recast.plugins.oracle import Oracle
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.scanner import Adjudicator, Scanner
from recast.plugins.store import EvidenceStore, FindingStore
from recast.plugins.transform import Transform
from recast.plugins.verifier import Verifier
from recast.registry import KINDS, Registry
from recast.run import _NOT_STAGES, _NOT_STEPS, _STEPS, run_recipe

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
    # The oracle's name, not its cache key: the key folds in the compiler's
    # version and flags, so two machines that verified the same claim would
    # disagree about it and every CI run would report a diff.
    assert unit["oracle"] == "fake-oracle"
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


def test_the_summary_excludes_what_differs_between_machines(tmp_path: Path) -> None:
    """Timestamps, paths, and the oracle's build-specific cache key all vary
    by machine or by run. A committed record of them manufactures a diff on
    every invocation, and the manifests in the evidence store are where the
    per-run provenance belongs."""
    import json

    stages = _stages(
        Stage("oracle", "fake-oracle"),
        Stage("verifier", "fake.pass", gate=True),
    )
    blob = json.dumps(run_recipe(FakeRecipe(stages), tmp_path, registry=_registry()).summary())
    assert str(tmp_path) not in blob
    assert "timestamp" not in blob
    assert "shared-key" not in blob  # the oracle's key is provenance, not a claim


# --- more than one frontend ---------------------------------------------------


class OtherFrontend(Frontend):
    """A second language in the same tree. Independent of the first: it reads
    the source itself and never sees what the other one learned."""

    name = "other-frontend"

    def discover(self, root: Path) -> list[Unit]:
        return [Unit(uid="other:gamma", kind="module")]

    def analyze(self, unit: Unit, root: Path) -> Facts:
        return Facts(unit=unit.uid, interface={"module": "gamma", "by": self.name})


class ColludingFrontend(OtherFrontend):
    """Claims a uid the first frontend already discovered."""

    name = "colliding-frontend"

    def discover(self, root: Path) -> list[Unit]:
        return [Unit(uid="fake:alpha", kind="module")]


def _multi_registry() -> Registry:
    registry = _registry()
    registry.register("frontend", "other-frontend", OtherFrontend)
    registry.register("frontend", "colliding-frontend", ColludingFrontend)
    return registry


def _multi_stages(*frontends: str) -> list[Stage]:
    return [
        Stage("executor", "fake-exec"),
        *[Stage("frontend", name) for name in frontends],
        Stage("transform", "fake.transform"),
        Stage("store", "fake-store"),
    ]


def test_every_declared_frontend_contributes_its_units(tmp_path: Path) -> None:
    """The union, which is what lets one project hold more than one language.

    Before this, the runner took the first frontend stage and dropped the rest
    silently -- the second language's units never appeared and nothing said so.
    """
    recipe = FakeRecipe(_multi_stages("fake-frontend", "other-frontend"))
    run = run_recipe(recipe, tmp_path, {}, registry=_multi_registry())
    assert [u.unit.uid for u in run.units] == ["fake:alpha", "fake:beta", "other:gamma"]


def test_each_unit_is_analyzed_by_the_frontend_that_found_it(tmp_path: Path) -> None:
    """Ownership, not declaration order: handing one frontend's Unit to another
    is how a C file gets analyzed as Fortran."""
    recipe = FakeRecipe(_multi_stages("fake-frontend", "other-frontend"))
    run = run_recipe(recipe, tmp_path, {}, registry=_multi_registry())
    owners = {
        unit_run.unit.uid: [o.plugin for o in unit_run.outcomes if o.kind == "frontend"]
        for unit_run in run.units
    }
    assert owners == {
        "fake:alpha": ["fake-frontend"],
        "fake:beta": ["fake-frontend"],
        "other:gamma": ["other-frontend"],
    }


def test_two_frontends_claiming_one_unit_is_refused(tmp_path: Path) -> None:
    """First-wins would make the run reproducible only by accident of
    declaration order: the Unit carries one of their Facts and nothing records
    which."""
    recipe = FakeRecipe(_multi_stages("fake-frontend", "colliding-frontend"))
    with pytest.raises(ConfigError, match="both discovered 'fake:alpha'"):
        run_recipe(recipe, tmp_path, {}, registry=_multi_registry())


def test_declaring_one_frontend_twice_is_refused(tmp_path: Path) -> None:
    """Frontends do not chain, so a repeat is a duplicate or a misunderstanding."""
    recipe = FakeRecipe(_multi_stages("fake-frontend", "fake-frontend"))
    with pytest.raises(ConfigError, match="more than once"):
        run_recipe(recipe, tmp_path, {}, registry=_multi_registry())


def test_a_named_unit_still_resolves_across_frontends(tmp_path: Path) -> None:
    recipe = FakeRecipe(_multi_stages("fake-frontend", "other-frontend"))
    run = run_recipe(recipe, tmp_path, {"units": ["other:gamma"]}, registry=_multi_registry())
    assert [u.unit.uid for u in run.units] == ["other:gamma"]


# --- findings: scanners, adjudicators, and the store they belong in -----------
#
# The runner walked none of these until now: ``_walk_stage`` fell through to
# ``skipped`` for both kinds, so the ``audit`` recipe reported that every unit
# passed while its scanners were never called. These tests are the difference
# between that and a gate.


class FakeScanner(Scanner):
    name = "fake.scan"
    family = "audit"

    def scan(self, unit: Unit, facts: Facts, workspace: Path, config: dict[str, Any]):
        for n in range(2):
            yield Finding(
                uid=f"{unit.uid.replace(':', '_')}-{n}",
                unit=unit.uid,
                scanner=self.name,
                title=f"plausible thing {n}",
                severity=Severity.HIGH,
            )


class SilentScanner(Scanner):
    """A clean scan. Distinct from a scan that did not happen -- which is
    finding 5 in P5's list, and is not what this file can settle."""

    name = "fake.silent"

    def scan(self, unit: Unit, facts: Facts, workspace: Path, config: dict[str, Any]):
        return []


class ConfirmingAdjudicator(Adjudicator):
    name = "fake.confirm"
    verdict: ClassVar[Disclosure] = Disclosure.CONFIRMED

    def adjudicate(self, finding: Finding, workspace: Path, config: dict[str, Any]) -> Finding:
        finding.disclosure = type(self).verdict
        return finding


class RefutingAdjudicator(ConfirmingAdjudicator):
    name = "fake.refute"
    verdict: ClassVar[Disclosure] = Disclosure.REFUTED


class NeverAdjudicator(Adjudicator):
    name = "fake.never"

    def adjudicate(self, finding: Finding, workspace: Path, config: dict[str, Any]) -> Finding:
        raise AssertionError("adjudicated a finding that does not exist")


class MemoryFindingStore(FindingStore):
    name = "fake-findings"
    written: ClassVar[list[Finding]] = []
    roots: ClassVar[list[Path]] = []

    def __init__(self, **config: Any) -> None:
        type(self).roots.append(Path(config["root"]))

    def put(self, finding: Finding) -> str:
        self.guard(finding)
        type(self).written.append(finding)
        return f"finding://{finding.uid}"

    def get(self, uid: str) -> Finding:
        raise NotImplementedError

    def query(self, **selectors: Any):
        raise NotImplementedError


def _audit_registry() -> Registry:
    registry = _registry()
    registry.register("scanner", "fake.scan", FakeScanner)
    registry.register("scanner", "fake.silent", SilentScanner)
    registry.register("adjudicator", "fake.confirm", ConfirmingAdjudicator)
    registry.register("adjudicator", "fake.refute", RefutingAdjudicator)
    registry.register("adjudicator", "fake.never", NeverAdjudicator)
    registry.register("store", "fake-findings", MemoryFindingStore)
    return registry


def _audit_stages(*middle: Stage) -> list[Stage]:
    """The audit shape: no executor, no transform, findings out rather than
    candidates."""
    return [
        Stage("frontend", "fake-frontend"),
        Stage("scanner", "fake.scan"),
        *middle,
        Stage("store", "fake-findings"),
    ]


@pytest.fixture(autouse=True)
def _fresh_finding_counters() -> None:
    MemoryFindingStore.written = []
    MemoryFindingStore.roots = []


def test_a_scanner_stage_is_walked_rather_than_reported_as_skipped(tmp_path: Path) -> None:
    run = run_recipe(FakeRecipe(_audit_stages()), tmp_path, registry=_audit_registry())
    outcome = next(o for o in run.units[0].outcomes if o.kind == "scanner")
    assert outcome.status == "ok"
    assert outcome.detail == "2 finding(s)"
    assert [f.title for f in run.units[0].findings] == ["plausible thing 0", "plausible thing 1"]


def test_a_plausible_finding_alone_does_not_fail_the_run(tmp_path: Path) -> None:
    """Precision is the adjudicator's job. A scanner that yields freely -- which
    ``Scanner.scan`` asks it to -- would otherwise fail every run it improved."""
    run = run_recipe(FakeRecipe(_audit_stages()), tmp_path, registry=_audit_registry())
    assert run.passed


def test_an_adjudicator_revises_the_findings_it_was_given(tmp_path: Path) -> None:
    stages = _audit_stages(Stage("adjudicator", "fake.refute", gate=True))
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_audit_registry())
    outcome = next(o for o in run.units[0].outcomes if o.kind == "adjudicator")
    assert outcome.status == "ok"
    assert outcome.detail == "2 refuted"
    assert all(f.disclosure is Disclosure.REFUTED for f in run.units[0].findings)
    assert run.passed


def test_a_confirmed_finding_fails_the_gate_it_was_declared_as(tmp_path: Path) -> None:
    """An audit that reports what it found and passes anyway is a report, not a
    gate -- and reporting all-passed was the whole defect."""
    stages = _audit_stages(Stage("adjudicator", "fake.confirm", gate=True))
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_audit_registry())
    assert not run.passed
    assert run.units[0].stopped_by == "fake.confirm"


def test_a_failed_adjudication_gate_still_records_what_it_found(tmp_path: Path) -> None:
    """Same rule as a failed verifier gate: the store stages still run. A
    confirmed vulnerability that stopped the unit and then vanished is the one
    record nobody can afford to lose."""
    stages = _audit_stages(Stage("adjudicator", "fake.confirm", gate=True))
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_audit_registry())
    assert len(MemoryFindingStore.written) == 2 * len(run.units)
    assert all(f.disclosure is Disclosure.CONFIRMED for f in MemoryFindingStore.written)
    assert run.units[0].records == ["finding://fake_alpha-0", "finding://fake_alpha-1"]


def test_an_adjudicator_with_nothing_to_adjudicate_is_not_called(tmp_path: Path) -> None:
    stages = [
        Stage("frontend", "fake-frontend"),
        Stage("scanner", "fake.silent"),
        Stage("adjudicator", "fake.never", gate=True),
        Stage("store", "fake-findings"),
    ]
    run = run_recipe(FakeRecipe(stages), tmp_path, registry=_audit_registry())
    outcome = next(o for o in run.units[0].outcomes if o.kind == "adjudicator")
    assert outcome.status == "skipped"
    assert outcome.detail == "no findings to adjudicate"
    assert run.passed


# --- the two stores are two stores -------------------------------------------


def _both_stores_stages() -> list[Stage]:
    return [
        Stage("executor", "fake-exec"),
        Stage("frontend", "fake-frontend"),
        Stage("transform", "fake.transform"),
        Stage("scanner", "fake.scan"),
        Stage("verifier", "fake.pass", gate=True),
        Stage("store", "fake-store"),
        Stage("store", "fake-findings"),
    ]


def test_findings_go_to_the_finding_store_and_verdicts_to_the_evidence_store(
    tmp_path: Path,
) -> None:
    """Walking a FindingStore over ``verdicts`` recorded nothing and reported
    success. The two stores hold different things under different access rules,
    so the runner picks by the store's kind, not by the stage's position."""
    run = run_recipe(FakeRecipe(_both_stores_stages()), tmp_path, registry=_audit_registry())
    assert run.passed
    unit_run = run.units[0]
    assert [type(w).__name__ for w in MemoryStore.written] == ["Evidence"] * len(run.units)
    assert len(MemoryFindingStore.written) == 2 * len(run.units)
    assert len(unit_run.evidence) == 1
    assert len(unit_run.records) == 2
    assert not set(unit_run.evidence) & set(unit_run.records)


def test_each_store_reports_what_it_actually_recorded(tmp_path: Path) -> None:
    run = run_recipe(FakeRecipe(_both_stores_stages()), tmp_path, registry=_audit_registry())
    details = [o.detail for o in run.units[0].outcomes if o.kind == "store"]
    assert details == ["1 verdict(s) recorded", "2 finding(s) recorded"]


def test_the_finding_store_is_not_rooted_where_the_evidence_store_is(tmp_path: Path) -> None:
    """A FindingStore may not share the evidence directory. Findings default to
    ``Access.EMBARGOED``, and ``FilesystemFindingStore`` refuses a root that is
    readable by anyone but its owner -- which the evidence directory is."""
    run_recipe(FakeRecipe(_both_stores_stages()), tmp_path, registry=_audit_registry())
    assert MemoryFindingStore.roots[0].name == "findings"
    assert MemoryFindingStore.roots[0].parent == tmp_path / WORKSPACE_DIRNAME


def test_the_summary_says_nothing_about_findings(tmp_path: Path) -> None:
    """This file is written to be committed. A count is still a statement about
    embargoed material, so it is absent along with the findings themselves."""
    run = run_recipe(FakeRecipe(_audit_stages()), tmp_path, registry=_audit_registry())
    assert run.units[0].findings
    rendered = json.dumps(run.summary())
    assert "finding" not in rendered
    assert "plausible thing" not in rendered


# --- what a recipe has to declare, and what it may not -----------------------


def test_a_recipe_that_neither_materializes_nor_awards_needs_no_executor(
    tmp_path: Path,
) -> None:
    """``Stage`` asks for an executor from "a recipe that materializes an oracle
    or awards a verdict". Demanding one unconditionally made ``audit``
    unrunnable as shipped."""
    run = run_recipe(FakeRecipe(_audit_stages()), tmp_path, registry=_audit_registry())
    assert run.passed
    assert not any(o.kind == "executor" for o in run.units[0].outcomes)


def test_a_recipe_that_awards_a_verdict_still_needs_an_executor(tmp_path: Path) -> None:
    stages = [
        Stage("frontend", "fake-frontend"),
        Stage("transform", "fake.transform"),
        Stage("verifier", "fake.pass", gate=True),
        Stage("store", "fake-store"),
    ]
    with pytest.raises(ConfigError, match="no executor stage"):
        run_recipe(FakeRecipe(stages), tmp_path, registry=_audit_registry())


def test_a_stage_kind_the_runner_does_not_walk_is_refused_before_any_work(
    tmp_path: Path,
) -> None:
    """It used to answer ``skipped`` -- the word an uninstalled optional plugin
    gets -- so installed-and-unwalked and absent were indistinguishable."""
    stages = _stages(Stage("mystery", "whatever"))
    with pytest.raises(ConfigError, match="does not walk"):
        run_recipe(FakeRecipe(stages), tmp_path, registry=_audit_registry())


def test_a_kind_that_is_not_a_stage_at_all_is_refused_rather_than_ignored(
    tmp_path: Path,
) -> None:
    """An ``agent`` is consulted by a non-deterministic Transform rather than
    scheduled, so declaring one as a stage is a misunderstanding, not a no-op."""
    registry = _audit_registry()
    registry.register("agent", "fake-agent", object)
    with pytest.raises(ConfigError, match="does not walk"):
        run_recipe(FakeRecipe(_stages(Stage("agent", "fake-agent"))), tmp_path, registry=registry)


def test_an_unknown_kind_is_named_as_a_kind_and_not_as_a_missing_plugin(
    tmp_path: Path,
) -> None:
    """Availability is checked second on purpose: a kind nothing registers has
    no plugins either, so checking that first names the wrong problem."""
    with pytest.raises(ConfigError, match="stage kind"):
        run_recipe(
            FakeRecipe(_stages(Stage("mystery", "whatever"))), tmp_path, registry=_audit_registry()
        )


def test_every_registered_kind_is_a_step_a_non_step_or_not_a_stage() -> None:
    """The three sets partition ``KINDS``, so a tenth kind cannot be added
    without deciding which it is -- the decision this whole section is about."""
    assert _STEPS | _NOT_STEPS | _NOT_STAGES == set(KINDS)
    assert not _STEPS & _NOT_STEPS
    assert not _STEPS & _NOT_STAGES
    assert not _NOT_STEPS & _NOT_STAGES
