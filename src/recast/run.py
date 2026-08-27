"""The run engine: walk a recipe's stages and leave Evidence behind.

This is the piece that turns eight passing plugins into one command. A
Recipe declares which plugin fills each slot; this module drives the spine
-- discover, analyze, transform, verify, record -- and owns exactly the
decisions that belong to no plugin:

* Order and fail-fast. Stages run in the recipe's order. A gating Verifier
  that fails stops the Unit there -- no retry, no re-run of earlier stages,
  because a Verdict never flows back into a Transform (see ``Stage.gate``).
* Optionality. A missing optional plugin downgrades the run and says so; a
  missing required one fails before any work starts.
* The oracle cache. ``Oracle.key`` folds in everything that can move the
  reference, so two Units whose facts hash the same share one build.
* Evidence. Every Verdict awarded becomes one Evidence record in the store
  -- the model's rule is that a Candidate without Evidence is a draft, and
  the runner is where that rule gets teeth. Verdicts are recorded whether
  they passed or failed: a gate that failed and was recorded is audit trail,
  a gate that failed and vanished is a rumor.

The runner is deliberately ignorant of what any stage computes. It knows the
contracts, the order, and what to write down.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shutil
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from recast import OUTPUT_DIRNAME, WORKSPACE_DIRNAME, __version__
from recast.errors import ConfigError, RecastError, ScannerUnavailable
from recast.model import (
    Candidate,
    Disclosure,
    Evidence,
    Facts,
    Finding,
    OracleRef,
    Severity,
    Unit,
    Verdict,
)
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.store import FindingStore
from recast.registry import REGISTRY, Registry

__all__ = ["RecipeRun", "RunStatus", "UnitRun", "missing_tools", "run_recipe"]

NO_ORACLE = OracleRef(unit="", oracle="none", key="", handle=None)
"""What a Verifier receives before any Oracle has materialized. Static
verifiers ignore it; an oracle-backed verifier handed this fails closed on
its own rules, which is the correct outcome for a recipe ordered wrongly."""


class RunStatus(StrEnum):
    """What a run is entitled to claim.

    Three states rather than a boolean, because the two ways of not passing are
    not the same answer and folding them together is how "nothing ran" comes
    out as "nothing wrong". A ``FAILED`` run checked something and did not like
    it. An ``INCOMPLETE`` run did not check.

    ``INCOMPLETE`` is not a pass: ``passed`` is False for it, so anything
    already gating on ``passed`` keeps gating correctly without being told
    about this enum. What the enum buys is the ability to *say which*, in the
    exit status, in the last line of ``recast run``, and in a waiver.

    When a run is both, it reports ``INCOMPLETE``. See ``_SEVERITY``.
    """

    PASSED = "passed"
    INCOMPLETE = "incomplete"
    FAILED = "failed"


# Worst wins when a run is summarized from its units, and a unit from its
# stages. Ordered rather than compared by name, so adding a state is a
# deliberate placement rather than an accident of spelling.
#
# INCOMPLETE outranks FAILED, which is the ordering `hpc-devsecops` has been
# running in production with and states in its SECURITY.md: incomplete exits 2,
# findings exit 1, only a completed clean check exits 0. The argument is that a
# run with a check that did not complete has an incomplete findings list, so
# announcing the findings implies a completeness the run does not have -- the
# operator's next move is to make the run complete, not to read a list that may
# be missing the worst entry. Both are non-zero either way; what this decides is
# the headline and which of the two non-zero codes a caller sees.
_SEVERITY = {RunStatus.PASSED: 0, RunStatus.FAILED: 1, RunStatus.INCOMPLETE: 2}


@dataclass
class StageOutcome:
    """What one stage did for one unit."""

    kind: str
    plugin: str
    status: str
    """``ok`` | ``failed`` | ``skipped`` | ``incomplete``.

    ``skipped`` and ``incomplete`` are deliberately different words. Skipped is
    a plugin the operator left out of the environment and declared optional;
    incomplete is a plugin that is installed, was asked, and could not answer.
    Reusing one word for both is what let a scanner that never ran look like a
    scanner that found nothing.
    """

    detail: str = ""

    waived: bool = False
    """True when ``incomplete`` was allowed for this plugin by config.

    The outcome still says ``incomplete`` and still prints. A waiver changes
    what the *run* concludes, never what the stage reports -- a waiver that
    edited the record would be indistinguishable from the stage having run.
    """


@dataclass
class UnitRun:
    """Everything that happened to one Unit."""

    unit: Unit
    outcomes: list[StageOutcome] = field(default_factory=list)
    candidate: Candidate | None = None
    verdicts: list[Verdict] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    """What the Scanners found, as the Adjudicator left them.

    Kept beside ``verdicts`` rather than merged into them, because they answer
    different questions and go to different stores under different access
    rules. A run has one or the other in practice: the ``audit`` recipe fills
    this, the modernization recipes fill ``verdicts``.
    """

    evidence: list[str] = field(default_factory=list)
    """Store URIs, one per recorded Verdict."""

    records: list[str] = field(default_factory=list)
    """FindingStore URIs, one per recorded Finding. Separate from
    ``evidence`` because the two stores have different access classes and
    conflating their URIs would lose that."""

    oracle: OracleRef = field(default_factory=lambda: NO_ORACLE)
    """The reference this unit's oracle-backed verifiers compare against."""

    stopped_by: str | None = None
    """The stage that ended this unit's run early, if any."""

    @property
    def status(self) -> RunStatus:
        """Worst of what happened here, by ``_SEVERITY``.

        Collected and ranked rather than returned from the first matching
        branch, so this ordering and the run's are the same one table. Written
        as a chain of ``if``s it was not: flipping ``_SEVERITY`` left this
        answering by declaration order, which is exactly the drift the table
        exists to prevent.
        """
        states = {RunStatus.PASSED}
        if self.stopped_by is not None or any(o.status == "failed" for o in self.outcomes):
            states.add(RunStatus.FAILED)
        if any(o.status == "incomplete" and not o.waived for o in self.outcomes):
            states.add(RunStatus.INCOMPLETE)
        return max(states, key=lambda s: _SEVERITY[s])

    @property
    def passed(self) -> bool:
        return self.status is RunStatus.PASSED


@dataclass
class RecipeRun:
    """One invocation of ``run_recipe``: the units and the shared context."""

    recipe: str
    root: Path
    workspace: Path
    units: list[UnitRun] = field(default_factory=list)

    @property
    def status(self) -> RunStatus:
        """The worst of the units, and ``INCOMPLETE`` when there are none.

        A run that discovered nothing to walk used to report the same False as
        a run whose gate failed. It has not checked anything, which is exactly
        what ``INCOMPLETE`` is for -- and the usual cause is a frontend pointed
        at the wrong tree, which is a fact about the invocation rather than
        about the code.
        """
        if not self.units:
            return RunStatus.INCOMPLETE
        return max((u.status for u in self.units), key=lambda s: _SEVERITY[s])

    @property
    def passed(self) -> bool:
        return self.status is RunStatus.PASSED

    def summary(self) -> dict[str, Any]:
        """The run's verification status, as a record worth committing.

        Distinct from the Evidence manifests, and the distinction matters. A
        manifest is one immutable record of one run, content-addressed and
        append-only -- an audit trail, which accumulates a file per attempt
        including the attempts that failed. This is the *current state*: one
        entry per unit and verifier, regenerated rather than appended, so a
        repository can commit it like a lockfile and a diff means something --
        a confidence that dropped, an oracle that moved, a unit that stopped
        being covered.

        Deliberately excludes wall-clock time, paths, and the oracle's cache
        key -- the key folds in the compiler's identity and flags, so two
        machines that verified the same claim would disagree about it. Two runs
        over the same revisions produce the same summary on any machine, which
        is what lets CI regenerate it and fail on a diff; when a number does
        change, the change is the finding rather than the weather.

        The cost of that stability is worth naming. Every confidence in here is
        a claim *relative to an environment* -- a different compiler, libm or
        device can cost a bit-exact verdict without anything about the candidate
        changing -- and this file is the one place that deliberately leaves the
        environment out. So read it as an index of what was claimed, not as the
        claim in full: the conditions are in the Evidence manifest, which
        records them per run and is not diffed against anything.
        """
        return {
            "schema": 1,
            "recipe": self.recipe,
            "units": [
                {
                    "unit": unit_run.unit.uid,
                    "candidate": unit_run.candidate.digest() if unit_run.candidate else None,
                    "transform": unit_run.candidate.transform if unit_run.candidate else None,
                    "deferred": len(unit_run.candidate.deferred) if unit_run.candidate else None,
                    # The oracle's *name*, not its cache key. The key folds in
                    # the compiler's version and flags, so it legitimately
                    # differs between two machines that both verified the same
                    # claim -- which would make this file unstable and its
                    # diffs meaningless. Build-specific provenance belongs to
                    # the run's Evidence manifest, which records all of it.
                    "oracle": unit_run.oracle.oracle if unit_run.oracle.key else None,
                    "stopped_by": unit_run.stopped_by,
                    # Findings are deliberately absent, including their count.
                    # This file is written to be committed; a Finding defaults
                    # to Access.EMBARGOED, and SECURITY.md is explicit that
                    # nothing reaches a public store or a CI log before
                    # disclosure completes. A count is still a statement about
                    # embargoed material. The FindingStore holds them, and
                    # holds them at 0700.
                    "verdicts": [
                        {
                            "verifier": verdict.verifier,
                            "confidence": verdict.confidence.value,
                            "passed": verdict.passed,
                            "metrics": {
                                key: value
                                for key, value in sorted(verdict.metrics.items())
                                if isinstance(value, (int, float, str, bool))
                            },
                            "detail": verdict.detail,
                        }
                        for verdict in unit_run.verdicts
                    ],
                }
                for unit_run in sorted(self.units, key=lambda u: u.unit.uid)
            ],
        }


def run_recipe(
    recipe: Recipe,
    root: Path,
    config: dict[str, Any] | None = None,
    *,
    registry: Registry = REGISTRY,
) -> RecipeRun:
    """Walk ``recipe`` over the source tree at ``root``.

    ``config`` carries the operator's tables. Two conventions: top-level keys
    are the run's (``units``, ``workspace``), and ``config["stages"][plugin]``
    is merged over the recipe's own per-stage config -- ranges, dims, setup
    calls, waivers all arrive that way, because they are facts about a target,
    not about the engine.
    """
    config = dict(config or {})
    # Absolute from the start: stages hand paths to jobs that run elsewhere,
    # and a relative path is only meaningful in the directory it was typed in.
    root = Path(root).resolve()
    problems = recipe.validate(config)
    if problems:
        raise ConfigError(f"recipe {recipe.name!r} config: " + "; ".join(problems))

    stages = recipe.stages(config)
    _require_walkable(recipe, stages)
    _require_available(stages, registry)
    _require_one_transform(recipe, stages)
    waived = _waived(recipe, stages, config)
    _require_valid_bars(recipe, stages, config)

    output = output_root(root, config)
    workspace = Path(config.get("workspace") or output / recipe.name)
    workspace.mkdir(parents=True, exist_ok=True)
    run = RecipeRun(recipe=recipe.name, root=root, workspace=workspace)

    def stage_config(stage: Stage) -> dict[str, Any]:
        """The operator's per-stage table over the recipe's own."""
        return {**stage.config, **config.get("stages", {}).get(stage.plugin, {})}

    def call_config(stage: Stage) -> dict[str, Any]:
        """Per-call stages also learn where the source tree is; a plugin that
        is *constructed* from config (executor, frontend, store) takes paths
        per call instead, so its constructor sees only its own keys.

        ``range`` rides along the same way: a revision range is a fact about
        the invocation -- this push, these commits -- and a scanner that can
        scope to one reads it, while one that cannot (composition describes
        the whole tree regardless, by design) ignores it."""
        invocation = {"range": config["range"]} if config.get("range") else {}
        return {"root": root, "output": output, **invocation, **stage_config(stage)}

    # An executor stage is not a step: it declares the executor every stage
    # that runs something receives as an argument. The requirement is
    # conditional on there being such a stage -- a recipe of frontend and
    # store alone needs none -- and the set of kinds that are handed one is
    # exactly ``_HANDED_AN_EXECUTOR``, which is the contract's list rather
    # than the runner's guess. Scanners and adjudicators joined it the day the
    # in-tree gitleaks wrapper was found calling subprocess because the
    # contract had given it nothing else.
    executor_stage = next((s for s in stages if s.kind == "executor"), None)
    handed = sorted({s.kind for s in stages if s.kind in _HANDED_AN_EXECUTOR})
    if executor_stage is None and handed:
        raise ConfigError(
            f"recipe {recipe.name!r} declares {handed} stage(s), which are handed an "
            "executor, but declares no executor stage"
        )
    executor = (
        registry.get("executor", executor_stage.plugin)(**stage_config(executor_stage))
        if executor_stage is not None
        else None
    )
    executor_name = executor_stage.plugin if executor_stage is not None else ""

    frontend_stages = [s for s in stages if s.kind == "frontend"]
    if not frontend_stages:
        raise ConfigError(f"recipe {recipe.name!r} declares no frontend stage")
    repeated = sorted(
        {
            s.plugin
            for s in frontend_stages
            if [t.plugin for t in frontend_stages].count(s.plugin) > 1
        }
    )
    if repeated:
        raise ConfigError(
            f"recipe {recipe.name!r} declares frontend {repeated} more than once. "
            "Frontends do not chain -- a second declaration of one would read the same "
            "tree again and claim the same units -- so this is either a duplicate or an "
            "attempt at layering, which belongs inside a Frontend of your own."
        )
    frontends = {
        stage.plugin: registry.get("frontend", stage.plugin)(**stage_config(stage))
        for stage in frontend_stages
    }

    walked = [s for s in stages if s.kind not in _NOT_STEPS]
    oracle_cache: dict[str, OracleRef] = {}

    def walk(unit_run: UnitRun, facts: Facts, steps: list[Stage], unit_workspace: Path) -> None:
        for stage in steps:
            outcome = _walk_stage(
                stage,
                call_config(stage),
                registry,
                recipe,
                executor_name,
                unit_run.unit,
                facts,
                unit_run,
                unit_workspace,
                executor,
                oracle_cache,
                waived,
            )
            unit_run.outcomes.append(outcome)
            if stage.kind == "oracle" and outcome.status == "ok":
                unit_run.oracle = oracle_cache[outcome.detail]
            # A failed scanner gate fails the unit but does not stop it. The
            # stop exists so that an hour is not spent verifying a candidate
            # that already failed; scanners are independent checks with nothing
            # downstream that needs a clean one, and hpc-devsecops runs every
            # check before it blocks so the operator gets the whole list rather
            # than the first item of it.
            if outcome.status == "failed" and stage.kind == "scanner":
                continue
            if outcome.status == "failed" and (stage.gate or stage.kind in ("transform", "oracle")):
                unit_run.stopped_by = stage.plugin
                # Fail fast, but not silently: the store stages still run, so
                # the failing Verdict is recorded. A gate that failed and was
                # recorded is audit trail; one that failed and vanished is a
                # rumor.
                for later in steps[steps.index(stage) + 1 :]:
                    if later.kind == "store":
                        unit_run.outcomes.append(
                            _walk_stage(
                                later,
                                call_config(later),
                                registry,
                                recipe,
                                executor_name,
                                unit_run.unit,
                                facts,
                                unit_run,
                                unit_workspace,
                                executor,
                                oracle_cache,
                                waived,
                            )
                        )
                break

    # Repository scanners first, once, against a Unit that stands for the
    # tree. They get the same adjudicator and store stages as everything else,
    # so a finding about history is adjudicated, stored and counted exactly
    # like a finding about a file -- which is the reason the tree is a Unit
    # here rather than a special case threaded through every later stage.
    # Unit-subject scanners are left out of this walk, and repository ones
    # out of the per-unit walks below; neither is recorded as skipped on the
    # other's subject, because it was not asked.
    repository_scanners = _repository_scanners(walked, registry)
    if repository_scanners:
        tree = Unit(uid=f"repository:{root.resolve().name}", kind="repository")
        tree_run = UnitRun(unit=tree)
        run.units.append(tree_run)
        tree_workspace = workspace / "repository"
        tree_workspace.mkdir(parents=True, exist_ok=True)
        steps = [
            s
            for s in walked
            if (s.kind == "scanner" and s.plugin in repository_scanners)
            or s.kind in ("adjudicator", "store")
        ]
        walk(tree_run, Facts(unit=tree.uid), steps, tree_workspace)
    unit_steps = [
        s for s in walked if not (s.kind == "scanner" and s.plugin in repository_scanners)
    ]

    for unit, owner in _selected_units(frontends, root, recipe, config):
        unit_run = UnitRun(unit=unit)
        run.units.append(unit_run)
        unit_workspace = workspace / unit.uid.replace(":", "_").replace("/", "_")
        unit_workspace.mkdir(parents=True, exist_ok=True)

        try:
            facts = frontends[owner].analyze(unit, root)
        except RecastError as error:
            unit_run.outcomes.append(StageOutcome("frontend", owner, "failed", str(error)))
            unit_run.stopped_by = owner
            continue
        unit_run.outcomes.append(StageOutcome("frontend", owner, "ok"))
        walk(unit_run, facts, unit_steps, unit_workspace)
    return run


def _walk_stage(
    stage: Stage,
    config: dict[str, Any],
    registry: Registry,
    recipe: Recipe,
    executor_name: str,
    unit: Unit,
    facts: Facts,
    unit_run: UnitRun,
    workspace: Path,
    executor: Any,
    oracle_cache: dict[str, OracleRef],
    waived: frozenset[str] = frozenset(),
) -> StageOutcome:
    if stage.optional and stage.plugin not in registry.names(stage.kind):
        return StageOutcome(stage.kind, stage.plugin, "skipped", "optional plugin not installed")
    factory = registry.get(stage.kind, stage.plugin)

    if stage.kind == "transform":
        transform = factory()
        # A field the frontend produced may legitimately be empty: a module of
        # parameters alone has no subprograms, so no effects and no call
        # graph, and is still a translation (its constants module).
        nothing_to_analyze = not facts.interface.get("subprograms")
        missing = [
            name
            for name in transform.requires
            if not getattr(facts, name, None) and not (nothing_to_analyze and name != "interface")
        ]
        if missing:
            return StageOutcome(
                stage.kind,
                stage.plugin,
                "failed",
                f"facts missing required fields {missing}; the frontend did not "
                "produce what this transform declared it needs",
            )
        if not transform.applicable(unit, facts):
            return StageOutcome(stage.kind, stage.plugin, "skipped", "not applicable to this unit")
        try:
            unit_run.candidate = transform.apply(unit, facts, config)
        except RecastError as error:
            return StageOutcome(stage.kind, stage.plugin, "failed", str(error))
        deferred = len(unit_run.candidate.deferred)
        return StageOutcome(
            stage.kind, stage.plugin, "ok", f"{deferred} deferred block(s)" if deferred else ""
        )

    if stage.kind == "oracle":
        oracle = factory()
        try:
            key = oracle.key(unit, facts, config)
            if key not in oracle_cache:
                oracle_cache[key] = oracle.materialize(unit, facts, workspace, executor, config)
        except RecastError as error:
            return StageOutcome(stage.kind, stage.plugin, "failed", str(error))
        return StageOutcome(stage.kind, stage.plugin, "ok", key)

    if stage.kind == "verifier":
        if unit_run.candidate is None:
            return StageOutcome(
                stage.kind, stage.plugin, "failed", "no candidate to verify; transform never ran"
            )
        verifier = factory()
        verdict = verifier.verify(
            unit, unit_run.candidate, unit_run.oracle, workspace, executor, config
        )
        unit_run.verdicts.append(verdict)
        status = "ok" if verdict.passed else "failed"
        return StageOutcome(
            stage.kind, stage.plugin, status, f"{verdict.confidence.value}: {verdict.detail}"
        )

    if stage.kind == "scanner":
        scanner = factory()
        try:
            # Listed here rather than consumed lazily downstream: ``scan`` may
            # be a generator, and a generator that raises on its third item has
            # not run either. The exception has to surface where the stage's
            # status is decided.
            found = list(scanner.scan(unit, facts, workspace, executor, config))
        except ScannerUnavailable as error:
            return _incomplete(stage, str(error), waived)
        unit_run.findings.extend(found)
        if not stage.gate:
            return StageOutcome(stage.kind, stage.plugin, "ok", f"{len(found)} finding(s)")
        # A scanner declared as a gate is hpc-devsecops's shape: what it found
        # is the verdict, at the bar the scanner declares for its own tool.
        bar = Severity(config.get("blocks_on", scanner.blocks_on))
        blocking = [f for f in found if _SEVERITY_ORDER[f.severity] >= _SEVERITY_ORDER[bar]]
        return StageOutcome(
            stage.kind,
            stage.plugin,
            "failed" if blocking else "ok",
            f"{len(found)} finding(s), {len(blocking)} at or above {bar.value}",
        )

    if stage.kind == "adjudicator":
        if not unit_run.findings:
            return StageOutcome(stage.kind, stage.plugin, "skipped", "no findings to adjudicate")
        adjudicator = factory()
        try:
            adjudicated = [
                adjudicator.adjudicate(finding, workspace, executor, config)
                for finding in unit_run.findings
            ]
        except ScannerUnavailable as error:
            # The findings stay as the scanners left them -- PLAUSIBLE, and
            # still on their way to the store. Dropping them because nobody
            # could adjudicate them would lose the only record that there was
            # something to adjudicate.
            return _incomplete(stage, str(error), waived)
        unit_run.findings = adjudicated
        confirmed = [f for f in unit_run.findings if f.disclosure is Disclosure.CONFIRMED]
        tally = Counter(f.disclosure.value for f in unit_run.findings)
        detail = ", ".join(f"{n} {name}" for name, n in sorted(tally.items()))
        # A confirmed finding is a real defect that nobody has fixed, so an
        # adjudicator stage declared as a gate fails on one. That is the same
        # thing hpc-devsecops does when it exits non-zero on findings, and the
        # reason the stage is worth gating on at all: an audit that reports
        # what it found and passes anyway is a report, not a gate.
        return StageOutcome(
            stage.kind,
            stage.plugin,
            "failed" if confirmed else "ok",
            detail,
        )

    if stage.kind == "store":
        store = _build_store(factory, config)
        if isinstance(store, FindingStore):
            # Findings, not Evidence. The two stores hold different things
            # under different access rules -- ``FindingStore.guard`` refuses a
            # record more sensitive than it can hold -- and walking a
            # FindingStore against ``verdicts`` recorded nothing while
            # reporting success.
            for finding in unit_run.findings:
                unit_run.records.append(store.put(finding))
            return StageOutcome(
                stage.kind, stage.plugin, "ok", f"{len(unit_run.findings)} finding(s) recorded"
            )
        for verdict in unit_run.verdicts:
            evidence = _evidence(recipe, executor_name, unit, unit_run, verdict)
            unit_run.evidence.append(store.put(evidence))
        return StageOutcome(
            stage.kind, stage.plugin, "ok", f"{len(unit_run.verdicts)} verdict(s) recorded"
        )

    # Not reachable from a validated recipe -- ``_require_walkable`` refuses
    # these before any work starts. Kept as a failure rather than a skip
    # because the two are not the same answer, and this branch is what said
    # "skipped" for every scanner and adjudicator stage until now.
    return StageOutcome(
        stage.kind, stage.plugin, "failed", f"the runner does not walk {stage.kind!r} stages"
    )


def _selected_units(
    frontends: dict[str, Any], root: Path, recipe: Recipe, config: dict[str, Any]
) -> list[tuple[Unit, str]]:
    """Every frontend's units, unioned, each paired with the one that owns it.

    Frontends are independent of each other: each reads the tree on its own and
    the sets are unioned, which is what lets one project hold more than one
    language. None of them sees another's Facts -- layering a domain's conventions
    onto Fortran analysis, say, happens inside a Frontend, not between two of
    them, which is why ``analyze`` takes no upstream Facts.

    Two frontends claiming one uid is an error rather than a first-wins. The
    Unit would carry one of their Facts and nothing downstream records which,
    so the run would be reproducible only by accident of declaration order.
    """
    discovered: list[tuple[Unit, str]] = []
    claimed: dict[str, str] = {}
    for name, frontend in frontends.items():
        for unit in frontend.discover(root):
            if unit.uid in claimed:
                raise ConfigError(
                    f"recipe {recipe.name!r}: frontends {claimed[unit.uid]!r} and {name!r} "
                    f"both discovered {unit.uid!r}. A Unit has one set of Facts and no "
                    "record of which frontend produced them, so one of the two has to be "
                    "narrowed -- by config, or by not declaring both."
                )
            claimed[unit.uid] = name
            discovered.append((unit, name))

    wanted = config.get("units")
    if wanted:
        by_uid = {unit.uid: (unit, name) for unit, name in discovered}
        missing = [uid for uid in wanted if uid not in by_uid]
        if missing:
            raise ConfigError(f"units {missing} not found; discovered {sorted(by_uid)}")
        return [by_uid[uid] for uid in wanted]
    # Top-level units by default: a subprogram Unit exists for transforms at
    # that granularity, and analyzing both would do every file's work twice.
    return [(unit, name) for unit, name in discovered if unit.parent is None]


# The steps the runner walks per unit, in the order the recipe lists them.
_STEPS = frozenset({"transform", "oracle", "verifier", "scanner", "adjudicator", "store"})

# Kinds that belong in a recipe without being a step. ``executor`` declares the
# executor the oracles and verifiers receive; ``frontend`` runs once per unit
# before the walk. Both are read before the walk starts, and left out of it.
_NOT_STEPS = frozenset({"executor", "frontend"})

# Registered kinds that are not stages at all, and so are refused rather than
# ignored: an ``agent`` is consulted by a non-deterministic Transform rather
# than scheduled, and a ``recipe`` is what this *is*, not something it can
# contain. Naming them here rather than letting them fall through as unknown
# kinds is what keeps the three sets a partition of ``registry.KINDS``, so a
# tenth kind cannot be added without deciding which of the three it is.
_NOT_STAGES = frozenset({"agent", "recipe"})


# Kinds whose plugins take an ``Executor`` argument. A recipe declaring any of
# them declares an executor; ``Stage``'s docstring says the same.
_HANDED_AN_EXECUTOR = frozenset({"oracle", "verifier", "scanner", "adjudicator"})

_SEVERITY_ORDER = {s: i for i, s in enumerate(Severity)}
"""INFO < LOW < MEDIUM < HIGH < CRITICAL, in the order ``Severity`` declares them."""


def _repository_scanners(stages: list[Stage], registry: Registry) -> frozenset[str]:
    """Names of the declared scanners whose ``subject`` is the repository.

    Asks the plugin, because the subject is the scanner's declaration and not
    the recipe's -- a recipe does not know, and should not have to say, that
    gitleaks reads history. An optional scanner that is not installed has no
    subject to ask about and is left to ``_walk_stage`` to report as skipped.
    """
    names = set()
    for stage in stages:
        if stage.kind != "scanner" or stage.plugin not in registry.names("scanner"):
            continue
        if getattr(registry.get("scanner", stage.plugin)(), "subject", "unit") == "repository":
            names.add(stage.plugin)
    return frozenset(names)


def missing_tools(
    stages: list[Stage], config: dict[str, Any], *, registry: Registry = REGISTRY
) -> dict[str, str]:
    """Stage plugin -> why it cannot run here, for the stages that declare a ``tool``.

    The preflight. A scanner that wraps gitleaks will say so at scan time by
    raising ``ScannerUnavailable``, and the run will be ``incomplete`` -- two
    stages in, after the frontend has read the tree. ``recast plan`` asks this
    first, which is the same preference the rest of this repository states:
    a problem worth reporting in a second rather than later.

    Only for installed plugins of the kinds that wrap tools. A plugin that is
    not registered is a different problem and ``plan`` already reports it.
    """
    out: dict[str, str] = {}
    for stage in stages:
        if stage.kind not in ("scanner", "adjudicator"):
            continue
        if stage.plugin not in registry.names(stage.kind):
            continue
        plugin = registry.get(stage.kind, stage.plugin)()
        declared = getattr(plugin, "tool", None)
        if not declared:
            continue
        tools = (declared,) if isinstance(declared, str) else tuple(declared)
        stage_config = {**stage.config, **config.get("stages", {}).get(stage.plugin, {})}
        absent = [
            stage_config.get(tool, tool)
            for tool in tools
            if shutil.which(stage_config.get(tool, tool)) is None
        ]
        if absent:
            out[stage.plugin] = f"{' and '.join(absent)} not on PATH"
    return out


def _incomplete(stage: Stage, detail: str, waived: frozenset[str]) -> StageOutcome:
    """The stage could not run. Neither a pass nor a failure, and not silent."""
    allowed = stage.plugin in waived
    return StageOutcome(
        stage.kind,
        stage.plugin,
        "incomplete",
        f"{detail} (waived)" if allowed else detail,
        waived=allowed,
    )


def _waived(recipe: Recipe, stages: list[Stage], config: dict[str, Any]) -> frozenset[str]:
    """Plugins whose ``incomplete`` the operator has agreed not to count.

    ``config["allow_incomplete"]`` is a list of plugin names. It exists because
    the alternative to a waiver is worse than either: an operator whose LLM
    audit scanner has no API key on this machine, and whose only way to get a
    green run is to delete the stage from the recipe, has been given a reason
    to make the recipe lie rather than the run.

    Three things it deliberately does not do. It does not silence the outcome
    -- the stage still reports ``incomplete``, and still says ``(waived)``, so
    the difference between a waived run and a clean one is visible in the same
    place it would otherwise be invisible. It does not accept a name no stage
    declares, because a waiver that matches nothing reads as coverage and is
    the failure `docs/disclosure-ledger.md` warns about for hygiene patterns.
    And it does not apply to a gate: waiving a gate's unavailability means the
    gate can be absent from a passing run, which is the same argument that
    makes an optional gate a contradiction.
    """
    names = config.get("allow_incomplete") or []
    if isinstance(names, str):
        raise ConfigError("allow_incomplete is a list of plugin names, not a single string")
    declared = {stage.plugin for stage in stages}
    unknown = sorted(name for name in names if name not in declared)
    if unknown:
        raise ConfigError(
            f"allow_incomplete names {unknown}, which recipe {recipe.name!r} does not declare; "
            "a waiver that matches nothing reads as coverage"
        )
    gates = sorted(s.plugin for s in stages if s.gate and s.plugin in set(names))
    if gates:
        raise ConfigError(
            f"allow_incomplete names gate stage(s) {gates}; a gate that may be absent from a "
            "passing run is not a gate. Drop the gate or drop the waiver -- deliberately, "
            "and in the recipe where a reader can see it."
        )
    return frozenset(names)


def _require_valid_bars(recipe: Recipe, stages: list[Stage], config: dict[str, Any]) -> None:
    """``config["blocks_on"]`` has to name a ``Severity`` -- checked here so a
    typo fails in a second rather than as a ``ValueError`` two scans in."""
    for stage in stages:
        if stage.kind != "scanner":
            continue
        merged = {**stage.config, **config.get("stages", {}).get(stage.plugin, {})}
        if "blocks_on" not in merged:
            continue
        try:
            Severity(merged["blocks_on"])
        except ValueError as error:
            raise ConfigError(
                f"recipe {recipe.name!r}, stage {stage.plugin!r}: blocks_on must be one of "
                f"{[s.value for s in Severity]}, not {merged['blocks_on']!r}"
            ) from error


def _require_walkable(recipe: Recipe, stages: list[Stage]) -> None:
    """Refuse a stage the runner would not walk, before anything runs.

    ``_walk_stage`` used to answer ``skipped`` for a kind it did not handle --
    the same word an uninstalled optional plugin gets -- so a recipe could
    declare a scanner, an adjudicator and a gate, walk none of them, and report
    that every unit passed. Failing here instead is the same reasoning
    ``Recipe.validate`` gives for itself: a problem worth reporting in a second
    rather than three hours in.

    Runs before ``_require_available`` deliberately. A kind the runner has never
    heard of has no registered plugins either, so checking availability first
    would report a missing plugin -- naming the plugin, which is fine, as the
    problem, which it is not.
    """
    unwalkable = sorted({s.kind for s in stages if s.kind not in _STEPS | _NOT_STEPS})
    if unwalkable:
        raise ConfigError(
            f"recipe {recipe.name!r} declares stage kind(s) {unwalkable} that the "
            f"runner does not walk; stages are {sorted(_STEPS | _NOT_STEPS)}"
        )


def _findings_root(root: Path) -> Path:
    """Where embargoed findings go when the operator has not said.

    Deliberately *not* under ``root``. Everything else the engine writes belongs
    beside the source it describes, and the evidence store is meant to be
    committed -- but a ``Finding`` defaults to ``Access.EMBARGOED``, and a
    default that puts one inside the working tree is one ``git add -A`` from
    publishing it. ``FilesystemFindingStore`` refuses such a root outright; this
    is what keeps the shipped ``audit`` recipe from meeting that refusal on
    every run rather than only when someone misconfigures it.

    Per project, because two checkouts' findings are not interchangeable, and
    keyed by the absolute path rather than by the directory name, because two
    clones of one repository are two projects here and would otherwise share a
    directory. ``RECAST_FINDINGS_HOME`` overrides the base for anyone who keeps
    embargoed material somewhere specific -- an encrypted volume, a host that is
    not this one.
    """
    home = os.environ.get("RECAST_FINDINGS_HOME")
    base = Path(home).expanduser() if home else Path.home() / WORKSPACE_DIRNAME / "findings"
    resolved = root.expanduser().resolve()
    key = hashlib.sha256(str(resolved).encode()).hexdigest()[:12]
    return base / f"{resolved.name or 'root'}-{key}"


def output_root(root: Path, config: dict[str, Any]) -> Path:
    """Where this run's candidates and evidence go: ``output/<project>/``.

    Outside the source tree by default, and named after it rather than hidden
    inside it. Two reasons, and the second is the one that bites: generated
    code written into a checkout is one ``git add -A`` from being committed as
    if it were source, and a ``Frontend`` that discovers it on the next pass
    offers to translate the last run's scaffolding.

    The project segment is the tree's own directory name, so
    ``examples/toy_physics`` and ``corpus/.build/numfor`` land in
    ``output/toy_physics`` and ``output/numfor`` -- readable, and the thing a
    person asking "where did it go" already knows. Two trees whose basenames
    collide share a directory; keying by a path hash instead would be correct
    and unreadable, and readable is what this directory is for. Override when
    that matters.

    Two overrides, and they are not the same thing. ``config["output"]`` names
    this run's directory outright, project segment included.
    ``RECAST_OUTPUT_HOME`` moves only the base that the project segment hangs
    under -- the same shape ``RECAST_FINDINGS_HOME`` has, and what a test
    harness wants: every run in the process lands somewhere disposable
    without any of them naming a path.
    """
    configured = config.get("output")
    if configured:
        return Path(configured).expanduser().resolve()
    home = os.environ.get("RECAST_OUTPUT_HOME")
    base = Path(home).expanduser() if home else Path.cwd() / OUTPUT_DIRNAME
    return (base / (root.name or "root")).resolve()


def _build_store(factory: Any, config: dict[str, Any]) -> Any:
    """Construct a store, rooted where its access class belongs.

    A FindingStore may not share the evidence directory, and may not sit under
    the project root at all -- see ``_findings_root``. The kind is taken from
    the registered class where there is one, and both shipped stores are
    classes. A store supplied as a factory *function* is built with the evidence
    root, which is the safe direction to be wrong in: a FindingStore built there
    refuses to operate rather than quietly writing an embargoed record into the
    checkout.
    """
    findings = isinstance(factory, type) and issubclass(factory, FindingStore)
    root = Path(config.get("root", "."))
    output = Path(config.get("output") or output_root(root, {}))
    default = _findings_root(root) if findings else output / "evidence"
    return factory(**_store_config(config, default))


def _store_config(config: dict[str, Any], default_root: Path) -> dict[str, Any]:
    """What a store's constructor sees: its own keys, and a resolved root.

    A store is the one plugin constructed inside the walk, so the per-call
    facts ``call_config`` adds -- ``root``, ``range`` -- arrive here and have
    to be taken back out. ``range`` was not, and the first pre-push hook
    invocation with a real range was what found it: a constructor given a
    keyword it has no parameter for does not skip the key, it refuses to
    construct.
    """
    prepared = dict(config)
    prepared.pop("range", None)
    prepared.pop("output", None)
    root = Path(prepared.pop("root", "."))
    store_root = Path(prepared.pop("store_root", default_root))
    if not store_root.is_absolute():
        store_root = root / store_root
    prepared["root"] = store_root
    return prepared


def _evidence(
    recipe: Recipe, executor_name: str, unit: Unit, unit_run: UnitRun, verdict: Verdict
) -> Evidence:
    candidate = unit_run.candidate
    oracle_ref = unit_run.oracle
    return Evidence(
        unit=unit.uid,
        verdict=verdict,
        recipe=recipe.name,
        executor=executor_name,
        artifact={
            "name": unit.uid,
            "files": sorted(str(p) for p in (candidate.files if candidate else {})),
            "digest": candidate.digest() if candidate else "",
            "transform": candidate.transform if candidate else "",
        },
        reference={"oracle": oracle_ref.oracle, "key": oracle_ref.key},
        environment={
            "engine": f"recast {__version__}",
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        meta={"timestamp": datetime.now(UTC).isoformat()},
    )


def _require_one_transform(recipe: Recipe, stages: list[Stage]) -> None:
    """One Transform produces the Candidate, in one pass.

    Stacking two transform stages does not compose them. The runner keeps one
    Candidate per Unit, so the second stage's would replace the first's -- and
    with it the first's files *and* its ``deferred`` list, which is precisely
    the hand-off anyone stacking them was reaching for. Composition happens
    inside a Transform instead: rules first, an ``AgentProvider`` for the sites
    they refused, one Candidate out. See ``plugins/transform.py``.

    Refusing here rather than at declaration time because a recipe may branch
    on config, so the stage list is only knowable once the config is.
    """
    transforms = [stage.plugin for stage in stages if stage.kind == "transform"]
    if len(transforms) > 1:
        raise ConfigError(
            f"recipe {recipe.name!r} declares {len(transforms)} transform stages "
            f"({', '.join(transforms)}); a Unit has one Candidate, so all but the last "
            "would be discarded along with what they deferred. Compose inside one "
            "Transform."
        )


def _require_available(stages: list[Stage], registry: Registry) -> None:
    missing = [
        f"{stage.kind}:{stage.plugin}"
        for stage in stages
        if not stage.optional and stage.plugin not in registry.names(stage.kind)
    ]
    if missing:
        raise ConfigError(
            f"required plugin(s) not registered: {', '.join(missing)}; "
            "see `recast plan` for the full table"
        )
