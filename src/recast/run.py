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
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recast import WORKSPACE_DIRNAME, __version__
from recast.errors import ConfigError, RecastError
from recast.model import (
    Candidate,
    Disclosure,
    Evidence,
    Facts,
    Finding,
    OracleRef,
    Unit,
    Verdict,
)
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.store import FindingStore
from recast.registry import REGISTRY, Registry

__all__ = ["RecipeRun", "UnitRun", "run_recipe"]

NO_ORACLE = OracleRef(unit="", oracle="none", key="", handle=None)
"""What a Verifier receives before any Oracle has materialized. Static
verifiers ignore it; an oracle-backed verifier handed this fails closed on
its own rules, which is the correct outcome for a recipe ordered wrongly."""


@dataclass
class StageOutcome:
    """What one stage did for one unit."""

    kind: str
    plugin: str
    status: str
    """``ok`` | ``failed`` | ``skipped``."""

    detail: str = ""


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
    def passed(self) -> bool:
        return self.stopped_by is None and all(o.status != "failed" for o in self.outcomes)


@dataclass
class RecipeRun:
    """One invocation of ``run_recipe``: the units and the shared context."""

    recipe: str
    root: Path
    workspace: Path
    units: list[UnitRun] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.units) and all(u.passed for u in self.units)

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

    workspace = Path(config.get("workspace") or root / WORKSPACE_DIRNAME / recipe.name)
    workspace.mkdir(parents=True, exist_ok=True)
    run = RecipeRun(recipe=recipe.name, root=root, workspace=workspace)

    def stage_config(stage: Stage) -> dict[str, Any]:
        """The operator's per-stage table over the recipe's own."""
        return {**stage.config, **config.get("stages", {}).get(stage.plugin, {})}

    def call_config(stage: Stage) -> dict[str, Any]:
        """Per-call stages also learn where the source tree is; a plugin that
        is *constructed* from config (executor, frontend, store) takes paths
        per call instead, so its constructor sees only its own keys."""
        return {"root": root, **stage_config(stage)}

    # An executor stage is not a step. ``Stage`` says what it is for: "it
    # declares the executor the run's Oracles and Verifiers receive as an
    # argument... A recipe that materializes an oracle or awards a verdict has
    # to declare one." So the requirement is conditional, and demanding one
    # unconditionally made the ``audit`` recipe -- which produces Findings and
    # neither materializes nor awards -- unrunnable as shipped. conformance/
    # already read the contract this way and skipped its executor checks for
    # that recipe; the runner was the one out of step.
    executor_stage = next((s for s in stages if s.kind == "executor"), None)
    if executor_stage is None and any(s.kind in ("oracle", "verifier") for s in stages):
        raise ConfigError(
            f"recipe {recipe.name!r} materializes an oracle or awards a verdict "
            "but declares no executor stage"
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

        for stage in walked:
            outcome = _walk_stage(
                stage,
                call_config(stage),
                registry,
                recipe,
                executor_name,
                unit,
                facts,
                unit_run,
                unit_workspace,
                executor,
                oracle_cache,
            )
            unit_run.outcomes.append(outcome)
            if stage.kind == "oracle" and outcome.status == "ok":
                unit_run.oracle = oracle_cache[outcome.detail]
            if outcome.status == "failed" and (stage.gate or stage.kind in ("transform", "oracle")):
                unit_run.stopped_by = stage.plugin
                # Fail fast, but not silently: the store stages still run, so
                # the failing Verdict is recorded. A gate that failed and was
                # recorded is audit trail; one that failed and vanished is a
                # rumor.
                for later in walked[walked.index(stage) + 1 :]:
                    if later.kind == "store":
                        unit_run.outcomes.append(
                            _walk_stage(
                                later,
                                call_config(later),
                                registry,
                                recipe,
                                executor_name,
                                unit,
                                facts,
                                unit_run,
                                unit_workspace,
                                executor,
                                oracle_cache,
                            )
                        )
                break
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
) -> StageOutcome:
    if stage.optional and stage.plugin not in registry.names(stage.kind):
        return StageOutcome(stage.kind, stage.plugin, "skipped", "optional plugin not installed")
    factory = registry.get(stage.kind, stage.plugin)

    if stage.kind == "transform":
        transform = factory()
        missing = [name for name in transform.requires if not getattr(facts, name, None)]
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
        found = list(scanner.scan(unit, facts, workspace, config))
        unit_run.findings.extend(found)
        return StageOutcome(stage.kind, stage.plugin, "ok", f"{len(found)} finding(s)")

    if stage.kind == "adjudicator":
        if not unit_run.findings:
            return StageOutcome(stage.kind, stage.plugin, "skipped", "no findings to adjudicate")
        adjudicator = factory()
        unit_run.findings = [
            adjudicator.adjudicate(finding, workspace, config) for finding in unit_run.findings
        ]
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
    language. None of them sees another's Facts -- layering CESM conventions
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
    default = _findings_root(root) if findings else root / WORKSPACE_DIRNAME / "evidence"
    return factory(**_store_config(config, default))


def _store_config(config: dict[str, Any], default_root: Path) -> dict[str, Any]:
    prepared = dict(config)
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
