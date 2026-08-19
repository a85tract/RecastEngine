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

import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from recast import __version__
from recast.errors import ConfigError, RecastError
from recast.model import Candidate, Evidence, Facts, OracleRef, Unit, Verdict
from recast.plugins.recipe import Recipe, Stage
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
    evidence: list[str] = field(default_factory=list)
    """Store URIs, one per recorded Verdict."""

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

        Deliberately excludes wall-clock time and paths. Two runs over the same
        revisions produce the same summary, so committing it does not manufacture
        a change on every invocation; when a number does change, the change is
        the finding.
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
                    "oracle": (
                        {"name": unit_run.oracle.oracle, "key": unit_run.oracle.key}
                        if unit_run.oracle.key
                        else None
                    ),
                    "stopped_by": unit_run.stopped_by,
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
    _require_available(stages, registry)

    workspace = Path(config.get("workspace") or root / ".recast" / recipe.name)
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

    executor_stage = next((s for s in stages if s.kind == "executor"), None)
    if executor_stage is None:
        raise ConfigError(f"recipe {recipe.name!r} declares no executor stage")
    executor = registry.get("executor", executor_stage.plugin)(**stage_config(executor_stage))

    frontend_stage = next((s for s in stages if s.kind == "frontend"), None)
    if frontend_stage is None:
        raise ConfigError(f"recipe {recipe.name!r} declares no frontend stage")
    frontend = registry.get("frontend", frontend_stage.plugin)(**stage_config(frontend_stage))

    walked = [s for s in stages if s.kind not in ("executor", "frontend")]
    oracle_cache: dict[str, OracleRef] = {}

    for unit in _selected_units(frontend, root, config):
        unit_run = UnitRun(unit=unit)
        run.units.append(unit_run)
        unit_workspace = workspace / unit.uid.replace(":", "_").replace("/", "_")
        unit_workspace.mkdir(parents=True, exist_ok=True)

        try:
            facts = frontend.analyze(unit, root)
        except RecastError as error:
            unit_run.outcomes.append(
                StageOutcome(frontend_stage.kind, frontend_stage.plugin, "failed", str(error))
            )
            unit_run.stopped_by = frontend_stage.plugin
            continue
        unit_run.outcomes.append(StageOutcome(frontend_stage.kind, frontend_stage.plugin, "ok"))

        for stage in walked:
            outcome = _walk_stage(
                stage,
                call_config(stage),
                registry,
                recipe,
                executor_stage.plugin,
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
                                executor_stage.plugin,
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

    if stage.kind == "store":
        store = factory(**_store_config(config))  # config still carries root here
        for verdict in unit_run.verdicts:
            evidence = _evidence(recipe, executor_name, unit, unit_run, verdict)
            unit_run.evidence.append(store.put(evidence))
        return StageOutcome(
            stage.kind, stage.plugin, "ok", f"{len(unit_run.verdicts)} verdict(s) recorded"
        )

    return StageOutcome(stage.kind, stage.plugin, "skipped", f"kind {stage.kind!r} not walked")


def _selected_units(frontend: Any, root: Path, config: dict[str, Any]) -> list[Unit]:
    discovered = list(frontend.discover(root))
    wanted = config.get("units")
    if wanted:
        by_uid = {u.uid: u for u in discovered}
        missing = [uid for uid in wanted if uid not in by_uid]
        if missing:
            raise ConfigError(f"units {missing} not found; discovered {sorted(by_uid)}")
        return [by_uid[uid] for uid in wanted]
    # Top-level units by default: a subprogram Unit exists for transforms at
    # that granularity, and analyzing both would do every file's work twice.
    return [u for u in discovered if u.parent is None]


def _store_config(config: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(config)
    root = Path(prepared.pop("root", "."))
    store_root = Path(prepared.pop("store_root", root / ".recast" / "evidence"))
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
