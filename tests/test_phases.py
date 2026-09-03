"""The durable transform/verify boundary is canonical and one-way."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, ClassVar

import pytest

import recast.phases as phase_api
from recast.engines import ArtifactContract, TranslationEngine
from recast.errors import ConfigError, PluginError
from recast.model import Candidate, Confidence, Facts, OracleRef, Unit, Verdict
from recast.observe import RunEvent, RunEventAction, RunEventEntity
from recast.phases import (
    CandidateBundle,
    EngineBinding,
    decode_candidate_bundle,
    transform_recipe,
    verify_recipe_candidates,
)
from recast.plugins.executor import Executor, Job, JobResult
from recast.plugins.frontend import Frontend
from recast.plugins.oracle import Oracle
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.store import EvidenceStore
from recast.plugins.transform import Transform
from recast.plugins.verifier import Verifier
from recast.registry import KINDS, Registry
from recast.run import StageOutcome, UnitRun

_SOURCE = "sha256:" + "1" * 64
_IMPLEMENTATION = "sha256:" + "2" * 64


class PhaseExecutor(Executor):
    name = "phase.executor"

    def submit(self, job: Job) -> str:
        del job
        return "unused"

    def wait(self, handle: str, timeout_s: float | None = None) -> JobResult:
        del handle, timeout_s
        return JobResult(0, "", "")


class PhaseFrontend(Frontend):
    name = "phase.frontend"

    def discover(self, root: Path) -> list[Unit]:
        del root
        return [
            Unit(
                uid="phase:alpha",
                kind="module",
                sources=(Path("src/alpha.f90"),),
                attrs={"profile": "generic"},
            ),
            Unit(
                uid="phase:alpha/step",
                kind="subprogram",
                sources=(Path("src/alpha.f90"),),
                parent="phase:alpha",
            ),
        ]

    def analyze(self, unit: Unit, root: Path) -> Facts:
        del root
        return Facts(
            unit=unit.uid,
            interface={"subprograms": [{"name": "step"}]},
            provenance={"source": "src/alpha.f90", "digest": "a" * 64},
        )


class PhaseTransform(Transform):
    name = "phase.transform.impl"
    requires = ("interface",)
    constructed: ClassVar[int] = 0
    applied: ClassVar[int] = 0
    last_config: ClassVar[dict[str, Any] | None] = None

    def __init__(self) -> None:
        type(self).constructed += 1

    def applicable(self, unit: Unit, facts: Facts) -> bool:
        del facts
        return unit.kind == "module"

    def apply(self, unit: Unit, facts: Facts, config: dict[str, Any]) -> Candidate:
        del facts
        type(self).applied += 1
        type(self).last_config = dict(config)
        deferred = ["phase:alpha/step/block: unsupported"] if config.get("defer") else []
        return Candidate(
            unit=unit.uid,
            transform=self.name,
            files={Path("python/alpha.py"): b"def step(x):\n    return x + 1\n"},
            deferred=deferred,
            notes={
                "blocks": [{"subprogram": "step", "status": "translated"}],
                "coverage": {"subprograms": ["step"]},
            },
        )


class MaliciousTransform(PhaseTransform):
    """Instantiation is already a contract violation during verification."""

    constructed: ClassVar[int] = 0
    applied: ClassVar[int] = 0

    def __init__(self) -> None:
        type(self).constructed += 1
        raise AssertionError("verification instantiated a Transform")


class PhaseVerifier(Verifier):
    name = "phase.gate"
    provides = Confidence.SAMPLED

    def verify(self, unit, candidate, oracle, workspace, executor, config) -> Verdict:
        del oracle, workspace, executor, config
        return Verdict(
            unit=unit.uid,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=Confidence.SAMPLED,
            metrics={"source_bytes": "DO NOT COPY THIS SOURCE"},
            detail="DO NOT COPY THIS SOURCE from /private/machine/root",
        )


class MutatingVerifier(PhaseVerifier):
    """Adversarial verifier used to exercise every mutable bundle boundary."""

    mutation: ClassVar[str] = ""
    target_bundle: ClassVar[CandidateBundle | None] = None

    def verify(self, unit, candidate, oracle, workspace, executor, config) -> Verdict:
        if self.mutation == "candidate":
            candidate.files[Path("python/alpha.py")] = b"tampered after transform\n"
        elif self.mutation == "unit":
            unit.attrs["tampered"] = True
        elif self.mutation == "facts":
            assert self.target_bundle is not None
            facts = self.target_bundle.units[0].facts
            assert facts is not None
            facts.extra["tampered"] = True
        elif self.mutation == "bundle":
            assert self.target_bundle is not None
            object.__setattr__(self.target_bundle, "recipe", "tampered")
        else:  # pragma: no cover - a broken test setup must fail loudly.
            raise AssertionError(f"unknown mutation mode {self.mutation!r}")
        return super().verify(unit, candidate, oracle, workspace, executor, config)


class BrokenOracle(Oracle):
    """An oracle whose build fails, as f2py does on Fortran it cannot wrap."""

    name = "phase.oracle"

    def key(self, unit, facts, config) -> str:
        del facts, config
        return f"{unit.uid}:oracle"

    def materialize(self, unit, facts, workspace, executor, config) -> OracleRef:
        del unit, facts, workspace, executor, config
        raise PluginError("crackfortran: DO NOT COPY THIS SOURCE at /private/machine/root")


class RecordingObserver:
    def __init__(self) -> None:
        self.events: list[RunEvent] = []

    def observe(self, event: RunEvent) -> None:
        self.events.append(event)


class PhaseStore(EvidenceStore):
    name = "phase.store"
    records: ClassVar[list[Any]] = []

    def __init__(self, **config: Any) -> None:
        del config

    def put(self, evidence: Any) -> str:
        type(self).records.append(evidence)
        return "/private/machine/root/evidence.json"

    def get(self, uri: str) -> Any:
        raise NotImplementedError(uri)

    def query(self, **selectors: Any) -> list[Any]:
        raise NotImplementedError(selectors)


class PhaseRecipe(Recipe):
    name = "phase"
    engine_id = "example.phase-engine"

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", "phase.executor"),
            Stage("frontend", "phase.frontend"),
            Stage("transform", "phase.transform", config={"defer": config.get("defer", False)}),
            Stage("verifier", "phase.gate", gate=True),
            Stage("store", "phase.store"),
        ]


class StoppingRecipe(PhaseRecipe):
    """The real shape of a rejected run: the oracle fails before the gate."""

    name = "stopping"

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", "phase.executor"),
            Stage("frontend", "phase.frontend"),
            Stage("transform", "phase.transform", config={"defer": config.get("defer", False)}),
            Stage("oracle", "phase.oracle"),
            Stage("verifier", "phase.gate", gate=True),
            Stage("store", "phase.store"),
        ]


class HostileVerificationRecipe(Recipe):
    """Every executable hook would instantiate a Transform if verification called it."""

    name = "hostile"

    def validate(self, config: dict[str, Any]) -> list[str]:
        del config
        MaliciousTransform()
        return []

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        del config
        MaliciousTransform()
        return []

    def resolved_engine_id(self, config: dict[str, Any]) -> str | None:
        del config
        MaliciousTransform()
        return None


def _engine() -> TranslationEngine:
    return TranslationEngine(
        id="example.phase-engine",
        version="1",
        implementation_digest=_IMPLEMENTATION,
        default_recipe="phase",
        input_artifact_contract=ArtifactContract(
            id="example.source.input",
            version="1",
            media_type="application/vnd.example.source-tree",
            language="input",
        ),
        output_artifact_contract=ArtifactContract(
            id="example.source.output",
            version="1",
            media_type="application/vnd.example.source-tree",
            language="output",
        ),
        required_gates=("phase.gate",),
        owning_repository="https://example.invalid/phase",
    )


def _registry() -> Registry:
    registry = Registry()
    registry._loaded.update(KINDS)
    registry.register("executor", "phase.executor", PhaseExecutor)
    registry.register("frontend", "phase.frontend", PhaseFrontend)
    registry.register("transform", "phase.transform", PhaseTransform)
    registry.register("verifier", "phase.gate", PhaseVerifier)
    registry.register("oracle", "phase.oracle", BrokenOracle)
    registry.register("store", "phase.store", PhaseStore)
    engine = _engine()
    registry.register("engine", engine.id, engine)
    return registry


@pytest.fixture(autouse=True)
def _reset() -> None:
    PhaseTransform.constructed = 0
    PhaseTransform.applied = 0
    PhaseTransform.last_config = None
    MaliciousTransform.constructed = 0
    MaliciousTransform.applied = 0
    MutatingVerifier.mutation = ""
    MutatingVerifier.target_bundle = None
    PhaseStore.records = []


def _bundle(
    tmp_path: Path, *, config: dict[str, Any] | None = None
) -> tuple[CandidateBundle, Registry]:
    registry = _registry()
    bundle = transform_recipe(
        PhaseRecipe(),
        tmp_path,
        config,
        source_artifact_digest=_SOURCE,
        registry=registry,
        output=tmp_path / "runtime-output",
        workspace=tmp_path / "runtime-workspace",
    )
    return bundle, registry


def test_candidate_bundle_is_canonical_portable_and_round_trips(tmp_path: Path) -> None:
    first, _ = _bundle(tmp_path / "checkout-a")
    second, _ = _bundle(tmp_path / "a-different-machine-root")

    encoded = first.to_json()
    decoded = decode_candidate_bundle(encoded)

    assert decoded == first
    assert decoded.to_json() == encoded
    assert decoded.digest() == first.digest() == second.digest()
    document = json.loads(encoded)
    descriptor = document["units"][0]["candidate"]["files"][0]
    assert descriptor["path"] == "python/alpha.py"
    assert descriptor["blob_digest"].startswith("sha256:")
    assert descriptor["media_type"] == "application/octet-stream"
    assert str(tmp_path) not in encoded.decode()
    assert "timestamp" not in document


def test_compiler_semantics_is_frozen_into_transform_and_golden_oracle_config() -> None:
    semantic = {"compiler_semantics": "gfortran"}

    transform = phase_api._resolved_stage_config(Stage("transform", "phase.transform"), semantic)
    oracle = phase_api._resolved_stage_config(Stage("oracle", "f2py-golden"), semantic)

    assert transform == {"profile": "gfortran"}
    assert oracle == {"fc": "gfortran"}
    with pytest.raises(ConfigError, match="contradicts compiler_semantics"):
        phase_api._resolved_stage_config(
            Stage("transform", "phase.transform", config={"profile": "ifx"}),
            semantic,
        )


def test_transform_phase_receives_the_bound_compiler_profile(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path, config={"compiler_semantics": "gfortran"})

    assert bundle.semantic_config_digest == phase_api.canonical_digest(
        {"compiler_semantics": "gfortran"}
    )
    assert PhaseTransform.last_config is not None
    assert PhaseTransform.last_config["profile"] == "gfortran"


def test_unit_run_positional_constructor_remains_backward_compatible() -> None:
    unit = Unit(uid="phase:legacy", kind="module")
    outcome = StageOutcome("frontend", "phase.frontend", "ok")

    run = UnitRun(unit, [outcome])

    assert run.outcomes == [outcome]
    assert run.facts is None


def test_decoder_rejects_tampering_unknown_fields_and_noncanonical_json(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    document = json.loads(bundle.to_json())
    document["blobs"][0]["data"] = "eA=="
    tampered = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ConfigError, match="does not match"):
        decode_candidate_bundle(tampered)

    document = json.loads(bundle.to_json())
    document["unknown"] = True
    unknown = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ConfigError, match="unknown"):
        decode_candidate_bundle(unknown)

    pretty = json.dumps(json.loads(bundle.to_json()), indent=2)
    with pytest.raises(ConfigError, match="not in canonical JSON form"):
        decode_candidate_bundle(pretty)


def test_decoder_rejects_nonportable_paths_and_media_type_drift(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    document = json.loads(bundle.to_json())
    document["units"][0]["candidate"]["files"][0]["path"] = "../escape.py"
    payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ConfigError, match="portable relative path"):
        decode_candidate_bundle(payload)

    document = json.loads(bundle.to_json())
    document["blobs"][0]["media_type"] = "text/plain"
    payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()
    with pytest.raises(ConfigError, match="application/octet-stream"):
        decode_candidate_bundle(payload)


def test_encoder_enforces_the_inline_blob_size_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(phase_api, "MAX_INLINE_BLOB_BYTES", 2)

    with pytest.raises(ConfigError, match="inline limit is 2"):
        transform_recipe(
            PhaseRecipe(),
            tmp_path,
            source_artifact_digest=_SOURCE,
            registry=_registry(),
        )


def test_verification_cannot_resolve_instantiate_or_call_transform(tmp_path: Path) -> None:
    bundle, registry = _bundle(tmp_path)
    assert PhaseTransform.constructed == 1
    assert PhaseTransform.applied == 1
    registry.register("transform", "phase.transform", MaliciousTransform, replace=True)

    report = verify_recipe_candidates(
        PhaseRecipe(),
        tmp_path,
        bundle,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=bundle.engine,
        project_required_gates=("phase.gate",),
        required_units=("phase:alpha",),
        required_subprograms=("step",),
        registry=registry,
        output=tmp_path / "verify-output",
        workspace=tmp_path / "verify-workspace",
    )

    assert report.accepted
    assert report.bundle_digest == bundle.digest()
    assert MaliciousTransform.constructed == 0
    assert MaliciousTransform.applied == 0
    assert report.gates[0].required_by == ("engine", "project")
    assert report.gates[0].passed_units == ("phase:alpha",)
    assert report.required_units[0].accepted
    subprogram = next(item for item in report.subprograms if item.selector == "step")
    assert subprogram.required and subprogram.owner_units == ("phase:alpha",)
    assert subprogram.accepted


def test_report_omits_source_bytes_details_metrics_store_uris_roots_and_time(
    tmp_path: Path,
) -> None:
    bundle, registry = _bundle(tmp_path)
    report = verify_recipe_candidates(
        PhaseRecipe(),
        tmp_path,
        bundle,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=EngineBinding.from_engine(_engine()),
        registry=registry,
    )

    encoded = report.to_json().decode()
    assert report.accepted
    assert "DO NOT COPY THIS SOURCE" not in encoded
    assert "/private/machine/root" not in encoded
    assert str(tmp_path) not in encoded
    assert "timestamp" not in encoded
    assert "metrics" not in encoded
    assert "detail" not in encoded


def test_report_names_the_stage_that_stopped_a_unit_and_observer_gets_its_detail(
    tmp_path: Path,
) -> None:
    registry = _registry()
    bundle = transform_recipe(
        StoppingRecipe(),
        tmp_path,
        None,
        source_artifact_digest=_SOURCE,
        registry=registry,
        output=tmp_path / "runtime-output",
        workspace=tmp_path / "runtime-workspace",
    )
    observer = RecordingObserver()
    report = verify_recipe_candidates(
        StoppingRecipe(),
        tmp_path,
        bundle,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=EngineBinding.from_engine(_engine()),
        registry=registry,
        observer=observer,
    )

    assert not report.accepted
    assert "gate.phase.gate" in report.reason_codes
    (gate,) = report.gates
    assert gate.missing_units == ("phase:alpha",)
    unit = next(item for item in report.units if item.uid == "phase:alpha")
    # The gate never ran; the report says which stage is to blame, by name only.
    assert unit.stopped_by == "phase.oracle"
    assert unit.to_dict()["stopped_by"] == "phase.oracle"
    encoded = report.to_json().decode()
    assert "crackfortran" not in encoded
    assert "DO NOT COPY THIS SOURCE" not in encoded
    assert "/private/machine/root" not in encoded

    # The observer, and only the observer, is told why.
    oracle_finished = next(
        event
        for event in observer.events
        if event.entity is RunEventEntity.STAGE
        and event.action is RunEventAction.FINISHED
        and event.stage_plugin == "phase.oracle"
    )
    assert (oracle_finished.status, oracle_finished.reason_code) == ("failed", "stage_failed")
    assert "crackfortran: DO NOT COPY THIS SOURCE" in oracle_finished.reason
    gate_finished = next(
        event
        for event in observer.events
        if event.entity is RunEventEntity.STAGE
        and event.action is RunEventAction.FINISHED
        and event.stage_plugin == "phase.gate"
    )
    assert (gate_finished.status, gate_finished.reason_code) == ("skipped", "upstream_stop")
    assert "phase.oracle" in gate_finished.reason
    # Suppressed stages are bracketed like every other: a start precedes the skip.
    gate_lifecycle = [
        (event.action, event.status, event.reason_code)
        for event in observer.events
        if event.entity is RunEventEntity.STAGE and event.stage_plugin == "phase.gate"
    ]
    assert gate_lifecycle == [
        (RunEventAction.STARTED, "running", "stage_considered"),
        (RunEventAction.FINISHED, "skipped", "upstream_stop"),
    ]
    unit_finished = next(
        event
        for event in observer.events
        if event.entity is RunEventEntity.UNIT
        and event.action is RunEventAction.FINISHED
        and event.unit_id == "phase:alpha"
    )
    assert (unit_finished.status, unit_finished.reason_code) == ("failed", "unit_stopped")
    assert "stopped by 'phase.oracle': crackfortran" in unit_finished.reason
    assert [(e.entity, e.action) for e in observer.events[:1]] == [
        (RunEventEntity.RUN, RunEventAction.STARTED)
    ]
    assert (observer.events[-1].entity, observer.events[-1].status) == (
        RunEventEntity.RUN,
        "failed",
    )
    assert [event.sequence for event in observer.events] == list(range(1, len(observer.events) + 1))

    # A unit the gate passed reports no stop, and a run without an observer is unchanged.
    accepted_bundle, accepted_registry = _bundle(tmp_path)
    accepted = verify_recipe_candidates(
        PhaseRecipe(),
        tmp_path,
        accepted_bundle,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=EngineBinding.from_engine(_engine()),
        registry=accepted_registry,
    )
    assert accepted.accepted
    assert all(item.stopped_by is None for item in accepted.units)
    assert all(item.to_dict()["stopped_by"] is None for item in accepted.units)


def test_untransformed_unit_fails_the_verification_run_it_was_never_verified_in(
    tmp_path: Path,
) -> None:
    bundle, registry = _bundle(tmp_path)
    (transformed,) = bundle.units
    untransformed = replace(
        bundle, units=(replace(transformed, transform_status="skipped", candidate=None),)
    )
    observer = RecordingObserver()
    report = verify_recipe_candidates(
        PhaseRecipe(),
        tmp_path,
        untransformed,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=EngineBinding.from_engine(_engine()),
        registry=registry,
        observer=observer,
    )

    assert not report.accepted
    assert "binding.transform_coverage" in report.reason_codes
    (unit,) = report.units
    assert (unit.stage_status, unit.stopped_by) == ("incomplete", None)
    # The run's terminal event agrees with the unit's, not with an outcome-less
    # UnitRun that would have counted as passed.
    lifecycle = [
        (event.entity, event.action, event.status, event.reason_code)
        for event in observer.events
        if event.entity in {RunEventEntity.RUN, RunEventEntity.UNIT}
    ]
    assert lifecycle == [
        (RunEventEntity.RUN, RunEventAction.STARTED, "running", "verification_requested"),
        (RunEventEntity.UNIT, RunEventAction.STARTED, "running", "unit_bundled"),
        (RunEventEntity.UNIT, RunEventAction.FINISHED, "incomplete", "unit_not_transformed"),
        (RunEventEntity.RUN, RunEventAction.FINISHED, "failed", "verification_walked"),
    ]
    assert observer.events[-1].reason == "1 incomplete"
    assert not any(event.entity is RunEventEntity.STAGE for event in observer.events)


def test_verification_never_executes_caller_recipe_hooks(tmp_path: Path) -> None:
    bundle, registry = _bundle(tmp_path)

    report = verify_recipe_candidates(
        HostileVerificationRecipe(),
        tmp_path,
        bundle,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=bundle.engine,
        registry=registry,
    )

    assert report.accepted
    assert report.recipe == bundle.recipe
    assert MaliciousTransform.constructed == 0
    assert MaliciousTransform.applied == 0


@pytest.mark.parametrize("mutation", ["candidate", "unit", "facts", "bundle"])
def test_verification_rejects_any_reachable_bundle_mutation(tmp_path: Path, mutation: str) -> None:
    bundle, registry = _bundle(tmp_path)
    MutatingVerifier.mutation = mutation
    MutatingVerifier.target_bundle = bundle
    registry.register("verifier", "phase.gate", MutatingVerifier, replace=True)

    with pytest.raises(PluginError, match=r"mutated the .*CandidateBundle"):
        verify_recipe_candidates(
            HostileVerificationRecipe(),
            tmp_path,
            bundle,
            expected_source_artifact_digest=_SOURCE,
            expected_engine=bundle.engine,
            registry=registry,
        )


def test_decoder_rejects_non_verification_stage_in_frozen_plan(tmp_path: Path) -> None:
    bundle, _ = _bundle(tmp_path)
    document = json.loads(bundle.to_json())
    document["verification_plan"]["stages"].append(
        {
            "kind": "transform",
            "plugin": "malicious.transform",
            "config": {},
            "optional": False,
            "gate": False,
        }
    )
    payload = json.dumps(document, separators=(",", ":"), sort_keys=True).encode()

    with pytest.raises(ConfigError, match="cannot execute stage kind 'transform'"):
        decode_candidate_bundle(payload)


def test_acceptance_fails_closed_on_source_engine_gate_subprogram_or_deferred(
    tmp_path: Path,
) -> None:
    deferred_bundle, registry = _bundle(tmp_path, config={"defer": True})
    wrong_engine = EngineBinding(
        id=deferred_bundle.engine.id,  # type: ignore[union-attr]
        manifest_digest="sha256:" + "f" * 64,
        implementation_digest=_IMPLEMENTATION,
        input_contract_digest=deferred_bundle.engine.input_contract_digest,  # type: ignore[union-attr]
        output_contract_digest=deferred_bundle.engine.output_contract_digest,  # type: ignore[union-attr]
    )
    report = verify_recipe_candidates(
        PhaseRecipe(),
        tmp_path,
        deferred_bundle,
        {"defer": True},
        expected_source_artifact_digest="sha256:" + "9" * 64,
        expected_engine=wrong_engine,
        project_required_gates=("project.missing-gate",),
        required_units=("phase:missing-unit",),
        required_subprograms=("phase:missing",),
        registry=registry,
    )

    assert not report.accepted
    assert "binding.source_artifact" in report.reason_codes
    assert "binding.expected_engine" in report.reason_codes
    assert "binding.required_gate_declarations" in report.reason_codes
    assert "gate.project.missing-gate" in report.reason_codes
    assert "deferred.nonempty" in report.reason_codes
    assert "unit.required_coverage" in report.reason_codes
    assert "subprogram.required_coverage" in report.reason_codes


def test_required_subprogram_needs_an_explicit_well_formed_coverage_ledger(
    tmp_path: Path,
) -> None:
    bundle, registry = _bundle(tmp_path)
    candidate = bundle.units[0].candidate
    assert candidate is not None
    candidate.notes.pop("coverage")

    report = verify_recipe_candidates(
        PhaseRecipe(),
        tmp_path,
        bundle,
        expected_source_artifact_digest=_SOURCE,
        expected_engine=bundle.engine,
        required_subprograms=("step",),
        registry=registry,
    )

    assert not report.accepted
    assert "subprogram.coverage_ledger" in report.reason_codes
    assert "subprogram.required_coverage" in report.reason_codes


def test_phase_config_separates_machine_paths_from_semantic_identity(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="dedicated keyword"):
        transform_recipe(
            PhaseRecipe(),
            tmp_path,
            {"workspace": str(tmp_path / "machine-path")},
            source_artifact_digest=_SOURCE,
            registry=_registry(),
        )
