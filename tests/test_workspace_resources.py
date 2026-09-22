"""A stage input that lives in the run's workspace, declared without a host path.

The identity a phase produces must not depend on where the caller happened to
put its workspace. These run the real phase API twice over the same tiny
recipe, changing **only** the operational workspace, and check that everything
portable is identical while the stage is handed a path under each root.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, ClassVar

import pytest

from recast.errors import ConfigError
from recast.model import OracleRef
from recast.phases import (
    EngineBinding,
    transform_recipe,
    verify_recipe_candidates,
)
from recast.plugins.oracle import Oracle
from recast.plugins.recipe import Recipe, Stage
from recast.registry import Registry
from recast.workspace_resources import (
    WORKSPACE_MARKER,
    resolve_workspace_resources,
    workspace_resource,
)
from tests.test_phases import (  # the existing minimal plugins
    PhaseTransform,
    _engine,
    _registry,
)

RECORDING = "vendor/recording"
PAYLOAD = b"# PROBE example.unit:\nINPUT: 1\n"
PINNED = hashlib.sha256(PAYLOAD).hexdigest()


# ---------------------------------------------------------------- the resolver


def test_a_declaration_becomes_a_path_under_the_given_workspace(tmp_path: Path) -> None:
    resolved = resolve_workspace_resources(workspace_resource(RECORDING), tmp_path)
    assert resolved == str(tmp_path / RECORDING)


def test_declarations_are_found_wherever_a_config_puts_them(tmp_path: Path) -> None:
    value = {"a": [workspace_resource("one"), {"b": workspace_resource("two")}], "c": 3}
    resolved = resolve_workspace_resources(value, tmp_path)
    assert resolved["a"][0] == str(tmp_path / "one")
    assert resolved["a"][1]["b"] == str(tmp_path / "two")
    assert resolved["c"] == 3


def test_without_a_workspace_the_declaration_is_left_alone() -> None:
    declared = workspace_resource(RECORDING)
    assert resolve_workspace_resources(declared, None) == declared


def test_a_location_that_escapes_the_workspace_is_refused(tmp_path: Path) -> None:
    for bad in ("/etc/passwd", "../outside", "a/../../b"):
        with pytest.raises(ConfigError, match="stay inside the workspace"):
            resolve_workspace_resources({WORKSPACE_MARKER: bad}, tmp_path)


def test_a_marker_that_is_not_a_relative_string_is_refused(tmp_path: Path) -> None:
    for bad in (None, 3, "", "   "):
        with pytest.raises(ConfigError, match="relative location"):
            resolve_workspace_resources({WORKSPACE_MARKER: bad}, tmp_path)


def test_a_marker_with_extra_keys_is_not_a_marker(tmp_path: Path) -> None:
    value = {WORKSPACE_MARKER: RECORDING, "else": 1}
    assert resolve_workspace_resources(value, tmp_path) == value


# ------------------------------------------------------- the phases, end to end


class PinnedRecordingOracle(Oracle):
    """Reads the recording the stage was pointed at, and checks it is the pinned one."""

    name = "phase.recording"
    seen: ClassVar[list[str]] = []

    def key(self, unit, facts, config) -> str:
        del facts
        return f"{unit.uid}:{config.get('pin')}"

    def materialize(self, unit, facts, workspace, executor, config) -> OracleRef:
        del facts, workspace, executor
        directory = Path(config["dumps"])
        type(self).seen.append(str(directory))
        payloads = sorted(directory.glob("*.txt")) if directory.is_dir() else []
        if not payloads:
            raise ConfigError(f"no recording at {directory}")
        digest = hashlib.sha256(payloads[0].read_bytes()).hexdigest()
        if digest != config["pin"]:
            raise ConfigError(
                f"the recording at {directory} is {digest[:16]}, "
                f"not the pinned {config['pin'][:16]}"
            )
        return OracleRef(
            unit=unit.uid,
            oracle=self.name,
            key=self.key(unit, None, config),
            handle={"input_source": "recorded", "samples": []},
        )


class RecordingRecipe(Recipe):
    """The tiny recipe: one stage needs a recording, and never names a host path."""

    name = "recording-phase"
    engine_id = "example.phase-engine"

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", "phase.executor"),
            Stage("frontend", "phase.frontend"),
            Stage(
                "transform",
                "phase.transform",
                config={
                    "defer": False,
                    "dumps": workspace_resource(RECORDING),
                    "pin": config.get("pin", PINNED),
                },
            ),
            Stage(
                "oracle",
                "phase.recording",
                config={"dumps": workspace_resource(RECORDING), "pin": config.get("pin", PINNED)},
            ),
            Stage("verifier", "phase.gate", gate=True),
            Stage("store", "phase.store"),
        ]


def registry() -> Registry:
    found = _registry()
    found.register("recipe", RecordingRecipe.name, RecordingRecipe)
    found.register("oracle", PinnedRecordingOracle.name, PinnedRecordingOracle)
    return found


def source_tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "unit.txt").write_text("source")
    return root


def put_recording(workspace: Path, payload: bytes = PAYLOAD) -> Path:
    """What the project profile does: materialize into the leased workspace."""

    directory = workspace / RECORDING
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "call_0001.txt").write_bytes(payload)
    return directory


def transform_at(tmp_path: Path, name: str):
    root = source_tree(tmp_path / "src")
    workspace = tmp_path / name / "workspace"
    workspace.mkdir(parents=True)
    put_recording(workspace)
    bundle = transform_recipe(
        RecordingRecipe(),
        root,
        {"units": ["phase:alpha"]},
        source_artifact_digest="sha256:" + "1" * 64,
        registry=registry(),
        output=tmp_path / name / "out",
        workspace=workspace,
    )
    return bundle, workspace


def test_only_the_operational_workspace_changes_nothing_portable(tmp_path: Path) -> None:
    PhaseTransform.last_config = None
    first, first_workspace = transform_at(tmp_path, "lease-a")
    first_seen = PhaseTransform.last_config["dumps"]
    PhaseTransform.last_config = None
    second, second_workspace = transform_at(tmp_path, "a-much-longer-lease-name-b")
    second_seen = PhaseTransform.last_config["dumps"]

    # the two runs were handed different paths...
    assert first_seen == str(first_workspace / RECORDING)
    assert second_seen == str(second_workspace / RECORDING)
    assert first_seen != second_seen

    # ...and produced the same portable identity
    assert first.digest() == second.digest()
    assert first.semantic_config_digest == second.semantic_config_digest
    assert first.verification_plan == second.verification_plan
    plan_config = next(
        dict(stage.config)
        for stage in first.verification_plan.stages
        if stage.plugin == "phase.recording"
    )
    assert dict(plan_config["dumps"]) == {WORKSPACE_MARKER: RECORDING}, (
        "the frozen plan keeps the declaration, not a path"
    )
    assert str(first_workspace) not in repr(plan_config)
    assert str(second_workspace) not in repr(plan_config)


def verify_at(tmp_path: Path, bundle, name: str, payload: bytes = PAYLOAD, place: bool = True):
    root = source_tree(tmp_path / "src")
    workspace = tmp_path / name / "verify-workspace"
    workspace.mkdir(parents=True)
    if place:
        put_recording(workspace, payload)
    return verify_recipe_candidates(
        RecordingRecipe(),
        root,
        bundle,
        {"units": ["phase:alpha"]},
        expected_source_artifact_digest="sha256:" + "1" * 64,
        expected_engine=EngineBinding.from_engine(_engine()),
        registry=registry(),
        output=tmp_path / name / "out",
        workspace=workspace,
    )


def test_verification_consumes_the_recording_at_whichever_root_it_is_given(tmp_path: Path) -> None:
    bundle, _ = transform_at(tmp_path, "lease-a")
    PinnedRecordingOracle.seen = []
    first = verify_at(tmp_path, bundle, "verify-one")
    second = verify_at(tmp_path, bundle, "verify-two")
    assert len(PinnedRecordingOracle.seen) == 2
    assert PinnedRecordingOracle.seen[0] != PinnedRecordingOracle.seen[1]
    assert PinnedRecordingOracle.seen[0].endswith(RECORDING)
    assert first.accepted == second.accepted
    assert [item["name"] for item in json.loads(first.to_json())["bindings"]] == [
        item["name"] for item in json.loads(second.to_json())["bindings"]
    ]


def test_a_missing_recording_is_refused(tmp_path: Path) -> None:
    bundle, _ = transform_at(tmp_path, "lease-a")
    report = verify_at(tmp_path, bundle, "verify-missing", place=False)
    assert report.accepted is False


def test_a_different_recording_is_refused(tmp_path: Path) -> None:
    bundle, _ = transform_at(tmp_path, "lease-a")
    report = verify_at(
        tmp_path, bundle, "verify-other", payload=b"# PROBE example.unit:\nINPUT: 2\n"
    )
    assert report.accepted is False


class ResourceExecutor:
    """An executor that declares a workspace resource of its own."""

    name = "phase.resource-executor"
    seen: ClassVar[list[Any]] = []

    def __init__(self, **config: Any) -> None:
        type(self).seen.append(config.get("scratch"))

    def submit(self, job) -> str:
        del job
        return "unused"

    def wait(self, handle: str, timeout_s: float | None = None):
        raise NotImplementedError(handle)


class ExecutorResourceRecipe(RecordingRecipe):
    name = "recording-phase-executor"

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        stages = super().stages(config)
        return [
            Stage(
                "executor",
                "phase.resource-executor",
                config={"scratch": workspace_resource("vendor/executor")},
            )
            if stage.kind == "executor"
            else stage
            for stage in stages
        ]


def test_an_executor_stage_resource_is_resolved_too(tmp_path: Path) -> None:
    # Reported by WORKSPACE_RESOURCE_REVIEW: the verification executor was
    # constructed before the workspace was chosen, so it received the
    # declaration rather than a path.
    found = registry()
    found.register("recipe", ExecutorResourceRecipe.name, ExecutorResourceRecipe)
    found.register("executor", ResourceExecutor.name, ResourceExecutor)
    root = source_tree(tmp_path / "src")
    workspace = tmp_path / "lease" / "workspace"
    workspace.mkdir(parents=True)
    put_recording(workspace)
    bundle = transform_recipe(
        ExecutorResourceRecipe(),
        root,
        {"units": ["phase:alpha"]},
        source_artifact_digest="sha256:" + "1" * 64,
        registry=found,
        output=tmp_path / "out",
        workspace=workspace,
    )
    verify_workspace = tmp_path / "verify" / "workspace"
    verify_workspace.mkdir(parents=True)
    put_recording(verify_workspace)
    ResourceExecutor.seen = []
    verify_recipe_candidates(
        ExecutorResourceRecipe(),
        root,
        bundle,
        {"units": ["phase:alpha"]},
        expected_source_artifact_digest="sha256:" + "1" * 64,
        expected_engine=EngineBinding.from_engine(_engine()),
        registry=found,
        output=tmp_path / "verify-out",
        workspace=verify_workspace,
    )
    assert ResourceExecutor.seen, "the executor was constructed"
    assert ResourceExecutor.seen[-1] == str(verify_workspace / "vendor/executor")
