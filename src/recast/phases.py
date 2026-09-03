"""Portable transformation and verification phase boundaries.

``run_recipe`` remains the convenient in-process walk.  Campaign runners need
the same contracts on opposite sides of a durable artifact boundary, though:
one worker may create candidates while another, independently provisioned
worker verifies them.  This module provides that boundary without turning a
Verdict into feedback for a Transform.

The boundary is deliberately asymmetric.  :func:`transform_recipe` is the
only entry point here that resolves a ``transform`` plugin.  The verification
entry point walks a fixed allow-list (executor, oracle, verifier, store) through
a registry facade which rejects every other kind.  Transformation freezes an
inert verification plan in the bundle; verification never calls the supplied
Recipe object and never resolves or instantiates its Transform factory.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Any, ClassVar, cast

from recast.engines import TranslationEngine, canonical_digest, get_engine
from recast.errors import ConfigError, PluginError
from recast.model import Candidate, Facts, Patch, Unit, Verdict
from recast.observe import RunEventAction, RunEventEntity, RunObserver
from recast.plugins.recipe import Recipe, Stage
from recast.registry import REGISTRY, Registry
from recast.run import (
    StageOutcome,
    UnitRun,
    _emit_unit_finished,
    _exception_reason,
    _RunEventEmitter,
    _walk_stage,
    output_root,
    run_recipe,
)

__all__ = [
    "BindingCheck",
    "CandidateBundle",
    "CandidateUnit",
    "DeferredCoverage",
    "EngineBinding",
    "GateCoverage",
    "GateResult",
    "RequiredUnitCoverage",
    "SubprogramCoverage",
    "UnitVerification",
    "VerificationPlan",
    "VerificationReport",
    "VerificationStage",
    "decode_candidate_bundle",
    "encode_candidate_bundle",
    "transform_recipe",
    "verify_recipe_candidates",
]


_BUNDLE_SCHEMA = "recast.candidate-bundle.v2"
_VERIFICATION_PLAN_SCHEMA = "recast.verification-plan.v1"
_REPORT_SCHEMA = "recast.verification-report.v1"
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_BLOB_MEDIA_TYPE = "application/octet-stream"
_OPERATIONAL_CONFIG = frozenset({"output", "workspace", "store_root"})
_TRANSFORM_STATUSES = frozenset({"ok", "skipped", "failed", "not_run"})
_GATE_STATUSES = frozenset({"passed", "failed", "missing", "unrecorded"})
_VERIFICATION_STAGE_KINDS = frozenset({"executor", "oracle", "verifier", "store"})

# A CandidateBundle is a control-plane message, not an unbounded archive.
# Larger generated trees should use a content-store implementation which keeps
# these same descriptors and externalizes the blobs.
MAX_INLINE_BLOB_BYTES = 64 * 1024 * 1024
MAX_INLINE_BLOBS_BYTES = 256 * 1024 * 1024
MAX_INLINE_PATCH_BYTES = 16 * 1024 * 1024
MAX_ENCODED_BUNDLE_BYTES = 350 * 1024 * 1024


def _canonical_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            _json_value(value, "document"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ConfigError(f"phase document is not canonical JSON: {error}") from error


def _json_value(value: object, context: str) -> object:
    """Copy a value into the JSON vocabulary and reject lossy encodings."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{context} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        out: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ConfigError(f"{context} has a non-string key {key!r}")
            out[key] = _json_value(item, f"{context}.{key}")
        return out
    if isinstance(value, (list, tuple)):
        return [_json_value(item, f"{context}[]") for item in value]
    raise ConfigError(f"{context} contains unsupported JSON value {type(value).__name__}")


def _freeze_json(value: object, context: str) -> object:
    """Make a validated JSON value recursively immutable."""

    copied = _json_value(value, context)
    if isinstance(copied, dict):
        return MappingProxyType(
            {key: _freeze_json(item, f"{context}.{key}") for key, item in copied.items()}
        )
    if isinstance(copied, list):
        return tuple(_freeze_json(item, f"{context}[]") for item in copied)
    return copied


def _digest_bytes(payload: bytes) -> str:
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _require_digest(value: str, context: str) -> str:
    if not _SHA256.fullmatch(value):
        raise ConfigError(f"{context} must be a lowercase sha256 digest")
    return value


def _portable_path(value: Path | str, context: str) -> str:
    text = value.as_posix() if isinstance(value, Path) else value
    if not text or text == "." or "\\" in text or "\x00" in text:
        raise ConfigError(f"{context} must be a non-empty portable relative path")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != text:
        raise ConfigError(f"{context} must be a normalized portable relative path, not {text!r}")
    return text


def _semantic_config(config: Mapping[str, object]) -> dict[str, object]:
    operational = sorted(_OPERATIONAL_CONFIG.intersection(config))
    if operational:
        raise ConfigError(
            f"phase config contains operational key(s) {operational}; pass output/workspace "
            "through the dedicated keyword arguments so bundle identity stays machine-independent"
        )
    value = _json_value(config, "config")
    if not isinstance(value, dict):  # Mapping above makes this defensive only.
        raise ConfigError("phase config must be a JSON object")
    return value


@dataclass(frozen=True)
class EngineBinding:
    """Exact engine and artifact-contract identity carried across the phase."""

    id: str
    manifest_digest: str
    implementation_digest: str
    input_contract_digest: str
    output_contract_digest: str

    def __post_init__(self) -> None:
        if not self.id:
            raise ConfigError("engine binding id must not be empty")
        for name in (
            "manifest_digest",
            "implementation_digest",
            "input_contract_digest",
            "output_contract_digest",
        ):
            _require_digest(cast(str, getattr(self, name)), f"engine binding {name}")

    @classmethod
    def from_engine(cls, engine: TranslationEngine) -> EngineBinding:
        return cls(
            id=engine.id,
            manifest_digest=engine.digest(),
            implementation_digest=engine.implementation_digest,
            input_contract_digest=engine.input_artifact_contract.digest(),
            output_contract_digest=engine.output_artifact_contract.digest(),
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "manifest_digest": self.manifest_digest,
            "implementation_digest": self.implementation_digest,
            "input_contract_digest": self.input_contract_digest,
            "output_contract_digest": self.output_contract_digest,
        }


@dataclass(frozen=True, slots=True)
class VerificationStage:
    """One inert, recursively frozen verification-stage declaration."""

    kind: str
    plugin: str
    config: Mapping[str, object]
    optional: bool = False
    gate: bool = False

    def __post_init__(self) -> None:
        if self.kind not in _VERIFICATION_STAGE_KINDS:
            raise ConfigError(
                f"verification plan cannot execute stage kind {self.kind!r}; allowed kinds are "
                f"{sorted(_VERIFICATION_STAGE_KINDS)}"
            )
        if not self.plugin:
            raise ConfigError("verification plan stage plugin must not be empty")
        if type(self.optional) is not bool or type(self.gate) is not bool:
            raise ConfigError("verification plan optional and gate fields must be booleans")
        frozen = _freeze_json(self.config, f"verification stage {self.plugin}.config")
        if not isinstance(frozen, Mapping):
            raise ConfigError(f"verification stage {self.plugin!r} config must be an object")
        object.__setattr__(self, "config", frozen)

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "plugin": self.plugin,
            "config": _json_value(self.config, f"verification stage {self.plugin}.config"),
            "optional": self.optional,
            "gate": self.gate,
        }

    def to_stage(self) -> Stage:
        config = _json_value(self.config, f"verification stage {self.plugin}.config")
        assert isinstance(config, dict)
        return Stage(
            kind=self.kind,
            plugin=self.plugin,
            config=cast(dict[str, Any], config),
            optional=self.optional,
            gate=self.gate,
        )


@dataclass(frozen=True, slots=True)
class VerificationPlan:
    """Canonical recipe projection which contains no executable Transform."""

    recipe: str
    engine_id: str | None
    frontend_plugins: tuple[str, ...]
    transform_plugin: str
    stages: tuple[VerificationStage, ...]

    schema: ClassVar[str] = _VERIFICATION_PLAN_SCHEMA

    def __post_init__(self) -> None:
        if not self.recipe:
            raise ConfigError("verification plan recipe must not be empty")
        if self.engine_id is not None and not self.engine_id:
            raise ConfigError("verification plan engine_id must be non-empty when present")
        if not self.frontend_plugins or any(not item for item in self.frontend_plugins):
            raise ConfigError("verification plan must name every frontend plugin")
        if len(set(self.frontend_plugins)) != len(self.frontend_plugins):
            raise ConfigError("verification plan repeats a frontend plugin")
        if not self.transform_plugin:
            raise ConfigError("verification plan transform_plugin must not be empty")
        if any(type(stage) is not VerificationStage for stage in self.stages):
            raise ConfigError("verification plan stages must be VerificationStage values")
        executors = [stage for stage in self.stages if stage.kind == "executor"]
        handed = [stage for stage in self.stages if stage.kind in {"oracle", "verifier"}]
        if handed and len(executors) != 1:
            raise ConfigError(
                "verification plan requires exactly one executor when an oracle or verifier "
                "is declared"
            )
        if executors and self.stages[0] is not executors[0]:
            raise ConfigError("verification plan executor declaration must be first")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "recipe": self.recipe,
            "engine_id": self.engine_id,
            "frontend_plugins": list(self.frontend_plugins),
            "transform_plugin": self.transform_plugin,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    def digest(self) -> str:
        return _digest_bytes(_canonical_bytes(self.to_dict()))


@dataclass(frozen=True)
class CandidateUnit:
    """One selected unit and the exact frontend/transform result for it."""

    unit: Unit
    facts: Facts | None
    transform_status: str
    candidate: Candidate | None
    outcome_digest: str

    def __post_init__(self) -> None:
        if self.transform_status not in _TRANSFORM_STATUSES:
            raise ConfigError(f"unknown transform status {self.transform_status!r}")
        _require_digest(self.outcome_digest, "candidate unit outcome_digest")
        if self.facts is not None and self.facts.unit != self.unit.uid:
            raise ConfigError(
                f"facts for {self.facts.unit!r} are bound to candidate unit {self.unit.uid!r}"
            )
        if self.candidate is not None and self.candidate.unit != self.unit.uid:
            raise ConfigError(
                f"candidate for {self.candidate.unit!r} is bound to unit {self.unit.uid!r}"
            )
        if self.transform_status == "ok" and (self.facts is None or self.candidate is None):
            raise ConfigError("an ok transform result requires both facts and a candidate")
        if self.transform_status != "ok" and self.candidate is not None:
            raise ConfigError("only an ok transform result may carry a candidate")


@dataclass(frozen=True)
class CandidateBundle:
    """Self-contained, canonical candidates produced by one recipe transform.

    Machine paths and timestamps are intentionally absent.  Generated files
    are addressed by digest and use portable relative paths; their inline bytes
    are a bounded transport representation, not their identity.
    """

    recipe: str
    semantic_config_digest: str
    source_artifact_digest: str
    engine: EngineBinding | None
    frontend_plugins: tuple[str, ...]
    transform_plugin: str
    verification_plan: VerificationPlan
    discovered_units: tuple[Unit, ...]
    units: tuple[CandidateUnit, ...]

    schema: ClassVar[str] = _BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if not self.recipe:
            raise ConfigError("candidate bundle recipe must not be empty")
        _require_digest(self.semantic_config_digest, "candidate bundle semantic_config_digest")
        _require_digest(self.source_artifact_digest, "candidate bundle source_artifact_digest")
        if not self.frontend_plugins or any(not item for item in self.frontend_plugins):
            raise ConfigError("candidate bundle must name every frontend plugin")
        if len(set(self.frontend_plugins)) != len(self.frontend_plugins):
            raise ConfigError("candidate bundle repeats a frontend plugin")
        if not self.transform_plugin:
            raise ConfigError("candidate bundle transform_plugin must not be empty")
        expected_engine_id = self.engine.id if self.engine is not None else None
        if (
            self.verification_plan.recipe != self.recipe
            or self.verification_plan.engine_id != expected_engine_id
            or self.verification_plan.frontend_plugins != self.frontend_plugins
            or self.verification_plan.transform_plugin != self.transform_plugin
        ):
            raise ConfigError(
                "candidate bundle verification plan does not match its recipe/engine/stage "
                "provenance"
            )

        discovered = [unit.uid for unit in self.discovered_units]
        if len(discovered) != len(set(discovered)):
            raise ConfigError("candidate bundle repeats a discovered unit uid")
        selected = [item.unit.uid for item in self.units]
        if len(selected) != len(set(selected)):
            raise ConfigError("candidate bundle repeats a selected unit uid")
        missing = sorted(set(selected).difference(discovered))
        if missing:
            raise ConfigError(f"selected unit(s) {missing} are absent from discovery coverage")

        for unit in self.discovered_units:
            _unit_document(unit)
        total_blobs: dict[str, int] = {}
        for item in self.units:
            _unit_document(item.unit)
            if item.facts is not None:
                _facts_document(item.facts)
            candidate = item.candidate
            if candidate is None:
                continue
            _json_value(candidate.notes, f"candidate {candidate.unit}.notes")
            for index, deferred in enumerate(candidate.deferred):
                if not isinstance(deferred, str):
                    raise ConfigError(
                        f"candidate {candidate.unit}.deferred[{index}] must be a string"
                    )
            for path, content in candidate.files.items():
                _portable_path(path, f"candidate {candidate.unit} file")
                if len(content) > MAX_INLINE_BLOB_BYTES:
                    raise ConfigError(
                        f"candidate file {path} is {len(content)} bytes; inline limit is "
                        f"{MAX_INLINE_BLOB_BYTES}"
                    )
                total_blobs[_digest_bytes(content)] = len(content)
            for index, patch in enumerate(candidate.patches):
                _portable_path(patch.target, f"candidate {candidate.unit} patch[{index}].target")
                if len(patch.diff.encode("utf-8")) > MAX_INLINE_PATCH_BYTES:
                    raise ConfigError(
                        f"candidate patch {index} exceeds the {MAX_INLINE_PATCH_BYTES}-byte limit"
                    )
        if sum(total_blobs.values()) > MAX_INLINE_BLOBS_BYTES:
            raise ConfigError(
                f"candidate bundle has {sum(total_blobs.values())} unique blob bytes; inline "
                f"limit is {MAX_INLINE_BLOBS_BYTES}"
            )

    def to_dict(self) -> dict[str, object]:
        blobs: dict[str, bytes] = {}
        unit_documents: list[dict[str, object]] = []
        for item in sorted(self.units, key=lambda entry: entry.unit.uid):
            candidate_document: dict[str, object] | None = None
            if item.candidate is not None:
                candidate_document = _candidate_document(item.candidate, blobs)
            unit_documents.append(
                {
                    "unit": item.unit.uid,
                    "facts": _facts_document(item.facts) if item.facts is not None else None,
                    "transform_status": item.transform_status,
                    "outcome_digest": item.outcome_digest,
                    "candidate": candidate_document,
                }
            )
        blob_documents = [
            {
                "digest": digest,
                "size": len(content),
                "media_type": _BLOB_MEDIA_TYPE,
                "encoding": "base64",
                "data": base64.b64encode(content).decode("ascii"),
            }
            for digest, content in sorted(blobs.items())
        ]
        return {
            "schema": self.schema,
            "recipe": self.recipe,
            "semantic_config_digest": self.semantic_config_digest,
            "source_artifact_digest": self.source_artifact_digest,
            "engine": self.engine.to_dict() if self.engine is not None else None,
            "frontend_plugins": list(self.frontend_plugins),
            "transform_plugin": self.transform_plugin,
            "verification_plan": self.verification_plan.to_dict(),
            "discovered_units": [
                _unit_document(unit) for unit in sorted(self.discovered_units, key=lambda u: u.uid)
            ],
            "units": unit_documents,
            "blobs": blob_documents,
        }

    def to_json(self) -> bytes:
        payload = _canonical_bytes(self.to_dict())
        if len(payload) > MAX_ENCODED_BUNDLE_BYTES:
            raise ConfigError(
                f"encoded candidate bundle is {len(payload)} bytes; limit is "
                f"{MAX_ENCODED_BUNDLE_BYTES}"
            )
        return payload

    def digest(self) -> str:
        return _digest_bytes(self.to_json())

    @classmethod
    def from_json(cls, payload: bytes | str) -> CandidateBundle:
        return decode_candidate_bundle(payload)


def _unit_document(unit: Unit) -> dict[str, object]:
    return {
        "uid": unit.uid,
        "kind": unit.kind,
        "sources": sorted(
            _portable_path(path, f"unit {unit.uid}.sources") for path in unit.sources
        ),
        "parent": unit.parent,
        "attrs": _json_value(unit.attrs, f"unit {unit.uid}.attrs"),
    }


def _facts_document(facts: Facts) -> dict[str, object]:
    return {
        "unit": facts.unit,
        "interface": _json_value(facts.interface, f"facts {facts.unit}.interface"),
        "constants": _json_value(facts.constants, f"facts {facts.unit}.constants"),
        "callgraph": _json_value(facts.callgraph, f"facts {facts.unit}.callgraph"),
        "effects": _json_value(facts.effects, f"facts {facts.unit}.effects"),
        "provenance": _json_value(facts.provenance, f"facts {facts.unit}.provenance"),
        "extra": _json_value(facts.extra, f"facts {facts.unit}.extra"),
    }


def _candidate_document(candidate: Candidate, blobs: dict[str, bytes]) -> dict[str, object]:
    files: list[dict[str, object]] = []
    for path, content in sorted(candidate.files.items(), key=lambda item: item[0].as_posix()):
        digest = _digest_bytes(content)
        previous = blobs.setdefault(digest, content)
        if previous != content:  # Cryptographic collision: refuse rather than alias.
            raise ConfigError(f"two candidate blobs claim digest {digest}")
        files.append(
            {
                "path": _portable_path(path, f"candidate {candidate.unit} file"),
                "blob_digest": digest,
                "size": len(content),
                "media_type": _BLOB_MEDIA_TYPE,
            }
        )
    return {
        "unit": candidate.unit,
        "transform": candidate.transform,
        "candidate_digest": f"sha256:{candidate.digest()}",
        "files": files,
        "patches": [
            {
                "target": _portable_path(
                    patch.target, f"candidate {candidate.unit} patch[{index}].target"
                ),
                "diff": patch.diff,
                "order": patch.order,
            }
            for index, patch in enumerate(candidate.patches)
        ],
        "deferred": list(candidate.deferred),
        "notes": _json_value(candidate.notes, f"candidate {candidate.unit}.notes"),
    }


def encode_candidate_bundle(bundle: CandidateBundle) -> bytes:
    """Encode ``bundle`` as its one canonical UTF-8 JSON representation."""
    return bundle.to_json()


def _reject_constant(value: str) -> None:
    raise ConfigError(f"candidate bundle contains non-JSON constant {value}")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ConfigError(f"candidate bundle repeats object key {key!r}")
        out[key] = value
    return out


def decode_candidate_bundle(payload: bytes | str) -> CandidateBundle:
    """Decode and strictly validate a canonical CandidateBundle document."""
    encoded = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(encoded) > MAX_ENCODED_BUNDLE_BYTES:
        raise ConfigError(
            f"encoded candidate bundle is {len(encoded)} bytes; limit is {MAX_ENCODED_BUNDLE_BYTES}"
        )
    try:
        raw = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ConfigError("candidate bundle is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise ConfigError(f"candidate bundle is not valid JSON: {error.msg}") from error
    document = _object(
        raw,
        "candidate bundle",
        {
            "schema",
            "recipe",
            "semantic_config_digest",
            "source_artifact_digest",
            "engine",
            "frontend_plugins",
            "transform_plugin",
            "verification_plan",
            "discovered_units",
            "units",
            "blobs",
        },
    )
    if _string(document["schema"], "candidate bundle.schema") != _BUNDLE_SCHEMA:
        raise ConfigError(f"unsupported candidate bundle schema {document['schema']!r}")

    blob_table = _decode_blobs(document["blobs"])
    discovered_units = tuple(
        _decode_unit(item, f"candidate bundle.discovered_units[{index}]")
        for index, item in enumerate(_array(document["discovered_units"], "discovered_units"))
    )
    by_uid = {unit.uid: unit for unit in discovered_units}
    units = tuple(
        _decode_candidate_unit(item, index, by_uid, blob_table)
        for index, item in enumerate(_array(document["units"], "units"))
    )
    referenced = {
        _digest_bytes(content)
        for item in units
        if item.candidate is not None
        for content in item.candidate.files.values()
    }
    unused = sorted(set(blob_table).difference(referenced))
    if unused:
        raise ConfigError(f"candidate bundle carries unreferenced blob(s) {unused}")

    engine_raw = document["engine"]
    engine = None if engine_raw is None else _decode_engine(engine_raw)
    bundle = CandidateBundle(
        recipe=_string(document["recipe"], "candidate bundle.recipe"),
        semantic_config_digest=_require_digest(
            _string(document["semantic_config_digest"], "semantic_config_digest"),
            "candidate bundle semantic_config_digest",
        ),
        source_artifact_digest=_require_digest(
            _string(document["source_artifact_digest"], "source_artifact_digest"),
            "candidate bundle source_artifact_digest",
        ),
        engine=engine,
        frontend_plugins=tuple(
            _string(item, f"frontend_plugins[{index}]")
            for index, item in enumerate(_array(document["frontend_plugins"], "frontend_plugins"))
        ),
        transform_plugin=_string(document["transform_plugin"], "transform_plugin"),
        verification_plan=_decode_verification_plan(document["verification_plan"]),
        discovered_units=discovered_units,
        units=units,
    )
    if encoded != bundle.to_json():
        raise ConfigError("candidate bundle is valid but not in canonical JSON form")
    return bundle


def _object(value: object, context: str, keys: set[str]) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ConfigError(f"{context} must be an object")
    actual = set(value)
    missing = sorted(keys.difference(actual))
    unknown = sorted(actual.difference(keys))
    if missing or unknown:
        parts = []
        if missing:
            parts.append(f"missing {missing}")
        if unknown:
            parts.append(f"unknown {unknown}")
        raise ConfigError(f"{context} has " + " and ".join(parts))
    return cast(dict[str, object], value)


def _array(value: object, context: str) -> list[object]:
    if not isinstance(value, list):
        raise ConfigError(f"{context} must be an array")
    return value


def _string(value: object, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{context} must be a non-empty string")
    return value


def _integer(value: object, context: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ConfigError(f"{context} must be at least {minimum}")
    return value


def _nullable_string(value: object, context: str) -> str | None:
    return None if value is None else _string(value, context)


def _decode_engine(value: object) -> EngineBinding:
    document = _object(
        value,
        "candidate bundle.engine",
        {
            "id",
            "manifest_digest",
            "implementation_digest",
            "input_contract_digest",
            "output_contract_digest",
        },
    )
    return EngineBinding(
        id=_string(document["id"], "engine.id"),
        manifest_digest=_string(document["manifest_digest"], "engine.manifest_digest"),
        implementation_digest=_string(
            document["implementation_digest"], "engine.implementation_digest"
        ),
        input_contract_digest=_string(
            document["input_contract_digest"], "engine.input_contract_digest"
        ),
        output_contract_digest=_string(
            document["output_contract_digest"], "engine.output_contract_digest"
        ),
    )


def _decode_verification_plan(value: object) -> VerificationPlan:
    document = _object(
        value,
        "candidate bundle.verification_plan",
        {
            "schema",
            "recipe",
            "engine_id",
            "frontend_plugins",
            "transform_plugin",
            "stages",
        },
    )
    if document["schema"] != _VERIFICATION_PLAN_SCHEMA:
        raise ConfigError(f"unsupported verification plan schema {document['schema']!r}")
    stages: list[VerificationStage] = []
    for index, raw_stage in enumerate(_array(document["stages"], "verification_plan.stages")):
        context = f"verification_plan.stages[{index}]"
        stage = _object(raw_stage, context, {"kind", "plugin", "config", "optional", "gate"})
        config = stage["config"]
        if not isinstance(config, dict):
            raise ConfigError(f"{context}.config must be an object")
        optional = stage["optional"]
        gate = stage["gate"]
        if type(optional) is not bool or type(gate) is not bool:
            raise ConfigError(f"{context}.optional and gate must be booleans")
        stages.append(
            VerificationStage(
                kind=_string(stage["kind"], f"{context}.kind"),
                plugin=_string(stage["plugin"], f"{context}.plugin"),
                config=cast(dict[str, object], config),
                optional=optional,
                gate=gate,
            )
        )
    return VerificationPlan(
        recipe=_string(document["recipe"], "verification_plan.recipe"),
        engine_id=_nullable_string(document["engine_id"], "verification_plan.engine_id"),
        frontend_plugins=tuple(
            _string(item, f"verification_plan.frontend_plugins[{index}]")
            for index, item in enumerate(
                _array(document["frontend_plugins"], "verification_plan.frontend_plugins")
            )
        ),
        transform_plugin=_string(
            document["transform_plugin"], "verification_plan.transform_plugin"
        ),
        stages=tuple(stages),
    )


def _decode_blobs(value: object) -> dict[str, bytes]:
    blobs: dict[str, bytes] = {}
    total = 0
    for index, item in enumerate(_array(value, "candidate bundle.blobs")):
        context = f"candidate bundle.blobs[{index}]"
        document = _object(item, context, {"digest", "size", "media_type", "encoding", "data"})
        digest = _require_digest(_string(document["digest"], f"{context}.digest"), context)
        if digest in blobs:
            raise ConfigError(f"candidate bundle repeats blob {digest}")
        size = _integer(document["size"], f"{context}.size", minimum=0)
        if size > MAX_INLINE_BLOB_BYTES:
            raise ConfigError(f"{context} exceeds the {MAX_INLINE_BLOB_BYTES}-byte limit")
        if document["media_type"] != _BLOB_MEDIA_TYPE:
            raise ConfigError(f"{context}.media_type must be {_BLOB_MEDIA_TYPE!r}")
        if document["encoding"] != "base64":
            raise ConfigError(f"{context}.encoding must be 'base64'")
        data = _string(document["data"], f"{context}.data") if size else document["data"]
        if not isinstance(data, str):
            raise ConfigError(f"{context}.data must be a string")
        try:
            decoded = base64.b64decode(data, validate=True)
        except (binascii.Error, ValueError) as error:
            raise ConfigError(f"{context}.data is not canonical base64") from error
        if base64.b64encode(decoded).decode("ascii") != data:
            raise ConfigError(f"{context}.data is not canonical base64")
        if len(decoded) != size:
            raise ConfigError(f"{context}.size does not match decoded content")
        if _digest_bytes(decoded) != digest:
            raise ConfigError(f"{context}.digest does not match decoded content")
        blobs[digest] = decoded
        total += size
        if total > MAX_INLINE_BLOBS_BYTES:
            raise ConfigError(f"candidate bundle exceeds {MAX_INLINE_BLOBS_BYTES} blob bytes")
    return blobs


def _decode_unit(value: object, context: str) -> Unit:
    document = _object(value, context, {"uid", "kind", "sources", "parent", "attrs"})
    attrs = document["attrs"]
    if not isinstance(attrs, dict):
        raise ConfigError(f"{context}.attrs must be an object")
    return Unit(
        uid=_string(document["uid"], f"{context}.uid"),
        kind=_string(document["kind"], f"{context}.kind"),
        sources=tuple(
            Path(_portable_path(_string(item, f"{context}.sources[{index}]"), context))
            for index, item in enumerate(_array(document["sources"], f"{context}.sources"))
        ),
        parent=_nullable_string(document["parent"], f"{context}.parent"),
        attrs=cast(dict[str, Any], attrs),
    )


def _decode_facts(value: object, context: str) -> Facts:
    keys = {"unit", "interface", "constants", "callgraph", "effects", "provenance", "extra"}
    document = _object(value, context, keys)
    tables: dict[str, dict[str, Any]] = {}
    for key in keys.difference({"unit"}):
        table = document[key]
        if not isinstance(table, dict):
            raise ConfigError(f"{context}.{key} must be an object")
        tables[key] = cast(dict[str, Any], table)
    return Facts(unit=_string(document["unit"], f"{context}.unit"), **tables)


def _decode_candidate_unit(
    value: object,
    index: int,
    discovered: Mapping[str, Unit],
    blobs: Mapping[str, bytes],
) -> CandidateUnit:
    context = f"candidate bundle.units[{index}]"
    document = _object(
        value, context, {"unit", "facts", "transform_status", "outcome_digest", "candidate"}
    )
    uid = _string(document["unit"], f"{context}.unit")
    if uid not in discovered:
        raise ConfigError(f"{context}.unit {uid!r} was not discovered")
    facts = (
        None if document["facts"] is None else _decode_facts(document["facts"], f"{context}.facts")
    )
    candidate = (
        None
        if document["candidate"] is None
        else _decode_candidate(document["candidate"], f"{context}.candidate", blobs)
    )
    return CandidateUnit(
        unit=discovered[uid],
        facts=facts,
        transform_status=_string(document["transform_status"], f"{context}.transform_status"),
        candidate=candidate,
        outcome_digest=_string(document["outcome_digest"], f"{context}.outcome_digest"),
    )


def _decode_candidate(value: object, context: str, blobs: Mapping[str, bytes]) -> Candidate:
    document = _object(
        value,
        context,
        {"unit", "transform", "candidate_digest", "files", "patches", "deferred", "notes"},
    )
    files: dict[Path, bytes] = {}
    for index, item in enumerate(_array(document["files"], f"{context}.files")):
        item_context = f"{context}.files[{index}]"
        descriptor = _object(item, item_context, {"path", "blob_digest", "size", "media_type"})
        path = Path(
            _portable_path(_string(descriptor["path"], f"{item_context}.path"), item_context)
        )
        if path in files:
            raise ConfigError(f"{context} repeats file path {path}")
        digest = _require_digest(
            _string(descriptor["blob_digest"], f"{item_context}.blob_digest"), item_context
        )
        if descriptor["media_type"] != _BLOB_MEDIA_TYPE:
            raise ConfigError(f"{item_context}.media_type must be {_BLOB_MEDIA_TYPE!r}")
        if digest not in blobs:
            raise ConfigError(f"{item_context} references missing blob {digest}")
        content = blobs[digest]
        if _integer(descriptor["size"], f"{item_context}.size", minimum=0) != len(content):
            raise ConfigError(f"{item_context}.size does not match its blob")
        files[path] = content

    patches: list[Patch] = []
    for index, item in enumerate(_array(document["patches"], f"{context}.patches")):
        item_context = f"{context}.patches[{index}]"
        descriptor = _object(item, item_context, {"target", "diff", "order"})
        document_diff = descriptor["diff"]
        if not isinstance(document_diff, str):
            raise ConfigError(f"{item_context}.diff must be a string")
        diff = document_diff
        if len(diff.encode("utf-8")) > MAX_INLINE_PATCH_BYTES:
            raise ConfigError(f"{item_context}.diff exceeds the inline patch limit")
        patches.append(
            Patch(
                target=Path(
                    _portable_path(
                        _string(descriptor["target"], f"{item_context}.target"), item_context
                    )
                ),
                diff=diff,
                order=_integer(descriptor["order"], f"{item_context}.order"),
            )
        )
    deferred = [
        _string(item, f"{context}.deferred[{index}]")
        for index, item in enumerate(_array(document["deferred"], f"{context}.deferred"))
    ]
    notes = document["notes"]
    if not isinstance(notes, dict):
        raise ConfigError(f"{context}.notes must be an object")
    candidate = Candidate(
        unit=_string(document["unit"], f"{context}.unit"),
        transform=_string(document["transform"], f"{context}.transform"),
        files=files,
        patches=patches,
        deferred=deferred,
        notes=cast(dict[str, Any], notes),
    )
    claimed = _require_digest(
        _string(document["candidate_digest"], f"{context}.candidate_digest"), context
    )
    if candidate.digest() != claimed.removeprefix("sha256:"):
        # Candidate.digest predates the catalog convention and returns bare hex;
        # the portable document consistently labels the algorithm.
        raise ConfigError(f"{context}.candidate_digest does not match candidate content")
    return candidate


class _TransformPhaseRecipe(Recipe):
    """A fixed projection which can resolve only frontend and transform stages."""

    def __init__(self, original: Recipe, stages: Sequence[Stage]) -> None:
        self._original = original
        self._stages = list(stages)
        self.name = original.name
        self.summary = original.summary
        self.engine_id = original.engine_id

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        del config
        return list(self._stages)

    def validate(self, config: dict[str, Any]) -> list[str]:
        return self._original.validate(config)

    def resolved_engine_id(self, config: dict[str, Any]) -> str | None:
        return self._original.resolved_engine_id(config)


def _engine_binding(
    recipe: Recipe, config: dict[str, Any], registry: Registry
) -> EngineBinding | None:
    engine_id = recipe.resolved_engine_id(config)
    if engine_id is None:
        return None
    return EngineBinding.from_engine(get_engine(engine_id, registry=registry))


def _resolved_stage_config(
    stage: Stage, semantic_config: Mapping[str, object]
) -> dict[str, object]:
    declared = _json_value(stage.config, f"stage {stage.plugin}.config")
    if not isinstance(declared, dict):
        raise ConfigError(f"stage {stage.plugin!r} config must be an object")
    configured_stages = semantic_config.get("stages", {})
    if not isinstance(configured_stages, Mapping):
        raise ConfigError("config.stages must be an object")
    configured = configured_stages.get(stage.plugin, {})
    if not isinstance(configured, Mapping):
        raise ConfigError(f"config.stages[{stage.plugin!r}] must be an object")
    override = _json_value(configured, f"config.stages[{stage.plugin!r}]")
    if not isinstance(override, dict):  # Mapping above makes this defensive only.
        raise ConfigError(f"config.stages[{stage.plugin!r}] must be an object")
    effective: dict[str, object] = {**declared, **override}
    compiler_semantics = semantic_config.get("compiler_semantics")
    if compiler_semantics is not None:
        if not isinstance(compiler_semantics, str) or not compiler_semantics:
            raise ConfigError("config.compiler_semantics must be a non-empty string")
        # Compiler lowering and the compiler used for the golden reference are
        # one semantic choice.  Keeping it only at the top level would put the
        # value in CandidateBundle identity while neither plugin actually saw
        # it.  Derive the two typed stage fields here so they are frozen into
        # the verification plan. Campaign accepts only reviewed compiler names;
        # standalone callers retain Recast's normal trusted-config boundary.
        inherited: tuple[str, str] | None = None
        if stage.kind == "transform":
            inherited = ("profile", compiler_semantics)
        elif stage.kind == "oracle" and stage.plugin in ("f2py-golden", "f2py-golden-flat"):
            # The flat oracle compiles the same reference, plus the tree it
            # uses, with the same compiler: it reads ``fc`` for both builds.
            inherited = ("fc", compiler_semantics)
        if inherited is not None:
            key, value = inherited
            configured = effective.get(key)
            if configured is not None and configured != value:
                raise ConfigError(f"stage {stage.plugin!r} {key} contradicts compiler_semantics")
            effective[key] = value
    if stage.kind != "executor" and semantic_config.get("range"):
        effective = {"range": semantic_config["range"], **effective}
    return effective


def _build_verification_plan(
    *,
    recipe_name: str,
    engine_id: str | None,
    frontends: Sequence[Stage],
    transform: Stage,
    stages: Sequence[Stage],
    semantic_config: Mapping[str, object],
) -> VerificationPlan:
    unsupported = sorted(
        {stage.kind for stage in stages}.difference(
            {"frontend", "transform", *_VERIFICATION_STAGE_KINDS}
        )
    )
    if unsupported:
        raise ConfigError(
            "phased verification cannot execute recipe stage kind(s) " + ", ".join(unsupported)
        )
    declarations = tuple(
        VerificationStage(
            kind=stage.kind,
            plugin=stage.plugin,
            config=_resolved_stage_config(stage, semantic_config),
            optional=stage.optional,
            gate=stage.gate,
        )
        for stage in stages
        if stage.kind in _VERIFICATION_STAGE_KINDS
    )
    return VerificationPlan(
        recipe=recipe_name,
        engine_id=engine_id,
        frontend_plugins=tuple(stage.plugin for stage in frontends),
        transform_plugin=transform.plugin,
        stages=declarations,
    )


def _outcome_digest(outcome: StageOutcome | None) -> str:
    # Raw plugin detail may contain a machine path, a timestamp, or source
    # text.  The portable boundary records the stable classification only;
    # the observer/event stream remains the place for operator-facing detail.
    status = outcome.status if outcome is not None else "not_run"
    return canonical_digest({"status": status})


def transform_recipe(
    recipe: Recipe,
    root: Path,
    config: dict[str, Any] | None = None,
    *,
    source_artifact_digest: str,
    registry: Registry = REGISTRY,
    output: Path | None = None,
    workspace: Path | None = None,
) -> CandidateBundle:
    """Run only the frontend/transform half and return a portable bundle.

    ``source_artifact_digest`` is the caller's content-store identity for the
    exact source tree mounted at ``root``.  RecastEngine records and later
    checks that binding; it does not invent a second tree hashing convention.
    Operational paths are separate keyword arguments and never enter bundle
    identity.
    """
    semantic = _semantic_config(config or {})
    source_digest = _require_digest(source_artifact_digest, "source_artifact_digest")
    phase_config = cast(dict[str, Any], dict(semantic))
    stages = recipe.stages(phase_config)
    frontends = [stage for stage in stages if stage.kind == "frontend"]
    transforms = [stage for stage in stages if stage.kind == "transform"]
    if not frontends:
        raise ConfigError(f"recipe {recipe.name!r} declares no frontend stage")
    if len(transforms) != 1:
        raise ConfigError(
            f"transformation phase requires exactly one transform stage; recipe "
            f"{recipe.name!r} declares {len(transforms)}"
        )
    projection = _TransformPhaseRecipe(recipe, [*frontends, transforms[0]])
    runtime_config = dict(phase_config)
    # ``run_recipe`` normally merges only config["stages"] into plugin calls.
    # Resolve trusted semantic inheritance here as well as in the frozen
    # verification plan; otherwise compiler_semantics would enter the bundle
    # digest while the Transform silently used its own default profile.
    configured_stages = semantic.get("stages", {})
    if not isinstance(configured_stages, Mapping):
        raise ConfigError("config.stages must be an object")
    runtime_config["stages"] = {
        **configured_stages,
        **{
            stage.plugin: _resolved_stage_config(stage, semantic)
            for stage in (*frontends, transforms[0])
        },
    }
    if output is not None:
        runtime_config["output"] = Path(output).resolve()
    if workspace is not None:
        runtime_config["workspace"] = Path(workspace).resolve()
    run = run_recipe(projection, Path(root), runtime_config, registry=registry)
    units = tuple(_candidate_unit(unit_run) for unit_run in run.units)
    engine = _engine_binding(recipe, phase_config, registry)
    plan = _build_verification_plan(
        recipe_name=recipe.name,
        engine_id=engine.id if engine is not None else None,
        frontends=frontends,
        transform=transforms[0],
        stages=stages,
        semantic_config=semantic,
    )
    return CandidateBundle(
        recipe=recipe.name,
        semantic_config_digest=canonical_digest(semantic),
        source_artifact_digest=source_digest,
        engine=engine,
        frontend_plugins=tuple(stage.plugin for stage in frontends),
        transform_plugin=transforms[0].plugin,
        verification_plan=plan,
        discovered_units=tuple(run.discovered_units),
        units=units,
    )


def _candidate_unit(unit_run: UnitRun) -> CandidateUnit:
    outcome = next((item for item in unit_run.outcomes if item.kind == "transform"), None)
    status = outcome.status if outcome is not None else "not_run"
    if status not in _TRANSFORM_STATUSES:
        raise PluginError(f"runner returned unknown transform status {status!r}")
    return CandidateUnit(
        unit=unit_run.unit,
        facts=unit_run.facts,
        transform_status=status,
        candidate=unit_run.candidate,
        outcome_digest=_outcome_digest(outcome),
    )


class _VerificationRegistry(Registry):
    """Capability-limited registry used by the verification entry point."""

    _ALLOWED = frozenset({"executor", "oracle", "verifier", "store"})

    def __init__(self, delegate: Registry) -> None:
        self._delegate = delegate

    def get(self, kind: str, name: str) -> Any:
        if kind not in self._ALLOWED:
            raise ConfigError(
                f"verification phase cannot resolve plugin kind {kind!r}; allowed kinds are "
                f"{sorted(self._ALLOWED)}"
            )
        return self._delegate.get(kind, name)

    def names(self, kind: str) -> tuple[str, ...]:
        if kind not in self._ALLOWED:
            raise ConfigError(
                f"verification phase cannot discover plugin kind {kind!r}; allowed kinds are "
                f"{sorted(self._ALLOWED)}"
            )
        return self._delegate.names(kind)

    def broken(self) -> dict[str, str]:
        return self._delegate.broken()


class _VerificationPlanRecipe(Recipe):
    """Inert identity used only by the existing Evidence constructor."""

    def __init__(self, name: str) -> None:
        self.name = name

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        del config
        raise AssertionError("a frozen verification plan must not ask a Recipe for stages")


@dataclass(frozen=True)
class BindingCheck:
    name: str
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "passed": self.passed}


@dataclass(frozen=True)
class GateResult:
    gate: str
    status: str
    confidence: str | None
    candidate_digest: str | None

    def __post_init__(self) -> None:
        if self.status not in _GATE_STATUSES:
            raise ConfigError(f"unknown gate result status {self.status!r}")

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "status": self.status,
            "confidence": self.confidence,
            "candidate_digest": self.candidate_digest,
        }


@dataclass(frozen=True)
class UnitVerification:
    uid: str
    kind: str
    parent: str | None
    transform_status: str
    candidate_digest: str | None
    deferred_count: int
    evidence_count: int
    evidence_complete: bool
    stage_status: str
    gates: tuple[GateResult, ...]
    stopped_by: str | None = None
    """The verification stage (oracle or gate) that ended this unit early.

    The plugin *name* only, as ``RecipeRun.summary()`` records it.  Its
    failure detail may quote source or a machine path, so it is delivered to
    the ``observer`` of :func:`verify_recipe_candidates` and never enters the
    report.  Without this name a gate the unit never reached is reported as
    ``missing`` with nothing to say why.
    """

    def to_dict(self) -> dict[str, object]:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "parent": self.parent,
            "transform_status": self.transform_status,
            "candidate_digest": self.candidate_digest,
            "deferred_count": self.deferred_count,
            "evidence_count": self.evidence_count,
            "evidence_complete": self.evidence_complete,
            "stage_status": self.stage_status,
            "gates": [gate.to_dict() for gate in self.gates],
            "stopped_by": self.stopped_by,
        }


@dataclass(frozen=True)
class GateCoverage:
    gate: str
    required_by: tuple[str, ...]
    passed_units: tuple[str, ...]
    failed_units: tuple[str, ...]
    missing_units: tuple[str, ...]
    unrecorded_units: tuple[str, ...]
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "gate": self.gate,
            "required_by": list(self.required_by),
            "passed_units": list(self.passed_units),
            "failed_units": list(self.failed_units),
            "missing_units": list(self.missing_units),
            "unrecorded_units": list(self.unrecorded_units),
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class RequiredUnitCoverage:
    uid: str
    selected: bool
    transformed: bool
    gates_passed: bool
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "uid": self.uid,
            "selected": self.selected,
            "transformed": self.transformed,
            "gates_passed": self.gates_passed,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class SubprogramCoverage:
    selector: str
    required: bool
    observed: bool
    owner_units: tuple[str, ...]
    gates_passed: bool
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "selector": self.selector,
            "required": self.required,
            "observed": self.observed,
            "owner_units": list(self.owner_units),
            "gates_passed": self.gates_passed,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class DeferredCoverage:
    required_empty: bool
    total: int
    units: tuple[tuple[str, int], ...]
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "required_empty": self.required_empty,
            "total": self.total,
            "units": [{"uid": uid, "count": count} for uid, count in self.units],
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class VerificationReport:
    """Root/time/source-byte-free acceptance proof for one CandidateBundle."""

    bundle_digest: str
    recipe: str
    source_artifact_digest: str
    engine: EngineBinding | None
    bindings: tuple[BindingCheck, ...]
    units: tuple[UnitVerification, ...]
    required_units: tuple[RequiredUnitCoverage, ...]
    gates: tuple[GateCoverage, ...]
    subprograms: tuple[SubprogramCoverage, ...]
    deferred: DeferredCoverage
    accepted: bool
    reason_codes: tuple[str, ...]

    schema: ClassVar[str] = _REPORT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "bundle_digest": self.bundle_digest,
            "recipe": self.recipe,
            "source_artifact_digest": self.source_artifact_digest,
            "engine": self.engine.to_dict() if self.engine is not None else None,
            "bindings": [item.to_dict() for item in self.bindings],
            "units": [item.to_dict() for item in self.units],
            "required_units": [item.to_dict() for item in self.required_units],
            "gates": [item.to_dict() for item in self.gates],
            "subprograms": [item.to_dict() for item in self.subprograms],
            "deferred": self.deferred.to_dict(),
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
        }

    def to_json(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def digest(self) -> str:
        return _digest_bytes(self.to_json())


def _composition_checks(
    config: Mapping[str, object],
    bundle: CandidateBundle,
    expected_source_artifact_digest: str,
    expected_engine: EngineBinding | None,
    registry: Registry,
) -> tuple[list[BindingCheck], TranslationEngine | None]:
    plan = bundle.verification_plan
    checks = [
        BindingCheck("recipe", bundle.recipe == plan.recipe),
        BindingCheck("semantic_config", bundle.semantic_config_digest == canonical_digest(config)),
        BindingCheck(
            "source_artifact", bundle.source_artifact_digest == expected_source_artifact_digest
        ),
        BindingCheck("frontends", bundle.frontend_plugins == plan.frontend_plugins),
        BindingCheck("transform_declaration", bundle.transform_plugin == plan.transform_plugin),
        BindingCheck("expected_engine", bundle.engine == expected_engine),
        BindingCheck(
            "verification_stage_kinds",
            all(stage.kind in _VERIFICATION_STAGE_KINDS for stage in plan.stages),
        ),
    ]
    installed: TranslationEngine | None = None
    if bundle.engine is None:
        checks.append(BindingCheck("installed_engine", expected_engine is None))
    else:
        installed = get_engine(bundle.engine.id, registry=registry)
        checks.append(
            BindingCheck("installed_engine", EngineBinding.from_engine(installed) == bundle.engine)
        )
        checks.append(
            BindingCheck(
                "recipe_engine",
                plan.engine_id == bundle.engine.id,
            )
        )
    return checks, installed


def _verification_run(
    root: Path,
    bundle: CandidateBundle,
    original_bundle: CandidateBundle,
    bundle_snapshot: bytes,
    candidate_digests: Mapping[str, str],
    registry: Registry,
    output: Path | None,
    workspace: Path | None,
    observer: RunObserver | None = None,
) -> dict[str, UnitRun]:
    safe_registry = _VerificationRegistry(registry)
    recipe = _VerificationPlanRecipe(bundle.recipe)
    events = _RunEventEmitter(recipe.name, observer)
    stages = tuple(stage.to_stage() for stage in bundle.verification_plan.stages)
    verification_stages = [
        (index, stage)
        for index, stage in enumerate(stages)
        if stage.kind in {"oracle", "verifier", "store"}
    ]
    for _, stage in verification_stages:
        if not stage.optional and stage.plugin not in safe_registry.names(stage.kind):
            raise ConfigError(
                f"required verification plugin {stage.kind}:{stage.plugin} is not registered"
            )

    executor_stages = [stage for stage in stages if stage.kind == "executor"]
    handed = [stage for _, stage in verification_stages if stage.kind in {"oracle", "verifier"}]
    if handed and len(executor_stages) != 1:
        raise ConfigError(
            f"verification phase requires exactly one executor stage; recipe {recipe.name!r} "
            f"declares {len(executor_stages)}"
        )

    def stage_config(stage: Stage) -> dict[str, Any]:
        copied = _json_value(stage.config, f"verification stage {stage.plugin}.config")
        if not isinstance(copied, dict):  # VerificationStage makes this defensive only.
            raise ConfigError(f"verification stage {stage.plugin!r} config must be an object")
        return cast(dict[str, Any], copied)

    executor: Any = None
    executor_name = ""
    if executor_stages:
        executor_stage = executor_stages[0]
        try:
            executor = safe_registry.get("executor", executor_stage.plugin)(
                **stage_config(executor_stage)
            )
        finally:
            _require_bundle_unchanged(
                original_bundle,
                bundle,
                bundle_snapshot,
                f"executor {executor_stage.plugin!r}",
            )
        executor_name = executor_stage.plugin

    resolved_root = Path(root).resolve()
    runtime_config: dict[str, Any] = {}
    if output is not None:
        runtime_config["output"] = Path(output).resolve()
    destination = output_root(resolved_root, runtime_config)
    verification_workspace = (
        Path(workspace).resolve()
        if workspace is not None
        else destination / f"{recipe.name}-verification"
    )
    verification_workspace.mkdir(parents=True, exist_ok=True)
    oracle_cache: dict[str, Any] = {}
    results: dict[str, UnitRun] = {}

    def call_config(stage: Stage) -> dict[str, Any]:
        return {
            "root": resolved_root,
            "output": destination,
            **stage_config(stage),
        }

    def walk_one(
        stage_index: int, stage: Stage, item: CandidateUnit, unit_run: UnitRun, unit_workspace: Path
    ) -> StageOutcome:
        """Walk one stage bracketed by the same observer lifecycle ``run_recipe`` emits."""
        assert item.facts is not None  # walk_one is only reached for transformed units
        events.emit(
            RunEventEntity.STAGE,
            RunEventAction.STARTED,
            status="running",
            reason_code="stage_scheduled",
            reason=f"{stage.kind} plugin {stage.plugin!r} scheduled",
            unit_id=unit_run.unit.uid,
            stage_index=stage_index,
            stage_kind=stage.kind,
            stage_plugin=stage.plugin,
        )
        try:
            try:
                outcome = _walk_stage(
                    stage,
                    call_config(stage),
                    safe_registry,
                    recipe,
                    executor_name,
                    item.unit,
                    item.facts,
                    unit_run,
                    unit_workspace,
                    executor,
                    oracle_cache,
                    events=events,
                    stage_index=stage_index,
                )
            finally:
                _require_bundle_unchanged(
                    original_bundle,
                    bundle,
                    bundle_snapshot,
                    f"{stage.kind} {stage.plugin!r}",
                )
        except BaseException as error:
            events.emit(
                RunEventEntity.STAGE,
                RunEventAction.FINISHED,
                status="aborted",
                reason_code="stage_exception",
                reason=_exception_reason(error),
                unit_id=unit_run.unit.uid,
                stage_index=stage_index,
                stage_kind=stage.kind,
                stage_plugin=stage.plugin,
            )
            raise
        # ``reason`` is the plugin's raw detail -- an oracle build log, a
        # verifier's explanation -- which is exactly what the report must not
        # carry and exactly what an operator repairing the unit needs.
        events.emit(
            RunEventEntity.STAGE,
            RunEventAction.FINISHED,
            status=outcome.status,
            reason_code=f"stage_{outcome.status}",
            reason=outcome.detail,
            unit_id=unit_run.unit.uid,
            stage_index=stage_index,
            stage_kind=stage.kind,
            stage_plugin=stage.plugin,
        )
        return outcome

    def not_run(stage_index: int, stage: Stage, unit: Unit, stopped_by: str) -> None:
        """Expose a declared stage suppressed by an upstream stop, bracketed
        as ``run_recipe`` brackets it: a finish always follows a start."""
        events.emit(
            RunEventEntity.STAGE,
            RunEventAction.STARTED,
            status="running",
            reason_code="stage_considered",
            reason=f"considering {stage.kind} plugin {stage.plugin!r}",
            unit_id=unit.uid,
            stage_index=stage_index,
            stage_kind=stage.kind,
            stage_plugin=stage.plugin,
        )
        events.emit(
            RunEventEntity.STAGE,
            RunEventAction.FINISHED,
            status="skipped",
            reason_code="upstream_stop",
            reason=f"not run because {stopped_by!r} stopped the unit",
            unit_id=unit.uid,
            stage_index=stage_index,
            stage_kind=stage.kind,
            stage_plugin=stage.plugin,
        )

    def walk_unit(item: CandidateUnit, unit_run: UnitRun) -> None:
        assert item.facts is not None  # checked by the caller before dispatch
        unit_workspace = (
            verification_workspace / hashlib.sha256(item.unit.uid.encode()).hexdigest()[:16]
        )
        unit_workspace.mkdir(parents=True, exist_ok=True)
        for position, (stage_index, stage) in enumerate(verification_stages):
            before_verdicts = len(unit_run.verdicts)
            outcome = walk_one(stage_index, stage, item, unit_run, unit_workspace)
            unit_run.outcomes.append(outcome)
            if stage.kind == "oracle" and outcome.status == "ok":
                unit_run.oracle = oracle_cache[outcome.detail]
            if stage.kind == "verifier" and len(unit_run.verdicts) > before_verdicts:
                verdict = unit_run.verdicts[-1]
                _validate_verdict(stage, item, verdict, candidate_digests[item.unit.uid])
            if outcome.status == "failed" and (stage.gate or stage.kind == "oracle"):
                unit_run.stopped_by = stage.plugin
                for later_index, later in verification_stages[position + 1 :]:
                    if later.kind != "store":
                        not_run(later_index, later, item.unit, stage.plugin)
                        continue
                    later_outcome = walk_one(later_index, later, item, unit_run, unit_workspace)
                    unit_run.outcomes.append(later_outcome)
                break

    events.emit(
        RunEventEntity.RUN,
        RunEventAction.STARTED,
        status="running",
        reason_code="verification_requested",
        reason=f"verification of recipe {recipe.name!r} candidates requested",
    )
    # What each unit's finishing event said, so the run's own says the same:
    # a unit that was never transformed has no outcomes and would otherwise
    # count as passed.
    finished: list[str] = []
    try:
        for item in bundle.units:
            unit_run = UnitRun(unit=item.unit, facts=item.facts, candidate=item.candidate)
            results[item.unit.uid] = unit_run
            events.emit(
                RunEventEntity.UNIT,
                RunEventAction.STARTED,
                status="running",
                reason_code="unit_bundled",
                reason=f"transform status {item.transform_status!r}",
                unit_id=item.unit.uid,
            )
            if item.transform_status != "ok" or item.facts is None or item.candidate is None:
                events.emit(
                    RunEventEntity.UNIT,
                    RunEventAction.FINISHED,
                    status="incomplete",
                    reason_code="unit_not_transformed",
                    reason=f"transform status {item.transform_status!r}: no candidate to verify",
                    unit_id=item.unit.uid,
                )
                finished.append("incomplete")
                continue
            try:
                walk_unit(item, unit_run)
            except BaseException as error:
                events.emit(
                    RunEventEntity.UNIT,
                    RunEventAction.FINISHED,
                    status="aborted",
                    reason_code="unit_exception",
                    reason=_exception_reason(error),
                    unit_id=item.unit.uid,
                )
                raise
            _emit_unit_finished(events, unit_run)
            finished.append(unit_run.status.value)
    except BaseException as error:
        events.emit(
            RunEventEntity.RUN,
            RunEventAction.FINISHED,
            status="aborted",
            reason_code="verification_exception",
            reason=_exception_reason(error),
        )
        raise
    if events.enabled:
        tally = Counter(finished)
        events.emit(
            RunEventEntity.RUN,
            RunEventAction.FINISHED,
            status="passed" if all(status == "passed" for status in finished) else "failed",
            reason_code="verification_walked",
            reason=", ".join(f"{count} {status}" for status, count in sorted(tally.items()))
            or "no units",
        )
    return results


def _require_bundle_unchanged(
    original_bundle: CandidateBundle,
    verification_bundle: CandidateBundle,
    expected: bytes,
    boundary: str,
) -> None:
    for label, candidate_bundle in (
        ("input", original_bundle),
        ("verification copy", verification_bundle),
    ):
        try:
            actual = candidate_bundle.to_json()
        except Exception as error:
            raise PluginError(
                f"{boundary} mutated the {label} CandidateBundle into an invalid state"
            ) from error
        if actual != expected:
            raise PluginError(
                f"{boundary} mutated the {label} CandidateBundle or one of its reachable "
                "Unit/Facts/Candidate values"
            )


def _validate_verdict(
    stage: Stage,
    item: CandidateUnit,
    verdict: Verdict,
    expected_candidate_digest: str,
) -> None:
    candidate = item.candidate
    if candidate is None:  # Defensive: caller only invokes for an ok candidate.
        raise PluginError("verifier produced a verdict without a candidate")
    if verdict.unit != item.unit.uid:
        raise PluginError(
            f"verifier {stage.plugin!r} returned unit {verdict.unit!r}, expected {item.unit.uid!r}"
        )
    if verdict.verifier != stage.plugin:
        raise PluginError(f"verifier stage {stage.plugin!r} returned identity {verdict.verifier!r}")
    if verdict.candidate != expected_candidate_digest:
        raise PluginError(
            f"verifier {stage.plugin!r} judged candidate {verdict.candidate!r}, "
            f"expected transform product {expected_candidate_digest!r}"
        )


def _gate_result(
    gate: str,
    unit_run: UnitRun,
    evidence_complete: bool,
) -> GateResult:
    verdicts = [item for item in unit_run.verdicts if item.verifier == gate]
    if len(verdicts) != 1:
        return GateResult(gate, "missing", None, None)
    verdict = verdicts[0]
    if not verdict.passed:
        status = "failed"
    elif not evidence_complete:
        status = "unrecorded"
    else:
        status = "passed"
    return GateResult(gate, status, verdict.confidence.value, f"sha256:{verdict.candidate}")


def _candidate_subprogram_coverage(candidate: Candidate) -> tuple[frozenset[str], bool]:
    """Read the language-neutral, source-free Candidate coverage protocol."""
    coverage = candidate.notes.get("coverage")
    if not isinstance(coverage, dict):
        return frozenset(), False
    subprograms = coverage.get("subprograms")
    if not isinstance(subprograms, list):
        return frozenset(), False
    if not all(isinstance(item, str) and item for item in subprograms):
        return frozenset(), False
    if len(subprograms) != len(set(subprograms)):
        return frozenset(), False
    return frozenset(subprograms), True


def verify_recipe_candidates(
    recipe: Recipe,
    root: Path,
    bundle: CandidateBundle,
    config: dict[str, Any] | None = None,
    *,
    expected_source_artifact_digest: str,
    expected_engine: EngineBinding | None,
    project_required_gates: Iterable[str] = (),
    required_units: Iterable[str] = (),
    required_subprograms: Iterable[str] = (),
    require_no_deferred: bool = True,
    registry: Registry = REGISTRY,
    output: Path | None = None,
    workspace: Path | None = None,
    observer: RunObserver | None = None,
) -> VerificationReport:
    """Verify a CandidateBundle without consulting executable Recipe code.

    Acceptance requires exact source/config/recipe/engine bindings, every
    engine and project gate for every selected unit, recorded Evidence, the
    requested subprogram coverage, and (by default) an empty deferred ledger.
    The returned report contains digests/counts/statuses only: no source bytes,
    verifier detail, metrics, store URI, machine root, or timestamp.  A unit
    that a failed oracle or gate stopped early names that stage in
    ``stopped_by``; the stage's own detail -- the reason the later gates never
    ran -- is delivered only to ``observer``, under the same ordered lifecycle
    and embargo :func:`recast.run.run_recipe` gives it.

    ``recipe`` remains a positional compatibility argument, but is deliberately
    ignored.  The transform phase freezes the safe verification declaration in
    ``bundle.verification_plan``; calling any method (or even reading an
    attribute) on a caller-supplied Recipe here would reopen the Transform
    capability boundary.
    """
    bundle_snapshot = bundle.to_json()
    verification_bundle = decode_candidate_bundle(bundle_snapshot)
    bundle_digest = _digest_bytes(bundle_snapshot)
    candidate_digests = {
        item.unit.uid: item.candidate.digest()
        for item in verification_bundle.units
        if item.candidate is not None
    }
    semantic = _semantic_config(config or {})
    expected_source = _require_digest(
        expected_source_artifact_digest, "expected_source_artifact_digest"
    )
    phase_config = cast(dict[str, Any], dict(semantic))
    stages = tuple(stage.to_stage() for stage in verification_bundle.verification_plan.stages)
    checks, installed_engine = _composition_checks(
        phase_config,
        verification_bundle,
        expected_source,
        expected_engine,
        registry,
    )
    engine_gates = installed_engine.required_gates if installed_engine is not None else ()
    project_gates = tuple(project_required_gates)
    if any(not gate for gate in project_gates):
        raise ConfigError("project_required_gates contains an empty gate name")
    origins: dict[str, set[str]] = {}
    for gate in engine_gates:
        origins.setdefault(gate, set()).add("engine")
    for gate in project_gates:
        origins.setdefault(gate, set()).add("project")
    required_gates = tuple(sorted(origins))
    checks.append(BindingCheck("required_gates", bool(required_gates)))
    gate_stage_names = [stage.plugin for stage in stages if stage.kind == "verifier" and stage.gate]
    checks.append(
        BindingCheck(
            "required_gate_declarations",
            all(gate_stage_names.count(gate) == 1 for gate in required_gates),
        )
    )

    runs = _verification_run(
        Path(root),
        verification_bundle,
        bundle,
        bundle_snapshot,
        candidate_digests,
        registry,
        output,
        workspace,
        observer,
    )
    unit_reports: list[UnitVerification] = []
    by_unit_gate: dict[str, dict[str, GateResult]] = {}
    all_transformed = bool(verification_bundle.units)
    all_evidence = bool(verification_bundle.units)
    verification_passed = bool(verification_bundle.units)
    for item in sorted(verification_bundle.units, key=lambda entry: entry.unit.uid):
        unit_run = runs[item.unit.uid]
        store_ok = any(
            outcome.kind == "store" and outcome.status == "ok" for outcome in unit_run.outcomes
        )
        evidence_complete = (
            bool(unit_run.verdicts)
            and store_ok
            and (len(unit_run.evidence) >= len(unit_run.verdicts))
        )
        gate_results = tuple(
            _gate_result(gate, unit_run, evidence_complete) for gate in required_gates
        )
        by_unit_gate[item.unit.uid] = {gate.gate: gate for gate in gate_results}
        candidate_digest = (
            f"sha256:{candidate_digests[item.unit.uid]}" if item.candidate is not None else None
        )
        deferred_count = len(item.candidate.deferred) if item.candidate is not None else 0
        stage_status = unit_run.status.value if item.transform_status == "ok" else "incomplete"
        unit_reports.append(
            UnitVerification(
                uid=item.unit.uid,
                kind=item.unit.kind,
                parent=item.unit.parent,
                transform_status=item.transform_status,
                candidate_digest=candidate_digest,
                deferred_count=deferred_count,
                evidence_count=len(unit_run.evidence),
                evidence_complete=evidence_complete,
                stage_status=stage_status,
                gates=gate_results,
                stopped_by=unit_run.stopped_by,
            )
        )
        all_transformed = all_transformed and item.transform_status == "ok"
        all_evidence = all_evidence and evidence_complete
        verification_passed = verification_passed and stage_status == "passed"

    checks.extend(
        [
            BindingCheck("selected_units", bool(verification_bundle.units)),
            BindingCheck("transform_coverage", all_transformed),
            BindingCheck("evidence_coverage", all_evidence),
            BindingCheck("verification_stages", verification_passed),
        ]
    )
    gate_reports: list[GateCoverage] = []
    for gate in required_gates:
        statuses = {
            uid: by_unit_gate.get(uid, {}).get(gate, GateResult(gate, "missing", None, None)).status
            for uid in sorted(item.unit.uid for item in verification_bundle.units)
        }
        passed = tuple(uid for uid, status in statuses.items() if status == "passed")
        failed = tuple(uid for uid, status in statuses.items() if status == "failed")
        missing = tuple(uid for uid, status in statuses.items() if status == "missing")
        unrecorded = tuple(uid for uid, status in statuses.items() if status == "unrecorded")
        gate_reports.append(
            GateCoverage(
                gate=gate,
                required_by=tuple(sorted(origins[gate])),
                passed_units=passed,
                failed_units=failed,
                missing_units=missing,
                unrecorded_units=unrecorded,
                accepted=len(passed) == len(verification_bundle.units)
                and bool(verification_bundle.units),
            )
        )

    required_unit_set = set(required_units)
    if "" in required_unit_set:
        raise ConfigError("required_units contains an empty uid")
    selected_by_uid = {item.unit.uid: item for item in verification_bundle.units}
    required_unit_reports: list[RequiredUnitCoverage] = []
    for uid in sorted(required_unit_set):
        selected_item = selected_by_uid.get(uid)
        transformed = selected_item is not None and selected_item.transform_status == "ok"
        gates_passed = selected_item is not None and all(
            by_unit_gate.get(uid, {}).get(gate, GateResult(gate, "missing", None, None)).status
            == "passed"
            for gate in required_gates
        )
        required_unit_reports.append(
            RequiredUnitCoverage(
                uid=uid,
                selected=selected_item is not None,
                transformed=transformed,
                gates_passed=gates_passed,
                accepted=selected_item is not None and transformed and gates_passed,
            )
        )
    required_units_accepted = all(item.accepted for item in required_unit_reports)

    deferred_units = tuple(
        (item.unit.uid, len(item.candidate.deferred))
        for item in sorted(verification_bundle.units, key=lambda entry: entry.unit.uid)
        if item.candidate is not None and item.candidate.deferred
    )
    deferred_total = sum(count for _, count in deferred_units)
    deferred = DeferredCoverage(
        required_empty=require_no_deferred,
        total=deferred_total,
        units=deferred_units,
        accepted=not require_no_deferred or deferred_total == 0,
    )

    required_subprogram_set = set(required_subprograms)
    if "" in required_subprogram_set:
        raise ConfigError("required_subprograms contains an empty uid")
    discovered_subprograms = {
        unit.uid for unit in verification_bundle.discovered_units if unit.kind == "subprogram"
    }
    coverage_owners: dict[str, set[str]] = {}
    coverage_ledgers_valid = True
    for item in verification_bundle.units:
        if item.candidate is None:
            coverage_ledgers_valid = False
            continue
        covered, valid = _candidate_subprogram_coverage(item.candidate)
        coverage_ledgers_valid = coverage_ledgers_valid and valid
        for selector in covered:
            coverage_owners.setdefault(selector, set()).add(item.unit.uid)
    if not required_subprogram_set:
        # A caller with no project subprogram policy need not require this
        # optional Candidate.notes protocol.
        coverage_ledgers_valid = True
    coverage_ids = sorted(discovered_subprograms | required_subprogram_set | set(coverage_owners))
    subprogram_reports: list[SubprogramCoverage] = []
    for selector in coverage_ids:
        owners = tuple(sorted(coverage_owners.get(selector, set())))
        observed = bool(owners)
        gates_passed = bool(owners) and all(
            all(
                by_unit_gate.get(owner, {})
                .get(gate, GateResult(gate, "missing", None, None))
                .status
                == "passed"
                for gate in required_gates
            )
            for owner in owners
        )
        required = selector in required_subprogram_set
        accepted = observed and gates_passed
        subprogram_reports.append(
            SubprogramCoverage(
                selector=selector,
                required=required,
                observed=observed,
                owner_units=owners,
                gates_passed=gates_passed,
                accepted=accepted,
            )
        )
    required_subprograms_accepted = all(
        item.accepted for item in subprogram_reports if item.required
    )

    binding_ok = all(item.passed for item in checks)
    gates_ok = bool(gate_reports) and all(item.accepted for item in gate_reports)
    accepted = (
        binding_ok
        and gates_ok
        and deferred.accepted
        and required_units_accepted
        and coverage_ledgers_valid
        and required_subprograms_accepted
    )
    reasons = [f"binding.{item.name}" for item in checks if not item.passed]
    reasons.extend(f"gate.{item.gate}" for item in gate_reports if not item.accepted)
    if not deferred.accepted:
        reasons.append("deferred.nonempty")
    if not required_units_accepted:
        reasons.append("unit.required_coverage")
    if not coverage_ledgers_valid:
        reasons.append("subprogram.coverage_ledger")
    if not required_subprograms_accepted:
        reasons.append("subprogram.required_coverage")
    _require_bundle_unchanged(
        bundle,
        verification_bundle,
        bundle_snapshot,
        "verification report construction",
    )
    return VerificationReport(
        bundle_digest=bundle_digest,
        recipe=verification_bundle.recipe,
        source_artifact_digest=verification_bundle.source_artifact_digest,
        engine=verification_bundle.engine,
        bindings=tuple(checks),
        units=tuple(unit_reports),
        required_units=tuple(required_unit_reports),
        gates=tuple(gate_reports),
        subprograms=tuple(subprogram_reports),
        deferred=deferred,
        accepted=accepted,
        reason_codes=tuple(sorted(set(reasons))),
    )
