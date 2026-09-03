"""Declarations for translation engines and their artifact boundaries.

An engine is catalog metadata, not a stage the runner executes.  It gives an
outer orchestrator enough information to select a recipe and to connect its
input and output artifacts without teaching the orchestrator about Fortran,
NumPy, or a domain extension.  Executable behaviour remains in the plugins the
recipe names.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from types import MappingProxyType
from typing import Any

from recast.errors import PluginError
from recast.registry import REGISTRY, Registry

__all__ = [
    "ArtifactContract",
    "TranslationEngine",
    "canonical_digest",
    "catalog_document",
    "engines",
    "fortran_numpy_engine",
    "get_engine",
    "python_jax_engine",
    "python_numba_engine",
]

_IDENTIFIER = re.compile(r"^[a-z0-9]+(?:[._-][a-z0-9]+)*$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_CATALOG_SCHEMA = "recast.translation-engine-catalog.v1"
_ENGINE_SCHEMA = "recast.translation-engine.v1"
_ARTIFACT_SCHEMA = "recast.artifact-contract.v1"


def _canonical_bytes(value: object) -> bytes:
    """Encode JSON with the stable rules used by catalog and manifest digests."""
    try:
        document = json.dumps(
            _thaw_json(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise PluginError(f"engine declaration is not canonical JSON: {exc}") from exc
    return document.encode("utf-8")


def canonical_digest(value: object) -> str:
    """Return the canonical ``sha256:`` digest for a JSON-compatible value."""
    return f"sha256:{hashlib.sha256(_canonical_bytes(value)).hexdigest()}"


def _freeze_json(value: object, *, path: str) -> object:
    """Copy a JSON value into immutable mappings and tuples.

    Frozen dataclasses alone do not protect a nested config-schema dictionary.
    Copying here makes a manifest's digest stable for its whole lifetime.
    """
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise PluginError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise PluginError(f"{path} has a non-string key {key!r}")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    raise PluginError(f"{path} contains unsupported JSON value {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require_identifier(value: str, *, field_name: str) -> None:
    if not _IDENTIFIER.fullmatch(value):
        raise PluginError(
            f"translation engine {field_name} {value!r} is not a lowercase dotted identifier"
        )


@dataclass(frozen=True)
class ArtifactContract:
    """The exact artifact shape accepted or produced by an engine.

    Compatibility is equality of the serialized contract, not a guess based on
    a filename extension.  ``profile`` distinguishes, for example, generic
    Python source from the NumPy subset another engine may accept.
    """

    id: str
    version: str
    media_type: str
    language: str
    profile: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.id, field_name="artifact contract id")
        if not self.version:
            raise PluginError("artifact contract version must not be empty")
        if not self.media_type or "/" not in self.media_type:
            raise PluginError(f"artifact contract media_type {self.media_type!r} is invalid")
        if not self.language:
            raise PluginError("artifact contract language must not be empty")
        if self.profile is not None and not self.profile:
            raise PluginError("artifact contract profile must be non-empty when present")

    def to_dict(self) -> dict[str, object]:
        return {**self.declaration(), "digest": self.digest()}

    def declaration(self) -> dict[str, object]:
        return {
            "schema": _ARTIFACT_SCHEMA,
            "id": self.id,
            "version": self.version,
            "media_type": self.media_type,
            "language": self.language,
            **({"profile": self.profile} if self.profile is not None else {}),
        }

    def digest(self) -> str:
        return canonical_digest(self.declaration())


@dataclass(frozen=True)
class TranslationEngine:
    """Immutable declaration of one input-to-output translation engine."""

    id: str
    version: str
    implementation_digest: str
    default_recipe: str
    input_artifact_contract: ArtifactContract
    output_artifact_contract: ArtifactContract
    config_schema: Mapping[str, object] = field(default_factory=dict)
    default_config: Mapping[str, object] = field(default_factory=dict)
    required_gates: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()
    owning_repository: str = ""

    def __post_init__(self) -> None:
        _require_identifier(self.id, field_name="id")
        if not self.version:
            raise PluginError(f"translation engine {self.id!r} has an empty version")
        if not _SHA256.fullmatch(self.implementation_digest):
            raise PluginError(
                f"translation engine {self.id!r} implementation_digest must be sha256:<64 hex>"
            )
        if not self.default_recipe:
            raise PluginError(f"translation engine {self.id!r} has no default_recipe")
        if not self.owning_repository:
            raise PluginError(f"translation engine {self.id!r} has no owning_repository")
        if not self.required_gates:
            raise PluginError(f"translation engine {self.id!r} declares no required_gates")
        if len(set(self.required_gates)) != len(self.required_gates):
            raise PluginError(f"translation engine {self.id!r} repeats a required gate")
        if any(not gate for gate in self.required_gates):
            raise PluginError(f"translation engine {self.id!r} has an empty required gate")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise PluginError(f"translation engine {self.id!r} repeats a capability")
        for capability in self.capabilities:
            _require_identifier(capability, field_name="capability")

        frozen_schema = _freeze_json(self.config_schema, path="config_schema")
        frozen_default = _freeze_json(self.default_config, path="default_config")
        if not isinstance(frozen_schema, Mapping) or not isinstance(frozen_default, Mapping):
            raise PluginError("config_schema and default_config must be JSON objects")
        object.__setattr__(self, "config_schema", frozen_schema)
        object.__setattr__(self, "default_config", frozen_default)

    @property
    def name(self) -> str:
        """Plugin identity, equal to ``id`` for registry conformance."""
        return self.id

    def declaration(self) -> dict[str, object]:
        """The canonical digest subject, without a self-referential digest."""
        return {
            "schema": _ENGINE_SCHEMA,
            "id": self.id,
            "version": self.version,
            "implementation_digest": self.implementation_digest,
            "default_recipe": self.default_recipe,
            "input_artifact_contract": self.input_artifact_contract.to_dict(),
            "output_artifact_contract": self.output_artifact_contract.to_dict(),
            "config_schema": _thaw_json(self.config_schema),
            "config_schema_digest": self.config_schema_digest,
            "default_config": _thaw_json(self.default_config),
            "required_gates": list(self.required_gates),
            "capabilities": list(self.capabilities),
            "owning_repository": self.owning_repository,
        }

    @property
    def config_schema_digest(self) -> str:
        return canonical_digest(self.config_schema)

    def digest(self) -> str:
        return canonical_digest(self.declaration())

    def to_dict(self) -> dict[str, object]:
        return {**self.declaration(), "digest": self.digest()}

    def __hash__(self) -> int:
        # ``config_schema`` is a mapping proxy and therefore not directly
        # hashable. The canonical digest is the immutable value identity.
        return hash(self.digest())


def _as_engine(value: Any, origin: str) -> TranslationEngine:
    resolved = value() if callable(value) and not isinstance(value, TranslationEngine) else value
    if not isinstance(resolved, TranslationEngine):
        raise PluginError(f"{origin} returned {type(resolved).__name__}, not TranslationEngine")
    return resolved


def engines(*, registry: Registry = REGISTRY) -> tuple[TranslationEngine, ...]:
    """Return every installed engine, ordered by its stable identifier."""
    installed: list[TranslationEngine] = []
    for registered in registry.names("engine"):
        engine = _as_engine(registry.get("engine", registered), f"engine {registered!r}")
        if engine.id != registered:
            raise PluginError(
                f"engine entry point {registered!r} declares id {engine.id!r}; they must match"
            )
        installed.append(engine)
    return tuple(sorted(installed, key=lambda engine: engine.id))


def get_engine(name: str, *, registry: Registry = REGISTRY) -> TranslationEngine:
    engine = _as_engine(registry.get("engine", name), f"engine {name!r}")
    if engine.id != name:
        raise PluginError(f"engine entry point {name!r} declares id {engine.id!r}; they must match")
    return engine


def catalog_document(*, registry: Registry = REGISTRY) -> dict[str, object]:
    """Export the installed catalog with one digest over the ordered manifests."""
    payload: dict[str, object] = {
        "schema": _CATALOG_SCHEMA,
        "engines": [engine.to_dict() for engine in engines(registry=registry)],
    }
    return {**payload, "digest": canonical_digest(payload)}


def _installed_code_digest() -> str:
    """Digest the installed engine package conservatively, relative paths included.

    The built-in translation uses several core contracts and plugins.  Hashing
    all shipped Python modules is intentionally broader than trying to guess
    which helper is semantically relevant, and works in wheels and editable
    installs without embedding an absolute checkout path.
    """
    package_root = Path(__file__).parent
    digest = hashlib.sha256()
    paths = sorted(
        package_root.rglob("*.py"),
        key=lambda item: item.relative_to(package_root).as_posix(),
    )
    for path in paths:
        relative = path.relative_to(package_root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    # Entry-point mappings are executable composition, not packaging trivia:
    # changing which frontend/transform/verifier a name resolves to changes the
    # engine even when every module byte remains identical.  Wheel and editable
    # installs both expose this stable, path-free metadata.
    try:
        entry_points = distribution("recast-engine").read_text("entry_points.txt")
    except PackageNotFoundError:  # Source-only imports still get a useful digest.
        entry_points = None
    if entry_points is not None:
        relative = b"distribution/entry_points.txt"
        content = entry_points.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"sha256:{digest.hexdigest()}"


def fortran_numpy_engine() -> TranslationEngine:
    """The shipped Fortran source-tree to Python/NumPy engine declaration."""
    return TranslationEngine(
        id="recast.fortran-python.numpy",
        version="1",
        implementation_digest=_installed_code_digest(),
        default_recipe="translate",
        input_artifact_contract=ArtifactContract(
            id="recast.source-tree.fortran",
            version="1",
            media_type="application/vnd.recast.source-tree",
            language="fortran",
        ),
        output_artifact_contract=ArtifactContract(
            id="recast.source-tree.python.numpy",
            version="1",
            media_type="application/vnd.recast.source-tree",
            language="python",
            profile="numpy",
        ),
        config_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "target": {"const": "numpy"},
                "frontend": {"const": "fortran"},
                # This catalog entry is also consumed by remote/control-plane
                # launchers.  Selecting an executor is operational authority,
                # not a semantic translation option, so the built-in engine
                # exposes only its reviewed local implementation.
                "executor": {"const": "local"},
            },
            # Project and deployment policy may add their own *derived* values
            # after catalog validation.  Browser/operator input must not smuggle
            # stage configuration, paths, commands, flags, or environments
            # through this engine manifest.
            "additionalProperties": False,
        },
        default_config={"target": "numpy", "frontend": "fortran", "executor": "local"},
        required_gates=("static.rwset", "differential.bitexact"),
        capabilities=("translation", "deterministic", "numerical-verification"),
        owning_repository="https://github.com/a85tract/RecastEngine",
    )


def _python_accelerator_engine(target: str) -> TranslationEngine:
    """Build one Python/NumPy source-tree accelerator declaration."""
    if target not in {"numba", "jax"}:
        raise PluginError(f"unknown built-in Python accelerator {target!r}")
    return TranslationEngine(
        id=f"recast.python-numpy.{target}",
        version="1",
        implementation_digest=_installed_code_digest(),
        default_recipe=f"python-to-{target}",
        input_artifact_contract=ArtifactContract(
            id="recast.source-tree.python.numpy",
            version="1",
            media_type="application/vnd.recast.source-tree",
            language="python",
            profile="numpy",
        ),
        output_artifact_contract=ArtifactContract(
            id=f"recast.source-tree.python.{target}",
            version="1",
            media_type="application/vnd.recast.source-tree",
            language="python",
            profile=target,
        ),
        config_schema={
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "target": {"const": target},
                "frontend": {"const": "python-numpy"},
                "executor": {"const": "local"},
            },
            "required": ["target", "frontend", "executor"],
            "additionalProperties": False,
        },
        default_config={"target": target, "frontend": "python-numpy", "executor": "local"},
        required_gates=("static.complete", f"differential.python-{target}"),
        capabilities=("translation", "deterministic", "numerical-verification", target),
        owning_repository="https://github.com/a85tract/RecastEngine",
    )


def python_numba_engine() -> TranslationEngine:
    """The Python/NumPy source-tree to Numba engine declaration."""
    return _python_accelerator_engine("numba")


def python_jax_engine() -> TranslationEngine:
    """The Python/NumPy source-tree to JAX engine declaration."""
    return _python_accelerator_engine("jax")
