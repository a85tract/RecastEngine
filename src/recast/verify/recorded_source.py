"""A physical recording tied to this chain's source, with no upstream translation.

The installed caller resolves the recorder's native provenance format. These
domain-independent types bind its result to the actual source, observation
authority and complete workload extent. They do not certify numerical output;
the ordinary and workload verifiers still execute independently.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from recast.errors import ConfigError
from recast.verify._forward_workload import WorkloadDeclaration

_DIGEST = re.compile(r"sha256:[0-9a-f]{64}\Z")
MAX_SOURCE_FILES = 4096
MAX_SOURCE_BYTES = 512 << 20
MAX_SOURCE_FILE_BYTES = 64 << 20


def _digest(value: Any, label: str) -> None:
    if not isinstance(value, str) or not _DIGEST.fullmatch(value):
        raise ConfigError(f"recorded source {label} must be a canonical sha256 digest")


def _identifier(value: Any, label: str) -> None:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise ConfigError(f"recorded source {label} must be a bounded identifier")


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()


def _hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_canonical(value)).hexdigest()


@dataclass(frozen=True)
class RecordedSourceBoundaryReceipt:
    """The launch's immutable authorization for an existing source recording."""

    kind: str
    anchor_digest: str
    downstream_engine_id: str
    downstream_source_artifact_digest: str
    recording_provenance_digest: str
    tape_digest: str
    policy_digest: str
    observation_keys_digest: str
    provider_id: str
    provider_digest: str
    profile_digest: str

    def __post_init__(self) -> None:
        if self.kind != "recorded-source":
            raise ConfigError("a recorded source receipt must have kind 'recorded-source'")
        for name in ("downstream_engine_id", "provider_id"):
            _identifier(getattr(self, name), name)
        for name in (
            "anchor_digest",
            "downstream_source_artifact_digest",
            "recording_provenance_digest",
            "tape_digest",
            "policy_digest",
            "observation_keys_digest",
            "provider_digest",
            "profile_digest",
        ):
            _digest(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def digest(self) -> str:
        return _hash(self.to_dict())


@dataclass(frozen=True)
class RecordedSourceFile:
    """A canonical selected source file, as resolved from the caller's CAS."""

    path: str
    digest: str
    size: int
    mode: str = "0644"

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path or len(self.path) > 4096:
            raise ConfigError("recorded source file path is invalid")
        path = PurePosixPath(self.path)
        if path.is_absolute() or ".." in path.parts or path.as_posix() != self.path:
            raise ConfigError("recorded source file path must be canonical and relative")
        _digest(self.digest, "file digest")
        if type(self.size) is not int or not 0 <= self.size <= MAX_SOURCE_FILE_BYTES:
            raise ConfigError("recorded source file size exceeds the bound")
        if self.mode not in {"0644", "0755"}:
            raise ConfigError("recorded source file mode is invalid")


@dataclass(frozen=True)
class RecordedSourceEvidence:
    """Checked facts returned only by the exact installed recording profile.

    These are not free-standing acceptance flags. The trusted caller must
    resolve the receipt's pinned provenance, native recorder execution and
    source mapping, and must retain that entire closure. The engine checks
    the resolved facts against the actual request and live source again.
    """

    receipt: RecordedSourceBoundaryReceipt
    source_files: tuple[RecordedSourceFile, ...]
    recording_manifest_digest: str
    recording_run_digest: str
    origin_source_digest: str
    producer_digest: str
    checker_digest: str
    coverage_digest: str

    def __post_init__(self) -> None:
        if type(self.receipt) is not RecordedSourceBoundaryReceipt:
            raise ConfigError("recorded source evidence has no typed receipt")
        if (
            type(self.source_files) is not tuple
            or not 1 <= len(self.source_files) <= MAX_SOURCE_FILES
            or any(type(item) is not RecordedSourceFile for item in self.source_files)
        ):
            raise ConfigError("recorded source evidence needs a bounded typed file inventory")
        paths = [item.path for item in self.source_files]
        if paths != sorted(set(paths)):
            raise ConfigError("recorded source evidence file paths must be unique and sorted")
        if sum(item.size for item in self.source_files) > MAX_SOURCE_BYTES:
            raise ConfigError("recorded source evidence files exceed the total bound")
        for name in (
            "recording_manifest_digest",
            "recording_run_digest",
            "origin_source_digest",
            "producer_digest",
            "checker_digest",
            "coverage_digest",
        ):
            _digest(getattr(self, name), name)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recorded_source_coverage_digest(declaration: WorkloadDeclaration) -> str:
    """Scientific extent, independent of candidate entry-point reconstruction."""

    fixture = declaration.fixtures
    return _hash(
        {
            "case": fixture.case,
            "probe": fixture.probe,
            "first_sequence": fixture.first_sequence,
            "final_sequence": fixture.final_sequence,
            "steps": fixture.steps,
            "required_inputs": list(fixture.required_inputs),
            "required_outputs": list(fixture.required_outputs),
            "outputs": list(declaration.interface.outputs),
            "static_signatures": list(declaration.interface.static_signatures),
            "coverage_labels": list(fixture.coverage_labels),
            "states": list(fixture.states),
            "state_digests": dict(fixture.state_digests),
            "observation_keys_digest": fixture.observation_keys_digest,
            "observation_key_count": fixture.observation_key_count,
        }
    )


def require_recorded_source_files(root: Path, evidence: RecordedSourceEvidence) -> None:
    """Read only declared source files; refuse substitutions before and after dispatch."""

    root = Path(root)
    if root.is_symlink() or not root.is_dir():
        raise ConfigError("recorded source root is not a regular directory")
    for item in evidence.source_files:
        path = root / item.path
        parents = (path, *path.parents)
        if any(parent.is_symlink() for parent in parents if parent != root.parent):
            raise ConfigError(f"recorded source file {item.path!r} traverses a symlink")
        if not path.is_file() or path.stat().st_size != item.size:
            raise ConfigError(f"recorded source file {item.path!r} differs from its inventory")
        payload = path.read_bytes()
        digest = "sha256:" + hashlib.sha256(payload).hexdigest()
        if len(payload) != item.size or digest != item.digest:
            raise ConfigError(f"recorded source file {item.path!r} differs from its content digest")


def require_recorded_source_boundary(
    receipt: RecordedSourceBoundaryReceipt,
    evidence: RecordedSourceEvidence | None,
    *,
    anchor: Any,
    declaration: WorkloadDeclaration,
    source_artifact_digest: str,
    engine_id: str | None,
    root: Path,
) -> None:
    if type(evidence) is not RecordedSourceEvidence or evidence.receipt != receipt:
        raise ConfigError("recorded source boundary has no exact installed provenance evidence")
    if anchor.kind != "recorded-tape" or anchor.digest() != receipt.anchor_digest:
        raise ConfigError("recorded source boundary names a different physical anchor")
    if (
        receipt.downstream_source_artifact_digest != source_artifact_digest
        or anchor.source_artifact_digest != evidence.origin_source_digest
    ):
        raise ConfigError("recorded source boundary names a different actual source artifact")
    if engine_id is None or receipt.downstream_engine_id != engine_id:
        raise ConfigError("recorded source boundary names a different installed engine")
    if receipt.tape_digest != declaration.fixtures.tape_digest or (
        receipt.tape_digest != anchor.tape_digest
    ):
        raise ConfigError("recorded source boundary names a different physical tape")
    if receipt.policy_digest != declaration.policy_digest:
        raise ConfigError("recorded source boundary names a different numerical policy")
    if receipt.observation_keys_digest != declaration.fixtures.observation_keys_digest:
        raise ConfigError("recorded source boundary names different observation keys")
    if evidence.coverage_digest != recorded_source_coverage_digest(declaration):
        raise ConfigError("recorded source boundary does not cover the declared workload")
    require_recorded_source_files(root, evidence)
