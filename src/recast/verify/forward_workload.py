"""Verified full-forward workload objectives, beside ordinary bundle gates.

``recast.verify.python_jax_objectives`` checks one exported scalar float64
function and the latency of its compiled value-and-gradient. That is a real
objective and it is unchanged here, but it is not a forward workload: it says
nothing about a package that evolves named array state over many steps, and
its result must never be relabelled as though it did.

This module adds the other objective. :func:`verify_bundle_forward_objectives`
takes the complete candidate bundle, the exact original translation, the
immediate predecessor and a typed reference to the original scientific anchor,
binds every identity that makes those four things one chain, stages each
package separately, dispatches to an *installed registered* trusted workload
verifier through the existing :class:`~recast.plugins.executor.Executor`, and
returns an immutable ``recast.forward-workload-report.v1``.

Three rules shape the whole file.

**It supplements; it never substitutes.** The candidate's ordinary
:func:`~recast.phases.verify_recipe_candidates` report is a required argument,
must be bound to this exact bundle, and must be accepted. A workload adjunct
cannot stand in for a missing unit gate, and this is where that is enforced
rather than assumed.

**The plugin measures; this module concludes.** Thresholds, required modes and
required coverage come from a declaration whose digest was authorized before
the candidate ran. Speed, numerical and derivative conclusions are recomputed
here from validated measurements. A parent's accepted report is permission to
use the parent as input, never evidence about this leaf's computation.

**Nothing the candidate supplies is authority.** Policy, fixtures, tapes,
import roots, entry points and helper ownership are read from the trusted
declaration or from the bundle's frozen verification plan. The candidate
supplies bytes to execute and nothing else.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, cast

from recast.engines import TranslationEngine
from recast.errors import ConfigError, PluginError
from recast.model import OracleRef, Verdict
from recast.phases import (
    BindingCheck,
    CandidateBundle,
    CandidateUnit,
    EngineBinding,
    VerificationReport,
    _canonical_bytes,
    _composition_checks,
    _digest_bytes,
    _freeze_json,
    _json_value,
    _portable_path,
    _require_bundle_unchanged,
    _require_digest,
    _semantic_config,
    decode_candidate_bundle,
)
from recast.plugins.executor import Executor
from recast.plugins.verifier import Verifier
from recast.registry import REGISTRY, PluginOrigin, Registry
from recast.verify._forward_workload import (
    ANCHOR_KINDS,
    BOUNDARY_RECEIPT_KINDS,
    FIXTURE_SCHEMA,
    MAX_FIXTURE_BYTES,
    MIN_LINK_RECEIPTS,
    NONPRODUCTION_PREFIX,
    REPORT_SCHEMA,
    SUBJECT_ROLES,
    AdapterConfiguration,
    ComparisonObservation,
    CoverageResult,
    DerivativeObservation,
    EvidenceSubject,
    ForwardMetrics,
    ObjectiveSet,
    SpeedConclusion,
    TimingObservation,
    WorkloadDeclaration,
    WorkloadResult,
    evaluate_coverage,
    evaluate_derivatives,
    evaluate_numerics,
    evaluate_speed,
    evaluate_timing_requirements,
    fixture_section_digest,
    project_metrics,
    validate_adapter_configuration,
    validate_workload_declaration,
    validate_workload_result,
)
from recast.verify._python_accelerator_protocol import ProtocolError, decode_document
from recast.verify.recorded_source import (
    RecordedSourceBoundaryReceipt,
    RecordedSourceEvidence,
    RecordedSourceFile,
    recorded_source_coverage_digest,
    require_recorded_source_boundary,
    require_recorded_source_files,
)

__all__ = [
    "NONPRODUCTION_PREFIX",
    "AnchorReference",
    "AssembledPackage",
    "BoundaryLink",
    "ChainIdentity",
    "ForwardWorkloadReport",
    "InheritedObjectives",
    "RecordedSourceBoundaryReceipt",
    "RecordedSourceEvidence",
    "RecordedSourceFile",
    "TimingBaselineReference",
    "TransformationBoundaryReceipt",
    "WorkloadSubject",
    # A caller building an adapter configuration needs to compute the same
    # source digests the engine will check it against.
    "adapter_source_digest",
    "recorded_source_coverage_digest",
    "verify_bundle_forward_objectives",
]

_PACKAGE_DIGEST_DOMAIN = b"recast.forward-workload-package.v1\0"
_SOURCE_DIGEST_DOMAIN = b"recast.forward-workload-source.v1\0"
MAX_IMPLEMENTATION_BYTES = 8 * 1024 * 1024
"""One implementation module. Generous for source, small enough to bound."""
_RESULT_METRIC = "forward_workload"
"""Where the trusted plugin puts its result document inside ``Verdict.metrics``."""


def implementation_inventory(modules: Sequence[str]) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Hash the actual installed bytes of the named modules.

    ``PluginOrigin`` is attribution: it reads a distribution name and version
    out of installed metadata and says explicitly that this is neither a
    signature nor a content check. Matching a name and a version therefore
    cannot establish which code will run. This is the defined procedure that
    can: resolve each declared module to its file without importing it, hash
    the bytes, and digest the canonical ``(module, digest)`` inventory.

    The launch computes and authorizes that digest and puts it in the
    declaration. The adjunct recomputes it from what is actually installed and
    refuses to dispatch on a mismatch, so neither an echoed manifest digest nor
    a plugin's own constant can stand in for the bytes.
    """

    rows: list[tuple[str, str]] = []
    for name in sorted(set(modules)):
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError, AttributeError) as error:
            raise ConfigError(
                f"workload implementation module {name!r} cannot be resolved"
            ) from error
        if spec is None or spec.origin is None or spec.origin in {"built-in", "frozen"}:
            raise ConfigError(f"workload implementation module {name!r} has no file to hash")
        path = Path(spec.origin)
        if path.is_symlink() or not path.is_file():
            raise ConfigError(f"workload implementation module {name!r} is not a regular file")
        size = path.stat().st_size
        if size == 0 or size > MAX_IMPLEMENTATION_BYTES:
            raise ConfigError(
                f"workload implementation module {name!r} is {size} bytes; the bound is "
                f"{MAX_IMPLEMENTATION_BYTES}"
            )
        rows.append((name, _digest_bytes(path.read_bytes())))
    digest = _digest_bytes(_canonical_bytes([[name, value] for name, value in rows]))
    return digest, tuple(rows)


def _require_implementation(label: str, modules: Sequence[str], expected: str) -> tuple[str, ...]:
    observed, rows = implementation_inventory(modules)
    if observed != expected:
        raise ConfigError(
            f"the installed {label} bytes digest {observed} where the declaration authorized "
            f"{expected}; the code that would run is not the code that was approved"
        )
    return tuple(f"{name}@{value}" for name, value in rows)


# --- typed immutable references ---------------------------------------------


@dataclass(frozen=True)
class AnchorReference:
    """The original scientific authority, which is not a timing baseline.

    For a Python/NumPy project this is the unchanged source artifact. For
    CLUBB it is additionally the pinned Fortran and its recorded tape. A
    cross-language anchor legitimately carries a different engine and source
    identity from the chain it anchors, so the relationship is established by
    an authorized composition receipt rather than by whichever digests happen
    to match.
    """

    kind: str
    language: str
    engine_id: str | None = None
    source_artifact_digest: str | None = None
    bundle_digest: str | None = None
    report_digest: str | None = None
    tape_digest: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in ANCHOR_KINDS:
            raise ConfigError(f"workload anchor kind must be one of {sorted(ANCHOR_KINDS)}")
        if not self.language or len(self.language) > 128:
            raise ConfigError("workload anchor must declare its bounded source language")
        for name in ("source_artifact_digest", "bundle_digest", "report_digest", "tape_digest"):
            value = getattr(self, name)
            if value is not None:
                _require_digest(cast(str, value), f"workload anchor {name}")
        if self.engine_id is not None and not self.engine_id:
            raise ConfigError("workload anchor engine_id must be non-empty when present")
        required: dict[str, tuple[str, ...]] = {
            "source-artifact": ("source_artifact_digest",),
            "candidate-bundle": ("source_artifact_digest", "bundle_digest", "report_digest"),
            "recorded-tape": ("tape_digest",),
        }
        for name in required[self.kind]:
            if getattr(self, name) is None:
                raise ConfigError(f"a {self.kind} workload anchor requires {name}")
        if self.kind == "source-artifact" and (
            self.bundle_digest is not None or self.report_digest is not None
        ):
            raise ConfigError("a source-artifact workload anchor has no upstream bundle or report")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "language": self.language,
            "engine_id": self.engine_id,
            "source_artifact_digest": self.source_artifact_digest,
            "bundle_digest": self.bundle_digest,
            "report_digest": self.report_digest,
            "tape_digest": self.tape_digest,
        }

    def digest(self) -> str:
        return _digest_bytes(_canonical_bytes(self.to_dict()))


@dataclass(frozen=True)
class TransformationBoundaryReceipt:
    """Authorized evidence that a claimed transformation boundary was verified.

    Two shapes, both real. ``cross-engine`` is one accepted run's bundle
    materialized as the next run's source. ``recipe-intermediate`` is the case
    the corrected design recommends: a recipe such as ``port-clubb`` already
    emits a concrete NumPy closure on its way to JAX, and an immutable
    path/content inventory extracted from that accepted port bundle -- with
    independently accepted comparisons for each link -- is sufficient. Neither
    a separate Campaign launch nor an artificial intermediate bundle is
    required to satisfy this API.

    Transform names are retained separately from inventories on purpose:
    ``Candidate.digest()`` folds in the transform name, so two transforms that
    emit identical NumPy bytes have different candidate digests, and it is the
    bytes that matter.
    """

    kind: str
    upstream_run: str
    upstream_engine_id: str
    upstream_bundle_digest: str
    upstream_report_digest: str
    upstream_output_contract_digest: str
    downstream_engine_id: str
    downstream_input_contract_digest: str
    downstream_source_artifact_digest: str
    exported_tree_digest: str
    path_mapping_digest: str
    anchor_digest: str
    upstream_transform: str = ""
    downstream_transform: str = ""
    upstream_source_artifact_digest: str | None = None
    extracted_inventory_digest: str | None = None
    link_receipts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in BOUNDARY_RECEIPT_KINDS:
            raise ConfigError(
                f"transformation boundary receipt kind must be one of "
                f"{sorted(BOUNDARY_RECEIPT_KINDS)}"
            )
        if self.kind == "recipe-intermediate":
            if not self.upstream_transform or not self.downstream_transform:
                raise ConfigError(
                    "a recipe-intermediate receipt must name both transforms, because equal "
                    "emitted bytes can come from differently named transforms"
                )
            for name in ("upstream_source_artifact_digest", "extracted_inventory_digest"):
                value = getattr(self, name)
                if value is None:
                    raise ConfigError(f"a recipe-intermediate receipt requires {name}")
                _require_digest(cast(str, value), f"composition receipt {name}")
            if self.extracted_inventory_digest == self.upstream_source_artifact_digest:
                # The emitted NumPy closure is not the Fortran source it came
                # from; treating one digest as both is how an unverified
                # intermediate would slip through as already proven.
                raise ConfigError(
                    "a recipe-intermediate receipt must distinguish the extracted intermediate "
                    "inventory from the original source artifact it was emitted from"
                )
            if len(set(self.link_receipts)) != len(self.link_receipts):
                raise ConfigError("a recipe-intermediate receipt repeats a link receipt")
            if len(self.link_receipts) < MIN_LINK_RECEIPTS:
                raise ConfigError(
                    f"a recipe-intermediate receipt needs at least {MIN_LINK_RECEIPTS} "
                    "independently accepted link receipts, one per claimed boundary"
                )
            for digest in self.link_receipts:
                _require_digest(digest, "composition receipt link_receipts[]")
        for name in ("upstream_run", "upstream_engine_id", "downstream_engine_id"):
            value = cast(str, getattr(self, name))
            if not value or len(value) > 256:
                raise ConfigError(f"composition receipt {name} must be a bounded identifier")
        for name in (
            "upstream_bundle_digest",
            "upstream_report_digest",
            "upstream_output_contract_digest",
            "downstream_input_contract_digest",
            "downstream_source_artifact_digest",
            "exported_tree_digest",
            "path_mapping_digest",
            "anchor_digest",
        ):
            _require_digest(cast(str, getattr(self, name)), f"composition receipt {name}")

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "upstream_run": self.upstream_run,
            "upstream_engine_id": self.upstream_engine_id,
            "upstream_bundle_digest": self.upstream_bundle_digest,
            "upstream_report_digest": self.upstream_report_digest,
            "upstream_output_contract_digest": self.upstream_output_contract_digest,
            "downstream_engine_id": self.downstream_engine_id,
            "downstream_input_contract_digest": self.downstream_input_contract_digest,
            "downstream_source_artifact_digest": self.downstream_source_artifact_digest,
            "exported_tree_digest": self.exported_tree_digest,
            "path_mapping_digest": self.path_mapping_digest,
            "anchor_digest": self.anchor_digest,
            "upstream_transform": self.upstream_transform,
            "downstream_transform": self.downstream_transform,
            "upstream_source_artifact_digest": self.upstream_source_artifact_digest,
            "extracted_inventory_digest": self.extracted_inventory_digest,
            "link_receipts": list(self.link_receipts),
        }

    def digest(self) -> str:
        return _digest_bytes(_canonical_bytes(self.to_dict()))


@dataclass(frozen=True)
class BoundaryLink:
    """One independently accepted comparison across a transformation boundary.

    ``TransformationBoundaryReceipt.link_receipts`` used to be a tuple of
    digest-shaped strings that nothing resolved, so any well-formed hex stood
    in as proof that Fortran-to-NumPy and NumPy-to-initial-JAX had each been
    checked. These are the resolved objects: the caller supplies them, the
    receipt's digests must be exactly theirs, each must be accepted under the
    declaration's own policy, and they must chain end to end.

    ``upstream`` and ``downstream`` are inventory or artifact digests, not
    languages: a link says "these exact bytes were compared against those
    exact bytes and the comparison was accepted".
    """

    boundary: str
    upstream: str
    downstream: str
    report_digest: str
    policy_digest: str
    observation_keys_digest: str
    accepted: bool

    def __post_init__(self) -> None:
        if not self.boundary or len(self.boundary) > 256:
            raise ConfigError("boundary link must name its boundary")
        for name in (
            "upstream",
            "downstream",
            "report_digest",
            "policy_digest",
            "observation_keys_digest",
        ):
            _require_digest(cast(str, getattr(self, name)), f"boundary link {name}")
        if self.upstream == self.downstream:
            raise ConfigError("a boundary link must compare two different artifacts")
        if type(self.accepted) is not bool:
            raise ConfigError("boundary link acceptance must be a boolean")

    def to_dict(self) -> dict[str, object]:
        return {
            "boundary": self.boundary,
            "upstream": self.upstream,
            "downstream": self.downstream,
            "report_digest": self.report_digest,
            "policy_digest": self.policy_digest,
            "observation_keys_digest": self.observation_keys_digest,
            "accepted": self.accepted,
        }

    def digest(self) -> str:
        return _digest_bytes(_canonical_bytes(self.to_dict()))


@dataclass(frozen=True)
class ChainIdentity:
    """Which selected node this report is about, and which reports precede it.

    The two predecessor digests do different jobs and are therefore separate
    fields. ``predecessor_report_digest`` is the *ordinary*
    :class:`~recast.phases.VerificationReport`, which authorizes using that
    package as input. ``predecessor_workload_report_digest`` is the
    predecessor's *workload* report, which is the only thing that can say what
    full-workload objectives it was actually held to. An ordinary report cannot
    establish a forward-workload objective and is not asked to.

    Generation and ancestry values here are identities the engine can check
    against the supplied objects. It cannot tell from them whether a database
    attachment was retried, stopped or superseded: current-generation checks
    and the transaction fences at launch, publication and approval belong to
    the controller.
    """

    attachment: str
    spec_digest: str
    generation: int
    chain_id: str
    predecessor_report_digest: str
    predecessor_workload_report_digest: str | None = None

    def __post_init__(self) -> None:
        for name in ("attachment", "chain_id"):
            value = cast(str, getattr(self, name))
            if not value or len(value) > 256:
                raise ConfigError(f"workload chain {name} must be a bounded identifier")
        _require_digest(self.spec_digest, "workload chain spec_digest")
        _require_digest(self.predecessor_report_digest, "workload chain predecessor_report_digest")
        if self.predecessor_workload_report_digest is not None:
            _require_digest(
                self.predecessor_workload_report_digest,
                "workload chain predecessor_workload_report_digest",
            )
        if type(self.generation) is not int or not (1 <= self.generation <= 4096):
            raise ConfigError("workload chain generation must be a positive bounded integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "attachment": self.attachment,
            "spec_digest": self.spec_digest,
            "generation": self.generation,
            "chain_id": self.chain_id,
            "predecessor_report_digest": self.predecessor_report_digest,
            "predecessor_workload_report_digest": self.predecessor_workload_report_digest,
        }


@dataclass(frozen=True)
class InheritedObjectives:
    """The objective set an authorized upstream already established.

    A leaf cannot be trusted to describe what it inherited, and neither can an
    ordinary verification report. Only two things are authoritative: the
    objective set the launch froze for the whole chain, and the accepted
    workload report of the immediate predecessor. ``source`` says which, and
    ``receipt_digest`` is the identity that has to match it.
    """

    source: str
    receipt_digest: str
    authority_digest: str
    objectives: ObjectiveSet

    def __post_init__(self) -> None:
        if self.source not in {"launch", "predecessor_workload"}:
            raise ConfigError(
                "inherited objectives must come from the launch or the predecessor's workload "
                "report, not from this request"
            )
        _require_digest(self.receipt_digest, "inherited objectives receipt_digest")
        _require_digest(self.authority_digest, "inherited objectives authority_digest")
        if type(self.objectives) is not ObjectiveSet:
            raise ConfigError("inherited objectives must carry a complete ObjectiveSet")

    def to_dict(self) -> dict[str, object]:
        return {
            "source": self.source,
            "receipt_digest": self.receipt_digest,
            "authority_digest": self.authority_digest,
            "objectives": self.objectives.to_dict(),
        }


@dataclass(frozen=True)
class TimingBaselineReference:
    """Which subject the speed objectives are measured against, and why.

    The original scientific anchor and the original JAX timing baseline are
    different identities and this keeps them apart. The user's benchmark names
    a *frozen* mechanically translated JAX package; a package regenerated at a
    newer engine revision may not be the same bytes, so the baseline is
    labelled and its package digest can be pinned. A pinned digest that does
    not match the package which actually ran is refused rather than
    substituted.
    """

    label: str
    role: str
    frozen_package_digest: str | None = None

    def __post_init__(self) -> None:
        if not self.label or len(self.label) > 256:
            raise ConfigError("timing baseline label must be a bounded identifier")
        if self.role not in SUBJECT_ROLES or self.role == "candidate":
            raise ConfigError(
                "the timing baseline must be a staged subject other than the candidate"
            )
        if self.frozen_package_digest is not None:
            _require_digest(self.frozen_package_digest, "timing baseline frozen_package_digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "role": self.role,
            "frozen_package_digest": self.frozen_package_digest,
        }


# --- staged packages ---------------------------------------------------------


@dataclass(frozen=True)
class AssembledPackage:
    """One subject's complete package, staged into its own import root.

    ``package_digest`` covers every staged file -- companion modules, emitted
    runtimes, constants and declared helpers -- not the entry module alone. A
    candidate that changes a helper and leaves the entry point alone changes
    this digest, which is the point.
    """

    role: str
    root: Path
    package_digest: str
    entry_module: str
    entry_function: str
    paths: tuple[str, ...]
    unit_count: int
    unique_bytes: int
    materialized_bytes: int
    owners: Mapping[str, str]

    def descriptor(self) -> dict[str, object]:
        """The trusted, path-explicit description handed to the plugin."""
        return {
            "role": self.role,
            "root": str(self.root),
            "entry_module": self.entry_module,
            "entry_function": self.entry_function,
            "package_digest": self.package_digest,
            # The inventory, not just its size: a plugin that re-digests the
            # package it was handed is checking the parent's claim rather
            # than inheriting it, and it cannot do that from a count.
            "paths": list(self.paths),
            "file_count": len(self.paths),
            "unique_bytes": self.unique_bytes,
            "materialized_bytes": self.materialized_bytes,
        }


@dataclass(frozen=True)
class WorkloadSubject:
    """The report's binding for one staged subject."""

    role: str
    bundle_digest: str
    ordinary_report_digest: str
    recipe: str
    engine: EngineBinding | None
    source_artifact_digest: str
    semantic_config_digest: str
    package_digest: str
    entry_module: str
    file_count: int
    unit_count: int
    unique_bytes: int
    materialized_bytes: int

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "bundle_digest": self.bundle_digest,
            "ordinary_report_digest": self.ordinary_report_digest,
            "recipe": self.recipe,
            "engine": self.engine.to_dict() if self.engine is not None else None,
            "source_artifact_digest": self.source_artifact_digest,
            "semantic_config_digest": self.semantic_config_digest,
            "package_digest": self.package_digest,
            "entry_module": self.entry_module,
            "file_count": self.file_count,
            "unit_count": self.unit_count,
            "unique_bytes": self.unique_bytes,
            "materialized_bytes": self.materialized_bytes,
        }


@dataclass(frozen=True)
class ForwardWorkloadReport:
    """Immutable acceptance proof for one candidate's forward workload.

    Digests, counts, statuses, validated numbers and reason codes only: no
    source bytes, machine root, timestamp or verifier detail text. Detail that
    does not belong in a control-plane message -- per-field, per-step or
    per-column results -- is referenced through ``evidence`` subjects.

    The acceptance truth table, in full. Each objective family has three
    states: ``passed``, ``failed`` and ``not_requested``.

    * ``numerical_status`` is never ``not_requested``: the declaration must
      require both the original anchor and the immediate predecessor.
    * ``derivative_status`` is ``not_requested`` only when the declaration
      names no required mode. A required mode that is missing, unsupported or
      unresolved is ``failed``; it cannot become a pass by omission.
    * ``performance_status`` is ``not_requested`` only when the declaration
      names no speed objective, and a missing measurement is ``failed``, never
      an invented one. A numerically correct but slower candidate keeps
      ``numerical_status="passed"`` and gets ``performance_status="failed"``.
    * ``accepted`` requires every family to be ``passed`` or
      ``not_requested``, every coverage row accepted, no reason codes, a
      permitted measurement class, and the plugin's own verdict to have passed.
    * ``production_admissible`` is narrower and separate: it additionally
      requires really-executed measurements from a plugin with a verified
      installed distribution origin, and **no** ``unsupported`` entry with
      the reserved ``nonproduction.`` prefix. A synthetic provider, a local
      registration, or a plugin that typed its own result as
      non-production can be ``accepted`` under a declaration that permits
      it and is still not admissible as production evidence.
    """

    workload: str
    workload_version: str
    workload_manifest_digest: str
    policy_digest: str
    adapter_digest: str
    package_contract_digest: str
    tape_digest: str
    verifier_plugin: str
    verifier_origin: Mapping[str, object]
    verifier_implementation: tuple[str, ...]
    adapter_implementation: tuple[str, ...]
    bundle_report_digest: str
    objectives: ObjectiveSet
    inherited: InheritedObjectives | None
    timing_baseline: TimingBaselineReference
    subjects: tuple[WorkloadSubject, ...]
    anchor: AnchorReference
    composition: TransformationBoundaryReceipt | RecordedSourceBoundaryReceipt | None
    boundary_links: tuple[BoundaryLink, ...]
    chain: ChainIdentity
    bindings: tuple[BindingCheck, ...]
    coverage: tuple[CoverageResult, ...]
    comparisons: tuple[ComparisonObservation, ...]
    derivatives: tuple[DerivativeObservation, ...]
    timings: tuple[TimingObservation, ...]
    speed: tuple[SpeedConclusion, ...]
    metrics: ForwardMetrics
    evidence: tuple[EvidenceSubject, ...]
    measurement_class: str
    numerical_status: str
    derivative_status: str
    performance_status: str
    production_admissible: bool
    accepted: bool
    reason_codes: tuple[str, ...]
    unsupported: tuple[str, ...]

    schema: ClassVar[str] = REPORT_SCHEMA

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "workload": self.workload,
            "workload_version": self.workload_version,
            "workload_manifest_digest": self.workload_manifest_digest,
            "policy_digest": self.policy_digest,
            "adapter_digest": self.adapter_digest,
            "package_contract_digest": self.package_contract_digest,
            "tape_digest": self.tape_digest,
            "verifier_plugin": self.verifier_plugin,
            "verifier_origin": _json_value(self.verifier_origin, "workload verifier origin"),
            "verifier_implementation": list(self.verifier_implementation),
            "adapter_implementation": list(self.adapter_implementation),
            "bundle_report_digest": self.bundle_report_digest,
            "objectives": self.objectives.to_dict(),
            "inherited": self.inherited.to_dict() if self.inherited is not None else None,
            "timing_baseline": self.timing_baseline.to_dict(),
            "subjects": [item.to_dict() for item in self.subjects],
            "anchor": self.anchor.to_dict(),
            "composition": self.composition.to_dict() if self.composition is not None else None,
            "boundary_links": [item.to_dict() for item in self.boundary_links],
            "chain": self.chain.to_dict(),
            "bindings": [item.to_dict() for item in self.bindings],
            "coverage": [item.to_dict() for item in self.coverage],
            "comparisons": [item.to_dict() for item in self.comparisons],
            "derivatives": [item.to_dict() for item in self.derivatives],
            "timings": [item.to_dict() for item in self.timings],
            "speed": [item.to_dict() for item in self.speed],
            "metrics": self.metrics.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "measurement_class": self.measurement_class,
            "numerical_status": self.numerical_status,
            "derivative_status": self.derivative_status,
            "performance_status": self.performance_status,
            "production_admissible": self.production_admissible,
            "accepted": self.accepted,
            "reason_codes": list(self.reason_codes),
            "unsupported": list(self.unsupported),
        }

    def to_json(self) -> bytes:
        return _canonical_bytes(self.to_dict())

    def digest(self) -> str:
        return _digest_bytes(self.to_json())


# --- package assembly --------------------------------------------------------


def _entry_names(entry_point: str) -> tuple[str, str]:
    module, _, function = entry_point.rpartition(".")
    if not module or not function:
        raise ConfigError(
            "workload entry_point must name its module and its callable, as 'module.callable'"
        )
    return module, function


def _assemble_package(
    role: str,
    bundle: CandidateBundle,
    declaration: WorkloadDeclaration,
    target: Path,
) -> AssembledPackage:
    """Stage every file of every candidate in one bundle into a fresh root.

    Duplicate import paths must resolve deterministically: the same path
    carrying the same bytes in two Candidates is one file with two owners,
    and the same path carrying different bytes is refused rather than
    resolved by staging order.
    """

    contract = declaration.package_contract
    # This subject's own entry, not one entry for all four. A reconstructed
    # named-state leaf and the flat controls it is compared with are
    # different implementations of the same physical workload, and staging
    # them all under the declaration's single entry point named the wrong
    # function for whichever of them was not flat.
    entry_module, entry_function = _entry_names(declaration.interface.entry_for(role))
    target.mkdir(parents=True, exist_ok=False)
    contents: dict[str, bytes] = {}
    owners: dict[str, str] = {}
    unique: dict[str, int] = {}
    materialized = 0
    unit_count = 0
    for item in sorted(bundle.units, key=lambda entry: entry.unit.uid):
        candidate = item.candidate
        if candidate is None:
            continue
        unit_count += 1
        if candidate.patches:
            raise ConfigError(
                f"{role} package unit {item.unit.uid!r} carries "
                f"{len(candidate.patches)} patch(es); a forward workload package must be "
                "complete files it can import"
            )
        for raw_path, content in candidate.files.items():
            path = _portable_path(raw_path, f"{role} package file")
            if not isinstance(content, bytes):
                raise ConfigError(f"{role} package file {path} is not bytes")
            if len(content) > contract.max_file_bytes:
                raise ConfigError(
                    f"{role} package file {path} is {len(content)} bytes; the declared ceiling "
                    f"is {contract.max_file_bytes}"
                )
            existing = contents.get(path)
            if existing is None:
                contents[path] = content
                owners[path] = item.unit.uid
                unique[_digest_bytes(content)] = len(content)
            elif existing != content:
                raise ConfigError(
                    f"{role} package resolves import path {path} to two different files; "
                    f"{owners[path]!r} and {item.unit.uid!r} both own it"
                )
            materialized += len(content)
    if not contents:
        raise ConfigError(f"{role} package contains no files to stage")
    if len(contents) > contract.max_files:
        raise ConfigError(
            f"{role} package stages {len(contents)} files; the declared ceiling is "
            f"{contract.max_files}"
        )
    unique_bytes = sum(unique.values())
    if unique_bytes > contract.max_total_bytes or materialized > contract.max_total_bytes:
        raise ConfigError(
            f"{role} package holds {unique_bytes} unique and {materialized} materialized bytes; "
            f"the declared ceiling is {contract.max_total_bytes}"
        )

    for path, content in sorted(contents.items()):
        destination = target / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)

    entry_relative = entry_module.replace(".", "/") + ".py"
    if entry_relative not in contents:
        raise ConfigError(
            f"{role} package does not contain its declared entry module {entry_module!r}"
        )
    for module in contract.entry_modules:
        required = module.replace(".", "/") + ".py"
        if required not in contents:
            raise ConfigError(f"{role} package drops frozen entry module {module!r}")
    # The subject's own entry function must be exported by the module that
    # will actually be imported for it. The declaration's frozen export list
    # still applies wherever it names the same module; a subject with its own
    # entry -- a named-state leaf, say -- exports that instead, and requiring
    # the flat name there would refuse exactly the case per-subject entries
    # exist for.
    expected = (
        contract.required_exports
        if entry_module == _entry_names(declaration.interface.entry_point)[0]
        else (entry_function,)
    )
    _require_exports(role, contents[entry_relative], entry_module, expected)

    digest = hashlib.sha256(_PACKAGE_DIGEST_DOMAIN)
    for path, content in sorted(contents.items()):
        encoded = path.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(hashlib.sha256(content).digest())
    return AssembledPackage(
        role=role,
        root=target,
        package_digest=f"sha256:{digest.hexdigest()}",
        entry_module=entry_module,
        entry_function=entry_function,
        paths=tuple(sorted(contents)),
        unit_count=unit_count,
        unique_bytes=unique_bytes,
        materialized_bytes=materialized,
        owners=MappingProxyType(dict(sorted(owners.items()))),
    )


def _require_exports(role: str, source: bytes, module: str, required: tuple[str, ...]) -> None:
    """Check the declared exports statically, without importing the package.

    Parsing is the whole check: importing a candidate module to ask what it
    exports would run candidate code on the controller's side of the
    boundary, and this runs before any isolated worker starts.
    """

    try:
        tree = ast.parse(source.decode(), filename=f"{module}.py")
    except (SyntaxError, UnicodeDecodeError) as error:
        raise ConfigError(f"{role} package entry module {module!r} does not parse") from error
    defined: set[str] = set()
    declared: set[str] | None = None
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    declared = _string_sequence(node.value)
                elif isinstance(target, ast.Name):
                    defined.add(target.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == "__all__":
                declared = _string_sequence(node.value)
            else:
                defined.add(node.target.id)
    available = defined if declared is None else defined & declared
    missing = sorted(set(required) - available)
    if missing:
        raise ConfigError(
            f"{role} package entry module {module!r} does not export required name(s) {missing}"
        )


def _string_sequence(value: ast.expr | None) -> set[str] | None:
    if not isinstance(value, (ast.List, ast.Tuple)):
        return None
    names: set[str] = set()
    for element in value.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            return None
        names.add(element.value)
    return names


def source_inventory(root: Path, bundle: CandidateBundle, anchor_module: str) -> str:
    """Digest the live source files the anchor authority actually reads.

    A ``source-artifact`` anchor is executed from the project root as it
    stands, so those bytes are part of the acceptance and belong inside the
    dispatch fence. Appending a line to the anchor's own module during
    dispatch previously produced an accepted report: the fence covered the
    staged packages and the read-only fixture but not the live source.

    The inventory is bounded by the bundle's own declared source coverage --
    every ``Unit.sources`` entry plus the declared anchor module -- rather
    than by walking an arbitrary tree.
    """

    names = {source.as_posix() for unit in bundle.discovered_units for source in unit.sources}
    names.update(source.as_posix() for item in bundle.units for source in item.unit.sources)
    names.add(anchor_module.replace(".", "/") + ".py")
    digest = hashlib.sha256(_SOURCE_DIGEST_DOMAIN)
    for name in sorted(names):
        target = root / name
        encoded = name.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        if target.is_symlink() or not target.is_file():
            digest.update(b"\0absent")
            continue
        digest.update(hashlib.sha256(target.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"


_ADAPTER_SOURCE_DOMAIN = b"recast.forward-workload-adapter-source.v1\0"


def adapter_source_digest(kind: str, path: Path) -> str:
    """Digest one declared adapter source: a file's bytes, or a whole tree.

    A ``tree`` is digested over every regular file under it, by sorted
    relative path, excluding derived bytecode. That is what gives a directory
    of recorded observations real authority: editing one file in it changes
    this value, so an expected value cannot be quietly rewritten between
    authorization and execution.
    """

    resolved = Path(path)
    digest = hashlib.sha256(_ADAPTER_SOURCE_DOMAIN + kind.encode() + b"\0")
    if kind == "file":
        if resolved.is_symlink() or not resolved.is_file():
            raise ConfigError(f"adapter source {path} is not a regular readable file")
        digest.update(hashlib.sha256(resolved.read_bytes()).digest())
        return f"sha256:{digest.hexdigest()}"
    if resolved.is_symlink() or not resolved.is_dir():
        raise ConfigError(f"adapter source {path} is not a readable directory")
    for item in sorted(resolved.rglob("*")):
        if not item.is_file() or "__pycache__" in item.parts:
            continue
        if item.is_symlink():
            raise ConfigError(f"adapter source tree {path} contains the symlink {item.name}")
        relative = item.relative_to(resolved).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return f"sha256:{digest.hexdigest()}"


def _require_adapter_configuration(
    declaration: WorkloadDeclaration, supplied: Mapping[str, object] | None
) -> tuple[AdapterConfiguration | None, dict[str, str]]:
    """Bind the adapter configuration to the declaration, and resolve it.

    Three separate statements have to hold, and each was a way to accept the
    wrong thing:

    * the declaration either authorizes a configuration or it does not, and
      supplying one it did not authorize -- or omitting one it did -- is a
      contradiction, not a default;
    * the supplied object's own digest must be the authorized one, so a
      launch freezes the adapter's configuration with everything else;
    * every named source must resolve *now* to the digest it declares. The
      path is a location; the digest is the authority.

    Nothing here interprets a value name or a source name. They are the
    extension's vocabulary.
    """

    authorized = declaration.adapter.configuration_digest
    if supplied is None:
        if authorized is not None:
            raise ConfigError(
                "the declaration authorizes an adapter configuration "
                f"({authorized}) and none was supplied"
            )
        return None, {}
    if authorized is None:
        raise ConfigError(
            "an adapter configuration was supplied and the declaration authorizes none"
        )
    configuration = validate_adapter_configuration(supplied)
    if configuration.digest() != authorized:
        raise ConfigError(
            f"the supplied adapter configuration digests to {configuration.digest()} and the "
            f"declaration authorizes {authorized}"
        )
    resolved: dict[str, str] = {}
    for name in sorted(configuration.sources):
        source = configuration.sources[name]
        observed = adapter_source_digest(source.kind, Path(source.path))
        if observed != source.digest:
            raise ConfigError(
                f"adapter source {name!r} at {source.path} digests to {observed} and the "
                f"authorized configuration declares {source.digest}"
            )
        resolved[name] = observed
    return configuration, resolved


def staged_inventory(package: AssembledPackage) -> str:
    """Re-digest one subject's package from the files on disk.

    ``package_digest`` is computed from the bundle's bytes as they were
    staged. This reads the staged tree back, so a file appended to between
    staging and the plugin's run -- or after it -- changes this value and
    fails the fence rather than being reported under the identity it had
    before.
    """

    digest = hashlib.sha256(_PACKAGE_DIGEST_DOMAIN)
    for path in package.paths:
        target = package.root / path
        if target.is_symlink() or not target.is_file():
            raise PluginError(f"staged {package.role} package file {path} is missing or replaced")
        encoded = path.encode()
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(hashlib.sha256(target.read_bytes()).digest())
    present = {
        item.relative_to(package.root).as_posix()
        for item in package.root.rglob("*")
        if item.is_file() and "__pycache__" not in item.parts
    }
    # Compiled bytecode is derived from the source files whose bytes are
    # hashed above, and importing the package is what creates it, so its
    # presence is expected rather than a change to the package.
    if present != set(package.paths):
        raise PluginError(
            f"staged {package.role} package gained or lost files: "
            f"{sorted(present.symmetric_difference(package.paths))}"
        )
    return f"sha256:{digest.hexdigest()}"


def _require_declared_package_edits(
    candidate: AssembledPackage,
    original: AssembledPackage,
    declaration: WorkloadDeclaration,
) -> None:
    """Only declared helpers may appear, and nothing may disappear.

    This is the engine-side half of the package edit contract. A helper the
    declaration did not name, a helper moved to a different import path, or a
    lost module all fail here -- the controller's export codec constructs
    changed-file descriptors, and an agent cannot widen the package by
    inventing one.
    """

    declared = set(declaration.package_contract.helper_files)
    added = set(candidate.paths) - set(original.paths)
    removed = set(original.paths) - set(candidate.paths)
    if removed:
        raise ConfigError(
            f"candidate package drops file(s) {sorted(removed)} the original translation owned"
        )
    undeclared = sorted(added - declared)
    if undeclared:
        raise ConfigError(f"candidate package adds undeclared file(s) {undeclared}")
    absent = sorted(declared - set(candidate.paths))
    if absent:
        raise ConfigError(f"candidate package is missing declared helper(s) {absent}")
    owner = declaration.package_contract.helper_owner
    if declared:
        wrong = sorted(path for path in declared if candidate.owners.get(path) != owner)
        if wrong:
            raise ConfigError(
                f"declared helper(s) {wrong} are not owned by the declared Candidate {owner!r}"
            )


def _require_fixture(observations: Path, declaration: WorkloadDeclaration) -> Path:
    """Bind the read-only observation fixture to the authorized digests.

    A forward workload needs two things a code bundle must not carry: the
    observation tape and the numerical policy. Both are staged beside the run
    and both are authorized independently, so a swapped tape or a loosened
    tolerance is caught here rather than becoming a favourable result. The
    engine hashes; only the plugin parses.
    """

    resolved = Path(observations).resolve()
    if resolved.is_symlink() or not resolved.is_file():
        raise ConfigError("workload observations must be a regular readable file")
    size = resolved.stat().st_size
    if size == 0 or size > MAX_FIXTURE_BYTES:
        raise ConfigError(
            f"workload observations are {size} bytes; the bound is {MAX_FIXTURE_BYTES}"
        )
    try:
        document = decode_document(resolved.read_bytes())
    except ProtocolError as error:
        raise ConfigError(
            f"workload observations are not one canonical document: {error}"
        ) from error
    if document.get("schema") != FIXTURE_SCHEMA or set(document) != {"schema", "policy", "tape"}:
        raise ConfigError(f"workload observations must be a {FIXTURE_SCHEMA} document")
    if fixture_section_digest(document["policy"]) != declaration.policy_digest:
        raise ConfigError("workload observation policy is not the authorized checker policy")
    if fixture_section_digest(document["tape"]) != declaration.fixtures.tape_digest:
        raise ConfigError("workload observation tape is not the authorized tape")
    return resolved


# --- identity binding --------------------------------------------------------

_UNCHANGED_ACROSS_SUBJECTS = (
    "recipe",
    "semantic_config_digest",
    "source_artifact_digest",
    "engine",
    "frontend_plugins",
    "transform_plugin",
    "verification_plan",
    "discovered_units",
)


@dataclass
class _Subject:
    """One decoded subject and the snapshot that proves it was not mutated."""

    role: str
    bundle: CandidateBundle
    snapshot: bytes = field(repr=False)
    saved: CandidateBundle = field(repr=False)

    @property
    def digest(self) -> str:
        return _digest_bytes(self.snapshot)


def _decode_subject(role: str, bundle: CandidateBundle) -> _Subject:
    snapshot = bundle.to_json()
    return _Subject(
        role=role, bundle=bundle, snapshot=snapshot, saved=decode_candidate_bundle(snapshot)
    )


def _require_same_chain(candidate: _Subject, other: _Subject) -> None:
    differing = [
        name
        for name in _UNCHANGED_ACROSS_SUBJECTS
        if getattr(candidate.saved, name) != getattr(other.saved, name)
    ]
    if differing:
        raise ConfigError(
            f"workload {other.role} differs from the candidate in {differing}; a forward "
            "workload compares packages built from one source, engine and verification plan"
        )
    original_units = {item.unit.uid: item for item in other.saved.units}
    if set(original_units) != {item.unit.uid for item in candidate.saved.units}:
        raise ConfigError(f"workload {other.role} unit coverage differs from the candidate")
    for item in candidate.saved.units:
        peer = original_units[item.unit.uid]
        if item.unit != peer.unit or item.facts != peer.facts:
            raise ConfigError(f"workload {other.role} source facts differ from the candidate")


def _plan_gates(bundle: CandidateBundle, engine: TranslationEngine | None) -> tuple[str, ...]:
    """Every gate the bundle's own frozen plan and engine require.

    Derived, not supplied: the adjunct must not accept an ordinary report's own
    idea of which gates were required. This is the same union
    :func:`~recast.phases.verify_recipe_candidates` builds.
    """

    declared = [
        stage.plugin
        for stage in bundle.verification_plan.stages
        if stage.kind == "verifier" and stage.gate
    ]
    required = set(declared) | set(engine.required_gates if engine is not None else ())
    return tuple(sorted(required))


def _require_bound_report(
    role: str,
    report: VerificationReport,
    subject: _Subject,
    *,
    expected_digest: str | None,
    owner_unit: str,
    engine: TranslationEngine | None,
) -> None:
    """An ordinary report is only permission when it really verified this bundle.

    ``accepted`` is a conclusion the report reached about itself, so it is the
    weakest thing here. Everything else is derived from the bundle and its
    frozen verification plan and then required of the report: the same recipe,
    source artifact and engine; the same selected units compared by uid; each
    unit's exact candidate digest; a passing result for *every* required gate
    on *every* unit, not only on the entry owner; complete recorded evidence;
    an empty deferred ledger where the report claims one; consistent aggregate
    gate coverage; and no reason codes beside an acceptance.

    Each of those was separately reproduced as an acceptance with a report
    that contradicted it -- an entry owner with no gate results at all, a
    companion unit whose stage had failed, and candidate digests belonging to
    other candidates.
    """

    if type(report) is not VerificationReport:
        raise ConfigError(f"workload {role} report must be a VerificationReport")
    for name in ("bindings", "units", "required_units", "gates", "subprograms", "reason_codes"):
        if type(getattr(report, name)) is not tuple:
            # A report built with a mutable sequence can be emptied after it
            # is validated and before its digest is bound.
            raise ConfigError(f"workload {role} report field {name!r} must be an immutable tuple")
    if report.bundle_digest != subject.digest:
        raise ConfigError(
            f"workload {role} report is bound to a different bundle; it cannot certify this one"
        )
    if not report.accepted:
        reasons = ", ".join(report.reason_codes) or "no reason"
        raise ConfigError(f"workload {role} report was not accepted ({reasons})")
    if report.reason_codes:
        raise ConfigError(
            f"workload {role} report claims acceptance and reports {list(report.reason_codes)}"
        )
    failed_bindings = sorted(item.name for item in report.bindings if not item.passed)
    if failed_bindings:
        # A report cannot claim acceptance over its own failed binding checks;
        # this was accepted with ``BindingCheck("recipe", False)`` present.
        raise ConfigError(
            f"workload {role} report claims acceptance with failed binding check(s) "
            f"{failed_bindings}"
        )
    if expected_digest is not None and report.digest() != expected_digest:
        raise ConfigError(f"workload {role} report digest differs from the recorded chain identity")

    saved = subject.saved
    if (
        report.recipe != saved.recipe
        or report.source_artifact_digest != saved.source_artifact_digest
        or report.engine != saved.engine
    ):
        raise ConfigError(
            f"workload {role} report describes a different recipe, source artifact or engine"
        )
    required = _plan_gates(saved, engine)
    if not required:
        raise ConfigError(f"workload {role} bundle's frozen plan declares no required gate")

    expected_digests = {
        item.unit.uid: (f"sha256:{item.candidate.digest()}" if item.candidate is not None else None)
        for item in saved.units
    }
    reported = [item.uid for item in report.units]
    if len(set(reported)) != len(reported):
        raise ConfigError(f"workload {role} report names a unit twice")
    if set(reported) != set(expected_digests):
        raise ConfigError(
            f"workload {role} report covers units {sorted(reported)}, "
            f"not {sorted(expected_digests)}"
        )
    for item in report.units:
        if item.candidate_digest != expected_digests[item.uid]:
            raise ConfigError(
                f"workload {role} report attributes unit {item.uid!r} to candidate "
                f"{item.candidate_digest!r}, not to this bundle's transform product"
            )
        if item.transform_status != "ok" or item.stage_status != "passed":
            raise ConfigError(
                f"workload {role} report shows unit {item.uid!r} as "
                f"{item.transform_status}/{item.stage_status}"
            )
        if not item.evidence_complete:
            raise ConfigError(
                f"workload {role} report has incomplete evidence for unit {item.uid!r}"
            )
        if item.deferred_count:
            raise ConfigError(
                f"workload {role} report leaves {item.deferred_count} deferred entr(ies) on "
                f"unit {item.uid!r}"
            )
        results = {gate.gate: gate for gate in item.gates}
        missing = sorted(set(required) - set(results))
        if missing:
            raise ConfigError(
                f"workload {role} report has no result for gate(s) {missing} on unit {item.uid!r}"
            )
        failing = sorted(name for name in required if results[name].status != "passed")
        if failing:
            raise ConfigError(
                f"workload {role} report shows unit {item.uid!r} not passing gate(s) {failing}"
            )
        # Each gate result names the candidate it judged. A gate that judged
        # some other candidate is not a gate on this one, and every gate's
        # digest was previously left unchecked.
        misattributed = sorted(
            name
            for name in required
            if results[name].candidate_digest != expected_digests[item.uid]
        )
        if misattributed:
            raise ConfigError(
                f"workload {role} report gate(s) {misattributed} on unit {item.uid!r} judged "
                f"a different candidate than this bundle's transform product"
            )
    if owner_unit not in set(reported):
        raise ConfigError(
            f"workload {role} report does not cover the declared entry owner {owner_unit!r}"
        )

    coverage = {item.gate: item for item in report.gates}
    if set(coverage) != set(required):
        raise ConfigError(
            f"workload {role} report covers gate(s) {sorted(coverage)}, not {list(required)}"
        )
    for name in required:
        row = coverage[name]
        if not row.accepted or set(row.passed_units) != set(expected_digests):
            raise ConfigError(
                f"workload {role} report gate {name!r} does not accept every selected unit"
            )
        if row.failed_units or row.missing_units or row.unrecorded_units:
            raise ConfigError(
                f"workload {role} report gate {name!r} records failed, missing or unrecorded units"
            )
    if not report.deferred.accepted:
        raise ConfigError(f"workload {role} report deferred coverage was not accepted")
    if any(not item.accepted for item in report.required_units):
        raise ConfigError(f"workload {role} report has an unaccepted required unit")
    if any(item.required and not item.accepted for item in report.subprograms):
        raise ConfigError(f"workload {role} report has an unaccepted required subprogram")


def _require_links(
    receipt: TransformationBoundaryReceipt,
    links: tuple[BoundaryLink, ...],
    declaration: WorkloadDeclaration,
) -> None:
    """Resolve the receipt's declared link receipts and check what they say.

    A digest-shaped string is not a comparison. Every digest the receipt
    declares must be one of these supplied links; every link must be
    accepted, taken under the declaration's own numerical policy, and cover
    the declared observation keys; and the links must form one connected path
    that starts at the receipt's original source artifact and verifies the
    exported intermediate closure on *both* sides -- at least one accepted
    comparison into it and at least one out of it. That is what makes
    "Fortran-to-NumPy and NumPy-to-initial-JAX were each independently
    accepted" a checked statement rather than an asserted one.
    """

    resolved = {link.digest(): link for link in links}
    if len(resolved) != len(links):
        raise ConfigError("the supplied boundary links repeat a link")
    declared = set(receipt.link_receipts)
    if declared != set(resolved):
        missing = sorted(declared - set(resolved))
        extra = sorted(set(resolved) - declared)
        raise ConfigError(
            "the supplied boundary links are not the ones the receipt declares "
            f"(unresolved {missing}, unexpected {extra})"
        )
    for link in links:
        if not link.accepted:
            raise ConfigError(f"boundary link {link.boundary!r} was not accepted")
        if link.policy_digest != declaration.policy_digest:
            raise ConfigError(
                f"boundary link {link.boundary!r} was taken under a different numerical policy"
            )
        if link.observation_keys_digest != declaration.fixtures.observation_keys_digest:
            raise ConfigError(
                f"boundary link {link.boundary!r} does not cover the declared observation keys"
            )
    ordered = sorted(links, key=lambda link: link.boundary)
    reachable = {receipt.upstream_source_artifact_digest}
    remaining = list(ordered)
    path: list[BoundaryLink] = []
    while remaining:
        for index, link in enumerate(remaining):
            if link.upstream in reachable:
                reachable.add(link.downstream)
                path.append(link)
                del remaining[index]
                break
        else:
            raise ConfigError(
                "the supplied boundary links do not form one connected path from the "
                f"receipt's original source artifact; {[link.boundary for link in remaining]} "
                "are unreachable"
            )
    inventory = receipt.extracted_inventory_digest
    into = [link for link in path if link.downstream == inventory]
    out_of = [link for link in path if link.upstream == inventory]
    if not into or not out_of:
        # The exported closure is an *intermediate*: it has to be verified on
        # both sides. One link ending there would only show it was produced,
        # and one link leaving there would only show what was made from it.
        raise ConfigError(
            "the supplied boundary links do not verify the exported intermediate inventory on "
            f"both sides ({len(into)} link(s) into it, {len(out_of)} out of it)"
        )


def _require_anchor(
    anchor: AnchorReference,
    receipt: TransformationBoundaryReceipt | RecordedSourceBoundaryReceipt | None,
    candidate: _Subject,
    declaration: WorkloadDeclaration,
    links: tuple[BoundaryLink, ...],
    *,
    recorded_source_evidence: RecordedSourceEvidence | None = None,
    source_root: Path | None = None,
) -> None:
    """Bind the anchor and, when a transformation boundary is claimed, its receipt.

    Two shapes, validated differently, because they are different claims.

    ``cross-engine`` is one accepted run's bundle materialized as the next
    run's source, so the exported tree really is the downstream source and
    that equality is the check.

    ``recipe-intermediate`` is the route the corrected design recommends:
    ``port-clubb`` emits a concrete NumPy closure on the way to JAX inside one
    request, so its extracted inventory is *not* the downstream source
    artifact -- that is still the original Fortran. Requiring the cross-engine
    equality here refused the very route it was meant to enable. This kind is
    also permitted beside an anchor that is this chain's own source: an
    internal boundary is a claimed boundary whatever the anchor is.
    """

    if anchor.kind == "recorded-tape" and anchor.tape_digest != declaration.fixtures.tape_digest:
        raise ConfigError("workload anchor tape differs from the declared observation authority")
    if isinstance(receipt, RecordedSourceBoundaryReceipt):
        if links:
            raise ConfigError("a recorded source boundary cannot claim intermediate links")
        if declaration.boundary_receipt_digest != receipt.digest():
            raise ConfigError(
                "the recorded source receipt is not the one the declaration authorized"
            )
        if source_root is None:
            raise ConfigError("a recorded source boundary needs the actual source root")
        engine = candidate.saved.engine
        require_recorded_source_boundary(
            receipt,
            recorded_source_evidence,
            anchor=anchor,
            declaration=declaration,
            source_artifact_digest=candidate.saved.source_artifact_digest,
            engine_id=None if engine is None else engine.id,
            root=source_root,
        )
        return
    if recorded_source_evidence is not None:
        raise ConfigError("recorded source evidence needs its own recorded source receipt")
    direct = (
        anchor.kind == "source-artifact"
        and anchor.source_artifact_digest == candidate.saved.source_artifact_digest
    )
    if receipt is None:
        if declaration.boundary_receipt_digest is not None:
            raise ConfigError(
                "the declaration authorizes a transformation boundary receipt that this "
                "request did not supply"
            )
        if not direct:
            raise ConfigError(
                "a workload anchor from another source or engine requires a transformation "
                "boundary receipt; matching digests alone do not establish the relationship"
            )
        return
    if declaration.boundary_receipt_digest is None:
        raise ConfigError("the declaration does not authorize any transformation boundary receipt")
    if receipt.digest() != declaration.boundary_receipt_digest:
        raise ConfigError(
            "the transformation boundary receipt is not the one the declaration authorized"
        )
    if receipt.anchor_digest != anchor.digest():
        raise ConfigError("the transformation boundary receipt authorizes a different anchor")
    if receipt.downstream_source_artifact_digest != candidate.saved.source_artifact_digest:
        raise ConfigError(
            "the transformation boundary receipt does not name this chain's source artifact"
        )
    if receipt.upstream_output_contract_digest != receipt.downstream_input_contract_digest:
        raise ConfigError(
            "the transformation boundary receipt's upstream output and downstream input "
            "contracts differ"
        )
    engine = candidate.saved.engine
    if engine is not None and receipt.downstream_engine_id != engine.id:
        raise ConfigError("the transformation boundary receipt names a different downstream engine")

    if links and receipt.kind != "recipe-intermediate":
        raise ConfigError(
            "boundary links describe a recipe-intermediate receipt's per-link comparisons"
        )
    if receipt.kind == "cross-engine":
        if direct:
            raise ConfigError(
                "a cross-engine receipt describes a handoff between two runs; this anchor is "
                "this chain's own source, so use kind 'recipe-intermediate' for an internal "
                "boundary"
            )
        if receipt.exported_tree_digest != receipt.downstream_source_artifact_digest:
            raise ConfigError(
                "the transformation boundary receipt's exported tree is not the source the "
                "downstream run bound"
            )
        if anchor.kind == "candidate-bundle":
            if receipt.upstream_bundle_digest != anchor.bundle_digest:
                raise ConfigError(
                    "the transformation boundary receipt names a different upstream bundle"
                )
            if receipt.upstream_report_digest != anchor.report_digest:
                raise ConfigError(
                    "the transformation boundary receipt names a different upstream report"
                )
            if anchor.engine_id is not None and receipt.upstream_engine_id != anchor.engine_id:
                raise ConfigError(
                    "the transformation boundary receipt names a different upstream engine"
                )
        return

    # recipe-intermediate
    if receipt.upstream_engine_id != receipt.downstream_engine_id:
        raise ConfigError(
            "a recipe-intermediate receipt describes one engine's internal boundary; use "
            "kind 'cross-engine' when the two sides really are different engines"
        )
    if receipt.upstream_source_artifact_digest != receipt.downstream_source_artifact_digest:
        raise ConfigError(
            "a recipe-intermediate receipt describes one request, so both sides share its "
            "original source artifact"
        )
    if receipt.exported_tree_digest != receipt.extracted_inventory_digest:
        raise ConfigError(
            "a recipe-intermediate receipt's exported tree must be the extracted intermediate "
            "inventory it binds"
        )
    if anchor.kind == "candidate-bundle" and receipt.upstream_bundle_digest != anchor.bundle_digest:
        raise ConfigError("the transformation boundary receipt names a different upstream bundle")
    _require_links(receipt, links, declaration)


def _require_inherited(
    declaration: WorkloadDeclaration,
    inherited: InheritedObjectives | None,
    chain: ChainIdentity,
    predecessor: _Subject,
    predecessor_report: VerificationReport,
    candidate_digest: str,
    anchor: AnchorReference,
    predecessor_workload_report: ForwardWorkloadReport | None,
) -> tuple[str, ...]:
    """Check this leaf against an authorized upstream objective set.

    Nothing in the request may weaken what an ancestor established, so the
    inherited set is authorized by exactly one of two things and bound to a
    controller-supplied identity either way.

    For ``predecessor_workload`` the parent's accepted workload report is the
    authority, and the binding is the **edge**: the parent's candidate bundle
    and package must be this launch's predecessor, taken against that bundle's
    ordinary report, under a distinct attachment in the same chain. A report
    certifying some other bundle was previously accepted as authority for this
    one. Its scientific authority has to match too -- same workload, policy,
    tape and original anchor -- or a differently-anchored parent could license
    this leaf.

    ``generation`` is deliberately *not* compared. It counts an attachment's
    retries, not a position in the chain, so requiring it to increase refused
    a valid optimize-then-reconstruct pair whose two distinct attachments were
    both on their first attempt.

    For ``launch`` the objective set self-identifies by its own digest, which
    establishes only internal consistency; ``authority_digest`` must therefore
    equal the controller's selected ``chain.spec_digest``, so a
    caller-fabricated set is not accepted as launch authorization.
    """

    if inherited is None:
        if predecessor_workload_report is not None:
            raise ConfigError(
                "a predecessor workload report was supplied with no inherited objective set to "
                "check this leaf against"
            )
        if chain.predecessor_workload_report_digest is not None:
            raise ConfigError(
                "the chain names a predecessor workload report but supplies neither the report "
                "nor the objectives it established"
            )
        return ()
    if inherited.source == "predecessor_workload":
        if predecessor_workload_report is None:
            raise ConfigError(
                "inherited objectives cite the predecessor's workload report, which was not "
                "supplied"
            )
        if type(predecessor_workload_report) is not ForwardWorkloadReport:
            raise ConfigError("the predecessor workload report has the wrong type")
        parent = predecessor_workload_report
        if not parent.accepted:
            raise ConfigError("the predecessor's workload report was not accepted")
        if parent.digest() != inherited.receipt_digest:
            raise ConfigError(
                "the inherited objective receipt does not identify the supplied predecessor "
                "workload report"
            )
        if chain.predecessor_workload_report_digest != inherited.receipt_digest:
            raise ConfigError(
                "the chain identity names a different predecessor workload report than the "
                "inherited objectives do"
            )
        if parent.objectives != inherited.objectives:
            raise ConfigError(
                "the inherited objective set is not the one the predecessor's workload report "
                "records"
            )
        subjects = {item.role: item for item in parent.subjects}
        parent_candidate = subjects.get("candidate")
        if parent_candidate is None or parent_candidate.bundle_digest != predecessor.digest:
            raise ConfigError(
                "the predecessor's workload report certifies a different candidate than this "
                "launch's predecessor bundle"
            )
        if parent_candidate.ordinary_report_digest != predecessor_report.digest():
            raise ConfigError(
                "the predecessor's workload report was taken against a different ordinary "
                "verification report for that bundle"
            )
        if parent.chain.chain_id != chain.chain_id:
            raise ConfigError("the predecessor's workload report belongs to a different chain")
        if parent.chain.attachment == chain.attachment:
            # The edge is what makes a parent a parent, and the edge is the
            # attachment: this node's predecessor bundle is that node's
            # candidate, and the two nodes are distinct attachments in one
            # chain. `generation` is an attachment's *retry* counter, not a
            # position in the chain, so a valid optimize -> reconstruct pair
            # whose distinct attachments are both on their first attempt is a
            # real chain and was wrongly refused by comparing generations.
            raise ConfigError(
                "the predecessor's workload report is this same attachment; a node cannot "
                "inherit its own objectives"
            )
        if parent_candidate.bundle_digest == candidate_digest:
            raise ConfigError(
                "the predecessor's workload report certifies this leaf's own candidate"
            )
        if inherited.authority_digest != parent.chain.spec_digest:
            raise ConfigError(
                "the inherited authority is not the spec the predecessor's workload report was "
                "selected under"
            )
        if (
            parent.workload != declaration.workload
            or parent.policy_digest != declaration.policy_digest
            or parent.tape_digest != declaration.fixtures.tape_digest
            or parent.anchor != anchor
        ):
            raise ConfigError(
                "the predecessor's workload report was taken under a different workload, "
                "policy, tape or original anchor"
            )
    else:
        if inherited.receipt_digest != inherited.objectives.digest():
            raise ConfigError(
                "a launch-authorized objective set must be identified by its own digest"
            )
        if inherited.authority_digest != chain.spec_digest:
            raise ConfigError(
                "a launch-authorized objective set must be bound to the controller's selected "
                "spec identity; a self-consistent object is not authorization"
            )
    return tuple(
        f"chain.dropped_requirement.{name}"
        for name in declaration.objectives.shortfalls_against(inherited.objectives)
    )


def _resolve_verifier(
    declaration: WorkloadDeclaration, registry: Registry
) -> tuple[Verifier, PluginOrigin]:
    """Resolve the declared workload verifier from installed registrations."""

    name = declaration.verifier.plugin
    origin = registry.origin("verifier", name)
    if origin.source == "distribution":
        if (
            origin.distribution_name != declaration.verifier.distribution_name
            or origin.distribution_version != declaration.verifier.distribution_version
        ):
            raise ConfigError(
                f"installed workload verifier {name!r} is "
                f"{origin.distribution_name} {origin.distribution_version}, not the declared "
                f"{declaration.verifier.distribution_name} "
                f"{declaration.verifier.distribution_version}"
            )
    elif not declaration.acceptance.permit_local_plugin_origin:
        raise ConfigError(
            f"workload verifier {name!r} has an unverified local registration and the "
            "declaration does not permit one"
        )
    verifier = registry.get("verifier", name)()
    if not isinstance(verifier, Verifier):
        raise ConfigError(f"workload verifier {name!r} does not implement the verifier contract")
    if getattr(verifier, "name", None) != name:
        raise ConfigError(f"workload verifier {name!r} reports the identity {verifier.name!r}")
    return verifier, origin


def _resolve_executor(bundle: CandidateBundle, registry: Registry) -> Executor:
    stages = [stage for stage in bundle.verification_plan.stages if stage.kind == "executor"]
    if len(stages) != 1:
        raise ConfigError("a forward workload needs the bundle's one frozen executor stage")
    options = _json_value(stages[0].config, "workload executor config")
    if not isinstance(options, dict):
        raise ConfigError("workload executor stage configuration is invalid")
    executor = registry.get("executor", stages[0].plugin)(**options)
    if not isinstance(executor, Executor):
        raise ConfigError("workload executor does not implement the executor contract")
    return executor


def _owner_unit(bundle: CandidateBundle, declaration: WorkloadDeclaration) -> CandidateUnit:
    owner = declaration.interface.owner_unit
    matches = [item for item in bundle.units if item.unit.uid == owner]
    if len(matches) != 1:
        raise ConfigError(
            f"the declared workload entry owner {owner!r} does not select exactly one selected unit"
        )
    item = matches[0]
    if item.transform_status != "ok" or item.candidate is None or item.facts is None:
        raise ConfigError(f"workload entry owner {owner!r} has no completed translation candidate")
    exports = item.facts.interface.get("exports", ())
    if not isinstance(exports, (list, tuple)):
        raise ConfigError(f"workload entry owner {owner!r} has an invalid export declaration")
    return item


# --- the adjunct -------------------------------------------------------------


def verify_bundle_forward_objectives(
    root: Path,
    bundle: CandidateBundle,
    workload_manifest: Mapping[str, object],
    *,
    observations: Path,
    bundle_report: VerificationReport,
    original_translation_bundle: CandidateBundle,
    original_translation_report: VerificationReport,
    predecessor_bundle: CandidateBundle,
    predecessor_report: VerificationReport,
    original_anchor: AnchorReference,
    chain: ChainIdentity,
    timing_baseline: TimingBaselineReference,
    expected_source_artifact_digest: str,
    expected_engine: EngineBinding | None,
    expected_workload_manifest_digest: str,
    boundary_receipt: TransformationBoundaryReceipt | RecordedSourceBoundaryReceipt | None = None,
    recorded_source_evidence: RecordedSourceEvidence | None = None,
    boundary_links: tuple[BoundaryLink, ...] = (),
    inherited_objectives: InheritedObjectives | None = None,
    predecessor_workload_report: ForwardWorkloadReport | None = None,
    config: Mapping[str, object] | None = None,
    adapter_configuration: Mapping[str, object] | None = None,
    registry: Registry = REGISTRY,
    workspace: Path | None = None,
) -> ForwardWorkloadReport:
    """Verify one candidate's complete forward workload against its chain.

    Raises :class:`~recast.errors.ConfigError` when the request itself cannot
    be trusted: an unauthorized declaration or observation fixture, installed
    verifier or adapter bytes that are not the authorized implementation, a
    plugin whose installed origin differs from the declaration, a subject
    built from another source or engine, an ordinary report that is
    unaccepted or does not cover this bundle's units and gates, a stale
    predecessor report, an inherited objective set no authorized upstream
    established, an anchor with no chaining transformation-boundary receipt, a
    package path collision, a missing declared helper or frozen export, an
    undeclared added file, a candidate carrying patches, or a pinned timing
    baseline that is not the package which ran. Those are contradictions, not
    measurements, and there is no honest report to write about them.

    Returns an unaccepted report, with reason codes, when the workload did
    run and its evidence falls short: a failed or missing required
    comparison, incomplete observation coverage, an unsupported or
    unclassified required derivative mode, a missing timing phase, a warm
    sample that still contains compilation, too few warm repetitions, an
    unmet speed objective, a measurement class the declaration does not
    permit, or an inherited requirement the leaf tried to drop.

    ``adapter_configuration`` is the one bound channel for a domain adapter's
    configuration: which of its APIs is being called, and where its trusted
    external authorities live. It is a
    ``recast.forward-workload-adapter-config.v1`` document of opaque scalars
    plus named filesystem sources with declared digests. Its own digest must
    equal ``adapter.configuration_digest`` in the declaration, and every
    source is resolved and hashed before dispatch and again afterwards. The
    engine never interprets a value key or a source name -- those are the
    extension's vocabulary. ``config`` is a different thing and is not this
    channel: it is the semantic configuration that participates in bundle
    identity, and it is deliberately not forwarded to a plugin.

    ``observations`` is the read-only fixture holding the tape and the
    numerical policy, each authorized by its own digest in the declaration.
    It is an engine argument rather than semantic config precisely so that
    bundle identity stays machine-independent and no candidate can name it.

    ``timing_baseline`` names which staged subject the speed objectives are
    measured against, and keeps that identity separate from
    ``original_anchor``: the scientific authority and the original JAX timing
    baseline are different things and are never conflated here.

    ``inherited_objectives`` may only come from the frozen launch objective set
    or the predecessor's accepted *workload* report -- an ordinary report
    cannot establish a full-workload objective, so the two report digests are
    separate fields on :class:`ChainIdentity`.

    ``verify_recipe_candidates`` remains required and separate. This adjunct
    reads its report as permission to proceed and adds an objective; it never
    stands in for a unit gate that did not run. Current-generation, ancestry
    and retry/stop fences stay with the controller: these value objects cannot
    say whether a live attachment was superseded.
    """

    declaration = validate_workload_declaration(workload_manifest)
    if declaration.digest != _require_digest(
        expected_workload_manifest_digest, "expected_workload_manifest_digest"
    ):
        raise ConfigError("the supplied workload declaration is not the one this launch authorized")
    fixture = _require_fixture(observations, declaration)
    if type(timing_baseline) is not TimingBaselineReference:
        raise ConfigError("a forward workload needs a declared timing-baseline reference")

    candidate = _decode_subject("candidate", bundle)
    original = _decode_subject("original_translation", original_translation_bundle)
    predecessor = _decode_subject("predecessor", predecessor_bundle)
    subjects = (candidate, original, predecessor)
    fenced: dict[str, Any] | None = None

    semantic = _semantic_config(config or {})
    adapter_config, adapter_sources = _require_adapter_configuration(
        declaration, adapter_configuration
    )
    expected_source = _require_digest(
        expected_source_artifact_digest, "expected_source_artifact_digest"
    )
    try:
        checks, installed_engine = _composition_checks(
            semantic, candidate.saved, expected_source, expected_engine, registry
        )
        if not all(check.passed for check in checks):
            failed = ", ".join(check.name for check in checks if not check.passed)
            raise ConfigError(f"workload candidate source/config/engine binding differs: {failed}")
        _require_same_chain(candidate, original)
        _require_same_chain(candidate, predecessor)
        owner_unit = declaration.interface.owner_unit
        reports = {
            "candidate": bundle_report,
            "original_translation": original_translation_report,
            "predecessor": predecessor_report,
        }
        _require_bound_report(
            "candidate",
            bundle_report,
            candidate,
            expected_digest=None,
            owner_unit=owner_unit,
            engine=installed_engine,
        )
        _require_bound_report(
            "original_translation",
            original_translation_report,
            original,
            expected_digest=None,
            owner_unit=owner_unit,
            engine=installed_engine,
        )
        _require_bound_report(
            "predecessor",
            predecessor_report,
            predecessor,
            expected_digest=chain.predecessor_report_digest,
            owner_unit=owner_unit,
            engine=installed_engine,
        )
        _require_anchor(
            original_anchor,
            boundary_receipt,
            candidate,
            declaration,
            tuple(boundary_links),
            recorded_source_evidence=recorded_source_evidence,
            source_root=Path(root),
        )
        chain_reasons = _require_inherited(
            declaration,
            inherited_objectives,
            chain,
            predecessor,
            predecessor_report,
            candidate.digest,
            original_anchor,
            predecessor_workload_report,
        )
        # Canonical bytes taken *after* validation and used for the digests the
        # report binds, so a report mutated during dispatch cannot be reported
        # under the identity it had when it was checked.
        report_snapshots = {role: item.to_json() for role, item in reports.items()}
        report_digests = {
            role: _digest_bytes(payload) for role, payload in report_snapshots.items()
        }
        owner = _owner_unit(candidate.saved, declaration)
        verifier, origin = _resolve_verifier(declaration, registry)
        verifier_bytes = _require_implementation(
            f"workload verifier {declaration.verifier.plugin!r}",
            declaration.verifier.implementation_modules,
            declaration.verifier.implementation_digest,
        )
        adapter_bytes = _require_implementation(
            "workload adapter",
            declaration.adapter.implementation_modules,
            declaration.adapter.digest,
        )
        executor = _resolve_executor(candidate.saved, registry)

        resolved_root = Path(root).resolve()
        target_workspace = (
            Path(workspace)
            if workspace is not None
            else resolved_root / ".recast" / "forward-workload"
        )
        target_workspace.mkdir(parents=True, exist_ok=True)
        job_root = Path(tempfile.mkdtemp(prefix="forward-workload-", dir=target_workspace))
        packages = {
            subject.role: _assemble_package(
                subject.role, subject.saved, declaration, job_root / "packages" / subject.role
            )
            for subject in subjects
        }
        _require_declared_package_edits(
            packages["candidate"], packages["original_translation"], declaration
        )
        if predecessor_workload_report is not None:
            parent_candidate = next(
                (item for item in predecessor_workload_report.subjects if item.role == "candidate"),
                None,
            )
            if (
                parent_candidate is None
                or parent_candidate.package_digest != packages["predecessor"].package_digest
            ):
                # Binding the parent's bundle digest is not enough: a report
                # naming a foreign *package* for the right bundle was accepted.
                raise ConfigError(
                    "the predecessor's workload report names a different assembled package "
                    "than the predecessor bundle stages"
                )
        pinned = timing_baseline.frozen_package_digest
        if pinned is not None and packages[timing_baseline.role].package_digest != pinned:
            raise ConfigError(
                f"the declared timing baseline {timing_baseline.label!r} pins package {pinned}, "
                f"but the staged {timing_baseline.role} package is "
                f"{packages[timing_baseline.role].package_digest}"
            )

        declaration_snapshot = _canonical_bytes(_json_value(declaration.document, "declaration"))
        staged = {role: staged_inventory(item) for role, item in packages.items()}
        for role, observed in staged.items():
            if observed != packages[role].package_digest:
                raise PluginError(
                    f"staged {role} package differs from the bundle it was staged from"
                )
        anchor_module = declaration.interface.anchor_entry_point.rsplit(".", 1)[0]
        fenced = {
            "packages": dict(packages),
            "staged": staged,
            "observations": observations,
            "declaration": declaration,
            "reports": report_snapshots,
            "root": resolved_root,
            "anchor_module": anchor_module,
            "source": source_inventory(resolved_root, candidate.saved, anchor_module),
            "adapter_configuration": adapter_config,
            "adapter_sources": adapter_sources,
        }
        verdict = _dispatch(
            verifier=verifier,
            owner=owner,
            declaration=declaration,
            anchor=original_anchor,
            packages=packages,
            source_root=resolved_root,
            fixture=fixture,
            job_root=job_root,
            executor=executor,
            adapter_configuration=adapter_config,
        )
        if _canonical_bytes(_json_value(declaration.document, "declaration")) != (
            declaration_snapshot
        ):
            raise PluginError("the workload verifier mutated the trusted declaration")
        result, protocol_reasons = _read_result(verdict, declaration, packages)
        return _conclude(
            declaration=declaration,
            chain=chain,
            anchor=original_anchor,
            boundary=boundary_receipt,
            links=tuple(boundary_links),
            inherited=inherited_objectives,
            timing_baseline=timing_baseline,
            bundle_report=bundle_report,
            reports=reports,
            subjects=subjects,
            packages=packages,
            origin=origin,
            report_digests=report_digests,
            verifier_bytes=verifier_bytes,
            adapter_bytes=adapter_bytes,
            bindings=tuple(checks),
            verdict=verdict,
            result=result,
            protocol_reasons=protocol_reasons + chain_reasons,
        )
    finally:
        for subject in subjects:
            _require_bundle_unchanged(
                subject.bundle, subject.saved, subject.snapshot, f"workload {subject.role}"
            )
        # Everything the acceptance rests on is rechecked here, including on a
        # failing dispatch: the staged packages that ran, the read-only
        # fixture that authorized the policy and tape, the installed bytes
        # that were dispatched to, and the ordinary reports whose digests the
        # result binds. Checking only the bundle objects left every one of
        # these mutable across the dispatch boundary.
        if fenced is not None:
            if recorded_source_evidence is not None:
                require_recorded_source_files(fenced["root"], recorded_source_evidence)
            for role, item in fenced["packages"].items():
                if staged_inventory(item) != fenced["staged"][role]:
                    raise PluginError(f"the staged {role} package changed during verification")
            _require_fixture(fenced["observations"], fenced["declaration"])
            _require_implementation(
                f"workload verifier {fenced['declaration'].verifier.plugin!r}",
                fenced["declaration"].verifier.implementation_modules,
                fenced["declaration"].verifier.implementation_digest,
            )
            _require_implementation(
                "workload adapter",
                fenced["declaration"].adapter.implementation_modules,
                fenced["declaration"].adapter.digest,
            )
            if (
                source_inventory(fenced["root"], candidate.saved, fenced["anchor_module"])
                != fenced["source"]
            ):
                raise PluginError(
                    "the project source the original anchor is executed from changed during "
                    "verification"
                )
            configuration = fenced["adapter_configuration"]
            if configuration is not None:
                for name, expected in fenced["adapter_sources"].items():
                    source = configuration.sources[name]
                    if adapter_source_digest(source.kind, Path(source.path)) != expected:
                        raise PluginError(f"adapter source {name!r} changed during verification")
            for role, payload in fenced["reports"].items():
                if reports[role].to_json() != payload:
                    raise PluginError(
                        f"the {role} ordinary verification report changed during verification"
                    )


def _dispatch(
    *,
    verifier: Verifier,
    owner: CandidateUnit,
    declaration: WorkloadDeclaration,
    anchor: AnchorReference,
    packages: Mapping[str, AssembledPackage],
    source_root: Path,
    fixture: Path,
    job_root: Path,
    executor: Executor,
    adapter_configuration: AdapterConfiguration | None = None,
) -> Verdict:
    """Call the trusted plugin through the existing Verifier contract.

    The plugin receives the root Candidate of the declared owner unit plus
    trusted descriptors for every assembled package. It does not receive, and
    cannot ask for, an import root or a policy the candidate chose.
    """

    candidate = owner.candidate
    if candidate is None:  # _owner_unit already refused this.
        raise ConfigError("workload entry owner has no candidate")
    anchor_root = (
        str(source_root)
        if anchor.kind == "source-artifact" and anchor.source_artifact_digest is not None
        else None
    )
    handle = _freeze_json(
        {
            "anchor": anchor.to_dict(),
            "root": anchor_root,
            "fixture": str(fixture),
            "tape_digest": declaration.fixtures.tape_digest,
            "policy_digest": declaration.policy_digest,
        },
        "workload anchor handle",
    )
    oracle = OracleRef(
        unit=owner.unit.uid,
        oracle=f"forward-workload:{declaration.workload}",
        key=f"{declaration.digest}:{anchor.digest()}",
        handle=handle,
        cost="batch",
    )
    stage_config: dict[str, Any] = {
        "workload": _json_value(declaration.document, "workload declaration"),
        "workload_manifest_digest": declaration.digest,
        "packages": {role: item.descriptor() for role, item in sorted(packages.items())},
        "workspace": str(job_root / "run"),
        # The one bound channel for domain adapter configuration. It is
        # frozen, its digest is the one the declaration authorized, and every
        # named source has already been resolved against its declared digest.
        # ``None`` when the declaration authorizes none.
        "adapter_configuration": (
            None
            if adapter_configuration is None
            else _freeze_json(adapter_configuration.to_dict(), "workload adapter configuration")
        ),
    }
    try:
        verdict = verifier.verify(
            owner.unit,
            candidate,
            oracle,
            job_root / "run",
            executor,
            stage_config,
        )
    except Exception as error:
        raise PluginError(
            f"workload verifier {verifier.name!r} raised instead of returning a verdict: {error}"
        ) from error
    if type(verdict) is not Verdict:
        raise PluginError(f"workload verifier {verifier.name!r} did not return a Verdict")
    if verdict.unit != owner.unit.uid:
        raise PluginError(
            f"workload verifier {verifier.name!r} judged unit {verdict.unit!r}, "
            f"expected {owner.unit.uid!r}"
        )
    if verdict.verifier != verifier.name:
        raise PluginError(
            f"workload verifier {verifier.name!r} returned identity {verdict.verifier!r}"
        )
    if verdict.candidate != candidate.digest():
        raise PluginError(
            f"workload verifier {verifier.name!r} judged candidate {verdict.candidate!r}, "
            f"expected the selected transform product"
        )
    return verdict


def _read_result(
    verdict: Verdict,
    declaration: WorkloadDeclaration,
    packages: Mapping[str, AssembledPackage],
) -> tuple[WorkloadResult | None, tuple[str, ...]]:
    """Validate the plugin's measurements, or say why they cannot be read."""

    document = verdict.metrics.get(_RESULT_METRIC)
    if document is None:
        return None, ("plugin.result_missing",)
    result = validate_workload_result(document, declaration)
    wrong = sorted(
        role
        for role in SUBJECT_ROLES
        if result.package_digests[role] != packages[role].package_digest
    )
    if wrong:
        raise PluginError(
            f"workload verifier reported package digest(s) for {wrong} that differ from the "
            "packages this adjunct staged"
        )
    return result, ()


def _conclude(
    *,
    declaration: WorkloadDeclaration,
    chain: ChainIdentity,
    anchor: AnchorReference,
    boundary: TransformationBoundaryReceipt | RecordedSourceBoundaryReceipt | None,
    links: tuple[BoundaryLink, ...],
    inherited: InheritedObjectives | None,
    timing_baseline: TimingBaselineReference,
    bundle_report: VerificationReport,
    reports: Mapping[str, VerificationReport],
    report_digests: Mapping[str, str],
    subjects: tuple[_Subject, ...],
    packages: Mapping[str, AssembledPackage],
    origin: PluginOrigin,
    verifier_bytes: tuple[str, ...],
    adapter_bytes: tuple[str, ...],
    bindings: tuple[BindingCheck, ...],
    verdict: Verdict,
    result: WorkloadResult | None,
    protocol_reasons: tuple[str, ...],
) -> ForwardWorkloadReport:
    """Recompute every conclusion from the validated measurements.

    See :class:`ForwardWorkloadReport` for the acceptance truth table this
    implements. Nothing here reads a conclusion the plugin reached.
    """

    reasons = list(protocol_reasons)
    if not verdict.passed:
        reasons.append("plugin.failed")

    coverage: tuple[CoverageResult, ...] = ()
    comparisons: tuple[ComparisonObservation, ...] = ()
    derivatives: tuple[DerivativeObservation, ...] = ()
    timings: tuple[TimingObservation, ...] = ()
    speed: tuple[SpeedConclusion, ...] = ()
    unsupported: tuple[str, ...] = ()
    evidence: tuple[EvidenceSubject, ...] = ()
    measurement_class = "missing"
    numerical_status = "failed"
    derivative_status = "not_requested" if not declaration.acceptance.required_modes else "failed"
    performance_status = "not_requested" if not declaration.acceptance.speed else "failed"
    metrics = ForwardMetrics(
        steps=declaration.fixtures.steps,
        states=len(declaration.fixtures.states),
        compared_outputs=None,
        max_ulp=None,
        max_relative_error=None,
        anchor_status="missing",
        predecessor_status="missing",
        warm_median_seconds=None,
        compile_seconds=None,
        whole_workload_seconds=None,
        speedup_against_original_translation=None,
        speedup_against_predecessor=None,
        required_modes_met=0,
        required_modes_total=len(declaration.acceptance.required_modes),
        evidence_subjects=0,
    )

    if result is not None:
        measurement_class = result.measurement_class
        if measurement_class not in declaration.acceptance.permitted_measurement_classes:
            reasons.append(f"measurement.class_not_permitted.{measurement_class}")
        coverage = evaluate_coverage(declaration, result)
        reasons.extend(f"coverage.{item.dimension}" for item in coverage if not item.accepted)
        numerical, numerical_reasons = evaluate_numerics(declaration, result)
        reasons.extend(numerical_reasons)
        numerical_status = "passed" if numerical else "failed"
        if declaration.acceptance.required_modes:
            derivative, derivative_reasons = evaluate_derivatives(declaration, result)
            reasons.extend(derivative_reasons)
            derivative_status = "passed" if derivative else "failed"
        timing_reasons = evaluate_timing_requirements(declaration, result)
        reasons.extend(timing_reasons)
        speed = evaluate_speed(declaration, result)
        reasons.extend(f"speed.{item.key}.{item.status}" for item in speed if item.status != "met")
        if declaration.acceptance.speed:
            performance_status = (
                "passed"
                if all(item.status == "met" for item in speed) and not timing_reasons
                else "failed"
            )
        elif timing_reasons:
            # No speed objective was requested, but a declared timing phase
            # that did not arrive is still a shortfall of this report.
            performance_status = "not_requested"
        comparisons = result.comparisons
        derivatives = result.derivatives
        timings = result.timings
        evidence = result.evidence
        unsupported = result.unsupported
        metrics = project_metrics(declaration, result, speed)

    statuses = (numerical_status, derivative_status, performance_status)
    accepted = (
        not reasons
        and verdict.passed
        and all(status in {"passed", "not_requested"} for status in statuses)
        and bool(coverage)
        and all(item.accepted for item in coverage)
    )
    production_admissible = (
        accepted
        and measurement_class == "executed"
        and origin.source == "distribution"
        and origin.verification == "distribution_metadata"
        # A plugin's typed self-exclusion. A domain adapter knows things
        # this layer cannot -- whether the runtime a measurement happened in
        # was the authorized one, for instance -- and needs a way to say
        # "this is a real, accepted result and it is not production
        # evidence". An ordinary `unsupported` note could not do that: the
        # decoder retains those labels and coverage only rejects the ones
        # that intersect a required case, signature, mode or output name, so
        # the note was retained and changed nothing. This prefix is
        # reserved, checked here, and carries no domain meaning of its own.
        and not any(item.startswith(NONPRODUCTION_PREFIX) for item in unsupported)
    )
    by_role = {subject.role: subject for subject in subjects}
    return ForwardWorkloadReport(
        workload=declaration.workload,
        workload_version=declaration.version,
        workload_manifest_digest=declaration.digest,
        policy_digest=declaration.policy_digest,
        adapter_digest=declaration.adapter.digest,
        package_contract_digest=declaration.package_contract.digest,
        tape_digest=declaration.fixtures.tape_digest,
        verifier_plugin=declaration.verifier.plugin,
        verifier_origin=MappingProxyType(dict(origin.as_dict())),
        verifier_implementation=verifier_bytes,
        adapter_implementation=adapter_bytes,
        bundle_report_digest=report_digests["candidate"],
        objectives=declaration.objectives,
        inherited=inherited,
        timing_baseline=timing_baseline,
        subjects=tuple(
            WorkloadSubject(
                role=role,
                bundle_digest=by_role[role].digest,
                ordinary_report_digest=report_digests[role],
                recipe=by_role[role].saved.recipe,
                engine=by_role[role].saved.engine,
                source_artifact_digest=by_role[role].saved.source_artifact_digest,
                semantic_config_digest=by_role[role].saved.semantic_config_digest,
                package_digest=packages[role].package_digest,
                entry_module=packages[role].entry_module,
                file_count=len(packages[role].paths),
                unit_count=packages[role].unit_count,
                unique_bytes=packages[role].unique_bytes,
                materialized_bytes=packages[role].materialized_bytes,
            )
            for role in SUBJECT_ROLES
        ),
        anchor=anchor,
        composition=boundary,
        boundary_links=links,
        chain=chain,
        bindings=bindings,
        coverage=coverage,
        comparisons=comparisons,
        derivatives=derivatives,
        timings=timings,
        speed=speed,
        metrics=metrics,
        evidence=evidence,
        measurement_class=measurement_class,
        numerical_status=numerical_status,
        derivative_status=derivative_status,
        performance_status=performance_status,
        production_admissible=production_admissible,
        accepted=accepted,
        reason_codes=tuple(sorted(set(reasons))),
        unsupported=unsupported,
    )
