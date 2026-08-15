"""Core vocabulary shared by every RecastEngine workload.

These types are deliberately small and language-neutral. Everything that is
specific to Fortran, to CESM, or to a particular accelerator lives in a plugin.

The spine every workload walks:

    discover -> analyze -> transform -> verify -> record
      Unit       Facts     Candidate   Verdict   Evidence

Cyber testing shares the first half of that spine but produces Findings rather
than Verdicts, and Findings carry an embargo. See ``Access``.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class Confidence(StrEnum):
    """How strong a Verdict is. Ordered weakest to strongest."""

    FAILED = "failed"
    """The candidate is wrong, or the comparison could not be run."""

    SAMPLED = "sampled"
    """Agreed on generated inputs within tolerance. No coverage claim."""

    TOLERANCED = "toleranced"
    """Agreed within a stated rtol/atol on the project's real inputs."""

    ULP_BOUNDED = "ulp_bounded"
    """Agreed with a proven bound on IEEE-754 ULP distance."""

    BIT_EXACT = "bit_exact"
    """Bit-for-bit identical to the oracle. The strongest empirical gate."""

    SYMBOLIC = "symbolic"
    """Proven equivalent in exact arithmetic (SymPy/mpmath/e-graph)."""


@dataclass(frozen=True)
class Unit:
    """An addressable piece of software under modernization.

    A Fortran module, a single subprogram, a CCPP scheme, a physics kernel, or
    a whole coupled component -- the granularity is the plugin's choice. The
    engine only requires that a Unit is stably addressable by ``uid``.
    """

    uid: str
    """Stable identifier, e.g. ``fortran:micro_mg2_0/micro_mg_tend``."""

    kind: str
    """Plugin-defined, e.g. ``module``, ``subprogram``, ``scheme``, ``component``."""

    sources: tuple[Path, ...] = ()
    """Files this Unit is derived from, relative to the project root."""

    parent: str | None = None
    """``uid`` of an enclosing Unit, if any. Lets plugins express hierarchy."""

    attrs: dict[str, Any] = field(default_factory=dict)
    """Frontend-specific metadata. Not interpreted by the engine."""


@dataclass
class Facts:
    """What analysis learned about a Unit, before any transform runs.

    Frontends fill in what they can and leave the rest empty; a Transform
    declares which keys it requires via ``Transform.requires``.
    """

    unit: str
    """``Unit.uid`` these facts describe."""

    interface: dict[str, Any] = field(default_factory=dict)
    """Signatures, argument intent/dims/dtype, module state, results."""

    constants: dict[str, Any] = field(default_factory=dict)
    """Named parameters and literal->symbol maps."""

    callgraph: dict[str, list[str]] = field(default_factory=dict)
    """Caller ``uid`` -> callee names, including unresolved externals."""

    effects: dict[str, Any] = field(default_factory=dict)
    """Read/write sets, aliasing, purity, side channels (I/O, MPI, halts)."""

    provenance: dict[str, Any] = field(default_factory=dict)
    """Upstream revision, license, preprocessor flags used to obtain the source."""

    extra: dict[str, Any] = field(default_factory=dict)
    """Anything else a frontend wants to pass to its own transforms."""


@dataclass
class Patch:
    """A unified-diff edit against an existing file.

    Refactoring workloads express most of their output this way rather than as
    whole new files -- freeCAM's control-plane carve-out is 16 ordered patches
    into upstream Fortran.
    """

    target: Path
    diff: str
    order: int = 0
    """Patches apply in ascending ``order``; ties keep declaration order."""


@dataclass
class Candidate:
    """A proposed modernization of one Unit. Not yet trusted.

    A Candidate is inert data. It is written to a workspace and gated by a
    Verifier before anything merges.
    """

    unit: str
    transform: str
    """Name of the Transform that produced this."""

    files: dict[Path, bytes] = field(default_factory=dict)
    """New or fully-replaced files, keyed by path relative to the workspace."""

    patches: list[Patch] = field(default_factory=list)
    """Edits to existing files."""

    deferred: list[str] = field(default_factory=list)
    """Sites the transform could not handle mechanically -- the agent queue.

    An empty list means the transform was fully deterministic for this Unit.
    """

    notes: dict[str, Any] = field(default_factory=dict)

    def digest(self) -> str:
        """Content hash. Identical inputs must yield an identical digest."""
        h = hashlib.sha256()
        h.update(self.unit.encode())
        h.update(self.transform.encode())
        for path in sorted(self.files):
            h.update(str(path).encode())
            h.update(self.files[path])
        for patch in sorted(self.patches, key=lambda p: (p.order, str(p.target))):
            h.update(str(patch.target).encode())
            h.update(patch.diff.encode())
        return h.hexdigest()


@dataclass
class OracleRef:
    """A materialized reference implementation, ready to be invoked.

    Materializing is usually expensive -- compiling an f2py truth module,
    replaying a captured dump, or standing up a pinned 512-rank baseline run --
    so the engine caches by ``key``.
    """

    unit: str
    oracle: str
    key: str
    """Cache key. Same key must mean same observable behaviour."""

    handle: Any
    """Opaque to the engine. The Verifier that consumes it defines the type."""

    cost: str = "unknown"
    """``cheap`` | ``build`` | ``batch`` -- lets schedulers order work sanely."""


@dataclass
class Verdict:
    """The outcome of comparing a Candidate against an OracleRef."""

    unit: str
    candidate: str
    """``Candidate.digest()`` that was judged."""

    verifier: str
    confidence: Confidence
    metrics: dict[str, Any] = field(default_factory=dict)
    """e.g. ``{"max_ulp": 0, "bit_exact": 512, "total_points": 512}``."""

    detail: str = ""

    @property
    def passed(self) -> bool:
        return self.confidence is not Confidence.FAILED


@dataclass
class Evidence:
    """An immutable, machine-readable record that some gate was actually run.

    Evidence is the product SciRecast actually ships. A Candidate without
    Evidence is a draft, regardless of how good it looks.

    RecastEngine does not define its own evidence format. CC-Test owns
    ``schemas/evidence-manifest.v1.json``; this type is a producer of it, and
    ``to_manifest`` is the only place the two vocabularies meet. If CC-Test
    revises the schema, change the mapping here -- not the engine's model.
    """

    unit: str
    verdict: Verdict
    recipe: str
    executor: str

    artifact: dict[str, Any] = field(default_factory=dict)
    """CC-Test ``artifact``: ``{name, repo, commit, version}`` of what was produced."""

    reference: dict[str, Any] = field(default_factory=dict)
    """CC-Test ``reference``: ``{model, commit_or_tag, provenance}`` -- the oracle."""

    environment: dict[str, Any] = field(default_factory=dict)
    """CC-Test ``environment``: machine, compiler, mpi, python, cuda, jax, numba, modules.

    Under-reporting this is the single most common way a bit-exact claim stops
    being reproducible. Executors are expected to fill it in, not the operator.
    """

    cases: list[dict[str, Any]] = field(default_factory=list)
    """CC-Test ``cases``: the per-case comparison rows behind ``result``."""

    artifacts: dict[str, str] = field(default_factory=dict)
    """Label -> store URI (logs, dumps, PBS job output, plots)."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Engine version, plugin versions, host, timestamp -- injected by the store."""

    def to_manifest(self, *, cc_test: dict[str, Any], timestamp: str) -> dict[str, Any]:
        """Render as a CC-Test ``evidence-manifest.v1`` document.

        ``evidence_class`` is ``complete`` only when the engine drove every step
        itself. Anything reconstructed from logs after the fact must say so.
        """
        return {
            "schema_version": 1,
            "evidence_class": self.meta.get("evidence_class", "complete"),
            "artifact": self.artifact,
            "reference": self.reference,
            "cc_test": cc_test,
            "environment": self.environment,
            "cases": self.cases,
            "result": {
                "verdict": self.verdict.confidence.value,
                "passed": self.verdict.passed,
                "verifier": self.verdict.verifier,
                "metrics": self.verdict.metrics,
                "detail": self.verdict.detail,
            },
            "timestamp": timestamp,
            "notes": json.dumps(
                {
                    "recast": {
                        "unit": self.unit,
                        "recipe": self.recipe,
                        "executor": self.executor,
                        "candidate": self.verdict.candidate,
                        "artifacts": self.artifacts,
                    }
                },
                sort_keys=True,
            ),
        }


class Access(StrEnum):
    """Who may see a record. Enforced by the engine, not by convention."""

    PUBLIC = "public"
    """Publishable. Correctness evidence is normally this."""

    INTERNAL = "internal"
    """Project members. Pre-publication benchmarks, draft manifests."""

    EMBARGOED = "embargoed"
    """Unpatched vulnerability. Sec-Track only.

    The engine refuses to write an ``EMBARGOED`` record into a store whose
    ``max_access`` is lower. This is a code-level check because the failure mode
    -- a 0-day landing in a public CI log -- is not recoverable by editing it
    out afterwards.
    """


class Severity(StrEnum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Disclosure(StrEnum):
    """Coordinated-disclosure lifecycle, mirroring Sec-Track's own states."""

    PLAUSIBLE = "plausible"
    """Reported by a scanner or agent, not yet adversarially verified."""

    CONFIRMED = "confirmed"
    DOWNGRADED = "downgraded"
    """Real defect, but the claimed impact did not hold up."""

    REFUTED = "refuted"
    REPORTED = "reported"
    """Sent to the owner. Embargo clock running."""

    FIXED = "fixed"
    PUBLISHED = "published"
    """Owner-coordinated release done. Only now may access drop to PUBLIC."""


@dataclass
class Finding:
    """A security finding. The cyber half of CC-Test produces these.

    Deliberately not a ``Verdict``: a Verdict is about whether a Candidate
    matches an oracle, a Finding is about a defect that exists regardless of any
    modernization. They flow to different stores with different access rules.
    """

    uid: str
    unit: str
    scanner: str
    title: str

    cwe: str | None = None
    severity: Severity = Severity.INFO
    disclosure: Disclosure = Disclosure.PLAUSIBLE
    access: Access = Access.EMBARGOED
    """Defaults to the safe end. A scanner must opt *down*, never up by omission."""

    location: dict[str, Any] = field(default_factory=dict)
    """``{"path": ..., "line": ..., "symbol": ...}``."""

    exploitability: str = "unknown"
    """e.g. ``dos``, ``integrity``, ``code-execution-self-config``, ``code-execution``."""

    evidence: dict[str, Any] = field(default_factory=dict)
    """Reproducer, sanitizer trace, build flags under which it holds."""

    upstream: str | None = None
    """Set when the defect belongs to a dependency rather than this project."""

    def publishable(self) -> bool:
        """True only when disclosure has actually completed."""
        return self.access is Access.PUBLIC and self.disclosure in (
            Disclosure.PUBLISHED,
            Disclosure.REFUTED,
        )
