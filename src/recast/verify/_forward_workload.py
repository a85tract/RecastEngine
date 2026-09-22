"""Frozen declaration, result and threshold arithmetic for forward workloads.

Separated from :mod:`recast.verify.forward_workload` for the reason
``_python_jax_objectives`` is separate from its adjunct: the schema and the
arithmetic are pure, so they are testable without staging a package or
starting a worker, and the adjunct that binds identities stays readable.

Two boundaries are load-bearing here.

**Trusted declaration, untrusted result.** A ``recast.forward-workload.v1``
declaration is authorized by digest before a launch; the plugin that runs the
workload afterwards supplies *measurements* only. Every threshold, coverage
requirement and required mode is read from the declaration, and every
conclusion is recomputed in this module from validated numbers. A plugin
cannot report that it met an objective.

**Names are data.** The declaration names an entry point, ordered state,
configuration, forcing and output fields, static signatures and differentiable
slots. This module counts, orders and compares those names; it never
interprets one. That is what keeps a domain workload -- CLUBB's 185 feedback
fields and 291 recorded driver inputs -- out of the core.
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType

from recast.errors import ConfigError
from recast.phases import _canonical_bytes, _digest_bytes, _freeze_json, _require_digest
from recast.verify._python_accelerator_protocol import ProtocolError

__all__ = [
    "BOUNDARY_RECEIPT_KINDS",
    "COVERAGE_DIMENSIONS",
    "FIXTURE_SCHEMA",
    "REPORT_SCHEMA",
    "RESULT_SCHEMA",
    "SUBJECT_ROLES",
    "TIMING_PHASES",
    "WORKLOAD_SCHEMA",
    "AdapterBinding",
    "ComparisonObservation",
    "CoverageResult",
    "DerivativeObservation",
    "EvidenceSubject",
    "FiniteDifferencePolicy",
    "ForwardMetrics",
    "ObjectiveSet",
    "SpeedConclusion",
    "SpeedRequirement",
    "TimingObservation",
    "WorkloadDeclaration",
    "WorkloadResult",
    "evaluate_coverage",
    "evaluate_derivatives",
    "evaluate_numerics",
    "evaluate_speed",
    "evaluate_timing_requirements",
    "fixture_section_digest",
    "observation_keys",
    "observation_keys_digest",
    "project_metrics",
    "validate_workload_declaration",
    "validate_workload_result",
]


WORKLOAD_SCHEMA = "recast.forward-workload.v1"
RESULT_SCHEMA = "recast.forward-workload-result.v1"
REPORT_SCHEMA = "recast.forward-workload-report.v1"
FIXTURE_SCHEMA = "recast.forward-workload-fixture.v1"
"""The read-only observation file: one ``policy`` and one ``tape`` section,
each addressed by the digest of its own canonical bytes."""

NONPRODUCTION_PREFIX = "nonproduction."
"""Reserved ``unsupported`` prefix: a plugin's typed self-exclusion.

A result can be real, accepted and still not be production evidence -- a
characterization run taken outside an enforced runtime contract, say. Only
the domain adapter knows that, and an ordinary ``unsupported`` note could
not express it: the decoder retains those labels and coverage only rejects
the ones intersecting a required case, signature, mode or output name, so
the note changed nothing. An entry with this prefix sets
``production_admissible`` to false and does not otherwise affect
acceptance. This layer attaches no domain meaning to what follows the
prefix; it only honours the exclusion.
"""

MAX_FIXTURE_BYTES = 64 * 1024 * 1024
"""A fixture is staged read-only beside the run, not carried in a bundle."""

SUBJECT_ROLES: tuple[str, ...] = ("candidate", "original_translation", "predecessor")
"""Packages this adjunct assembles and stages itself, in report order."""

ANCHOR_ROLE = "original_anchor"
TIMED_ROLES: tuple[str, ...] = (*SUBJECT_ROLES, ANCHOR_ROLE)
COMPARISON_SUBJECTS: tuple[str, ...] = (ANCHOR_ROLE, "original_translation", "predecessor")
COVERAGE_DIMENSIONS: tuple[str, ...] = ("case", "signature", "mode", "output")
"""Marginal dimensions. Required *combinations* are checked separately, by
observation-key digest per comparison: seeing an output once establishes
nothing about seeing it on every signature, state and step."""
ANCHOR_KINDS = frozenset({"source-artifact", "candidate-bundle", "recorded-tape"})
BOUNDARY_RECEIPT_KINDS = frozenset({"cross-engine", "recipe-intermediate"})
MIN_LINK_RECEIPTS = 2
"""A same-recipe intermediate claims two boundaries, so it proves two.

``port-clubb`` emits Fortran-to-NumPy and then NumPy-to-JAX inside one
request. Each link needs its own independently accepted comparison receipt;
one receipt covering "the whole port" is the gap the design named."""
"""How a verified transformation boundary was obtained. ``recipe-intermediate``
is the recommended route: one recipe already emits the intermediate, so an
extracted immutable inventory is the evidence, not a second launch."""
BOUNDARY_MODES = frozenset({"recorded_open_loop", "state_feedback_core_replay"})
MEASUREMENT_CLASSES = frozenset({"executed", "synthetic"})
"""``synthetic`` is a declared mock. It is never silently comparable to
``executed``: a declaration lists which classes it accepts, and the report
carries the class it got, so a fixture result cannot be shown as a real run."""

RESULT_STATUSES = frozenset({"passed", "failed", "missing", "unsupported"})
SPEED_OBJECTIVES = frozenset({"reduction", "regression_ceiling"})
SPEED_STATUSES = frozenset({"met", "unmet", "missing"})
RESET_POLICIES = frozenset({"per_round", "per_state", "never"})

TIMING_PHASES: tuple[str, ...] = (
    "preparation",
    # Host-side setup an adapter performs once, before the measured
    # boundary, on behalf of the package it is measuring: validating and
    # snapshotting inputs, or building the typed records a record-shaped
    # API takes. It is real work and it is not execution, so it is reported
    # apart from both rather than folded into the first execution or left
    # outside every clock. Optional: an adapter with no such boundary
    # reports null.
    "host_setup",
    "transfer_in",
    "trace",
    "lower",
    "compile",
    # A representation whose public dispatch and its ahead-of-time
    # executable are distinct objects has two outer first-call boundaries,
    # and they are not interchangeable: ``.trace()`` need not populate a
    # jitted object's own call cache, so the public first invocation can
    # carry a compilation the AOT path already paid for. Both are optional
    # and both are reported when they exist, so neither can be quietly
    # substituted for the other. ``first_execution`` stays what it was:
    # the first execution of the callable the warm samples time.
    "aot_first_call",
    "public_first_call",
    "first_execution",
    "warm_median",
    "warm_mad",
    "observation",
    "transfer_out",
    "whole_workload",
)
"""Every boundary the report can hold. Absence is ``None``, never zero."""

_NONNEGATIVE_PHASES = frozenset({"warm_mad"})
"""A spread of exactly zero is a real observation; a duration of zero is not.

``warm_mad`` is the *median* absolute deviation of the ordered warm samples,
and a run whose samples happen to be identical has a deviation of zero. A
duration of zero, by contrast, is a broken clock, so the two are validated by
different rules.
"""

EXECUTION_MODES = frozenset({"compiled", "interpreted"})
"""What a subject's timing row is allowed to contain.

A ``compiled`` subject must show tracing, lowering, compilation and a first
synchronized execution; an ``interpreted`` one must show none of them,
positively. That is the difference between *inapplicable* and *unmeasured*:
the declaration says which subject is in which mode, so a JAX package cannot
call its missing compile evidence inapplicable, and a NumPy anchor does not
fail for lacking phases it could never have.
"""

_COMPILED_ONLY_PHASES: tuple[str, ...] = ("trace", "lower", "compile")

# Bounded counts. A declaration is a control-plane message, so every list it
# carries has a ceiling; the ones below are generous for a whole-timestep
# scientific interface (CLUBB's is 476 inputs and 185 outputs) and still
# refuse an unbounded document.
MAX_NAMES = 4096
MAX_NAME_BYTES = 256
MAX_STATES = 64
MAX_STEPS = 1_000_000
MAX_SIGNATURES = 256
MAX_SPEED_REQUIREMENTS = 8
MAX_EVIDENCE_SUBJECTS = 256
MAX_DETAIL_BYTES = 4000
MAX_UNSUPPORTED = 256
MAX_WARM_SAMPLES = 4096
MAX_OBSERVATION_KEYS = 100_000_000
MAX_PACKAGE_FILES = 4096
MAX_PACKAGE_FILE_BYTES = 64 * 1024 * 1024
MAX_PACKAGE_TOTAL_BYTES = 512 * 1024 * 1024

_NAME = re.compile(r"^[^\s\x00][^\x00]*$")
_TOKEN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+:@-]{0,255}$")
_DOTTED = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$")
_ULP = re.compile(r"^(?:0|[1-9][0-9]{0,63})$")
_MEDIA_TYPE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}/[A-Za-z0-9][A-Za-z0-9!#$&^_.+-]{0,126}$"
)


def fixture_section_digest(section: object) -> str:
    """Digest one section of a ``recast.forward-workload-fixture.v1`` document.

    A forward workload needs two things the code bundle must not carry: the
    observation tape and the numerical policy. Both live in one read-only
    fixture beside the run, and both are addressed by the digest of their own
    canonical section, so the declaration authorizes each independently and
    the engine can check them without parsing either. The controller and the
    isolated worker call this same function, which is why it lives here.
    """

    return _digest_bytes(_canonical_bytes(section))


# --- strict primitives -------------------------------------------------------


def _mapping(
    value: object,
    context: str,
    keys: frozenset[str],
    *,
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, object]:
    """Exactly the declared keys, with named additions allowed to be absent.

    ``optional`` exists only so a field can be *added* to a schema without
    invalidating every already-frozen declaration that predates it. An
    optional key that is present is validated exactly like a required one; an
    unknown key is still refused.
    """

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{context} must be a string-keyed object")
    present = set(value)
    if present != keys:
        missing = sorted(keys - present - optional)
        unknown = sorted(present - keys)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if unknown:
            detail.append(f"unknown {unknown}")
        if detail:
            raise ConfigError(f"{context} fields are invalid: {'; '.join(detail)}")
    return value


def _text(value: object, context: str, *, pattern: re.Pattern[str] = _NAME) -> str:
    if not isinstance(value, str) or len(value.encode()) > MAX_NAME_BYTES:
        raise ConfigError(f"{context} must be a bounded string")
    if not pattern.fullmatch(value):
        raise ConfigError(f"{context} is not a well-formed name")
    return value


def _names(
    value: object,
    context: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_NAMES,
    pattern: re.Pattern[str] = _NAME,
) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigError(f"{context} must be an array of names")
    if not (minimum <= len(value) <= maximum):
        raise ConfigError(f"{context} must hold between {minimum} and {maximum} names")
    names = tuple(_text(item, f"{context}[]", pattern=pattern) for item in value)
    if len(set(names)) != len(names):
        raise ConfigError(f"{context} repeats a name")
    return names


def _integer(value: object, context: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int or not (minimum <= value <= maximum):
        raise ConfigError(f"{context} must be an integer in [{minimum}, {maximum}]")
    return value


def _flag(value: object, context: str) -> bool:
    if type(value) is not bool:
        raise ConfigError(f"{context} must be a boolean")
    return bool(value)


def _fraction(value: object, context: str, *, minimum: float, maximum: float) -> float:
    if type(value) is not float or not math.isfinite(value) or not (minimum <= value <= maximum):
        raise ConfigError(f"{context} must be a finite float in [{minimum}, {maximum}]")
    return value


def _subset(names: tuple[str, ...], universe: tuple[str, ...], context: str) -> tuple[str, ...]:
    unknown = sorted(set(names) - set(universe))
    if unknown:
        raise ConfigError(f"{context} names {unknown} which the interface does not declare")
    return names


# --- the trusted declaration -------------------------------------------------


@dataclass(frozen=True)
class VerifierBinding:
    """The exact installed plugin a launch authorized to run this workload.

    Distribution name and version are *attribution*: ``PluginOrigin`` reads
    them out of installed metadata and says so, and metadata is not a
    statement about code. ``implementation_modules`` is what makes this an
    identity check -- the adjunct resolves each named module to a file and
    hashes its bytes, and ``implementation_digest`` is the digest of that
    inventory. The launch computes and authorizes it; the adjunct observes
    the installed bytes and compares before dispatching anything.
    """

    plugin: str
    distribution_name: str
    distribution_version: str
    implementation_digest: str
    implementation_modules: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "plugin": self.plugin,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "implementation_digest": self.implementation_digest,
            "implementation_modules": list(self.implementation_modules),
        }


@dataclass(frozen=True)
class AdapterBinding:
    """The trusted flat-ABI/PyTree adapter, bound the same way."""

    digest: str
    entry_point: str
    implementation_modules: tuple[str, ...]
    #: The digest of the adapter configuration this declaration authorizes,
    #: or ``None`` when the adapter takes none. A declaration that names one
    #: will not dispatch without a matching supplied configuration, and a
    #: declaration that names none will not dispatch with one.
    configuration_digest: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "digest": self.digest,
            "entry_point": self.entry_point,
            "implementation_modules": list(self.implementation_modules),
            "configuration_digest": self.configuration_digest,
        }


ADAPTER_CONFIG_SCHEMA = "recast.forward-workload-adapter-config.v1"

_ADAPTER_CONFIG_DOMAIN = b"recast.forward-workload-adapter-config.v1\0"

MAX_ADAPTER_CONFIG_VALUES = 64
MAX_ADAPTER_CONFIG_SOURCES = 16
ADAPTER_SOURCE_KINDS = frozenset({"file", "tree"})


@dataclass(frozen=True)
class AdapterSource:
    """One named filesystem authority the adapter is allowed to read.

    ``kind`` is ``file`` for a single regular file, or ``tree`` for a
    directory whose whole content is digested. The engine resolves and hashes
    it and compares the result with ``digest`` before dispatch and again
    afterwards; it never looks at what the name means.
    """

    name: str
    kind: str
    path: str
    digest: str

    def to_dict(self) -> dict[str, object]:
        return {"name": self.name, "kind": self.kind, "path": self.path, "digest": self.digest}


@dataclass(frozen=True)
class AdapterConfiguration:
    """The trusted, immutable configuration a domain adapter may be given.

    A domain workload needs a few things the declaration cannot express in
    generic terms: which of its APIs is being called, and where its trusted
    external authorities live. Forwarding an arbitrary caller mapping to the
    plugin would make that an unbound channel -- anything could arrive, and
    nothing about it would be authorized. Forwarding nothing, as the previous
    revision did, silently drops the documented call's configuration and the
    real plugin then refuses.

    So the channel exists but is narrow and bound:

    * ``values`` are opaque scalars. No nesting, no paths, and the engine
      never interprets a key.
    * ``sources`` are named filesystem authorities with declared digests,
      which the engine resolves and verifies. A swapped anchor source or an
      edited observation tree fails here rather than becoming an expected
      value.
    * the whole object's digest must equal the one the declaration
      authorizes, so a launch freezes it along with everything else.

    Nothing domain-specific appears in the engine: ``values`` keys and
    ``sources`` names are the extension's vocabulary, checked for shape and
    identity only.
    """

    values: Mapping[str, str | int | float | bool]
    sources: Mapping[str, AdapterSource]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": ADAPTER_CONFIG_SCHEMA,
            "values": {name: self.values[name] for name in sorted(self.values)},
            "sources": {name: self.sources[name].to_dict() for name in sorted(self.sources)},
        }

    def digest(self) -> str:
        return _digest_bytes(_ADAPTER_CONFIG_DOMAIN + _canonical_bytes(self.to_dict()))


def validate_adapter_configuration(value: object) -> AdapterConfiguration:
    """Validate one adapter configuration document, or refuse it."""

    document = _mapping(
        value, "workload adapter configuration", frozenset({"schema", "values", "sources"})
    )
    if document["schema"] != ADAPTER_CONFIG_SCHEMA:
        raise ConfigError(
            f"workload adapter configuration schema must be {ADAPTER_CONFIG_SCHEMA!r}"
        )
    raw_values = document["values"]
    if not isinstance(raw_values, Mapping):
        raise ConfigError("workload adapter configuration values must be a mapping")
    if len(raw_values) > MAX_ADAPTER_CONFIG_VALUES:
        raise ConfigError(
            f"workload adapter configuration carries {len(raw_values)} values; the bound is "
            f"{MAX_ADAPTER_CONFIG_VALUES}"
        )
    values: dict[str, str | int | float | bool] = {}
    for key in sorted(raw_values):
        name = _text(key, "workload adapter configuration value name", pattern=_TOKEN)
        item = raw_values[key]
        # Scalars only. A nested mapping would be an unbound channel again,
        # and a path here would be an unverified filesystem authority.
        if isinstance(item, bool) or isinstance(item, (int, str)):
            values[name] = item
        elif isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise ConfigError(f"workload adapter configuration value {name!r} is not finite")
            values[name] = item
        else:
            raise ConfigError(
                f"workload adapter configuration value {name!r} is "
                f"{type(item).__name__}; only finite scalars are carried"
            )
    raw_sources = document["sources"]
    if not isinstance(raw_sources, Mapping):
        raise ConfigError("workload adapter configuration sources must be a mapping")
    if len(raw_sources) > MAX_ADAPTER_CONFIG_SOURCES:
        raise ConfigError(
            f"workload adapter configuration names {len(raw_sources)} sources; the bound is "
            f"{MAX_ADAPTER_CONFIG_SOURCES}"
        )
    sources: dict[str, AdapterSource] = {}
    for key in sorted(raw_sources):
        name = _text(key, "workload adapter configuration source name", pattern=_TOKEN)
        entry = _mapping(
            raw_sources[key],
            f"workload adapter configuration source {name!r}",
            frozenset({"kind", "path", "digest"}),
        )
        kind = _text(entry["kind"], f"adapter source {name!r} kind", pattern=_TOKEN)
        if kind not in ADAPTER_SOURCE_KINDS:
            raise ConfigError(
                f"adapter source {name!r} declares kind {kind!r}; it must be one of "
                f"{sorted(ADAPTER_SOURCE_KINDS)}"
            )
        path = str(entry["path"])
        if not path or "\0" in path:
            raise ConfigError(f"adapter source {name!r} declares an unusable path")
        sources[name] = AdapterSource(
            name=name,
            kind=kind,
            path=path,
            digest=_require_digest(
                _text(entry["digest"], f"adapter source {name!r} digest", pattern=_TOKEN),
                f"adapter source {name!r} digest",
            ),
        )
    return AdapterConfiguration(values=MappingProxyType(values), sources=MappingProxyType(sources))


@dataclass(frozen=True)
class WorkloadInterface:
    """Declared entry point and the ordered field names around it."""

    entry_point: str
    anchor_entry_point: str
    owner_unit: str
    state: tuple[str, ...]
    config: tuple[str, ...]
    forcing: tuple[str, ...]
    outputs: tuple[str, ...]
    static_signatures: tuple[str, ...]
    output_aliases: Mapping[str, str]
    #: Per-role execution binding, role -> ``module.function``. Physical
    #: workload identity is unchanged by this: the observations, outputs and
    #: objectives are the declaration's. Only *which implementation runs for
    #: which subject* differs, which is what lets a reconstructed named
    #: leaf be measured beside the flat controls it must be compared with.
    #: A role that is absent uses ``entry_point`` (or ``anchor_entry_point``
    #: for the anchor), so an older declaration means exactly what it did.
    subject_entry_points: Mapping[str, str] = field(default_factory=dict)
    #: Which framework the original anchor is executed under. Almost always
    #: ``numpy`` -- the interpreted original -- and that is the default, so
    #: an older declaration means exactly what it did. It is declared
    #: rather than assumed because a workload whose trusted original is
    #: itself a JAX package has no interpreted original to run, and
    #: recording one framework while executing another would make the
    #: anchor's environment record untrue.
    anchor_framework: str = "numpy"

    def entry_for(self, role: str) -> str:
        """The entry point this subject actually runs."""

        default = self.anchor_entry_point if role == ANCHOR_ROLE else self.entry_point
        return self.subject_entry_points.get(role, default)


@dataclass(frozen=True)
class WorkloadFixtures:
    """Content-addressed observation authority and its exact extent.

    ``state_digests`` is why two initial states are two states: names prove
    nothing, so each declared state carries the digest of its own encoded
    contents and the digests must differ.
    """

    tape_digest: str
    case: str
    probe: str
    first_sequence: int
    final_sequence: int
    steps: int
    required_inputs: tuple[str, ...]
    required_outputs: tuple[str, ...]
    coverage_labels: tuple[str, ...]
    states: tuple[str, ...]
    state_digests: Mapping[str, str]
    observation_keys_digest: str
    observation_key_count: int


@dataclass(frozen=True)
class WorkloadBoundary:
    """Which scientific workload this is, and what stays externally fixed.

    ``interleaved_probe`` names the states that must be run interleaved with a
    reset between repetitions, which is the check that catches one variant
    borrowing another's cached state. It is deliberately separable from the
    canonical trajectory: a single valid recording is not obliged to invent a
    second reference trajectory, and an empty probe says so explicitly.
    """

    mode: str
    state_feedback: tuple[str, ...]
    external_inputs: tuple[str, ...]
    observation_cadence: int
    reset: str
    interleaved_probe: tuple[str, ...]


@dataclass(frozen=True)
class SpeedRequirement:
    """One speed objective, with its threshold fixed before any timing.

    ``reduction`` requires ``(reference - candidate) / reference >= threshold``
    and a gap wider than ``noise_factor`` times the larger median absolute
    deviation. ``regression_ceiling`` requires
    ``(candidate - reference) / reference <= threshold`` *or* a gap inside that
    same margin. Both are computed from the ordered raw warm samples, never
    from a plugin's own conclusion.
    """

    objective: str
    reference: str
    threshold: float
    noise_factor: float

    @property
    def key(self) -> str:
        """Stable identity used to check a leaf against inherited objectives."""
        return f"{self.objective}:{self.reference}"

    def at_least_as_strict_as(self, other: SpeedRequirement) -> bool:
        """Whether this requirement demands everything ``other`` demanded.

        The noise factor points in opposite directions for the two
        objectives, which a single comparison got wrong. For ``reduction`` the
        gap must *exceed* ``noise_factor * MAD``, so a larger factor is a
        stricter demand and lowering it from three to zero is a weakening. For
        ``regression_ceiling`` the factor sizes an *exemption* that forgives a
        regression inside the noise, so a smaller factor is stricter.
        """

        if self.key != other.key:
            return False
        if self.objective == "reduction":
            return self.threshold >= other.threshold and self.noise_factor >= other.noise_factor
        return self.threshold <= other.threshold and self.noise_factor <= other.noise_factor

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "reference": self.reference,
            "threshold": self.threshold,
            "noise_factor": self.noise_factor,
        }


@dataclass(frozen=True)
class FiniteDifferencePolicy:
    """The declared derivative-checking policy, not a worker constant."""

    steps: tuple[float, ...]
    smooth_relative: float
    kink_relative: float
    classified_modes: tuple[str, ...]
    """Modes that must report smooth, kink and unresolved counts.

    Only a mode checked against a finite-difference sweep has that
    classification to report: a compilation-agreement check and a
    forward/reverse dot-product identity do not, and requiring counts from
    them would force a worker to invent numbers. Naming the modes keeps the
    requirement real -- a required unresolved count cannot then disappear
    into a null.
    """

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": list(self.steps),
            "smooth_relative": self.smooth_relative,
            "kink_relative": self.kink_relative,
            "classified_modes": list(self.classified_modes),
        }


@dataclass(frozen=True)
class WorkloadAcceptance:
    """What this leaf must satisfy on its own bytes."""

    required_comparisons: tuple[str, ...]
    required_modes: tuple[str, ...]
    differentiable_slots: tuple[str, ...]
    required_outputs: tuple[str, ...]
    readout: str
    permitted_measurement_classes: frozenset[str]
    permit_local_plugin_origin: bool
    speed: tuple[SpeedRequirement, ...]
    finite_difference: FiniteDifferencePolicy


@dataclass(frozen=True)
class WorkloadMeasurement:
    """Timing contract, declared before the candidate is timed."""

    boundary: str
    rounds: int
    warm_repetitions: int
    required_phases: Mapping[str, tuple[str, ...]]
    subject_modes: Mapping[str, str]
    compile_excluded_from_warm: bool
    # ``backend`` is the platform the measurement is only valid on, and is
    # compared with the backend the worker actually executed on.
    # ``environment_contract`` is a *declared identity* for the wider
    # environment -- library builds, thread controls, machine facts. The
    # adjunct cannot resolve that digest, so it binds it and says so rather
    # than presenting an unresolvable label as an environment check.
    backend: str
    environment_contract: str

    def mode(self, subject: str) -> str | None:
        return self.subject_modes.get(subject)

    def required(self, subject: str) -> tuple[str, ...]:
        mode = self.subject_modes.get(subject)
        return () if mode is None else self.required_phases.get(mode, ())


@dataclass(frozen=True)
class PackageContract:
    """What the assembled package must contain, and its ceilings."""

    digest: str
    entry_modules: tuple[str, ...]
    required_exports: tuple[str, ...]
    helper_files: tuple[str, ...]
    helper_owner: str
    max_files: int
    max_file_bytes: int
    max_total_bytes: int


@dataclass(frozen=True)
class ObjectiveSet:
    """Everything a node was held to, as one comparable identity.

    A chain inherits objectives, not objective *names*. Keeping ``vjp`` in the
    list while dropping the slot it applied to, or keeping a reduction
    objective while halving its threshold, is a weakening, and
    :meth:`shortfalls_against` is where that is caught.
    """

    required_modes: tuple[str, ...]
    required_comparisons: tuple[str, ...]
    differentiable_slots: tuple[str, ...]
    required_outputs: tuple[str, ...]
    readout: str
    measurement_boundary: str
    permitted_measurement_classes: tuple[str, ...]
    speed: tuple[SpeedRequirement, ...]
    observation_keys_digest: str
    finite_difference: FiniteDifferencePolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "required_modes": list(self.required_modes),
            "required_comparisons": list(self.required_comparisons),
            "differentiable_slots": list(self.differentiable_slots),
            "required_outputs": list(self.required_outputs),
            "readout": self.readout,
            "measurement_boundary": self.measurement_boundary,
            "permitted_measurement_classes": list(self.permitted_measurement_classes),
            "speed": [item.to_dict() for item in self.speed],
            "observation_keys_digest": self.observation_keys_digest,
            "finite_difference": self.finite_difference.to_dict(),
        }

    def digest(self) -> str:
        return _digest_bytes(_canonical_bytes(self.to_dict()))

    def shortfalls_against(self, inherited: ObjectiveSet) -> tuple[str, ...]:
        """Ways this set asks for less than ``inherited`` already did."""

        reasons: list[str] = []
        for label, mine, theirs in (
            ("mode", self.required_modes, inherited.required_modes),
            ("comparison", self.required_comparisons, inherited.required_comparisons),
            ("slot", self.differentiable_slots, inherited.differentiable_slots),
            ("output", self.required_outputs, inherited.required_outputs),
        ):
            for name in sorted(set(theirs) - set(mine)):
                reasons.append(f"{label}.{name}")
        if self.readout != inherited.readout:
            reasons.append("readout")
        if self.measurement_boundary != inherited.measurement_boundary:
            reasons.append("boundary")
        widened = sorted(
            set(self.permitted_measurement_classes) - set(inherited.permitted_measurement_classes)
        )
        for name in widened:
            reasons.append(f"measurement_class.{name}")
        if self.observation_keys_digest != inherited.observation_keys_digest:
            reasons.append("coverage")
        mine_speed = {item.key: item for item in self.speed}
        for requirement in inherited.speed:
            held = mine_speed.get(requirement.key)
            if held is None or not held.at_least_as_strict_as(requirement):
                reasons.append(f"speed.{requirement.key}")
        theirs_fd, mine_fd = inherited.finite_difference, self.finite_difference
        if (
            mine_fd.smooth_relative > theirs_fd.smooth_relative
            or mine_fd.kink_relative > theirs_fd.kink_relative
            or set(theirs_fd.classified_modes) - set(mine_fd.classified_modes)
            or set(theirs_fd.steps) - set(mine_fd.steps)
        ):
            reasons.append("finite_difference")
        return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class WorkloadDeclaration:
    """A validated ``recast.forward-workload.v1`` document.

    ``digest`` is over the canonical bytes of the document as supplied, so the
    launch-authorized digest is checked against what the engine actually read
    rather than against a re-serialization of selected fields.
    """

    workload: str
    version: str
    digest: str
    verifier: VerifierBinding
    adapter: AdapterBinding
    policy_digest: str
    interface: WorkloadInterface
    fixtures: WorkloadFixtures
    boundary: WorkloadBoundary
    acceptance: WorkloadAcceptance
    measurement: WorkloadMeasurement
    package_contract: PackageContract
    boundary_receipt_digest: str | None
    document: Mapping[str, object]

    @property
    def speed_keys(self) -> frozenset[str]:
        return frozenset(item.key for item in self.acceptance.speed)

    @property
    def objectives(self) -> ObjectiveSet:
        """The comparable objective identity this leaf is held to."""
        return ObjectiveSet(
            required_modes=self.acceptance.required_modes,
            required_comparisons=self.acceptance.required_comparisons,
            differentiable_slots=self.acceptance.differentiable_slots,
            required_outputs=self.acceptance.required_outputs,
            readout=self.acceptance.readout,
            measurement_boundary=self.measurement.boundary,
            permitted_measurement_classes=tuple(
                sorted(self.acceptance.permitted_measurement_classes)
            ),
            speed=self.acceptance.speed,
            observation_keys_digest=self.fixtures.observation_keys_digest,
            finite_difference=self.acceptance.finite_difference,
        )


def observation_keys(
    *,
    signatures: tuple[str, ...],
    states: tuple[str, ...],
    first_sequence: int,
    final_sequence: int,
    outputs: tuple[str, ...],
) -> tuple[str, ...]:
    """Every observation the declared dimensions require, in canonical order.

    Marginal coverage sets are not enough: seeing an output once says nothing
    about seeing it on every signature, state and step. These keys are the
    required *combinations*, and both the launch that authorizes the
    declaration and the plugin that runs the workload derive them from the
    same rule, so a selectively omitted or duplicated observation changes the
    digest.
    """

    return tuple(
        f"{signature}|{state}|{sequence}|{output}"
        for signature in sorted(signatures)
        for state in sorted(states)
        for sequence in range(first_sequence, final_sequence + 1)
        for output in sorted(outputs)
    )


def observation_keys_digest(keys: tuple[str, ...]) -> str:
    """Digest one observation-key set: sorted, with duplicates retained.

    Sorting makes the two sides agree without either having to reproduce the
    other's iteration order. Duplicates are *kept*, because an observation
    recorded twice is exactly the kind of padding this digest exists to catch.
    """

    return _digest_bytes(_canonical_bytes(sorted(keys)))


_DECLARATION_KEYS = frozenset(
    {
        "schema",
        "workload",
        "version",
        "verifier",
        "adapter",
        "policy",
        "interface",
        "fixtures",
        "boundary",
        "acceptance",
        "measurement",
        "composition",
        "package_contract",
    }
)


def validate_workload_declaration(value: object) -> WorkloadDeclaration:
    """Validate one trusted workload declaration, or refuse it.

    Every cross-field rule that a domain workload could otherwise smuggle past
    the engine is checked here: a discontinuous tape, an empty required
    coverage set, a required output the interface does not declare, state
    feedback in an open-loop workload, an interleaved probe over states whose
    contents are identical, an observation-key digest that does not follow
    from the declared dimensions, and a speed objective that references the
    candidate as its own baseline.
    """

    document = _mapping(value, "forward workload declaration", _DECLARATION_KEYS)
    if document["schema"] != WORKLOAD_SCHEMA:
        raise ConfigError(f"forward workload declaration schema must be {WORKLOAD_SCHEMA!r}")
    workload = _text(document["workload"], "workload", pattern=_TOKEN)
    version = _text(document["version"], "workload version", pattern=_TOKEN)

    verifier_document = _mapping(
        document["verifier"],
        "workload verifier binding",
        frozenset(
            {
                "plugin",
                "distribution_name",
                "distribution_version",
                "implementation_digest",
                "implementation_modules",
            }
        ),
    )
    verifier = VerifierBinding(
        plugin=_text(verifier_document["plugin"], "workload verifier plugin", pattern=_TOKEN),
        distribution_name=_text(
            verifier_document["distribution_name"],
            "workload verifier distribution",
            pattern=_TOKEN,
        ),
        distribution_version=_text(
            verifier_document["distribution_version"],
            "workload verifier distribution version",
            pattern=_TOKEN,
        ),
        implementation_digest=_require_digest(
            _text(
                verifier_document["implementation_digest"],
                "workload verifier implementation_digest",
                pattern=_TOKEN,
            ),
            "workload verifier implementation_digest",
        ),
        implementation_modules=_names(
            verifier_document["implementation_modules"],
            "workload verifier implementation_modules",
            minimum=1,
            maximum=MAX_SIGNATURES,
            pattern=_DOTTED,
        ),
    )

    adapter_document = _mapping(
        document["adapter"],
        "workload adapter",
        frozenset({"digest", "entry_point", "implementation_modules", "configuration_digest"}),
        optional=frozenset({"configuration_digest"}),
    )
    adapter = AdapterBinding(
        digest=_require_digest(
            _text(adapter_document["digest"], "workload adapter digest", pattern=_TOKEN),
            "workload adapter digest",
        ),
        entry_point=_text(
            adapter_document["entry_point"], "workload adapter entry_point", pattern=_TOKEN
        ),
        implementation_modules=_names(
            adapter_document["implementation_modules"],
            "workload adapter implementation_modules",
            minimum=1,
            maximum=MAX_SIGNATURES,
            pattern=_DOTTED,
        ),
        configuration_digest=(
            None
            if adapter_document.get("configuration_digest") is None
            else _require_digest(
                _text(
                    adapter_document["configuration_digest"],
                    "workload adapter configuration_digest",
                    pattern=_TOKEN,
                ),
                "workload adapter configuration_digest",
            )
        ),
    )
    policy = _mapping(document["policy"], "workload policy", frozenset({"digest"}))
    policy_digest = _require_digest(
        _text(policy["digest"], "workload policy digest", pattern=_TOKEN), "workload policy digest"
    )

    interface = _validate_interface(document["interface"])
    acceptance = _validate_acceptance(document["acceptance"], interface)
    fixtures = _validate_fixtures(document["fixtures"], interface, acceptance)
    boundary = _validate_boundary(document["boundary"], interface, fixtures)
    measurement = _validate_measurement(document["measurement"], acceptance)
    package_contract = _validate_package_contract(document["package_contract"], interface)

    composition = _mapping(
        document["composition"], "workload composition", frozenset({"receipt_digest"})
    )
    raw_receipt = composition["receipt_digest"]
    receipt_digest = (
        None
        if raw_receipt is None
        else _require_digest(
            _text(raw_receipt, "workload composition receipt_digest", pattern=_TOKEN),
            "workload composition receipt_digest",
        )
    )

    frozen = _freeze_json(document, "forward workload declaration")
    if not isinstance(frozen, Mapping):  # _mapping above makes this defensive only.
        raise ConfigError("forward workload declaration must be an object")
    return WorkloadDeclaration(
        workload=workload,
        version=version,
        digest=_digest_bytes(_canonical_bytes(document)),
        verifier=verifier,
        adapter=adapter,
        policy_digest=policy_digest,
        interface=interface,
        fixtures=fixtures,
        boundary=boundary,
        acceptance=acceptance,
        measurement=measurement,
        package_contract=package_contract,
        boundary_receipt_digest=receipt_digest,
        document=frozen,
    )


def _validate_interface(value: object) -> WorkloadInterface:
    document = _mapping(
        value,
        "workload interface",
        frozenset(
            {
                "entry_point",
                "anchor_entry_point",
                "subject_entry_points",
                "owner_unit",
                "state",
                "config",
                "forcing",
                "outputs",
                "static_signatures",
                "output_aliases",
                "anchor_framework",
            }
        ),
        # Added after the first frozen declarations. Absent means "every
        # subject uses the declared entry point" and "the anchor is the
        # interpreted NumPy original", which is what those declarations
        # meant.
        optional=frozenset({"subject_entry_points", "anchor_framework"}),
    )
    # One subject may legitimately be a different *implementation* of the
    # same physical workload: a reconstructed named-state leaf beside flat
    # controls, with its own module and function. Physical identity -- the
    # observations, the outputs, the objectives -- is unchanged; only the
    # execution binding differs, and it is declared per role rather than
    # assumed to be one entry for all four.
    raw_entries = document.get("subject_entry_points") or {}
    if not isinstance(raw_entries, Mapping):
        raise ConfigError("workload interface subject_entry_points must be a mapping")
    subject_entry_points: dict[str, str] = {}
    for key in sorted(raw_entries):
        role = _text(key, "workload interface subject_entry_points role", pattern=_TOKEN)
        if role not in set(SUBJECT_ROLES) | {ANCHOR_ROLE}:
            raise ConfigError(
                f"workload interface subject_entry_points names {role!r}, which is not a "
                f"subject role"
            )
        subject_entry_points[role] = _text(
            raw_entries[key],
            f"workload interface subject_entry_points[{role}]",
            pattern=_DOTTED,
        )

    anchor_framework = document.get("anchor_framework", "numpy")
    if anchor_framework not in {"numpy", "jax"}:
        raise ConfigError(
            f"workload interface anchor_framework is {anchor_framework!r}; the original anchor "
            "is executed under 'numpy' or 'jax'"
        )

    state = _names(document["state"], "workload interface state", minimum=1)
    config = _names(document["config"], "workload interface config")
    forcing = _names(document["forcing"], "workload interface forcing")
    outputs = _names(document["outputs"], "workload interface outputs", minimum=1)
    overlap = sorted(
        (set(state) & set(config)) | (set(state) & set(forcing)) | (set(config) & set(forcing))
    )
    if overlap:
        raise ConfigError(f"workload interface reuses input name(s) {overlap} in two roles")
    signatures = _names(
        document["static_signatures"],
        "workload interface static_signatures",
        minimum=1,
        maximum=MAX_SIGNATURES,
    )
    aliases_value = document["output_aliases"]
    if not isinstance(aliases_value, Mapping) or len(aliases_value) > MAX_NAMES:
        raise ConfigError("workload interface output_aliases must be a bounded object")
    aliases: dict[str, str] = {}
    for key, alias in aliases_value.items():
        name = _text(key, "workload interface output alias key")
        aliases[name] = _text(alias, f"workload interface output alias {name}")
    _subset(tuple(aliases), outputs, "workload interface output_aliases")
    collisions = sorted(set(aliases.values()) & set(outputs))
    if collisions:
        raise ConfigError(f"workload interface output alias(es) {collisions} shadow a real output")
    if len(set(aliases.values())) != len(aliases):
        # Two outputs behind one alias name is a schema that has quietly lost
        # an output, which is exactly what alias validation is here to stop.
        raise ConfigError("workload interface maps two outputs onto one alias name")
    entry_point = _text(document["entry_point"], "workload entry_point", pattern=_DOTTED)
    # The anchor is a separate package -- for a Python project the unchanged
    # source, for CLUBB the pinned Fortran replay -- so its entry point is
    # declared rather than guessed from the candidate's module name.
    anchor_entry_point = _text(
        document["anchor_entry_point"], "workload anchor_entry_point", pattern=_DOTTED
    )
    for name, item in (("entry_point", entry_point), ("anchor_entry_point", anchor_entry_point)):
        if "." not in item:
            raise ConfigError(
                f"workload {name} must name its module and its callable, as 'module.callable'"
            )
    return WorkloadInterface(
        entry_point=entry_point,
        anchor_entry_point=anchor_entry_point,
        owner_unit=_text(document["owner_unit"], "workload owner_unit"),
        state=state,
        config=config,
        forcing=forcing,
        outputs=outputs,
        static_signatures=signatures,
        output_aliases=MappingProxyType(aliases),
        subject_entry_points=MappingProxyType(subject_entry_points),
        anchor_framework=str(anchor_framework),
    )


def _validate_fixtures(
    value: object, interface: WorkloadInterface, acceptance: WorkloadAcceptance
) -> WorkloadFixtures:
    document = _mapping(
        value,
        "workload fixtures",
        frozenset(
            {
                "tape_digest",
                "case",
                "probe",
                "first_sequence",
                "final_sequence",
                "steps",
                "required_inputs",
                "required_outputs",
                "coverage_labels",
                "states",
                "state_digests",
                "observation_keys_digest",
                "observation_key_count",
            }
        ),
    )
    first = _integer(
        document["first_sequence"], "workload first_sequence", minimum=0, maximum=MAX_STEPS
    )
    final = _integer(
        document["final_sequence"], "workload final_sequence", minimum=0, maximum=MAX_STEPS
    )
    steps = _integer(document["steps"], "workload steps", minimum=1, maximum=MAX_STEPS)
    if final - first + 1 != steps:
        raise ConfigError(
            f"workload sequence [{first}, {final}] is not a contiguous run of {steps} step(s)"
        )
    inputs = _names(document["required_inputs"], "workload required_inputs", minimum=1)
    _subset(
        inputs,
        interface.state + interface.config + interface.forcing,
        "workload required_inputs",
    )
    outputs = _names(document["required_outputs"], "workload required_outputs", minimum=1)
    _subset(outputs, interface.outputs, "workload required_outputs")
    states = _names(
        document["states"], "workload states", minimum=1, maximum=MAX_STATES, pattern=_TOKEN
    )
    raw_digests = document["state_digests"]
    if not isinstance(raw_digests, Mapping) or set(raw_digests) != set(states):
        raise ConfigError("workload state_digests must bind the contents of every declared state")
    state_digests: dict[str, str] = {}
    for name in states:
        state_digests[name] = _require_digest(
            _text(raw_digests[name], f"workload state_digests[{name}]", pattern=_TOKEN),
            f"workload state_digests[{name}]",
        )
    if len(set(state_digests.values())) != len(states):
        # Two names for one set of numbers are one state.
        raise ConfigError("workload states must have distinct contents, not merely distinct names")

    expected = observation_keys(
        signatures=interface.static_signatures,
        states=states,
        first_sequence=first,
        final_sequence=final,
        outputs=acceptance.required_outputs,
    )
    declared_digest = _require_digest(
        _text(
            document["observation_keys_digest"],
            "workload observation_keys_digest",
            pattern=_TOKEN,
        ),
        "workload observation_keys_digest",
    )
    if declared_digest != observation_keys_digest(expected):
        raise ConfigError(
            "workload observation_keys_digest does not follow from the declared signatures, "
            "states, sequence range and required outputs"
        )
    declared_count = _integer(
        document["observation_key_count"],
        "workload observation_key_count",
        minimum=1,
        maximum=MAX_OBSERVATION_KEYS,
    )
    if declared_count != len(expected):
        raise ConfigError(
            f"workload observation_key_count is {declared_count}; the declared dimensions "
            f"require {len(expected)}"
        )
    return WorkloadFixtures(
        tape_digest=_require_digest(
            _text(document["tape_digest"], "workload tape_digest", pattern=_TOKEN),
            "workload tape_digest",
        ),
        case=_text(document["case"], "workload case", pattern=_TOKEN),
        probe=_text(document["probe"], "workload probe", pattern=_TOKEN),
        first_sequence=first,
        final_sequence=final,
        steps=steps,
        required_inputs=inputs,
        required_outputs=outputs,
        coverage_labels=_names(
            document["coverage_labels"], "workload coverage_labels", minimum=1, pattern=_TOKEN
        ),
        states=states,
        state_digests=MappingProxyType(state_digests),
        observation_keys_digest=declared_digest,
        observation_key_count=declared_count,
    )


def _validate_boundary(
    value: object, interface: WorkloadInterface, fixtures: WorkloadFixtures
) -> WorkloadBoundary:
    document = _mapping(
        value,
        "workload boundary",
        frozenset(
            {
                "mode",
                "state_feedback",
                "external_inputs",
                "observation_cadence",
                "reset",
                "interleaved_probe",
            }
        ),
    )
    mode = document["mode"]
    if mode not in BOUNDARY_MODES:
        raise ConfigError(f"workload boundary mode must be one of {sorted(BOUNDARY_MODES)}")
    feedback = _names(document["state_feedback"], "workload state_feedback")
    _subset(feedback, interface.state, "workload state_feedback")
    if mode == "state_feedback_core_replay" and not feedback:
        raise ConfigError("a state-feedback replay must name the fed-back state fields")
    if mode == "recorded_open_loop" and feedback:
        raise ConfigError("an open-loop workload must not feed state back")
    external = _names(document["external_inputs"], "workload external_inputs")
    _subset(external, interface.forcing, "workload external_inputs")
    reset = document["reset"]
    if reset not in RESET_POLICIES:
        raise ConfigError(f"workload reset must be one of {sorted(RESET_POLICIES)}")
    probe = _names(
        document["interleaved_probe"],
        "workload interleaved_probe",
        maximum=MAX_STATES,
        pattern=_TOKEN,
    )
    _subset(probe, fixtures.states, "workload interleaved_probe")
    if probe and len(probe) < 2:
        raise ConfigError("an interleaved probe needs at least two distinct states, or none")
    if probe and reset == "never":
        raise ConfigError("an interleaved probe that never resets proves nothing about state")
    return WorkloadBoundary(
        mode=str(mode),
        state_feedback=feedback,
        external_inputs=external,
        observation_cadence=_integer(
            document["observation_cadence"],
            "workload observation_cadence",
            minimum=1,
            maximum=MAX_STEPS,
        ),
        reset=str(reset),
        interleaved_probe=probe,
    )


def _validate_acceptance(value: object, interface: WorkloadInterface) -> WorkloadAcceptance:
    document = _mapping(
        value,
        "workload acceptance",
        frozenset(
            {
                "required_comparisons",
                "required_modes",
                "differentiable_slots",
                "required_outputs",
                "readout",
                "permitted_measurement_classes",
                "permit_local_plugin_origin",
                "speed",
                "finite_difference",
            }
        ),
    )
    comparisons = _names(
        document["required_comparisons"],
        "workload required_comparisons",
        minimum=1,
        pattern=_TOKEN,
    )
    unknown = sorted(set(comparisons) - set(COMPARISON_SUBJECTS))
    if unknown:
        raise ConfigError(f"workload required_comparisons names unknown subject(s) {unknown}")
    for mandatory in (ANCHOR_ROLE, "predecessor"):
        if mandatory not in comparisons:
            # Tolerances are not transitive. A chain of individually
            # acceptable drifts is exactly what an adjacent-only comparison
            # accepts, and an anchor-only comparison says nothing about the
            # link this node actually added.
            raise ConfigError(f"workload required_comparisons must include {mandatory!r}")
    modes = _names(document["required_modes"], "workload required_modes", pattern=_TOKEN)
    slots = _names(document["differentiable_slots"], "workload differentiable_slots")
    _subset(slots, interface.outputs, "workload differentiable_slots")
    if modes and not slots:
        raise ConfigError("a required derivative mode needs at least one differentiable slot")
    outputs = _names(
        document["required_outputs"], "workload acceptance required_outputs", minimum=1
    )
    _subset(outputs, interface.outputs, "workload acceptance required_outputs")
    readout = _text(document["readout"], "workload acceptance readout")
    if modes and readout not in slots:
        raise ConfigError("the frozen scalar readout must be one of the differentiable slots")
    classes = _names(
        document["permitted_measurement_classes"],
        "workload permitted_measurement_classes",
        minimum=1,
        pattern=_TOKEN,
    )
    unknown_classes = sorted(set(classes) - MEASUREMENT_CLASSES)
    if unknown_classes:
        raise ConfigError(
            f"workload permitted_measurement_classes names unknown class(es) {unknown_classes}"
        )
    raw_speed = document["speed"]
    if not isinstance(raw_speed, Sequence) or isinstance(raw_speed, (str, bytes)):
        raise ConfigError("workload acceptance speed must be an array")
    if len(raw_speed) > MAX_SPEED_REQUIREMENTS:
        raise ConfigError("workload acceptance declares too many speed requirements")
    speed: list[SpeedRequirement] = []
    for index, item in enumerate(raw_speed):
        entry = _mapping(
            item,
            f"workload speed requirement[{index}]",
            frozenset({"objective", "reference", "threshold", "noise_factor"}),
        )
        objective = entry["objective"]
        if objective not in SPEED_OBJECTIVES:
            raise ConfigError(f"workload speed objective must be one of {sorted(SPEED_OBJECTIVES)}")
        reference = _text(entry["reference"], "workload speed reference", pattern=_TOKEN)
        if reference not in SUBJECT_ROLES or reference == "candidate":
            raise ConfigError(
                "workload speed reference must be another staged subject, never the candidate"
            )
        speed.append(
            SpeedRequirement(
                objective=str(objective),
                reference=reference,
                threshold=_fraction(
                    entry["threshold"], "workload speed threshold", minimum=0.0, maximum=1.0
                ),
                noise_factor=_fraction(
                    entry["noise_factor"], "workload speed noise_factor", minimum=0.0, maximum=100.0
                ),
            )
        )
    keys = [item.key for item in speed]
    if len(set(keys)) != len(keys):
        raise ConfigError("workload acceptance repeats a speed objective")
    return WorkloadAcceptance(
        required_comparisons=comparisons,
        required_modes=modes,
        differentiable_slots=slots,
        required_outputs=outputs,
        readout=readout,
        permitted_measurement_classes=frozenset(classes),
        permit_local_plugin_origin=_flag(
            document["permit_local_plugin_origin"], "workload permit_local_plugin_origin"
        ),
        speed=tuple(speed),
        finite_difference=_validate_finite_difference(document["finite_difference"], modes),
    )


def _validate_finite_difference(value: object, modes: tuple[str, ...]) -> FiniteDifferencePolicy:
    document = _mapping(
        value,
        "workload finite_difference",
        frozenset({"steps", "smooth_relative", "kink_relative", "classified_modes"}),
    )
    raw = document["steps"]
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise ConfigError("workload finite_difference steps must be an array")
    if not raw or len(raw) > 64:
        raise ConfigError("workload finite_difference needs between one and 64 step sizes")
    steps: list[float] = []
    for item in raw:
        steps.append(_fraction(item, "workload finite_difference step", minimum=0.0, maximum=1.0))
        if steps[-1] <= 0.0:
            raise ConfigError("workload finite_difference step must be positive")
    if len(set(steps)) != len(steps):
        raise ConfigError("workload finite_difference repeats a step size")
    smooth = _fraction(
        document["smooth_relative"],
        "workload finite_difference smooth_relative",
        minimum=0.0,
        maximum=1.0,
    )
    kink = _fraction(
        document["kink_relative"],
        "workload finite_difference kink_relative",
        minimum=0.0,
        maximum=1.0,
    )
    if smooth <= 0.0 or kink <= 0.0:
        raise ConfigError("workload finite_difference thresholds must be positive")
    if smooth >= kink:
        raise ConfigError(
            "workload finite_difference smooth_relative must be tighter than kink_relative"
        )
    classified = _names(
        document["classified_modes"], "workload finite_difference classified_modes", pattern=_TOKEN
    )
    unknown = sorted(set(classified) - set(modes))
    if unknown:
        raise ConfigError(
            f"workload finite_difference classified_modes names {unknown}, which are not "
            "required derivative modes"
        )
    if modes and not steps:
        raise ConfigError("a required derivative mode needs a finite-difference step sweep")
    return FiniteDifferencePolicy(
        steps=tuple(steps),
        smooth_relative=smooth,
        kink_relative=kink,
        classified_modes=classified,
    )


def _validate_measurement(value: object, acceptance: WorkloadAcceptance) -> WorkloadMeasurement:
    document = _mapping(
        value,
        "workload measurement",
        frozenset(
            {
                "boundary",
                "rounds",
                "warm_repetitions",
                "required_phases",
                "subject_modes",
                "compile_excluded_from_warm",
                "backend",
                "environment_contract",
            }
        ),
    )
    raw_modes = document["subject_modes"]
    if not isinstance(raw_modes, Mapping) or set(raw_modes) != set(TIMED_ROLES):
        raise ConfigError(
            f"workload subject_modes must name an execution mode for each of {list(TIMED_ROLES)}"
        )
    modes: dict[str, str] = {}
    for subject in TIMED_ROLES:
        mode = raw_modes[subject]
        if mode not in EXECUTION_MODES:
            raise ConfigError(
                f"workload subject_modes[{subject!r}] must be one of {sorted(EXECUTION_MODES)}"
            )
        modes[subject] = str(mode)

    raw_phases = document["required_phases"]
    if not isinstance(raw_phases, Mapping) or not set(raw_phases) <= EXECUTION_MODES:
        raise ConfigError("workload required_phases must be keyed by execution mode")
    used = {modes[subject] for subject in SUBJECT_ROLES}
    missing_modes = sorted(used - set(raw_phases))
    if missing_modes:
        raise ConfigError(f"workload required_phases does not cover mode(s) {missing_modes}")
    phases: dict[str, tuple[str, ...]] = {}
    for mode, names in raw_phases.items():
        listed = _names(names, f"workload required_phases[{mode}]", pattern=_TOKEN)
        unknown = sorted(set(listed) - set(TIMING_PHASES))
        if unknown:
            raise ConfigError(
                f"workload required_phases[{mode}] names unknown boundary/boundaries {unknown}"
            )
        if mode == "interpreted":
            inapplicable = sorted(set(listed) & set(_COMPILED_ONLY_PHASES))
            if inapplicable:
                raise ConfigError(
                    f"workload required_phases[interpreted] requires {inapplicable}, which an "
                    "interpreted subject cannot have"
                )
        else:
            absent = sorted(set(_COMPILED_ONLY_PHASES) - set(listed))
            if absent:
                raise ConfigError(
                    f"workload required_phases[compiled] omits {absent}; a compiled mode must "
                    "show that the named workload really traced, lowered and compiled"
                )
        phases[str(mode)] = listed

    excluded = _flag(document["compile_excluded_from_warm"], "workload compile_excluded_from_warm")
    if acceptance.speed and not excluded:
        raise ConfigError(
            "a workload with a speed objective must exclude compilation from its warm samples"
        )
    for requirement in acceptance.speed:
        if "warm_median" not in phases.get(modes[requirement.reference], ()):
            raise ConfigError(
                f"workload speed objective {requirement.key!r} needs a required warm median for "
                f"its {requirement.reference} baseline"
            )
    return WorkloadMeasurement(
        boundary=_text(document["boundary"], "workload measurement boundary", pattern=_TOKEN),
        rounds=_integer(document["rounds"], "workload rounds", minimum=1, maximum=4096),
        warm_repetitions=_integer(
            document["warm_repetitions"], "workload warm_repetitions", minimum=1, maximum=4096
        ),
        required_phases=MappingProxyType(phases),
        subject_modes=MappingProxyType(modes),
        compile_excluded_from_warm=excluded,
        backend=_text(document["backend"], "workload measurement backend", pattern=_TOKEN),
        environment_contract=_require_digest(
            _text(
                document["environment_contract"],
                "workload environment_contract",
                pattern=_TOKEN,
            ),
            "workload environment_contract",
        ),
    )


def _validate_package_contract(value: object, interface: WorkloadInterface) -> PackageContract:
    document = _mapping(
        value,
        "workload package_contract",
        frozenset(
            {
                "digest",
                "entry_modules",
                "required_exports",
                "helper_files",
                "helper_owner",
                "max_files",
                "max_file_bytes",
                "max_total_bytes",
            }
        ),
    )
    entry_modules = _names(
        document["entry_modules"], "workload package entry_modules", minimum=1, pattern=_DOTTED
    )
    entry_module = interface.entry_point.rsplit(".", 1)[0]
    if entry_module not in entry_modules:
        raise ConfigError(
            "workload package_contract does not freeze the module its entry point lives in"
        )
    helpers = _names(document["helper_files"], "workload package helper_files")
    for helper in helpers:
        if helper.startswith("/") or ".." in helper.split("/") or "\\" in helper:
            raise ConfigError(f"workload package helper {helper!r} is not a portable relative path")
    owner = _text(document["helper_owner"], "workload package helper_owner") if helpers else ""
    if helpers and not owner:
        raise ConfigError("a declared package helper must name the Candidate that owns it")
    return PackageContract(
        digest=_require_digest(
            _text(document["digest"], "workload package_contract digest", pattern=_TOKEN),
            "workload package_contract digest",
        ),
        entry_modules=entry_modules,
        required_exports=_names(
            document["required_exports"], "workload package required_exports", minimum=1
        ),
        helper_files=helpers,
        helper_owner=owner,
        max_files=_integer(
            document["max_files"],
            "workload package max_files",
            minimum=1,
            maximum=MAX_PACKAGE_FILES,
        ),
        max_file_bytes=_integer(
            document["max_file_bytes"],
            "workload package max_file_bytes",
            minimum=1,
            maximum=MAX_PACKAGE_FILE_BYTES,
        ),
        max_total_bytes=_integer(
            document["max_total_bytes"],
            "workload package max_total_bytes",
            minimum=1,
            maximum=MAX_PACKAGE_TOTAL_BYTES,
        ),
    )


# --- the plugin's result -----------------------------------------------------


@dataclass(frozen=True)
class EvidenceSubject:
    """A digest-addressed reference to detail the report deliberately omits."""

    label: str
    digest: str
    media_type: str
    size: int

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "digest": self.digest,
            "media_type": self.media_type,
            "size": self.size,
        }


@dataclass(frozen=True)
class ComparisonObservation:
    """One subject-to-subject numerical comparison, as measured."""

    comparison: str
    status: str
    checked_outputs: int
    observed_key_count: int
    observed_keys_digest: str | None
    failed_outputs: tuple[str, ...]
    max_ulp: str | None
    max_absolute_error: float | None
    max_relative_error: float | None
    nonfinite_mismatches: int | None
    discrete_mismatches: int | None
    evidence: str | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "comparison": self.comparison,
            "status": self.status,
            "checked_outputs": self.checked_outputs,
            "observed_key_count": self.observed_key_count,
            "observed_keys_digest": self.observed_keys_digest,
            "failed_outputs": list(self.failed_outputs),
            "max_ulp": self.max_ulp,
            "max_absolute_error": self.max_absolute_error,
            "max_relative_error": self.max_relative_error,
            "nonfinite_mismatches": self.nonfinite_mismatches,
            "discrete_mismatches": self.discrete_mismatches,
            "evidence": self.evidence,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class DerivativeObservation:
    """One required derivative mode, as measured."""

    mode: str
    status: str
    scope: str
    covered_pairs: tuple[str, ...]
    slots: tuple[str, ...]
    smooth: int | None
    kink: int | None
    unresolved: int | None
    informative: int | None
    zero_sensitivity: int | None
    max_relative_error: float | None
    evidence: str | None
    detail: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "status": self.status,
            "scope": self.scope,
            "covered_pairs": list(self.covered_pairs),
            "slots": list(self.slots),
            "smooth": self.smooth,
            "kink": self.kink,
            "unresolved": self.unresolved,
            "informative": self.informative,
            "zero_sensitivity": self.zero_sensitivity,
            "max_relative_error": self.max_relative_error,
            "evidence": self.evidence,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TimingObservation:
    """Separated timing boundaries for one subject. Absence stays ``None``.

    ``warm_median`` and ``warm_mad`` are not taken on trust: the ordered raw
    samples come with them and the summary is recomputed from the samples, so
    a plugin cannot report a favourable median over measurements it did not
    make.
    """

    subject: str
    boundary: str
    execution_mode: str
    executed_backend: str
    rounds: int | None
    warm_samples: int | None
    warm_sample_seconds: tuple[float, ...]
    warm_sample_keys: tuple[str, ...]
    replay_validated: bool
    outputs_synchronized: tuple[str, ...]
    synchronized: bool
    signatures_compiled: int | None
    reset_between_rounds: bool
    compile_excluded_from_warm: bool
    phases: Mapping[str, float | None]

    def phase(self, name: str) -> float | None:
        return self.phases.get(name)

    def to_dict(self) -> dict[str, object]:
        return {
            "subject": self.subject,
            "boundary": self.boundary,
            "execution_mode": self.execution_mode,
            "executed_backend": self.executed_backend,
            "rounds": self.rounds,
            "warm_samples": self.warm_samples,
            # Serialized, not merely retained: a reordered or replaced raw
            # sample tuple has to change the report digest, or the summary is
            # bound and its evidence is not.
            "warm_sample_seconds": list(self.warm_sample_seconds),
            "warm_sample_keys": list(self.warm_sample_keys),
            "replay_validated": self.replay_validated,
            "outputs_synchronized": list(self.outputs_synchronized),
            "synchronized": self.synchronized,
            "signatures_compiled": self.signatures_compiled,
            "reset_between_rounds": self.reset_between_rounds,
            "compile_excluded_from_warm": self.compile_excluded_from_warm,
            "phases": {name: self.phases.get(name) for name in TIMING_PHASES},
        }


@dataclass(frozen=True)
class WorkloadResult:
    """A validated ``recast.forward-workload-result.v1`` document."""

    measurement_class: str
    workload_manifest_digest: str
    package_digests: Mapping[str, str]
    coverage: Mapping[str, tuple[str, ...]]
    comparisons: tuple[ComparisonObservation, ...]
    derivatives: tuple[DerivativeObservation, ...]
    timings: tuple[TimingObservation, ...]
    evidence: tuple[EvidenceSubject, ...]
    unsupported: tuple[str, ...]

    def timing(self, subject: str) -> TimingObservation | None:
        for item in self.timings:
            if item.subject == subject:
                return item
        return None


_RESULT_KEYS = frozenset(
    {
        "schema",
        "measurement_class",
        "workload_manifest_digest",
        "package_digests",
        "coverage",
        "comparisons",
        "derivatives",
        "timings",
        "evidence",
        "unsupported",
    }
)


def validate_workload_result(value: object, declaration: WorkloadDeclaration) -> WorkloadResult:
    """Validate the trusted plugin's measurements. Conclusions are not read.

    A malformed, nonfinite, negative or zero duration is refused rather than
    coerced, and a quantity the plugin did not measure must arrive as ``null``:
    the one thing a verification report must never contain is an invented
    number that reads like an observation.
    """

    document = _mapping(value, "forward workload result", _RESULT_KEYS)
    if document["schema"] != RESULT_SCHEMA:
        raise ProtocolError(f"forward workload result schema must be {RESULT_SCHEMA!r}")
    measurement_class = document["measurement_class"]
    if measurement_class not in MEASUREMENT_CLASSES:
        raise ProtocolError(
            f"forward workload measurement_class must be one of {sorted(MEASUREMENT_CLASSES)}"
        )
    manifest_digest = _result_digest(
        document["workload_manifest_digest"], "forward workload result manifest digest"
    )
    if manifest_digest != declaration.digest:
        raise ProtocolError("forward workload result names a different workload declaration")

    packages_value = document["package_digests"]
    if not isinstance(packages_value, Mapping) or set(packages_value) != set(SUBJECT_ROLES):
        raise ProtocolError("forward workload result must report every staged package digest")
    package_digests = {
        role: _result_digest(packages_value[role], f"forward workload {role} package digest")
        for role in SUBJECT_ROLES
    }

    coverage_value = document["coverage"]
    if not isinstance(coverage_value, Mapping) or set(coverage_value) != set(COVERAGE_DIMENSIONS):
        raise ProtocolError("forward workload result must report every coverage dimension")
    coverage = {
        dimension: _result_names(
            coverage_value[dimension], f"forward workload {dimension} coverage"
        )
        for dimension in COVERAGE_DIMENSIONS
    }

    comparisons = tuple(
        _validate_comparison(item, index)
        for index, item in enumerate(_result_sequence(document["comparisons"], "comparisons"))
    )
    if len({item.comparison for item in comparisons}) != len(comparisons):
        raise ProtocolError("forward workload result repeats a comparison subject")
    derivatives = tuple(
        _validate_derivative(item, index)
        for index, item in enumerate(_result_sequence(document["derivatives"], "derivatives"))
    )
    if len({item.mode for item in derivatives}) != len(derivatives):
        raise ProtocolError("forward workload result repeats a derivative mode")
    timings = tuple(
        _validate_timing(item, index, declaration)
        for index, item in enumerate(_result_sequence(document["timings"], "timings"))
    )
    if len({item.subject for item in timings}) != len(timings):
        raise ProtocolError("forward workload result repeats a timing subject")
    evidence = tuple(
        _validate_evidence(item, index)
        for index, item in enumerate(_result_sequence(document["evidence"], "evidence"))
    )
    if len(evidence) > MAX_EVIDENCE_SUBJECTS:
        raise ProtocolError("forward workload result declares too many evidence subjects")
    if len({item.label for item in evidence}) != len(evidence):
        raise ProtocolError("forward workload result repeats an evidence label")
    known = {item.digest for item in evidence}
    referenced = {item.evidence for item in comparisons if item.evidence is not None} | {
        item.evidence for item in derivatives if item.evidence is not None
    }
    dangling = sorted(referenced - known)
    if dangling:
        raise ProtocolError("forward workload result references unbound evidence subject(s)")

    unsupported = _result_names(
        document["unsupported"], "forward workload unsupported", maximum=MAX_UNSUPPORTED
    )
    return WorkloadResult(
        measurement_class=str(measurement_class),
        workload_manifest_digest=manifest_digest,
        package_digests=package_digests,
        coverage=coverage,
        comparisons=comparisons,
        derivatives=derivatives,
        timings=timings,
        evidence=evidence,
        unsupported=unsupported,
    )


def _result_sequence(value: object, context: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolError(f"forward workload result {context} must be an array")
    if len(value) > MAX_NAMES:
        raise ProtocolError(f"forward workload result {context} exceeds its bound")
    return value


def _result_digest(value: object, context: str) -> str:
    try:
        return _require_digest(_text(value, context, pattern=_TOKEN), context)
    except ConfigError as error:
        raise ProtocolError(str(error)) from error


def _result_names(value: object, context: str, *, maximum: int = MAX_NAMES) -> tuple[str, ...]:
    try:
        return _names(value, context, maximum=maximum)
    except ConfigError as error:
        raise ProtocolError(str(error)) from error


def _result_status(value: object, context: str) -> str:
    if value not in RESULT_STATUSES:
        raise ProtocolError(f"{context} must be one of {sorted(RESULT_STATUSES)}")
    return str(value)


def _result_detail(value: object, context: str) -> str:
    if not isinstance(value, str) or len(value.encode()) > MAX_DETAIL_BYTES:
        raise ProtocolError(f"{context} must be a bounded string")
    return value


def _result_count(value: object, context: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not (0 <= value <= 2**53):
        raise ProtocolError(f"{context} must be null or a nonnegative integer")
    return value


def _result_error(value: object, context: str) -> float | None:
    """A reported error magnitude: null, or a finite nonnegative float."""
    if value is None:
        return None
    if type(value) is not float or not math.isfinite(value) or value < 0.0:
        raise ProtocolError(f"{context} must be null or a finite nonnegative float")
    return value


def _result_evidence(value: object, context: str) -> str | None:
    return None if value is None else _result_digest(value, context)


def _validate_comparison(value: object, index: int) -> ComparisonObservation:
    context = f"forward workload comparison[{index}]"
    document = _mapping(
        value,
        context,
        frozenset(
            {
                "comparison",
                "status",
                "checked_outputs",
                "observed_key_count",
                "observed_keys_digest",
                "failed_outputs",
                "max_ulp",
                "max_absolute_error",
                "max_relative_error",
                "nonfinite_mismatches",
                "discrete_mismatches",
                "evidence",
                "detail",
            }
        ),
    )
    subject = document["comparison"]
    if subject not in COMPARISON_SUBJECTS:
        raise ProtocolError(f"{context} names an unknown comparison subject")
    status = _result_status(document["status"], f"{context}.status")
    checked = _result_count(document["checked_outputs"], f"{context}.checked_outputs")
    if checked is None:
        raise ProtocolError(f"{context}.checked_outputs must be an integer")
    failed = tuple(_result_names(document["failed_outputs"], f"{context}.failed_outputs"))
    raw_ulp = document["max_ulp"]
    max_ulp = None
    if raw_ulp is not None:
        if not isinstance(raw_ulp, str) or not _ULP.fullmatch(raw_ulp):
            # A decimal string, not a float: a 2**53-exceeding ULP distance
            # must survive the report without being rounded into a lie.
            raise ProtocolError(f"{context}.max_ulp must be null or a decimal integer string")
        max_ulp = raw_ulp
    observed_count = _result_count(document["observed_key_count"], f"{context}.observed_key_count")
    if observed_count is None:
        raise ProtocolError(f"{context}.observed_key_count must be an integer")
    observed_digest = (
        None
        if document["observed_keys_digest"] is None
        else _result_digest(document["observed_keys_digest"], f"{context}.observed_keys_digest")
    )
    if status == "passed":
        if checked == 0:
            raise ProtocolError(f"{context} passed without comparing any output")
        if failed:
            raise ProtocolError(f"{context} passed while naming failed outputs")
        if observed_digest is None:
            raise ProtocolError(f"{context} passed without identifying what it observed")
    return ComparisonObservation(
        comparison=str(subject),
        status=status,
        checked_outputs=checked,
        observed_key_count=observed_count,
        observed_keys_digest=observed_digest,
        failed_outputs=failed,
        max_ulp=max_ulp,
        max_absolute_error=_result_error(
            document["max_absolute_error"], f"{context}.max_absolute_error"
        ),
        max_relative_error=_result_error(
            document["max_relative_error"], f"{context}.max_relative_error"
        ),
        nonfinite_mismatches=_result_count(
            document["nonfinite_mismatches"], f"{context}.nonfinite_mismatches"
        ),
        discrete_mismatches=_result_count(
            document["discrete_mismatches"], f"{context}.discrete_mismatches"
        ),
        evidence=_result_evidence(document["evidence"], f"{context}.evidence"),
        detail=_result_detail(document["detail"], f"{context}.detail"),
    )


def _validate_derivative(value: object, index: int) -> DerivativeObservation:
    context = f"forward workload derivative[{index}]"
    document = _mapping(
        value,
        context,
        frozenset(
            {
                "mode",
                "status",
                "scope",
                "covered_pairs",
                "slots",
                "smooth",
                "kink",
                "unresolved",
                "informative",
                "zero_sensitivity",
                "max_relative_error",
                "evidence",
                "detail",
            }
        ),
    )
    status = _result_status(document["status"], f"{context}.status")
    slots = _result_names(document["slots"], f"{context}.slots") if document["slots"] else ()
    if status == "passed" and not slots:
        raise ProtocolError(f"{context} passed without naming a differentiated slot")
    covered = (
        _result_names(document["covered_pairs"], f"{context}.covered_pairs")
        if document["covered_pairs"]
        else ()
    )
    if status == "passed" and not covered:
        raise ProtocolError(f"{context} passed without naming the pair(s) it covered")
    return DerivativeObservation(
        mode=_result_names([document["mode"]], f"{context}.mode")[0],
        status=status,
        scope=_text(document["scope"], f"{context}.scope", pattern=_TOKEN),
        covered_pairs=covered,
        slots=slots,
        smooth=_result_count(document["smooth"], f"{context}.smooth"),
        kink=_result_count(document["kink"], f"{context}.kink"),
        unresolved=_result_count(document["unresolved"], f"{context}.unresolved"),
        informative=_result_count(document["informative"], f"{context}.informative"),
        zero_sensitivity=_result_count(document["zero_sensitivity"], f"{context}.zero_sensitivity"),
        max_relative_error=_result_error(
            document["max_relative_error"], f"{context}.max_relative_error"
        ),
        evidence=_result_evidence(document["evidence"], f"{context}.evidence"),
        detail=_result_detail(document["detail"], f"{context}.detail"),
    )


def _validate_timing(
    value: object, index: int, declaration: WorkloadDeclaration
) -> TimingObservation:
    context = f"forward workload timing[{index}]"
    document = _mapping(
        value,
        context,
        frozenset(
            {
                "subject",
                "boundary",
                "execution_mode",
                "executed_backend",
                "rounds",
                "warm_samples",
                "warm_sample_seconds",
                "warm_sample_keys",
                "replay_validated",
                "outputs_synchronized",
                "synchronized",
                "signatures_compiled",
                "reset_between_rounds",
                "compile_excluded_from_warm",
                "phases",
            }
        ),
    )
    subject = document["subject"]
    if subject not in TIMED_ROLES:
        raise ProtocolError(f"{context} names an unknown timing subject")
    boundary = _text(document["boundary"], f"{context}.boundary", pattern=_TOKEN)
    if boundary != declaration.measurement.boundary:
        raise ProtocolError(f"{context} was measured at a boundary the declaration did not fix")
    mode = document["execution_mode"]
    declared = declaration.measurement.mode(str(subject))
    if mode not in EXECUTION_MODES:
        raise ProtocolError(f"{context}.execution_mode must be one of {sorted(EXECUTION_MODES)}")
    if mode != declared:
        # The declaration decides which subject runs compiled. A plugin that
        # could choose would simply declare itself interpreted and escape
        # every compilation requirement.
        raise ProtocolError(
            f"{context} claims mode {mode!r} where the declaration fixed {declared!r}"
        )

    phases_value = document["phases"]
    if not isinstance(phases_value, Mapping) or set(phases_value) != set(TIMING_PHASES):
        raise ProtocolError(
            f"{context}.phases must report every boundary, using null when unmeasured"
        )
    phases: dict[str, float | None] = {}
    for name in TIMING_PHASES:
        raw = phases_value[name]
        if raw is None:
            phases[name] = None
            continue
        if type(raw) is not float or not math.isfinite(raw):
            raise ProtocolError(f"{context}.phases.{name} must be null or a finite float")
        if name in _NONNEGATIVE_PHASES:
            if raw < 0.0:
                raise ProtocolError(f"{context}.phases.{name} must not be negative")
        elif raw <= 0.0:
            raise ProtocolError(f"{context}.phases.{name} must be a positive duration")
        phases[name] = raw
    if mode == "interpreted":
        present = sorted(name for name in _COMPILED_ONLY_PHASES if phases[name] is not None)
        if present:
            raise ProtocolError(f"{context} is interpreted and cannot have measured {present}")
    whole, warm = phases["whole_workload"], phases["warm_median"]
    if whole is not None and warm is not None and whole < warm:
        raise ProtocolError(f"{context} reports a whole workload shorter than one warm sample")

    rounds = _result_count(document["rounds"], f"{context}.rounds")
    if rounds == 0:
        raise ProtocolError(f"{context}.rounds must be null or positive")
    samples = _result_count(document["warm_samples"], f"{context}.warm_samples")
    if samples == 0:
        raise ProtocolError(f"{context}.warm_samples must be null or positive")
    ordered = _warm_samples(document["warm_sample_seconds"], context)
    if samples is None and warm is not None:
        raise ProtocolError(f"{context} reports a warm median without a sample count")
    if (samples or 0) != len(ordered):
        raise ProtocolError(
            f"{context} reports {samples} warm sample(s) and {len(ordered)} measurement(s)"
        )
    if ordered:
        _require_summary(ordered, warm, phases["warm_mad"], context)
    elif warm is not None or phases["warm_mad"] is not None:
        raise ProtocolError(f"{context} summarizes warm samples it did not report")

    compiled = _result_count(document["signatures_compiled"], f"{context}.signatures_compiled")
    if mode == "interpreted" and compiled not in (None, 0):
        raise ProtocolError(f"{context} is interpreted and cannot have compiled a signature")
    keys = (
        _result_names(
            document["warm_sample_keys"], f"{context}.warm_sample_keys", maximum=MAX_WARM_SAMPLES
        )
        if document["warm_sample_keys"]
        else ()
    )
    if len(keys) != len(ordered):
        raise ProtocolError(
            f"{context} reports {len(ordered)} warm sample(s) and {len(keys)} sample identities"
        )
    return TimingObservation(
        subject=str(subject),
        boundary=boundary,
        execution_mode=str(mode),
        executed_backend=_text(
            document["executed_backend"], f"{context}.executed_backend", pattern=_TOKEN
        ),
        rounds=rounds,
        warm_samples=samples,
        warm_sample_seconds=ordered,
        warm_sample_keys=keys,
        replay_validated=_flag(document["replay_validated"], f"{context}.replay_validated"),
        outputs_synchronized=_result_names(
            document["outputs_synchronized"], f"{context}.outputs_synchronized"
        )
        if document["outputs_synchronized"]
        else (),
        synchronized=_flag(document["synchronized"], f"{context}.synchronized"),
        signatures_compiled=compiled,
        reset_between_rounds=_flag(
            document["reset_between_rounds"], f"{context}.reset_between_rounds"
        ),
        compile_excluded_from_warm=_flag(
            document["compile_excluded_from_warm"], f"{context}.compile_excluded_from_warm"
        ),
        phases=MappingProxyType(phases),
    )


def _warm_samples(value: object, context: str) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ProtocolError(f"{context}.warm_sample_seconds must be an array")
    if len(value) > MAX_WARM_SAMPLES:
        raise ProtocolError(f"{context}.warm_sample_seconds exceeds its bound")
    out: list[float] = []
    for item in value:
        if type(item) is not float or not math.isfinite(item) or item <= 0.0:
            raise ProtocolError(
                f"{context}.warm_sample_seconds holds a value that is not a positive duration"
            )
        out.append(item)
    return tuple(out)


def _require_summary(
    samples: tuple[float, ...], median: float | None, deviation: float | None, context: str
) -> None:
    """Recompute the reported summary from the samples the plugin supplied.

    A summary that does not follow from its own measurements is the easiest
    place to hide a speedup, so the median and the median absolute deviation
    are recomputed here rather than read.
    """

    if median is None or deviation is None:
        raise ProtocolError(f"{context} reported warm samples without summarizing them")
    observed = statistics.median(samples)
    spread = statistics.median([abs(item - observed) for item in samples])
    if not math.isclose(observed, median, rel_tol=1e-9, abs_tol=0.0):
        raise ProtocolError(
            f"{context} reports a warm median of {median!r}; its samples give {observed!r}"
        )
    if not math.isclose(spread, deviation, rel_tol=1e-9, abs_tol=1e-18):
        raise ProtocolError(
            f"{context} reports a deviation of {deviation!r}; its samples give {spread!r}"
        )


def _validate_evidence(value: object, index: int) -> EvidenceSubject:
    context = f"forward workload evidence[{index}]"
    document = _mapping(value, context, frozenset({"label", "digest", "media_type", "size"}))
    media_type = document["media_type"]
    if not isinstance(media_type, str) or not _MEDIA_TYPE.fullmatch(media_type):
        raise ProtocolError(f"{context}.media_type is not a well-formed media type")
    size = _result_count(document["size"], f"{context}.size")
    if size is None or size == 0:
        raise ProtocolError(f"{context}.size must be a positive byte count")
    return EvidenceSubject(
        label=_result_names([document["label"]], f"{context}.label")[0],
        digest=_result_digest(document["digest"], f"{context}.digest"),
        media_type=media_type,
        size=size,
    )


# --- conclusions, recomputed here -------------------------------------------


@dataclass(frozen=True)
class CoverageResult:
    """Required against observed coverage for one dimension."""

    dimension: str
    required: tuple[str, ...]
    observed: tuple[str, ...]
    missing: tuple[str, ...]
    unsupported: tuple[str, ...]
    accepted: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dimension": self.dimension,
            "required": list(self.required),
            "observed": list(self.observed),
            "missing": list(self.missing),
            "unsupported": list(self.unsupported),
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class SpeedConclusion:
    """One speed objective, decided here from validated measurements."""

    objective: str
    reference: str
    status: str
    threshold: float
    noise_factor: float
    reference_median_seconds: float | None
    candidate_median_seconds: float | None
    observed_ratio: float | None
    observed_change: float | None
    noise_margin: float | None
    detail: str

    @property
    def key(self) -> str:
        return f"{self.objective}:{self.reference}"

    def to_dict(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "reference": self.reference,
            "status": self.status,
            "threshold": self.threshold,
            "noise_factor": self.noise_factor,
            "reference_median_seconds": self.reference_median_seconds,
            "candidate_median_seconds": self.candidate_median_seconds,
            "observed_ratio": self.observed_ratio,
            "observed_change": self.observed_change,
            "noise_margin": self.noise_margin,
            "detail": self.detail,
        }


def evaluate_coverage(
    declaration: WorkloadDeclaration, result: WorkloadResult
) -> tuple[CoverageResult, ...]:
    """Compare required coverage against what the plugin says it covered.

    Empty observed coverage never accepts. A dimension the declaration
    requires and the run did not reach is ``missing``; one the plugin declared
    it cannot do is ``unsupported``. Neither is a pass.
    """

    required: dict[str, tuple[str, ...]] = {
        "case": declaration.fixtures.coverage_labels,
        "signature": declaration.interface.static_signatures,
        "mode": declaration.acceptance.required_modes,
        "output": declaration.acceptance.required_outputs,
    }
    unsupported = set(result.unsupported)
    out: list[CoverageResult] = []
    for dimension in COVERAGE_DIMENSIONS:
        want = required[dimension]
        observed = result.coverage[dimension]
        missing = tuple(sorted(set(want) - set(observed)))
        blocked = tuple(sorted(set(want) & unsupported))
        accepted = not missing and not blocked and (not want or bool(observed))
        out.append(
            CoverageResult(
                dimension=dimension,
                required=want,
                observed=observed,
                missing=missing,
                unsupported=blocked,
                accepted=accepted,
            )
        )
    return tuple(out)


def evaluate_numerics(
    declaration: WorkloadDeclaration, result: WorkloadResult
) -> tuple[bool, tuple[str, ...]]:
    """Decide the numerical objective from the required comparison set.

    Each required comparison is judged on its own, and on its *complete*
    observation-key set rather than on a count. That is the whole point of
    keeping the original anchor in the required set: a candidate whose drift
    is tolerable against its immediate predecessor and intolerable against the
    anchor fails here, at the leaf, rather than accumulating.
    """

    observed = {item.comparison: item for item in result.comparisons}
    expected_digest = declaration.fixtures.observation_keys_digest
    expected_count = declaration.fixtures.observation_key_count
    reasons: list[str] = []
    for subject in declaration.acceptance.required_comparisons:
        item = observed.get(subject)
        if item is None:
            reasons.append(f"numerical.{subject}.missing")
            continue
        if item.status != "passed":
            reasons.append(f"numerical.{subject}.{item.status}")
            continue
        if (
            item.observed_keys_digest != expected_digest
            or item.observed_key_count != expected_count
        ):
            # Empty, duplicated or selectively omitted observations all land
            # here: the digest is over the canonical required combinations.
            reasons.append(f"numerical.{subject}.incomplete_observations")
    return not reasons, tuple(sorted(set(reasons)))


def evaluate_derivatives(
    declaration: WorkloadDeclaration, result: WorkloadResult
) -> tuple[bool, tuple[str, ...]]:
    """Decide the derivative objective, including slot coverage.

    A candidate that got faster by replacing a bounded masked scan with a
    non-transposable loop reports its reverse mode as ``unsupported`` or drops
    the slot; both are unmet requirements here, not a partial success.
    """

    observed = {item.mode: item for item in result.derivatives}
    required_slots = set(declaration.acceptance.differentiable_slots)
    policy = declaration.acceptance.finite_difference
    required_pairs = {
        f"{signature}|{state}"
        for signature in declaration.interface.static_signatures
        for state in declaration.fixtures.states
    }
    reasons: list[str] = []
    for mode in declaration.acceptance.required_modes:
        item = observed.get(mode)
        if item is None:
            reasons.append(f"derivative.{mode}.missing")
            continue
        if item.status != "passed":
            reasons.append(f"derivative.{mode}.{item.status}")
            continue
        if required_slots - set(item.slots):
            reasons.append(f"derivative.{mode}.slot_coverage")
        if required_pairs - set(item.covered_pairs):
            # A mode checked on whichever signature and state sorted first is
            # not a mode checked on the declared coverage.
            reasons.append(f"derivative.{mode}.scope_coverage")
        if mode in policy.classified_modes and any(
            count is None for count in (item.smooth, item.kink, item.unresolved)
        ):
            # A required unresolved or kink check cannot vanish into a null.
            reasons.append(f"derivative.{mode}.classification")
        if mode in policy.classified_modes and not item.informative:
            # Agreement between two zeros is not evidence of a derivative.
            reasons.append(f"derivative.{mode}.uninformative")
    return not reasons, tuple(sorted(set(reasons)))


def evaluate_speed(
    declaration: WorkloadDeclaration, result: WorkloadResult
) -> tuple[SpeedConclusion, ...]:
    """Recompute every declared speed objective from validated medians.

    The plugin never states whether an objective was met. Thresholds come from
    the declaration, which was frozen before the candidate ran, and a change
    inside the declared noise margin is not an improvement.
    """

    candidate = result.timing("candidate")
    out: list[SpeedConclusion] = []
    for requirement in declaration.acceptance.speed:
        reference = result.timing(requirement.reference)
        out.append(_speed_conclusion(requirement, candidate, reference))
    return tuple(out)


def _speed_conclusion(
    requirement: SpeedRequirement,
    candidate: TimingObservation | None,
    reference: TimingObservation | None,
) -> SpeedConclusion:
    def missing(detail: str) -> SpeedConclusion:
        return SpeedConclusion(
            objective=requirement.objective,
            reference=requirement.reference,
            status="missing",
            threshold=requirement.threshold,
            noise_factor=requirement.noise_factor,
            reference_median_seconds=None,
            candidate_median_seconds=None,
            observed_ratio=None,
            observed_change=None,
            noise_margin=None,
            detail=detail,
        )

    if candidate is None or reference is None:
        return missing("a required timing subject was not measured")
    after = candidate.phase("warm_median")
    before = reference.phase("warm_median")
    after_mad = candidate.phase("warm_mad")
    before_mad = reference.phase("warm_mad")
    if after is None or before is None:
        return missing("a required warm median was not measured")
    if after_mad is None or before_mad is None:
        return missing("a required warm deviation was not measured")
    margin = requirement.noise_factor * max(after_mad, before_mad)
    if requirement.objective == "reduction":
        change = (before - after) / before
        met = change >= requirement.threshold and (before - after) > margin
        detail = (
            f"warm median fell {change:.4f} against {requirement.reference}; "
            f"required {requirement.threshold:.4f} outside a {margin:.6g}s margin"
        )
    else:
        change = (after - before) / before
        met = change <= requirement.threshold or (after - before) <= margin
        detail = (
            f"warm median rose {change:.4f} against {requirement.reference}; "
            f"ceiling {requirement.threshold:.4f} or a {margin:.6g}s margin"
        )
    return SpeedConclusion(
        objective=requirement.objective,
        reference=requirement.reference,
        status="met" if met else "unmet",
        threshold=requirement.threshold,
        noise_factor=requirement.noise_factor,
        reference_median_seconds=before,
        candidate_median_seconds=after,
        observed_ratio=before / after,
        observed_change=change,
        noise_margin=margin,
        detail=detail,
    )


def evaluate_timing_requirements(
    declaration: WorkloadDeclaration, result: WorkloadResult
) -> tuple[str, ...]:
    """Reason codes for a timing set that does not meet the declared contract.

    Applicability comes from the declaration's per-subject execution mode, so
    an interpreted anchor is not charged for compilation phases it cannot
    have, and a compiled subject cannot escape them by omission.
    """

    reasons: list[str] = []
    measurement = declaration.measurement
    for subject in SUBJECT_ROLES:
        required = measurement.required(subject)
        timing = result.timing(subject)
        if timing is None:
            if required:
                reasons.append(f"timing.{subject}.missing")
            continue
        for phase in required:
            if timing.phase(phase) is None:
                reasons.append(f"timing.{subject}.missing_phase.{phase}")
        if measurement.compile_excluded_from_warm and not timing.compile_excluded_from_warm:
            # A warm sample that still contains tracing, lowering or
            # compilation is the cheapest way to manufacture a speedup.
            reasons.append(f"timing.{subject}.compile_in_warm")
        if not timing.synchronized:
            reasons.append(f"timing.{subject}.not_synchronized")
        if timing.executed_backend != measurement.backend:
            # The declared backend is the platform the timing claim is only
            # meaningful on. It was previously retained as requested
            # authority and never compared with anything, so a measurement
            # taken on a different backend was admitted under the declared
            # one's name.
            reasons.append(f"timing.{subject}.backend")
        if timing.execution_mode == "compiled":
            expected = len(declaration.interface.static_signatures)
            if timing.signatures_compiled != expected:
                # Every declared static signature has to be compiled before a
                # warm measurement, or one of them is paying for a trace.
                reasons.append(f"timing.{subject}.signatures_compiled")
        if timing.rounds != measurement.rounds:
            # The declared round count is what had to run, not a value to
            # echo: a schedule that ignored it reports what it really did.
            reasons.append(f"timing.{subject}.rounds")
        if not timing.replay_validated:
            reasons.append(f"timing.{subject}.replay_not_validated")
        if set(declaration.interface.outputs) - set(timing.outputs_synchronized):
            # A diagnostic that is never fed back is still part of the
            # workload; stopping the clock before it lands measures less.
            reasons.append(f"timing.{subject}.incomplete_synchronization")
        warm = timing.phase("warm_median")
        if warm is not None and (
            timing.warm_samples is None or timing.warm_samples < measurement.warm_repetitions
        ):
            reasons.append(f"timing.{subject}.warm_repetitions")
        if declaration.boundary.reset != "never" and not timing.reset_between_rounds:
            reasons.append(f"timing.{subject}.state_not_reset")
    return tuple(sorted(set(reasons)))


@dataclass(frozen=True)
class ForwardMetrics:
    """The small projection a controller or UI can read without the detail.

    Everything here is either a validated measurement or ``None``. It is a
    view of the report, never a substitute for its coverage: a passing
    projection with an unaccepted coverage row is still an unaccepted report.
    """

    steps: int
    states: int
    compared_outputs: int | None
    max_ulp: str | None
    max_relative_error: float | None
    anchor_status: str
    predecessor_status: str
    warm_median_seconds: float | None
    compile_seconds: float | None
    whole_workload_seconds: float | None
    speedup_against_original_translation: float | None
    speedup_against_predecessor: float | None
    required_modes_met: int
    required_modes_total: int
    evidence_subjects: int

    def to_dict(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "states": self.states,
            "compared_outputs": self.compared_outputs,
            "max_ulp": self.max_ulp,
            "max_relative_error": self.max_relative_error,
            "anchor_status": self.anchor_status,
            "predecessor_status": self.predecessor_status,
            "warm_median_seconds": self.warm_median_seconds,
            "compile_seconds": self.compile_seconds,
            "whole_workload_seconds": self.whole_workload_seconds,
            "speedup_against_original_translation": self.speedup_against_original_translation,
            "speedup_against_predecessor": self.speedup_against_predecessor,
            "required_modes_met": self.required_modes_met,
            "required_modes_total": self.required_modes_total,
            "evidence_subjects": self.evidence_subjects,
        }


def project_metrics(
    declaration: WorkloadDeclaration,
    result: WorkloadResult,
    speed: tuple[SpeedConclusion, ...],
) -> ForwardMetrics:
    """Reduce the validated result to the projection a backend can carry."""

    comparisons = {item.comparison: item for item in result.comparisons}
    required = [comparisons.get(name) for name in declaration.acceptance.required_comparisons]
    present = [item for item in required if item is not None]
    ulps = [int(item.max_ulp) for item in present if item.max_ulp is not None]
    errors = [item.max_relative_error for item in present if item.max_relative_error is not None]
    checked = [item.checked_outputs for item in present]
    candidate = result.timing("candidate")
    ratios = {item.reference: item.observed_ratio for item in speed}
    modes = {item.mode: item for item in result.derivatives}
    met = sum(
        1
        for mode in declaration.acceptance.required_modes
        if mode in modes and modes[mode].status == "passed"
    )
    return ForwardMetrics(
        steps=declaration.fixtures.steps,
        states=len(declaration.fixtures.states),
        compared_outputs=min(checked) if checked else None,
        max_ulp=str(max(ulps)) if ulps else None,
        max_relative_error=max(errors) if errors else None,
        anchor_status=(
            comparisons[ANCHOR_ROLE].status if ANCHOR_ROLE in comparisons else "missing"
        ),
        predecessor_status=(
            comparisons["predecessor"].status if "predecessor" in comparisons else "missing"
        ),
        warm_median_seconds=candidate.phase("warm_median") if candidate else None,
        compile_seconds=candidate.phase("compile") if candidate else None,
        whole_workload_seconds=candidate.phase("whole_workload") if candidate else None,
        speedup_against_original_translation=ratios.get("original_translation"),
        speedup_against_predecessor=ratios.get("predecessor"),
        required_modes_met=met,
        required_modes_total=len(declaration.acceptance.required_modes),
        evidence_subjects=len(result.evidence),
    )
