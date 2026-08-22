"""What an author declares so the conformance checks can reach their plugins.

Two things share the word. The ``conformance/`` directory at the repository
root holds the *checks* -- a pytest suite. This module holds the *declaration*:
the small amount an author has to say about their plugins before those checks
can run against them. It ships inside the installed package, because the
declaration lives in the author's own repository and has to be importable
there; the suite is run from a checkout or an sdist of the engine.

Most kinds cannot be checked from the plugin alone. Asking whether a Verifier
fails closed means handing it a candidate, an oracle and a workspace, and only
the author knows what a valid one looks like for their plugin. So a ``PluginSet``
is a list of cases, one per plugin, each carrying the least material its checks
need -- and no more, because everything declared here is a thing the author has
to keep working.

A set is named to the suite three ways, tried in this order:

    --plugin-set recast                     the engine's own, always available
    --plugin-set <entry-point-name>         from the ``recast.conformance`` group
    --plugin-set yourpkg.conformance:SET    a dotted path, for local development

Declaring the entry point is what makes the first form work for an installed
extension::

    [project.entry-points."recast.conformance"]
    your-set = "yourpkg.conformance:PLUGIN_SET"
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module
from importlib.metadata import entry_points
from pathlib import Path
from typing import Any

from recast.errors import ConfigError
from recast.model import Candidate, Confidence, Facts, OracleRef, Unit
from recast.plugins.executor import Executor, Job
from recast.plugins.frontend import Frontend
from recast.plugins.oracle import Oracle
from recast.plugins.recipe import Recipe
from recast.plugins.scanner import Scanner
from recast.plugins.store import EvidenceStore, FindingStore
from recast.plugins.transform import Transform
from recast.plugins.verifier import Verifier

__all__ = [
    "ENTRY_POINT_GROUP",
    "EvidenceStoreCase",
    "ExecutorCase",
    "FindingStoreCase",
    "FrontendCase",
    "OracleCase",
    "PluginSet",
    "RecipeCase",
    "TransformCase",
    "TransformSubject",
    "VerifierCase",
    "load_plugin_set",
]

ENTRY_POINT_GROUP = "recast.conformance"


@dataclass(frozen=True)
class ExecutorCase:
    """One Executor to check.

    ``build`` defaults to the registered factory, so an installed executor
    needs only its name. ``probe`` is a job the executor can genuinely run --
    the default runs this interpreter, which is available wherever the suite
    is. ``unsatisfiable`` is the request this executor must refuse; the
    default is more than one node and more than one rank, which is what
    ``local`` cannot honestly deliver. A batch executor overrides it with
    something *it* cannot deliver, because an executor that refuses nothing is
    the failure mode this check exists for.
    """

    name: str
    build: Callable[[], Executor] | None = None
    probe: Callable[[Path], Job] | None = None
    unsatisfiable: Mapping[str, Any] = field(
        default_factory=lambda: {"nodes": 4, "ranks": 512, "queue": "conformance"}
    )


@dataclass(frozen=True)
class EvidenceStoreCase:
    """One EvidenceStore to check.

    ``build`` is handed a scratch directory the suite owns, so a check may
    inspect what landed in it. ``read_manifest`` turns a URI returned by
    ``put`` back into the document that was written; without it the manifest
    cannot be validated and the suite says so rather than passing quietly.
    """

    name: str
    build: Callable[[Path], EvidenceStore]
    read_manifest: Callable[[EvidenceStore, str], dict[str, Any]] | None = None


@dataclass(frozen=True)
class FindingStoreCase:
    """One FindingStore to check. ``build`` receives a scratch directory."""

    name: str
    build: Callable[[Path], FindingStore]


@dataclass(frozen=True)
class ScannerCase:
    """One Scanner to check.

    ``tool`` names the external binary it wraps, when it wraps one, and
    declaring it is what makes the interesting checks possible: the suite puts
    a fake of that name on PATH, and takes it away again, to see whether the
    plugin can tell "not installed" and "installed but producing garbage" apart
    from "clean". Both are the same confusion the ``Scanner`` contract exists
    to prevent, and neither is visible in a return value.

    A scanner that wraps nothing leaves ``tool`` None, and those checks skip by
    name rather than passing -- which is the arrangement that makes them useful
    on the day one does wrap something.

    ``tool`` defaults to the plugin's own declaration. ``fakes`` is for a
    scanner whose tool does not write SARIF to a report path -- grype answers
    in its own JSON on stdout, syft writes an SBOM -- and is called with a
    directory to populate and one of ``"clean"``, ``"garbage"``, ``"one"``:
    fakes that scan clean, that emit unparseable output, and that report
    exactly one result. Absent, the suite writes a SARIF-speaking fake of
    ``tool`` itself.
    """

    name: str
    build: Callable[[], Scanner] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    tool: str | tuple[str, ...] | None = None
    fakes: Callable[[Path, str], None] | None = None


@dataclass(frozen=True)
class RecipeCase:
    """One Recipe to check, with a config it accepts.

    The config has to be one ``validate`` passes: a recipe's stage list may
    branch on config, and checking the plan it produces for a config it would
    refuse checks a plan that can never run.
    """

    name: str
    build: Callable[[], Recipe] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class FrontendCase:
    """One Frontend to check, and a source tree to point it at.

    ``plant_tree`` is handed a scratch directory the suite owns and fills it
    with source. It has to be a copy rather than the real thing, because two of
    the checks are about what the frontend leaves behind and one of them writes
    into the tree on purpose -- neither is safe against a directory somebody
    is working in.

    ``plant_workspace_artifact`` puts a file the frontend *would* discover
    inside the engine's own workspace directory, and the check requires it not
    to be discovered there. Only the case can write that file, because what
    counts as discoverable is exactly the language knowledge the frontend has
    and the suite does not.
    """

    name: str
    plant_tree: Callable[[Path], None]
    build: Callable[[], Frontend] | None = None
    expect_uids: tuple[str, ...] = ()
    """Units that must turn up. A frontend that discovers nothing is
    deterministic, side-effect free, and useless."""

    preprocesses: bool = False
    """True if ``preprocess`` is overridden -- then it must record its flags."""

    plant_workspace_artifact: Callable[[Path], None] | None = None
    requires: tuple[str, ...] = ()
    requires_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class TransformSubject:
    """One thing to transform: the unit, its Facts, and where its source is.

    ``config`` is per-invocation rather than per-case because a Transform that
    reads source resolves it against ``config["root"]``, and the suite plants
    each subject in a scratch directory it makes up at call time. The case's own
    ``config`` is merged underneath this one.
    """

    unit: Unit
    facts: Facts
    config: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TransformCase:
    """One Transform to check, with a subject it handles and one it does not.

    ``subject`` returns a unit this Transform should translate, with the Facts
    a Frontend produced for it. ``defers`` returns one carrying a site the rules
    cannot handle: a partial Candidate with a populated ``deferred`` list is a
    normal, useful result, and the check is that the Transform produces one
    rather than raising. Leave ``defers`` unset only if nothing can defeat the
    rules, which is a claim worth being sure about.
    """

    name: str
    subject: Callable[[Path], TransformSubject]
    build: Callable[[], Transform] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    defers: Callable[[Path], TransformSubject] | None = None
    requires: tuple[str, ...] = ()
    requires_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class OracleCase:
    """One Oracle to check, and what its cache key is supposed to notice.

    ``key`` is the field with teeth. Two calls under one key must be
    behaviourally identical, so a key that fails to move when the reference
    moves does not produce a wrong answer -- it produces a stale right one,
    silently, for every Verdict downstream of it. The suite cannot guess what
    moves a given oracle's reference: ``fflags`` for a compiled one, rank count
    for a pinned run, a dump's revision for a replay. So ``moves_the_key`` is
    the author's own list, label to config overlay, and each entry must produce
    a different key than ``config`` does.

    ``move_the_source`` is the same claim about the thing being referenced
    rather than about how it is built, expressed as a change to ``Facts``.

    ``materializes`` opts into the checks that actually build: that a refusing
    executor produces a ``RecastError`` rather than an exception the runner does
    not catch, and that ``release`` can be called twice. Leave it off when
    building is not affordable where the suite runs, and say so in ``requires``
    or ``requires_commands`` when it depends on a toolchain, so the skip names
    the reason.
    """

    name: str
    unit: Unit
    facts: Callable[[], Facts]
    build: Callable[[], Oracle] | None = None
    config: Mapping[str, Any] = field(default_factory=dict)
    moves_the_key: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    move_the_source: Callable[[Facts], Facts] | None = None
    materializes: bool = False
    submits_jobs: bool = True
    """Whether materializing leaves this process through the executor.

    True for anything that builds -- a compiled reference, a pinned run -- and
    that is the default because those are the expensive ones the rule is
    about. A cheap oracle that derives its reference in-process submits
    nothing, so there is no executor for it to route around, and the check
    skips by name rather than passing.
    """

    executor: Callable[[], Executor] | None = None
    requires: tuple[str, ...] = ()
    requires_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerifierCase:
    """One Verifier to check, with a candidate it can actually judge.

    This is the kind that costs the most to declare, and the reason is the
    first check: before the suite can ask whether a broken candidate fails, it
    has to see a good one pass. Otherwise a Verifier that returns ``FAILED``
    unconditionally satisfies every other row in the table, and the gate that
    never passes anything is as useless as the gate that never fails anything
    -- it just fails in a direction nobody complains about.

    So ``candidate`` builds an artifact this Verifier should accept, and
    ``break_candidate`` returns one it must not. Break the *artifact*, not the
    bookkeeping: corrupting the notes that describe the candidate checks that
    the Verifier reads its own protocol, which is not the same claim.

    ``oracle`` is what the comparison is against, and ``None`` says this
    Verifier needs none -- the ``StaticVerifier`` case. ``submits_jobs`` says
    whether the comparison leaves this process through the executor; when it
    does, the suite hands it one that refuses everything and requires
    ``FAILED`` rather than a result. ``requires`` names modules the case
    itself needs, so a case that cannot run is skipped by name instead of
    failing for the wrong reason.
    """

    name: str
    candidate: Callable[[Path], Candidate]
    break_candidate: Callable[[Candidate], Candidate]
    build: Callable[[], Verifier] | None = None
    unit: Unit | None = None
    oracle: Callable[[Path, Executor], OracleRef] | None = None
    executor: Callable[[], Executor] | None = None
    submits_jobs: bool = False
    config: Mapping[str, Any] = field(default_factory=dict)
    expect: Confidence | None = None
    requires: tuple[str, ...] = ()
    requires_commands: tuple[str, ...] = ()


@dataclass(frozen=True)
class PluginSet:
    """Everything one distribution offers, as far as conformance can see it.

    A kind left empty is not a pass. The suite reports it as unexercised, and
    an author publishing against a partly-declared set is publishing against
    partly-run checks.
    """

    name: str
    executors: tuple[ExecutorCase, ...] = ()
    frontends: tuple[FrontendCase, ...] = ()
    transforms: tuple[TransformCase, ...] = ()
    oracles: tuple[OracleCase, ...] = ()
    verifiers: tuple[VerifierCase, ...] = ()
    evidence_stores: tuple[EvidenceStoreCase, ...] = ()
    finding_stores: tuple[FindingStoreCase, ...] = ()
    scanners: tuple[ScannerCase, ...] = ()
    recipes: tuple[RecipeCase, ...] = ()


def load_plugin_set(name: str) -> PluginSet:
    """Resolve ``--plugin-set`` to a declaration. See the module docstring."""
    if name == "recast":
        from recast.conformance.builtin import PLUGIN_SET

        return PLUGIN_SET

    for ep in entry_points(group=ENTRY_POINT_GROUP):
        if ep.name == name:
            return _as_plugin_set(ep.load(), f"entry point {name!r}")

    if ":" in name:
        module_name, _, attribute = name.partition(":")
        try:
            module = import_module(module_name)
        except ImportError as exc:
            raise ConfigError(
                f"plugin set {name!r}: cannot import {module_name!r} ({exc})"
            ) from exc
        try:
            return _as_plugin_set(getattr(module, attribute), f"{name!r}")
        except AttributeError as exc:
            raise ConfigError(f"plugin set {name!r}: {module_name!r} has no {attribute!r}") from exc

    known = sorted({"recast", *(ep.name for ep in entry_points(group=ENTRY_POINT_GROUP))})
    raise ConfigError(
        f"unknown plugin set {name!r}; declared sets are {known}. "
        "A set under development is named by dotted path, e.g. yourpkg.conformance:PLUGIN_SET"
    )


def _as_plugin_set(value: Any, origin: str) -> PluginSet:
    """A callable declaration is called; anything else must already be a set."""
    resolved = value() if callable(value) and not isinstance(value, PluginSet) else value
    if not isinstance(resolved, PluginSet):
        raise ConfigError(f"{origin} is {type(resolved).__name__}, not a PluginSet")
    return resolved
