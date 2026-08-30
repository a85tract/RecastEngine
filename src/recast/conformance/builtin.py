"""The engine's own plugin set: what ``--plugin-set recast`` resolves to.

This is the set the engine holds *itself* to, and it is the worked example an
out-of-tree author copies. Two things in it are worth reading before writing
your own.

The evidence store declares ``read_manifest``. Without it the suite cannot see
what the store wrote, and reports the manifest check as unexercised rather than
passing it -- a store that writes garbage and a store nobody can read back are
indistinguishable from outside, and only one of them is acceptable.

The recipes carry a config each. ``refactor`` names a batch executor because
its own ``validate`` refuses ``local``: its gate is a pinned multi-rank run,
so a plan produced under the default config is a plan that can never execute,
and checking that plan would check nothing.

The three verifier cases are compiler-free on purpose, ``differential.bitexact``
included. Its oracle here is a handful of Python rather than a compiled Fortran
module, which it accepts because an ``Oracle`` hands over an opaque handle and
this gate only ever calls what is on it. That is enough to exercise every rule
in the table -- a good candidate earns its verdict, a broken one fails, an
absent oracle fails closed -- without a toolchain, which matters because a
conformance run that silently skips on a machine without gfortran is a
conformance run that told you nothing. The real Fortran spine is not skipped
either; it is held by ``tests/test_f2py_oracle.py`` in CI's compiler job, and
that is the right place for it, because a compiled oracle is what *that* test
is about and not what this one is.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import unquote, urlparse

from recast.conformance import (
    EvidenceStoreCase,
    ExecutorCase,
    FindingStoreCase,
    FrontendCase,
    OracleCase,
    PluginSet,
    RecipeCase,
    ScannerCase,
    TransformCase,
    TransformSubject,
    VerifierCase,
)
from recast.model import Candidate, Confidence, Facts, OracleRef, Unit
from recast.plugins.executor import Executor
from recast.plugins.store import EvidenceStore
from recast.registry import REGISTRY
from recast.store.filesystem import FilesystemEvidenceStore, FilesystemFindingStore


def _evidence_store(root: Path) -> FilesystemEvidenceStore:
    return FilesystemEvidenceStore(root=root / "evidence")


def _read_manifest(store: EvidenceStore, uri: str) -> dict[str, Any]:
    """``FilesystemEvidenceStore`` returns a ``file:`` URI; the document is at it."""
    document: dict[str, Any] = json.loads(Path(unquote(urlparse(uri).path)).read_text())
    return document


def _finding_store(root: Path) -> FilesystemFindingStore:
    # A subdirectory that does not exist yet, so the store creates it with its
    # own mode. Handing it one that already exists would test the caller's
    # umask rather than the store.
    return FilesystemFindingStore(root=root / "findings")


# --- composition -------------------------------------------------------------

_GRYPE_ONE = {
    "matches": [
        {
            "vulnerability": {"id": "CVE-0000-0001", "severity": "Critical"},
            "artifact": {"name": "conformance-dep", "version": "1.0"},
        }
    ]
}


def _composition_fakes(bin_dir: Path, behaviour: str) -> None:
    """syft writes an SBOM to ``-o spdx-json=<path>``; grype answers in its
    own JSON on stdout. Neither speaks SARIF, so the suite's default fakes
    would check the wrong shape."""
    from recast.conformance.fake_tool import fake_tool

    fake_tool(bin_dir, "syft", payload='{"spdxVersion": "SPDX-2.3"}')
    if behaviour == "garbage":
        fake_tool(bin_dir, "grype", payload="panic: runtime error\n", exit_code=1, report_flags=())
    else:
        matches = _GRYPE_ONE if behaviour == "one" else {"matches": []}
        fake_tool(bin_dir, "grype", sarif=matches, report_flags=())


# --- static.rwset ------------------------------------------------------------

_RWSET_EMITTED = """\
def fill(out, n):
    acc = 0.0
    for i in range(1, n + 1):
        acc = acc + POOL[i]
        out[i] = acc
    return out
"""

# 1-based inclusive spans into the text above, as a Transform records them.
_RWSET_BLOCKS = [
    {"subprogram": "fill", "block": "B001", "reads": [], "writes": ["acc"], "lines": [2, 2]},
    {
        "subprogram": "fill",
        "block": "B002",
        "reads": ["acc", "i", "n", "pool"],
        "writes": ["acc", "i", "out"],
        "lines": [3, 5],
    },
]


def _rwset_candidate(workspace: Path) -> Candidate:
    return Candidate(
        unit="conformance:demo/fill",
        transform="conformance.translate",
        files={Path("demo_numpy.py"): _RWSET_EMITTED.encode()},
        notes={
            "rwset": {
                "file": "demo_numpy.py",
                "blocks": _RWSET_BLOCKS,
                "names": {"POOL": "pool"},
                "procedures": ["fill"],
                # ``range`` is the NumPy backend's, not a source symbol. A
                # backend saying which names are its own is how this gate
                # avoids guessing; see ``recast.verify.rwset.LITERALS``.
                "scaffolding": ["range"],
            }
        },
    )


def _rwset_break(candidate: Candidate) -> Candidate:
    """Read a different array. The translation still runs and still returns
    numbers -- which is the bug class this gate exists for, and the one a
    differential test is worst at catching."""
    emitted = _RWSET_EMITTED.replace("POOL[i]", "SPARE[i]")
    return replace(candidate, files={Path("demo_numpy.py"): emitted.encode()})


# --- symbolic.notary ---------------------------------------------------------


def _notary_candidate(workspace: Path) -> Candidate:
    return Candidate(
        unit="conformance:demo/expr",
        transform="conformance.rewriter",
        notes={
            "rewrites": [
                {
                    # Distributing is algebra: exact in exact arithmetic, and
                    # not bit-equal in float64. Exactly what this verifier is
                    # for, and what a differential gate would wrongly reject.
                    "site": "fill/B002",
                    "old": "a*(b + c)",
                    "new": "a*b + a*c",
                    "ranges": {"a": [1.0, 2.0], "b": [1.0, 2.0], "c": [1.0, 2.0]},
                }
            ]
        },
    )


def _notary_break(candidate: Candidate) -> Candidate:
    rewrite = dict(candidate.notes["rewrites"][0])
    rewrite["new"] = "a*b + a*c + 1"
    return replace(candidate, notes={"rewrites": [rewrite]})


# --- differential.bitexact ---------------------------------------------------

_BITEXACT_EMITTED = """\
_SIGNATURES = {
    "blend": {
        "kind": "function",
        "result": "value",
        "args": [
            {"name": "x", "intent": "IN", "dtype": "float64"},
            {"name": "y", "intent": "IN", "dtype": "float64"},
        ],
    }
}


def blend(x, y):
    return x * y + 0.5 * x
"""


def _bitexact_candidate(workspace: Path) -> Candidate:
    return Candidate(
        unit="conformance:demo/blend",
        transform="conformance.translate",
        files={Path("blend_numpy.py"): _BITEXACT_EMITTED.encode()},
    )


def _bitexact_break(candidate: Candidate) -> Candidate:
    """Off by one, everywhere. Not subtle, because what is being checked is
    that the gate reports a difference at all, not how small a one it can see
    -- ``tests/test_numpy_runtime.py`` is where ULP resolution is argued."""
    emitted = _BITEXACT_EMITTED.replace("x * y + 0.5 * x", "x * y + 0.5 * x + 1.0")
    return replace(candidate, files={Path("blend_numpy.py"): emitted.encode()})


# --- differential.tolerance --------------------------------------------------

_DECAY = (
    1.0,
    0.1353352832366127,
    0.01831563888873418,
    0.0024787521766663585,
    0.00033546262790251185,
    4.5399929762484854e-05,
    6.14421235332821e-06,
    8.315287191035679e-07,
)
"""Eight points spanning six decades, as literals rather than as a computation.

Both sides multiply by the *same* constants, so any difference in the verdict
comes from the perturbation and not from two spellings of ``exp``. Index 0 is
the whole signal and 4..7 sit below the gate's 1e-3 dominance line, which is
what lets one array exercise both tiers.
"""

_TIERED_EMITTED = """\
import math

_SIGNATURES = {{
    "decay": {{
        "kind": "function",
        "result": "y",
        "args": [{{"name": "x", "intent": "IN", "dtype": "float64"}}],
    }}
}}

_DECAY = (
    1.0,
    0.1353352832366127,
    0.01831563888873418,
    0.0024787521766663585,
    0.00033546262790251185,
    4.5399929762484854e-05,
    6.14421235332821e-06,
    8.315287191035679e-07,
)


def decay(x):
    y = [x * d for d in _DECAY]
{perturbation}
    return y
"""


def _tiered_source(perturbation: str) -> bytes:
    return _TIERED_EMITTED.format(perturbation=perturbation).encode()


def _tolerance_candidate(workspace: Path) -> Candidate:
    """Drifts, in the tail, by less than the ULP bound.

    Not bit-exact on purpose: a backend that could be bit-exact would be using
    the other gate, so a case whose good candidate agrees exactly would check
    this one on the one path it was not written for.
    """
    return Candidate(
        unit="conformance:demo/decay",
        transform="conformance.port",
        files={Path("decay_numpy.py"): _tiered_source("    y[7] = math.nextafter(y[7], math.inf)")},
    )


def _tolerance_break(candidate: Candidate) -> Candidate:
    """The same distance, moved into a dominant element instead of the tail.

    The pair is the point of the tiering: a relative tolerance alone cannot
    tell these two apart, because the relative difference is identical.
    """
    return replace(
        candidate,
        files={Path("decay_numpy.py"): _tiered_source("    y[0] *= 1.0 + 1e-13")},
    )


def _tolerance_oracle(workspace: Path, executor: Executor) -> OracleRef:
    def w_decay(x: Any) -> Any:
        return [x * d for d in _DECAY]

    return OracleRef(
        unit="conformance:demo/decay",
        oracle="conformance.python-truth",
        key="conformance:decay:1",
        handle={"module": SimpleNamespace(w_decay=w_decay), "wrappers": {"decay": "w_decay"}},
        cost="cheap",
    )


def _bitexact_oracle(workspace: Path, executor: Executor) -> OracleRef:
    """The reference, in Python. An ``Oracle`` hands over an opaque handle and
    the Verifier defines its type; this one wants ``{"module", "wrappers"}``
    and calls what it finds there, so a compiled module is one way to satisfy
    it and not the only way."""

    def w_blend(x: Any, y: Any) -> Any:
        return x * y + 0.5 * x

    return OracleRef(
        unit="conformance:demo/blend",
        oracle="conformance.python-truth",
        key="conformance:blend:1",
        handle={"module": SimpleNamespace(w_blend=w_blend), "wrappers": {"blend": "w_blend"}},
        cost="cheap",
    )


# --- f2py-golden -------------------------------------------------------------
#
# The one case in this set that reaches outside the package. It needs Fortran
# source, and inventing an interface dictionary by hand would be inventing the
# frontend's output too -- so it runs the real frontend over the repository's
# own example. That makes the case honest and makes it conditional: from a
# wheel there is no ``examples/`` and the case skips by name, which is the
# answer that says "not checked here" rather than "nothing found".

TOY_PHYSICS = Path(__file__).resolve().parents[3] / "examples" / "toy_physics"
"""``src/recast/conformance/builtin.py`` -> the repository root -> the example."""

F2PY_UNIT = "fortran:toy_physics"


def _toy_physics_facts() -> Facts:
    frontend = REGISTRY.get("frontend", "fortran")()
    unit = next(u for u in frontend.discover(TOY_PHYSICS) if u.uid == F2PY_UNIT)
    facts: Facts = frontend.analyze(unit, TOY_PHYSICS)
    return facts


def _decline(site: Any) -> None:
    """A behaviour hook that fills nothing. Declared for the key check, which
    asks whether the *presence* of one moves the reference -- it does, and its
    address must not, or the cache would miss on every run."""
    return None


def _different_source(facts: Facts) -> Facts:
    """The same build of a different file. The key folds the source's content
    digest, so moving it must move the key -- otherwise a rebuilt reference is
    served out of the cache for source it was never built from."""
    return replace(facts, provenance={**facts.provenance, "digest": "0" * 64})


# --- fortran, translate.numpy ------------------------------------------------
#
# Both read source, so both get a copy of the example planted in a scratch
# directory rather than the repository's own -- one frontend check writes into
# the tree on purpose, and one asks whether the frontend did.

_DEFERS = """\
module conformance_defers
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
contains

  subroutine mechanical(x, y)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: y
    y = 2.0_r8 * x + 1.0_r8
  end subroutine mechanical

  subroutine refused(x, y)
    real(r8), intent(in)  :: x
    real(r8), intent(out) :: y
    ! A formatted internal write is refused on purpose -- an edit descriptor
    ! is a rounding rule, and guessing one writes different digits than the
    ! Fortran did -- so this block goes to the agent queue while the one above
    ! still translates. The point of the case is the mixture: a Candidate that
    ! is partial rather than absent.
    character(len=32) :: buffer
    y = x
    write(buffer, '(F8.2)') y
  end subroutine refused

end module conformance_defers
"""


def _plant_toy_physics(root: Path) -> None:
    for source in sorted(TOY_PHYSICS.glob("*.f90")):
        (root / source.name).write_text(source.read_text())


def _plant_workspace_artifact(workspace: Path) -> None:
    """What an f2py oracle leaves behind: compilable Fortran, under the
    engine's own directory, that a frontend would otherwise read as source."""
    (workspace / "wrappers.f90").write_text((TOY_PHYSICS / "toy_physics.f90").read_text())


def _fortran_subject(scratch: Path, source: str | None, uid: str) -> TransformSubject:
    root = scratch / f"src-{uid.rsplit(':', 1)[-1]}"
    root.mkdir(parents=True, exist_ok=True)
    if source is None:
        _plant_toy_physics(root)
    else:
        (root / f"{uid.rsplit(':', 1)[-1]}.f90").write_text(source)
    frontend = REGISTRY.get("frontend", "fortran")()
    unit = next(u for u in frontend.discover(root) if u.uid == uid)
    facts: Facts = frontend.analyze(unit, root)
    return TransformSubject(unit=unit, facts=facts, config={"root": str(root)})


def _translatable(scratch: Path) -> TransformSubject:
    return _fortran_subject(scratch, None, F2PY_UNIT)


def _with_a_refused_block(scratch: Path) -> TransformSubject:
    return _fortran_subject(scratch, _DEFERS, "fortran:conformance_defers")


PLUGIN_SET = PluginSet(
    name="recast",
    executors=(ExecutorCase(name="local"),),
    frontends=(
        FrontendCase(
            name="fortran",
            plant_tree=_plant_toy_physics,
            expect_uids=(F2PY_UNIT, f"{F2PY_UNIT}/settle"),
            plant_workspace_artifact=_plant_workspace_artifact,
            requires=("fparser",),
        ),
    ),
    transforms=(
        TransformCase(
            name="translate.numpy",
            subject=_translatable,
            defers=_with_a_refused_block,
            requires=("fparser", "numpy"),
        ),
        TransformCase(
            # The tree translation under empty conventions is the file
            # translation: same rules, same refusals, plus the use-constants,
            # stand-ins and adapters a flat single-file unit has none of.
            name="translate.tree",
            subject=_translatable,
            defers=_with_a_refused_block,
            requires=("fparser", "numpy"),
        ),
    ),
    oracles=(
        OracleCase(
            # The one oracle here that needs no toolchain: it derives its
            # reference by running the engine's own transform, so the Oracle
            # rules are exercised on every machine rather than only where a
            # Fortran compiler happens to be.
            name="numpy-anchor",
            unit=Unit(uid=F2PY_UNIT, kind="module"),
            facts=_toy_physics_facts,
            config={"root": str(TOY_PHYSICS)},
            moves_the_key={
                "compiler profile": {"profile": "ifx"},
                "framework stub tables": {"function_stubs": {"outfld": "pass"}},
                "an agentic hook, by its presence": {"deferred_handler": _decline},
            },
            move_the_source=_different_source,
            materializes=True,
            # It derives the reference in this process: no compile, nothing
            # handed to the executor, so nothing for a refusal to stop.
            submits_jobs=False,
            requires=("numpy", "fparser"),
        ),
        OracleCase(
            # The only oracle that computes nothing. It reads a recording and
            # supplies both the inputs and the expected outputs, so it is also
            # the only one that makes the differential gate run backwards --
            # which is why it is checked here rather than left to its own
            # tests: the contract rules about keys and refusals apply to it
            # exactly as they do to the two that build.
            #
            # Its material is synthetic and says so in every file. No
            # production dump is committed in either repository, so a case
            # that waited for one would never run.
            name="dump-replay",
            unit=Unit(uid=F2PY_UNIT, kind="module"),
            facts=_toy_physics_facts,
            config={"root": str(TOY_PHYSICS), "dumps": str(TOY_PHYSICS / "dumps")},
            moves_the_key={
                # Which machine the recording is attributed to is part of what
                # the reference *is*: the same numbers recorded on another
                # device are another device's numbers.
                "the recording's attributed device": {"reference_device": "gpu:0"},
            },
            # The recording does not depend on the source, and the key folds
            # the source digest anyway -- see ``DumpReplayOracle.key``. A
            # recording of code that has since changed is the stale reference
            # this rule exists to catch.
            move_the_source=_different_source,
            materializes=True,
            # It reads files in this process: nothing compiled, nothing handed
            # to the executor, so there is no refusal for one to honour.
            submits_jobs=False,
            requires=("numpy", "fparser"),
        ),
        OracleCase(
            name="f2py-golden",
            unit=Unit(uid=F2PY_UNIT, kind="module"),
            facts=_toy_physics_facts,
            config={"root": str(TOY_PHYSICS)},
            moves_the_key={
                # Optimization level reorders arithmetic. Two references built
                # at different -O are not interchangeable, whatever the source.
                "compiler flags": {"fflags": "-O2"},
                # A smaller public surface is a different reference: the
                # wrappers are what the gate can call at all.
                "wrapped subprograms": {"subprograms": ["settle"]},
                "wrapper parameters": {"wrapper_parameters": {"n": 8}},
            },
            move_the_source=_different_source,
            materializes=True,
            requires=("numpy", "fparser"),
            # ``key`` asks the compiler its version before anything is built,
            # so even the cheap checks need one on PATH.
            requires_commands=("gfortran",),
        ),
        OracleCase(
            # The same reference, built behind a static library and a flat
            # adapter module; on a unit with no derived types the adapter is
            # only a re-export, and the contract is the same as f2py-golden's
            # plus the link flags, which change what the extension loads
            # against.
            name="f2py-golden-flat",
            unit=Unit(uid=F2PY_UNIT, kind="module"),
            facts=_toy_physics_facts,
            config={"root": str(TOY_PHYSICS)},
            moves_the_key={
                "compiler flags": {"fflags": "-O2"},
                "wrapped subprograms": {"subprograms": ["settle"]},
                "wrapper parameters": {"wrapper_parameters": {"n": 8}},
                "link flags": {"ldflags": "-lm"},
            },
            move_the_source=_different_source,
            materializes=True,
            requires=("numpy", "fparser"),
            requires_commands=("gfortran",),
        ),
    ),
    verifiers=(
        VerifierCase(
            name="static.rwset",
            candidate=_rwset_candidate,
            break_candidate=_rwset_break,
            expect=Confidence.SAMPLED,
        ),
        VerifierCase(
            name="symbolic.notary",
            candidate=_notary_candidate,
            break_candidate=_notary_break,
            expect=Confidence.SYMBOLIC,
            config={"samples": 64},
            requires=("sympy", "mpmath"),
        ),
        VerifierCase(
            name="differential.tolerance",
            candidate=_tolerance_candidate,
            break_candidate=_tolerance_break,
            oracle=_tolerance_oracle,
            expect=Confidence.ULP_BOUNDED,
            submits_jobs=False,
            requires=("numpy",),
        ),
        VerifierCase(
            name="differential.bitexact",
            candidate=_bitexact_candidate,
            break_candidate=_bitexact_break,
            oracle=_bitexact_oracle,
            expect=Confidence.BIT_EXACT,
            # It imports the candidate and calls it here, in this process. No
            # job is submitted, so there is no executor to route around.
            submits_jobs=False,
            requires=("numpy",),
        ),
    ),
    evidence_stores=(
        EvidenceStoreCase(
            name="fs-evidence",
            build=_evidence_store,
            read_manifest=_read_manifest,
        ),
    ),
    finding_stores=(FindingStoreCase(name="fs-findings", build=_finding_store),),
    scanners=(
        ScannerCase(name="secret"),
        ScannerCase(name="composition", fakes=_composition_fakes),
    ),
    recipes=(
        RecipeCase(name="translate"),
        RecipeCase(
            name="refactor",
            config={"reference_commit": "0" * 40, "executor": "conformance-batch"},
        ),
        RecipeCase(name="port", config={"dumps": ["reference.nc"]}),
        RecipeCase(name="audit"),
    ),
)
