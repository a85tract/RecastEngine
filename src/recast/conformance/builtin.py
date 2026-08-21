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
    ! ASSOCIATE has no statement rule, so this block goes to the agent queue
    ! while the one above still translates. The point of the case is the
    ! mixture: a Candidate that is partial rather than absent.
    associate (scaled => 2.0_r8 * x)
      y = scaled + 1.0_r8
    end associate
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
    ),
    oracles=(
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
