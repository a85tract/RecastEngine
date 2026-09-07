"""The engine's own plugin set: what ``--plugin-set recast`` resolves to.

This is the set the engine holds *itself* to, and it is the worked example an
out-of-tree author copies. Two things in it are worth reading before writing
your own.

The evidence store declares ``read_manifest``. Without it the suite cannot see
what the store wrote, and reports the manifest check as unexercised rather than
passing it -- a store that writes garbage and a store nobody can read back are
indistinguishable from outside, and only one of them is acceptable.

The recipes carry a config each. ``refactor-todo`` names a batch executor because
its own ``validate`` refuses ``local``: its gate is a pinned multi-rank run,
so a plan produced under the default config is a plan that can never execute,
and checking that plan would check nothing.

The seven verifier cases are compiler-free on purpose,
``differential.bitexact`` included. Its oracle here is a handful of Python
rather than a compiled Fortran module, which it accepts because an ``Oracle``
hands over an opaque handle and this gate only ever calls what is on it. The
Numba/JAX cases instead plant an inert source handle and exercise their two
isolated Executor jobs. Together those fixtures cover every rule in the table
-- a good candidate earns its verdict, a broken one fails, an absent oracle or
refusing executor fails closed -- without a Fortran toolchain. That matters
because a conformance run that silently skips on a machine without gfortran is
a conformance run that told you nothing. The real Fortran spine is not skipped
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
    EngineCase,
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


# --- static.complete ---------------------------------------------------------


def _complete_candidate(workspace: Path) -> Candidate:
    del workspace
    return Candidate(
        unit="conformance:demo/fill",
        transform="conformance.translate",
        files={Path("translated.py"): b"def fill():\n    return 1\n"},
    )


def _complete_break(candidate: Candidate) -> Candidate:
    # This verifier judges the completeness ledger itself.  Adding an
    # unresolved refusal is therefore the broken subject, not a corruption of
    # the bookkeeping that describes some other comparison.
    return replace(candidate, deferred=["fill/B001: conformance refusal"])


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
        "result_dtype": "float64",
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


"""Eight points spanning six decades, as literals rather than as a computation.

Both sides multiply by the *same* constants, so any difference in the verdict
comes from the perturbation and not from two spellings of ``exp``. Index 0 is
the whole signal and 4..7 sit below the gate's 1e-3 dominance line, which is
what lets one array exercise both tiers.
"""


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
# own shipped tree. That makes the case honest and makes it conditional: from a
# wheel there is no ``corpus/`` and the case skips by name, which is the
# answer that says "not checked here" rather than "nothing found".

TOY_PHYSICS = Path(__file__).resolve().parents[3] / "corpus" / "toy_physics"
"""``src/recast/conformance/builtin.py`` -> the repository root -> the shipped tree."""

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


# --- python-numpy: the frontend and the python-source oracle -----------------

_PYTHON_NUMPY = f"""\
from __future__ import annotations

{"import"} numpy as np

__all__ = ["blend"]

def _square(x: np.ndarray) -> np.ndarray:
    return x * x

def blend(x: np.ndarray, scale: float) -> np.ndarray:
    return np.sin(x) * scale + _square(x)
"""


def _plant_python_numpy(root: Path) -> None:
    (root / "kernel.py").write_text(_PYTHON_NUMPY)


def _plant_python_workspace_artifact(workspace: Path) -> None:
    (workspace / "generated.py").write_text(_PYTHON_NUMPY)


def _python_oracle_facts() -> Facts:
    return Facts(
        unit="python:kernel.py",
        interface={"module": "kernel", "source": "kernel.py", "exports": ["blend"]},
        provenance={"digest": "1" * 64, "source": "kernel.py", "frontend": "python-numpy"},
    )


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
    ! A formatted internal write whose format is a variable is refused on
    ! purpose -- an edit descriptor is a rounding rule, and one the rules
    ! cannot read at translation time cannot be rendered -- so this block
    ! goes to the agent queue while the one above still translates. The
    ! point of the case is the mixture: a Candidate that is partial rather
    ! than absent. (A literal format used to stand here; the rules grew a
    ! rendering for it, and a defers-case is a claim about the rules.)
    character(len=32) :: buffer
    character(len=8) :: fmt
    y = x
    fmt = '(F8.2)'
    write(buffer, fmt) y
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


"""The repository's own pair of probe-printing scripts, the reference and a
candidate that agrees with it; see its README."""

_PUBLIC = PluginSet(
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
        FrontendCase(
            name="python-numpy",
            plant_tree=_plant_python_numpy,
            expect_uids=("python:kernel.py",),
            plant_workspace_artifact=_plant_python_workspace_artifact,
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
            name="python-source",
            unit=Unit(uid="python:kernel.py", kind="python-module", sources=(Path("kernel.py"),)),
            facts=_python_oracle_facts,
            move_the_source=_different_source,
            materializes=False,
            submits_jobs=False,
        ),
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
            name="static.complete",
            candidate=_complete_candidate,
            break_candidate=_complete_break,
            expect=Confidence.SAMPLED,
        ),
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
    scanners=(
        ScannerCase(name="secret"),
        ScannerCase(name="composition", fakes=_composition_fakes),
    ),
    recipes=(
        RecipeCase(name="translate"),
        RecipeCase(
            name="refactor-todo",
            config={"reference_commit": "0" * 40, "executor": "conformance-batch"},
        ),
        RecipeCase(name="audit"),
    ),
    engines=(EngineCase(name="recast.fortran-python.numpy"),),
)


def _with_pro(base: PluginSet) -> PluginSet:
    """These cases plus those of ``builtin_optional``, when that module is
    installed; an installation may lack it."""
    try:
        from recast.conformance.builtin_optional import CASES
    except ImportError:
        return base
    merged = {
        field: getattr(base, field) + getattr(CASES, field)
        for field in (
            "executors",
            "frontends",
            "transforms",
            "oracles",
            "verifiers",
            "evidence_stores",
            "finding_stores",
            "scanners",
            "recipes",
            "engines",
        )
    }
    return replace(base, **merged)


PLUGIN_SET = _with_pro(_PUBLIC)
