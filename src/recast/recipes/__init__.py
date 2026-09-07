"""The shipped recipes.

Each one names the real project it was abstracted from. They are stage
declarations only -- the plugins they name arrive from ``recast-fortran``
(in-tree, P2) or a domain extension (P4).

Read these four side by side and the claim that the engine is domain-independent
becomes checkable: they differ only in which plugin fills each slot.
"""

from __future__ import annotations

from typing import Any

from recast.plugins.recipe import Recipe, Stage


class TranslateRecipe(Recipe):
    """Rule-driven language translation, gated on a compiled oracle.

    Abstracted from the source pipeline: Fortran to NumPy, then optionally
    to Numba or CUDA, with the untouched Fortran compiled through f2py as the
    reference and bit-exactness as the acceptance bar.

    The source language is a slot, not a property of the recipe: frontend and
    oracle default to Fortran and f2py because that is the one language this
    repository ships, and ``config["frontend"]`` and ``config["oracle"]``
    move both together. They have to move together -- the oracle compiles the
    same source the frontend read, so a frontend for a second language brings
    its own oracle with it.

    ``target: tree`` is the NumPy translation for a unit that ``use``s its
    siblings. ``translate.numpy`` emits the import of a sibling's translation
    and carries only its own files; the run walks the sibling first and
    names its candidate's directory to the gate (``companion_paths``), so
    the import resolves when the sibling is a unit of the same run.
    ``translate.tree`` bundles the siblings' translations into the candidate
    instead, and needs no extension tables for a tree of plain modules; the
    tables are for constants modules and framework stubs, which such a tree
    does not have.
    """

    name = "translate"
    summary = "Translate a source language to a target language, gated bit-exact."
    engine_id = "recast.fortran-python.numpy"

    def resolved_engine_id(self, config: dict[str, Any]) -> str | None:
        # The legacy recipe also exposes direct Fortran -> Numba/CUDA variants.
        # Those are not the future Python -> Numba/JAX engines and therefore do
        # not borrow the NumPy engine's identity.
        is_builtin = (
            config.get("target", "numpy") == "numpy"
            and config.get("frontend", "fortran") == "fortran"
        )
        return self.engine_id if is_builtin else None

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        target = config.get("target", "numpy")
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("transform", f"translate.{target}"),
            Stage("verifier", "static.rwset", gate=True),
            Stage("oracle", config.get("oracle", "f2py-golden")),
            Stage("verifier", "differential.bitexact", gate=True),
            Stage("verifier", "symbolic.notary", optional=True),
            Stage("store", "fs-evidence"),
        ]

    def validate(self, config: dict[str, Any]) -> list[str]:
        target = config.get("target", "numpy")
        known = {"numpy", "numba", "cuda", "tree"}
        return [] if target in known else [f"unknown target {target!r}; expected {sorted(known)}"]


class RefactorRecipe(Recipe):
    """Architectural refactoring of a monolith, gated on a pinned full-model run.

    Abstracted from a control-plane port: generate C-interoperable adapters and an ordered
    series of source patches that carve a Python control plane into a coupled model,
    leaving the numerical routines untouched, then prove the result still
    reproduces the pinned reference bit-for-bit at production rank count.

    Note the gate is a ``batch`` oracle. This recipe cannot complete on the local
    executor, and that is a property of the work, not a limitation to route
    around.

    The ``-todo`` suffix is the incompleteness made visible: all four of its
    workload slots -- ``refactor.carve``, ``static.no-numerics-moved``,
    ``pinned-run``, ``fullmodel.bitwise`` -- name plugins nothing ships yet.
    The suffix comes off when they land.
    """

    name = "refactor-todo"
    summary = "Restructure architecture without touching numerics; gate on a full run."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            # One transform, one Candidate: the adapters and the ordered
            # patches are halves of a single carve-out, and a Candidate carries
            # both -- ``files`` for what is generated, ``patches`` for what is
            # edited. Splitting them across two stages would have thrown the
            # first half away, since a Unit has one Candidate.
            Stage("transform", "refactor.carve"),
            Stage("verifier", "static.no-numerics-moved", gate=True),
            Stage("oracle", "pinned-run", config={"ranks": config.get("ranks", 512)}),
            Stage("verifier", "fullmodel.bitwise", gate=True),
            Stage("store", "fs-evidence"),
        ]

    def validate(self, config: dict[str, Any]) -> list[str]:
        problems = []
        if not config.get("reference_commit"):
            problems.append(
                "refactor-todo requires 'reference_commit': the pinned upstream revision"
            )
        # The gate is a batch oracle, so the default executor cannot finish this
        # run. Saying so here costs a second; finding out costs the build.
        if config.get("executor", "local") == "local":
            problems.append(
                "refactor-todo gates on a pinned multi-rank run; set 'executor' to a batch executor"
            )
        return problems


class AuditRecipe(Recipe):
    """The cyber half of CC-Test, in CC-Test's shape. Findings, not Candidates.

    Runs against any git repository -- ported or legacy, in this domain or not. Findings
    route to a FindingStore under embargo; nothing here writes to the public
    evidence store.

    The gates are the scanners themselves. ``hpc-devsecops`` has no
    adjudication step: a check that found something is the verdict, each at
    its own bar -- any secret, a Critical CVE -- and every check runs before
    anything blocks, so the operator gets the whole list. Everything else
    CC-Test does is the domain extension's, and it carries its own recipe for
    it the way it carries ``translate-cam``: the adversarial adjudicator this
    recipe used to gate on (Sec-Track's discovery-loop step), the LLM source
    audit, and the sanitizer build. The engine keeps the ``Adjudicator``
    contract and ships no implementation of it. This recipe declares exactly
    what this repository ships, and nothing it does not -- a public recipe
    naming even an optional slot for the extension's stages would be
    advertising a capability this repository does not have, because the
    maintainer's rule is that the LLM audit stays out of the public
    repository, and a stage name is part of the repository.

    ``config["range"]`` scopes the history scanner to a revision range, which
    is what the pre-push hook in ``tools/`` passes.
    """

    name = "audit"
    summary = "Secret scan and SBOM/CVE/VEX, gating the way hpc-devsecops does."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("scanner", "secret", gate=True),
            Stage("scanner", "composition", gate=True),
            Stage("store", "fs-findings"),
        ]


BUILTIN: dict[str, type[Recipe]] = {
    "translate": TranslateRecipe,
    "refactor-todo": RefactorRecipe,
    "audit": AuditRecipe,
}

# The recipes of the commercial tier -- the accelerator port and the two
# Python-accelerator recipes -- are declared beside these and absent from
# the public edition, which is the same tree with that module left out.
try:
    from recast.recipes.pro import BUILTIN as _PRO
    from recast.recipes.pro import (  # noqa: F401 -- re-exported for their tests
        PortRecipe,
        PythonToJaxRecipe,
        PythonToNumbaRecipe,
    )
except ImportError:  # the public edition
    _PRO = {}
BUILTIN.update(_PRO)
