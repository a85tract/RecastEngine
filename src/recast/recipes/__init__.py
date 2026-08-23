"""The four shipped recipes.

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
    """

    name = "translate"
    summary = "Translate a source language to a target language, gated bit-exact."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        target = config.get("target", "numpy")
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("transform", f"translate.{target}"),
            Stage("verifier", "static.rwset", gate=True),
            Stage("oracle", "f2py-golden"),
            Stage("verifier", "differential.bitexact", gate=True),
            Stage("verifier", "symbolic.notary", optional=True),
            Stage("store", "fs-evidence"),
        ]

    def validate(self, config: dict[str, Any]) -> list[str]:
        target = config.get("target", "numpy")
        known = {"numpy", "numba", "cuda"}
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
    """

    name = "refactor"
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
            problems.append("refactor requires 'reference_commit': the pinned upstream revision")
        # The gate is a batch oracle, so the default executor cannot finish this
        # run. Saying so here costs a second; finding out costs the build.
        if config.get("executor", "local") == "local":
            problems.append(
                "refactor gates on a pinned multi-rank run; set 'executor' to a batch executor"
            )
        return problems


class PortRecipe(Recipe):
    """CPU to accelerator porting, gated on captured dumps.

    Abstracted from a kernel port: rewrite a physics kernel for JAX or
    Numba, and gate it where bit-exactness is not available -- XLA's
    transcendentals are not libm's, so the honest ceiling is a ULP bound.

    The reference is the validated NumPy translation of the same unit by
    default, which makes the port's claim a chain: NumPy bit-exact against the
    Fortran, JAX ULP-bounded against the NumPy. ``config["oracle"]`` selects
    ``dump-replay`` instead for a unit with no such translation to anchor on --
    validated against inputs and outputs captured from a real Fortran run,
    because the regimes that break a port are the ones the model visits.
    """

    name = "port"
    summary = "Retarget a kernel to an accelerator; gate on captured production dumps."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        backend = config.get("backend", "jax")
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("transform", f"port.{backend}"),
            # Two references are possible and they answer different questions.
            # ``numpy-anchor`` is the validated NumPy translation of the same
            # unit, which the translate recipe has already held bit-exact
            # against the Fortran -- so a port gated on it inherits a chain
            # rather than a looser claim. ``dump-replay`` is for a unit that
            # has no such translation to anchor on.
            Stage("oracle", config.get("oracle", "numpy-anchor")),
            # The Candidate carries the ported module *and* the anchor it
            # host-delegates to, so the gate has to be told which one is under
            # judgement. Without this it would import the anchor and compare it
            # against itself.
            Stage(
                "verifier",
                "differential.tolerance",
                gate=True,
                config={"module_suffix": f"_{backend}.py"},
            ),
            Stage("verifier", "performance.benchmark", optional=True),
            Stage("store", "fs-evidence"),
        ]

    def validate(self, config: dict[str, Any]) -> list[str]:
        problems = []
        backend = config.get("backend", "jax")
        if backend not in {"jax", "numba", "cuda"}:
            problems.append(f"unknown backend {backend!r}")
        # Only the replay oracle needs captured dumps. The NumPy anchor derives
        # its reference from the same source the port was made from, so
        # demanding dumps for it would be asking for a file nothing reads.
        if config.get("oracle", "numpy-anchor") == "dump-replay" and not config.get("dumps"):
            problems.append("the dump-replay oracle requires 'dumps': captured inputs/outputs")
        return problems


class AuditRecipe(Recipe):
    """The cyber half of CC-Test, in CC-Test's shape. Findings, not Candidates.

    Runs against any git repository -- ported or legacy, in this domain or not. Findings
    route to a FindingStore under embargo; nothing here writes to the public
    evidence store.

    The gates are the scanners themselves. ``hpc-devsecops`` has no
    adjudication step: a check that found something is the verdict, each at
    its own bar -- any secret, a Critical CVE -- and every check runs before
    anything blocks, so the operator gets the whole list. The adversarial
    adjudicator this recipe used to gate on is Sec-Track's discovery-loop
    step, LLM-driven, and belongs with the other LLM stage in the domain
    extension's deeper recipe; the engine keeps the ``Adjudicator`` contract
    and ships no implementation of it.

    This recipe declares exactly what this repository ships, and nothing it
    does not. The other two of CC-Test's four families -- the LLM source
    audit and the sanitizer build -- are the domain extension's, and it
    carries its own recipe for them the way it carries ``translate-cam``. A
    public recipe naming an optional slot for them would be advertising a
    capability this repository does not have; the maintainer's rule is that
    the LLM audit stays out of the public repository, and a stage name is
    part of the repository. ``config["range"]`` scopes the history scanner to
    a revision range, which is what the pre-push hook in ``tools/`` passes.
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
    "refactor": RefactorRecipe,
    "port": PortRecipe,
    "audit": AuditRecipe,
}
