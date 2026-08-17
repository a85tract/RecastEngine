"""The four shipped recipes.

Each one names the real project it was abstracted from. They are stage
declarations only -- the plugins they name arrive from ``recast-fortran``
(in-tree, P2) or ``recast-cesm`` (the CESM case, P4).

Read these four side by side and the claim that the engine is domain-independent
becomes checkable: they differ only in which plugin fills each slot.
"""

from __future__ import annotations

from typing import Any

from recast.plugins.recipe import Recipe, Stage


class TranslateRecipe(Recipe):
    """Rule-driven language translation, gated on a compiled oracle.

    Abstracted from CESM-language-translator: Fortran to NumPy, then optionally
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

    Abstracted from freeCAM: generate C-interoperable adapters and an ordered
    series of source patches that carve a Python control plane into iCESM,
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
            Stage("transform", "refactor.adapters"),
            Stage("transform", "refactor.patches"),
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

    Abstracted from CESM-jax-kernels: rewrite a physics kernel for JAX or Numba,
    validated against inputs and outputs captured from a real Fortran run rather
    than against synthetic samples, because the regimes that break a port are the
    ones the model actually visits.
    """

    name = "port"
    summary = "Retarget a kernel to an accelerator; gate on captured production dumps."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        backend = config.get("backend", "jax")
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("transform", f"port.{backend}"),
            Stage("oracle", "dump-replay"),
            Stage("verifier", "differential.tolerance", gate=True),
            Stage("verifier", "performance.benchmark", optional=True),
            Stage("store", "fs-evidence"),
        ]

    def validate(self, config: dict[str, Any]) -> list[str]:
        problems = []
        backend = config.get("backend", "jax")
        if backend not in {"jax", "numba", "cuda"}:
            problems.append(f"unknown backend {backend!r}")
        if not config.get("dumps"):
            problems.append("port requires 'dumps': captured reference inputs/outputs")
        return problems


class AuditRecipe(Recipe):
    """The cyber half of CC-Test. Produces Findings, not Candidates.

    Runs against any git repository -- ported or legacy, CESM or not. Confirmed
    findings route to a FindingStore under embargo; nothing here writes to the
    public evidence store.
    """

    name = "audit"
    summary = "Secret scan, SBOM/CVE/VEX, LLM source audit, sanitizer builds."

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        stages = [
            Stage("frontend", config.get("frontend", "fortran")),
            Stage("scanner", "secret"),
            Stage("scanner", "composition"),
            Stage("scanner", "audit.llm", optional=True),
        ]
        if config.get("dynamic"):
            stages.append(Stage("scanner", "dynamic.asan", optional=True))
        stages += [
            Stage("adjudicator", "adversarial", gate=True),
            Stage("store", "fs-findings"),
        ]
        return stages


BUILTIN: dict[str, type[Recipe]] = {
    "translate": TranslateRecipe,
    "refactor": RefactorRecipe,
    "port": PortRecipe,
    "audit": AuditRecipe,
}
