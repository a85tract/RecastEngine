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

    The source language is a slot, not a property of the recipe: frontend and
    oracle default to Fortran and f2py because that is the one language this
    repository ships, and ``config["frontend"]`` and ``config["oracle"]``
    move both together. They have to move together -- the oracle compiles the
    same source the frontend read, so a frontend for a second language brings
    its own oracle with it.

    ``target: tree`` is the NumPy translation for a unit that ``use``s its
    siblings. ``translate.numpy`` emits the import of a sibling's translation
    and carries only its own files, so the differential gate -- which stages
    a candidate's own files and nothing else -- cannot import it and fails
    the unit before comparing a number. ``translate.tree`` bundles the
    siblings' translations into the candidate and needs no extension tables
    for a tree of plain modules; the tables are for constants modules and
    framework stubs, which such a tree does not have.
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


class _PythonAcceleratorRecipe(Recipe):
    """Shared fixed topology; concrete subclasses pin every semantic slot."""

    target: str

    def stages(self, config: dict[str, Any]) -> list[Stage]:
        return [
            Stage("executor", config.get("executor", "local")),
            Stage("frontend", config.get("frontend", "python-numpy")),
            Stage("transform", f"translate.python-{self.target}"),
            Stage("verifier", "static.complete", gate=True),
            Stage("oracle", "python-source"),
            Stage("verifier", f"differential.python-{self.target}", gate=True),
            Stage("store", "fs-evidence"),
        ]

    def validate(self, config: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        if config.get("target", self.target) != self.target:
            problems.append(f"{self.name} requires target={self.target!r}")
        if config.get("frontend", "python-numpy") != "python-numpy":
            problems.append(f"{self.name} requires the python-numpy frontend")
        return problems


class PythonToNumbaRecipe(_PythonAcceleratorRecipe):
    """Compile a Python/NumPy numerical module with conservative Numba njit."""

    name = "python-to-numba"
    summary = "Compile Python/NumPy functions with Numba and verify against the source."
    engine_id = "recast.python-numpy.numba"
    target = "numba"


class PythonToJaxRecipe(_PythonAcceleratorRecipe):
    """Lower a pure Python/NumPy numerical subset to JAX jit."""

    name = "python-to-jax"
    summary = "Lower Python/NumPy functions to JAX and verify against the source."
    engine_id = "recast.python-numpy.jax"
    target = "jax"


BUILTIN: dict[str, type[Recipe]] = {
    "translate": TranslateRecipe,
    "refactor-todo": RefactorRecipe,
    "port": PortRecipe,
    "audit": AuditRecipe,
    "python-to-numba": PythonToNumbaRecipe,
    "python-to-jax": PythonToJaxRecipe,
}
