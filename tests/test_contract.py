"""Tests for the plugin contract itself.

These guard the two properties the whole design rests on: that the core does not
depend on any domain, and that an embargoed finding cannot reach a public store.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys
from pathlib import Path

import pytest

import recast
from recast.errors import AccessViolation, PluginError, PluginNotFound
from recast.model import Access, Candidate, Disclosure, Finding, Patch, Severity
from recast.plugins.store import FindingStore
from recast.recipes import BUILTIN
from recast.registry import KINDS, Registry


class _NullFindingStore(FindingStore):
    name = "null"

    def __init__(self, max_access: Access) -> None:
        self.max_access = max_access
        self.written: list[Finding] = []

    def put(self, finding: Finding) -> str:
        self.guard(finding)
        self.written.append(finding)
        return f"null:{finding.uid}"

    def get(self, uid: str) -> Finding:  # pragma: no cover - not exercised
        raise NotImplementedError

    def query(self, **selectors: object):  # pragma: no cover - not exercised
        raise NotImplementedError


def _finding(**kw: object) -> Finding:
    base = {"uid": "F-1", "unit": "u", "scanner": "s", "title": "t"}
    return Finding(**{**base, **kw})  # type: ignore[arg-type]


# --- the core stays domain-independent ---------------------------------------


def test_core_imports_no_domain_packages() -> None:
    """No module under ``recast`` may import a domain or heavy backend package.

    This is the mechanical form of the claim in the README. If it fails, some
    CESM/Fortran/JAX specific code has leaked out of a plugin and into the core.
    """
    forbidden = {"numpy", "sympy", "mpmath", "numba", "jax", "anthropic", "netCDF4", "fparser"}
    # ``recast.fortran`` is the in-tree reference Frontend, so it is the one
    # place a source-language parser belongs. Exempting it by name is the point:
    # the day fparser appears anywhere else, this test says so.
    exempt = {
        "recast.fortran": {"fparser"},
        # The reference translation backend. These are not incidental to it:
        # NumPy is the target language's library and the code it emits imports
        # it, and mpmath is how a constant-argument intrinsic is folded at the
        # precision gfortran folds it at. Both belong to the emitted artifact,
        # not to the engine.
        "recast.transform.numpy": {"numpy", "mpmath"},
        # The notary's whole job is exact-arithmetic comparison; sympy and
        # mpmath are its instrument, imported lazily behind the [verify]
        # extra so a bare install still registers the plugin.
        "recast.verify.notary": {"sympy", "mpmath"},
        # The differential gate generates and compares NumPy arrays -- that
        # is the comparison, not a convenience. Lazy, same rule as above.
        "recast.verify.bitexact": {"numpy"},
        # The JAX backend, by the same rule that exempts the NumPy one: these
        # are the target language's libraries and the code it emits imports
        # them. Nothing here is imported by the engine -- the emitter is pure
        # AST work and reads the runtime's text off disk rather than importing
        # it, so translating to JAX does not require JAX.
        "recast.transform.jax": {"jax", "numpy"},
        # The Numba backend, same rule again: numba and numpy are the target's
        # libraries and the emitted kernels import them. The engine imports
        # neither -- the emitter reads the runtime's text off disk, so
        # translating to Numba does not require numba.
        "recast.transform.numba": {"numba", "numpy"},
    }
    root = Path(recast.__file__).parent
    offenders = []
    for mod in pkgutil.walk_packages([str(root)], prefix="recast."):
        source = Path(mod.module_finder.path, mod.name.rsplit(".", 1)[-1] + ".py")  # type: ignore[attr-defined]
        if not source.exists():
            continue
        allowed = {p for pkg, pkgs in exempt.items() if mod.name.startswith(pkg) for p in pkgs}
        for line in source.read_text().splitlines():
            stripped = line.strip()
            if not stripped.startswith(("import ", "from ")):
                continue
            top = stripped.split()[1].split(".")[0]
            if top in forbidden and top not in allowed:
                offenders.append(f"{mod.name}: {stripped}")
    assert not offenders, "core must not import domain packages:\n" + "\n".join(offenders)


def test_every_entry_point_registers_on_a_bare_install() -> None:
    """CI's bare job, run locally: with fparser, numpy and mpmath blocked,
    every entry-point factory must still import and construct. The missing
    extras surface on first use, named, with the install line -- never at
    registration, where they would break ``recast doctor`` for everyone."""
    code = """
import sys

class Block:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in ("fparser", "numpy", "mpmath", "sympy"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return None

sys.meta_path.insert(0, Block())
for cached in list(sys.modules):
    if cached.split(".")[0] in ("fparser", "numpy", "mpmath", "sympy"):
        del sys.modules[cached]

from importlib.metadata import entry_points
for group in ("recast.frontends", "recast.transforms", "recast.verifiers",
              "recast.executors", "recast.stores", "recast.recipes"):
    for ep in entry_points(group=group):
        ep.load()
print("ok")
"""
    finished = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert finished.returncode == 0, finished.stderr
    assert finished.stdout.strip() == "ok"


def test_every_builtin_recipe_declares_a_gate() -> None:
    """A recipe with no gating stage cannot produce trustworthy evidence."""
    for name, cls in BUILTIN.items():
        config = {"reference_commit": "x", "dumps": ["x"]}
        stages = cls().stages(config)
        assert any(s.gate for s in stages), f"recipe {name!r} has no gate"


def test_no_stage_is_both_gate_and_optional() -> None:
    """An optional gate is not a gate. Catch the contradiction at declaration."""
    for name, cls in BUILTIN.items():
        for stage in cls().stages({"reference_commit": "x", "dumps": ["x"]}):
            assert not (stage.gate and stage.optional), f"{name}: {stage.plugin}"


def test_recipe_declares_an_executor_when_it_needs_one() -> None:
    """Oracles and Verifiers are handed an executor, so the recipe must name one.

    ``audit`` is exempt: it has neither, and a Scanner does not take an executor
    yet -- see the open question in ``docs/architecture.md``.
    """
    config = {"reference_commit": "x", "dumps": ["x"], "executor": "batch-stub"}
    for name, cls in BUILTIN.items():
        stages = cls().stages(config)
        kinds = {s.kind for s in stages}
        if not kinds & {"oracle", "verifier"}:
            continue
        executors = [s for s in stages if s.kind == "executor"]
        assert len(executors) == 1, f"recipe {name!r} declares {len(executors)} executors"
        # Ambient, not a step: it has to be resolved before anything runs.
        assert stages[0].kind == "executor", f"recipe {name!r} does not declare it first"


def test_recipe_executor_is_not_hardcoded() -> None:
    """A site's executor name must never be baked into a shipped recipe."""
    config = {"reference_commit": "x", "dumps": ["x"], "executor": "pbs-site"}
    for name, cls in BUILTIN.items():
        for stage in cls().stages(config):
            if stage.kind == "executor":
                assert stage.plugin == "pbs-site", f"recipe {name!r} ignores configured executor"


def test_refactor_rejects_the_default_executor() -> None:
    """Its gate is a batch oracle; ``local`` cannot finish the run at all."""
    refactor = BUILTIN["refactor"]()
    problems = refactor.validate({"reference_commit": "x"})
    assert any("batch executor" in p for p in problems)
    assert refactor.validate({"reference_commit": "x", "executor": "pbs-stub"}) == []


def test_recipe_stage_kinds_are_known() -> None:
    for name, cls in BUILTIN.items():
        for stage in cls().stages({"reference_commit": "x", "dumps": ["x"]}):
            assert stage.kind in KINDS, f"{name}: unknown stage kind {stage.kind!r}"


def test_recipe_stages_are_reproducible() -> None:
    """Same config must yield the same plan, or Evidence cannot be replayed."""
    config = {"target": "numba", "backend": "jax", "dumps": ["d"], "reference_commit": "c"}
    for cls in BUILTIN.values():
        first = [(s.kind, s.plugin) for s in cls().stages(config)]
        second = [(s.kind, s.plugin) for s in cls().stages(config)]
        assert first == second


# --- access control is enforced, not documented ------------------------------


def test_embargoed_finding_rejected_by_public_store() -> None:
    store = _NullFindingStore(Access.PUBLIC)
    with pytest.raises(AccessViolation):
        store.put(_finding(access=Access.EMBARGOED, severity=Severity.HIGH))
    assert store.written == []


def test_public_finding_accepted_by_restricted_store() -> None:
    """Restriction is a ceiling, not an equality check."""
    store = _NullFindingStore(Access.EMBARGOED)
    store.put(_finding(access=Access.PUBLIC, disclosure=Disclosure.PUBLISHED))
    assert len(store.written) == 1


def test_finding_defaults_to_embargoed() -> None:
    """A scanner that forgets to classify must fail closed."""
    assert _finding().access is Access.EMBARGOED
    assert _finding().disclosure is Disclosure.PLAUSIBLE


def test_publishable_requires_completed_disclosure() -> None:
    assert not _finding(access=Access.PUBLIC, disclosure=Disclosure.CONFIRMED).publishable()
    assert _finding(access=Access.PUBLIC, disclosure=Disclosure.PUBLISHED).publishable()
    assert not _finding(access=Access.EMBARGOED, disclosure=Disclosure.PUBLISHED).publishable()


# --- registry behaviour -------------------------------------------------------


def test_registry_rejects_unknown_kind() -> None:
    reg = Registry()
    with pytest.raises(PluginError):
        reg.register("frobnicator", "x", object)


def test_registry_rejects_silent_override() -> None:
    reg = Registry()
    reg.register("transform", "t", object)
    with pytest.raises(PluginError):
        reg.register("transform", "t", object)
    reg.register("transform", "t", object, replace=True)


def test_registry_reports_available_names_on_miss() -> None:
    reg = Registry()
    reg.register("oracle", "f2py-golden", object)
    with pytest.raises(PluginNotFound, match="f2py-golden"):
        reg.get("oracle", "typo")


# --- candidate digest ---------------------------------------------------------


def test_candidate_digest_is_order_independent() -> None:
    a = Candidate("u", "t", files={Path("a"): b"1", Path("b"): b"2"})
    b = Candidate("u", "t", files={Path("b"): b"2", Path("a"): b"1"})
    assert a.digest() == b.digest()


def test_candidate_digest_covers_patches() -> None:
    base = Candidate("u", "t", patches=[Patch(Path("f"), "diff-a")])
    other = Candidate("u", "t", patches=[Patch(Path("f"), "diff-b")])
    assert base.digest() != other.digest()
