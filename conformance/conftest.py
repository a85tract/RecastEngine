"""Wiring: resolve ``--plugin-set`` and parametrize each kind's checks over it.

Every check in this suite takes a case fixture -- ``executor_case``,
``recipe_case``, and so on -- and the suite runs it once per case the set
declares. A kind the set does not declare produces no cases, and pytest reports
those checks as skipped for an empty parameter set. That is the intended
reading: unexercised, not passed.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from recast.conformance import PluginSet, load_plugin_set
from recast.model import Access, Confidence, Evidence, Finding, Verdict
from recast.plugins.executor import Executor, Job
from recast.plugins.recipe import Recipe
from recast.registry import REGISTRY

_CASE_FIXTURES = {
    "executor_case": "executors",
    "oracle_case": "oracles",
    "verifier_case": "verifiers",
    "evidence_store_case": "evidence_stores",
    "finding_store_case": "finding_stores",
    "recipe_case": "recipes",
}

_LOADED: dict[str, PluginSet] = {}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--plugin-set",
        action="store",
        default="recast",
        metavar="NAME",
        help=(
            "Which plugins to check: a declared set's name, or a dotted path "
            "such as yourpkg.conformance:PLUGIN_SET. Defaults to the engine's own."
        ),
    )


def _plugin_set(config: pytest.Config) -> PluginSet:
    name = str(config.getoption("--plugin-set"))
    if name not in _LOADED:
        _LOADED[name] = load_plugin_set(name)
    return _LOADED[name]


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    for fixture, attribute in _CASE_FIXTURES.items():
        if fixture not in metafunc.fixturenames:
            continue
        cases = getattr(_plugin_set(metafunc.config), attribute)
        metafunc.parametrize(fixture, cases, ids=[case.name for case in cases])


def pytest_report_header(config: pytest.Config) -> str:
    declared = _plugin_set(config)
    counts = ", ".join(
        f"{attribute.replace('_', ' ')}: {len(getattr(declared, attribute))}"
        for attribute in _CASE_FIXTURES.values()
    )
    return f"conformance: plugin set {declared.name!r} ({counts})"


# --- shared material ---------------------------------------------------------


@pytest.fixture
def build_executor() -> Any:
    """Construct the executor a case names, from the case or from the registry."""

    def build(case: Any) -> Executor:
        if case.build is not None:
            return case.build()
        executor: Executor = REGISTRY.get("executor", case.name)()
        return executor

    return build


@pytest.fixture
def build_recipe() -> Any:
    def build(case: Any) -> Recipe:
        if case.build is not None:
            return case.build()
        recipe: Recipe = REGISTRY.get("recipe", case.name)()
        return recipe

    return build


@pytest.fixture
def probe_job() -> Any:
    """A job any executor should be able to run: this interpreter, printing."""

    def build(case: Any, cwd: Path) -> Job:
        if case.probe is not None:
            job: Job = case.probe(cwd)
            return job
        return Job(
            argv=[sys.executable, "-c", "print('conformance probe')"],
            cwd=cwd,
            label="conformance-probe",
        )

    return build


@pytest.fixture
def sample_evidence() -> Any:
    """An Evidence record with every field a manifest needs populated."""

    def build(*, unit: str = "conformance:unit", digest: str = "cafe") -> Evidence:
        return Evidence(
            unit=unit,
            verdict=Verdict(
                unit=unit,
                candidate=digest,
                verifier="conformance.verifier",
                confidence=Confidence.BIT_EXACT,
                metrics={"bit_exact": 8, "total_points": 8},
                detail="",
            ),
            recipe="conformance",
            executor="conformance",
            artifact={"name": unit, "digest": digest},
            reference={"oracle": "conformance", "key": "k"},
            environment={"engine": "conformance", "python": sys.version.split()[0]},
            cases=[{"case": "1", "status": "pass"}],
            meta={"timestamp": "2026-01-01T00:00:00+00:00"},
        )

    return build


@pytest.fixture
def sample_finding() -> Any:
    def build(*, uid: str = "CONF-1", access: Access = Access.EMBARGOED) -> Finding:
        return Finding(
            uid=uid,
            unit="conformance:unit",
            scanner="conformance.scanner",
            title="a finding the suite invented",
            access=access,
        )

    return build


@pytest.fixture
def scratch(tmp_path: Path) -> Iterator[Path]:
    """A directory the suite owns, so a check may inspect what a store left in it."""
    root = tmp_path / "store"
    root.mkdir()
    yield root
