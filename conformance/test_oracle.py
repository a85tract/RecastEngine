"""Oracle: the cache key has to notice everything that moves the reference.

An Oracle's failure mode is not a wrong answer. It is a stale right one. If
``key`` misses something -- an optimization flag, a rank count, the source
itself -- the engine serves a cached reference built from different inputs, and
every Verdict downstream of it is a comparison against the wrong thing while
looking exactly like a comparison against the right one. Nothing later in the
pipeline can detect that, which is why the checks here are mostly about the key
and why the case's author, not the suite, supplies the list of what must move
it.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

from recast.conformance.doubles import RefusingExecutor
from recast.errors import RecastError
from recast.executors.local import LocalExecutor
from recast.model import Facts, OracleRef
from recast.plugins.oracle import Oracle
from recast.registry import REGISTRY


@pytest.fixture
def oracle(oracle_case: Any) -> Oracle:
    missing = [m for m in oracle_case.requires if importlib.util.find_spec(m) is None]
    missing += [c for c in oracle_case.requires_commands if shutil.which(c) is None]
    if missing:
        pytest.skip(f"case {oracle_case.name!r} needs {missing}, which are not available here")
    built: Oracle = (
        oracle_case.build() if oracle_case.build else REGISTRY.get("oracle", oracle_case.name)()
    )
    return built


@pytest.fixture
def facts(oracle_case: Any, oracle: Oracle) -> Facts:
    """Depends on ``oracle`` so the skip happens before the facts are built --
    a frontend that is not installed must not fail here as a missing import."""
    return oracle_case.facts()


def test_the_key_is_stable_for_the_same_inputs(
    oracle_case: Any, oracle: Oracle, facts: Facts
) -> None:
    """The other half of the cache's promise: a key that moves on its own means
    nothing is ever reused, and every gate pays for a rebuild it did not need."""
    config = dict(oracle_case.config)
    first = oracle.key(oracle_case.unit, facts, config)
    assert first == oracle.key(oracle_case.unit, facts, dict(oracle_case.config))
    assert isinstance(first, str) and first, "a key has to be a non-empty string"


def test_the_key_moves_when_the_reference_moves(
    oracle_case: Any, oracle: Oracle, facts: Facts
) -> None:
    """Every perturbation the case declares must produce a different key."""
    if not oracle_case.moves_the_key:
        pytest.skip(
            f"{oracle_case.name!r} declares nothing that moves its key, so this "
            "check has nothing to try -- unexercised, not passed"
        )
    base = oracle.key(oracle_case.unit, facts, dict(oracle_case.config))
    unmoved = [
        label
        for label, overlay in oracle_case.moves_the_key.items()
        if oracle.key(oracle_case.unit, facts, {**oracle_case.config, **overlay}) == base
    ]
    assert not unmoved, (
        f"{oracle_case.name!r} keys the same reference for a changed {unmoved}; a cache "
        "hit here serves a reference built from other inputs, and every Verdict that "
        "follows compares against the wrong thing"
    )


def test_the_key_moves_when_the_source_moves(
    oracle_case: Any, oracle: Oracle, facts: Facts
) -> None:
    if oracle_case.move_the_source is None:
        pytest.skip(f"{oracle_case.name!r} declares no way to move its source")
    config = dict(oracle_case.config)
    base = oracle.key(oracle_case.unit, facts, config)
    moved = oracle.key(oracle_case.unit, oracle_case.move_the_source(facts), config)
    assert moved != base, (
        "the source changed and the key did not; the next run is served a reference "
        "compiled from source that no longer exists"
    )


def test_a_refusing_executor_is_a_recast_error(
    oracle_case: Any, oracle: Oracle, facts: Facts, tmp_path: Path
) -> None:
    """Not a crash. The runner catches ``RecastError`` and marks this unit's
    oracle stage failed; anything else escapes it and takes the whole run down,
    so one refused build costs every other unit its verdict too."""
    if not oracle_case.materializes:
        pytest.skip(f"{oracle_case.name!r} does not materialize where the suite runs")

    with pytest.raises(RecastError):
        oracle.materialize(
            oracle_case.unit, facts, tmp_path, RefusingExecutor(), dict(oracle_case.config)
        )


def test_release_is_idempotent(
    oracle_case: Any, oracle: Oracle, facts: Facts, tmp_path: Path
) -> None:
    """Teardown runs after failures, where it may already have run once."""
    if not oracle_case.materializes:
        pytest.skip(f"{oracle_case.name!r} does not materialize where the suite runs")

    executor = oracle_case.executor() if oracle_case.executor else LocalExecutor()
    ref = oracle.materialize(oracle_case.unit, facts, tmp_path, executor, dict(oracle_case.config))
    assert isinstance(ref, OracleRef)
    assert ref.key == oracle.key(oracle_case.unit, facts, dict(oracle_case.config)), (
        "the ref is filed under a different key than ``key`` reports, so the cache "
        "will miss on the very next call and build it again"
    )
    oracle.release(ref)
    oracle.release(ref)
