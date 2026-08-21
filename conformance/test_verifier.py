"""Verifier: the gate. Every rule here is a way of failing closed.

A Verifier is the only thing in the engine that decides whether work is worth
anything, and its dangerous failure is silent: a gate that passes when it
should have failed looks exactly like a gate that worked. So the checks are
arranged around the ways a verdict can be produced without a comparison having
happened -- a broken artifact, an oracle that never materialized, an executor
that refused the job -- and each one requires ``FAILED`` rather than a weaker
pass.

The first check is not one of those. It requires the *good* candidate to earn
its verdict, and it exists because without it every other check on this page
is satisfiable by a Verifier that returns ``FAILED`` unconditionally.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any

import pytest

from recast.conformance.doubles import RefusingExecutor
from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence, Unit, Verdict
from recast.plugins.verifier import StaticVerifier, Verifier
from recast.registry import REGISTRY
from recast.run import NO_ORACLE


@pytest.fixture
def judge(verifier_case: Any, tmp_path: Path) -> Any:
    """Run the case's verifier, with each ingredient substitutable."""
    missing = [m for m in verifier_case.requires if importlib.util.find_spec(m) is None]
    missing += [c for c in verifier_case.requires_commands if shutil.which(c) is None]
    if missing:
        pytest.skip(f"case {verifier_case.name!r} needs {missing}, which are not available here")

    verifier: Verifier = (
        verifier_case.build()
        if verifier_case.build
        else REGISTRY.get("verifier", verifier_case.name)()
    )

    def run(candidate: Candidate, *, oracle: Any = None, executor: Any = None) -> Verdict:
        if oracle is None:
            oracle = (
                verifier_case.oracle(tmp_path, LocalExecutor())
                if verifier_case.oracle
                else NO_ORACLE
            )
        if executor is None:
            executor = verifier_case.executor() if verifier_case.executor else LocalExecutor()
        unit = verifier_case.unit or Unit(uid=candidate.unit, kind="subprogram")
        return verifier.verify(
            unit, candidate, oracle, tmp_path, executor, dict(verifier_case.config)
        )

    run.verifier = verifier  # type: ignore[attr-defined]
    return run


def test_a_good_candidate_earns_its_verdict(verifier_case: Any, judge: Any, tmp_path: Path) -> None:
    """Without this, every other check here passes on a gate that never passes."""
    verdict = judge(verifier_case.candidate(tmp_path))
    assert verdict.passed, (
        f"the case's own good candidate was rejected: {verdict.confidence.value}: "
        f"{verdict.detail}. Every other check on this verifier is vacuous until "
        "this one holds."
    )
    if verifier_case.expect is not None:
        assert verdict.confidence is verifier_case.expect, (
            f"expected {verifier_case.expect.value}, got {verdict.confidence.value}"
        )
    assert verdict.metrics, (
        "a verdict from a real comparison reports numbers, not just a conclusion"
    )


def test_a_broken_candidate_fails(verifier_case: Any, judge: Any, tmp_path: Path) -> None:
    broken = verifier_case.break_candidate(verifier_case.candidate(tmp_path))
    verdict = judge(broken)
    assert verdict.confidence is Confidence.FAILED, (
        f"a deliberately broken candidate earned {verdict.confidence.value}: {verdict.detail}"
    )
    assert verdict.detail, "a failing verdict has to say what it saw"
    assert verdict.metrics, (
        "the comparison ran and disagreed, so its numbers are the evidence for the "
        "verdict -- report them, not just the conclusion"
    )


def test_the_verdict_names_what_it_judged(verifier_case: Any, judge: Any, tmp_path: Path) -> None:
    """``Verdict.candidate`` is the digest that was judged, and Evidence keys off
    it. A verdict that names something else is filed against the wrong artifact."""
    candidate = verifier_case.candidate(tmp_path)
    verdict = judge(candidate)
    assert verdict.candidate == candidate.digest()
    assert verdict.verifier == judge.verifier.name
    assert verdict.unit == candidate.unit


def test_an_unavailable_oracle_fails_closed(verifier_case: Any, judge: Any, tmp_path: Path) -> None:
    """Never a weaker pass. A comparison that could not run is not a sampled one."""
    if isinstance(judge.verifier, StaticVerifier):
        pytest.skip(f"{verifier_case.name!r} is a StaticVerifier: it needs no oracle to be right")

    verdict = judge(verifier_case.candidate(tmp_path), oracle=NO_ORACLE)
    assert verdict.confidence is Confidence.FAILED, (
        f"with no oracle to compare against, {verifier_case.name!r} still awarded "
        f"{verdict.confidence.value}"
    )
    assert verdict.detail, "a failing verdict has to say what it saw"


def test_a_refusing_executor_fails_closed(verifier_case: Any, judge: Any, tmp_path: Path) -> None:
    """The executor said no. That is a `FAILED` verdict, not a smaller run, and
    not an exception: the runner does not catch one, so a raising Verifier takes
    the whole recipe down instead of recording why this unit stopped."""
    if not verifier_case.submits_jobs:
        pytest.skip(
            f"{verifier_case.name!r} declares that it submits no jobs, so there is "
            "no executor for it to route around"
        )

    verdict = judge(verifier_case.candidate(tmp_path), executor=RefusingExecutor())
    assert verdict.confidence is Confidence.FAILED, (
        f"{verifier_case.name!r} awarded {verdict.confidence.value} while the executor "
        "was refusing every job; the comparison it claims to have run did not happen here"
    )
    assert verdict.detail, "a failing verdict has to say what it saw"
