"""Tests for the opt-in ``static.complete`` promotion gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from recast.executors.local import LocalExecutor
from recast.model import Candidate, Confidence, Unit
from recast.plugins.verifier import StaticVerifier
from recast.recipes import TranslateRecipe
from recast.verify.complete import CandidateCompletenessVerifier, factory


def _candidate(*deferred: str) -> Candidate:
    return Candidate(
        unit="conformance:demo/fill",
        transform="conformance.translate",
        files={Path("translated.py"): b"def fill():\n    return 1\n"},
        deferred=list(deferred),
    )


@pytest.fixture
def verify(tmp_path: Path):
    verifier = factory()

    def run(candidate: Candidate, **config: Any):
        return verifier.check(
            Unit(uid=candidate.unit, kind="subprogram"),
            candidate,
            tmp_path,
            LocalExecutor(),
            config,
        )

    return run


def test_it_is_a_static_verifier() -> None:
    assert isinstance(factory(), StaticVerifier)
    assert factory().name == "static.complete"
    assert factory().provides is Confidence.SAMPLED


def test_a_candidate_with_no_deferred_entries_passes(verify) -> None:
    candidate = _candidate()
    verdict = verify(candidate)

    assert verdict.confidence is Confidence.SAMPLED
    assert verdict.candidate == candidate.digest()
    assert verdict.metrics["deferred_total"] == 0
    assert verdict.metrics["deferred_malformed"] == 0
    assert verdict.metrics["deferred_ledger_schema"] == "recast.deferred-ledger.v1"
    assert len(verdict.metrics["deferred_ledger_digest"]) == 64
    assert "declares no deferred" in verdict.detail


def test_a_deferred_entry_fails_closed(verify) -> None:
    deferred = "fill/B002: formatted internal write"
    verdict = verify(_candidate(deferred))

    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["deferred_total"] == 1
    assert "deferred_entries" not in verdict.metrics
    assert deferred not in json.dumps(verdict.metrics)
    assert deferred not in verdict.detail
    assert verdict.metrics["deferred_ledger_digest"] in verdict.detail


def test_stage_configuration_cannot_waive_a_deferred_entry(verify) -> None:
    deferred = "fill/B002: formatted internal write"
    verdict = verify(_candidate(deferred), waivers={deferred: "approved elsewhere"})

    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["deferred_total"] == 1
    assert deferred not in json.dumps(verdict.metrics)


def test_a_malformed_deferred_entry_fails_instead_of_aborting_the_run(verify) -> None:
    candidate = _candidate()
    candidate.deferred = [42]  # type: ignore[list-item]

    verdict = verify(candidate)

    assert verdict.confidence is Confidence.FAILED
    assert verdict.metrics["deferred_malformed"] == 1
    assert "42" not in json.dumps(verdict.metrics)
    assert "42" not in verdict.detail


def test_the_ledger_digest_binds_order_and_content_without_changing_artifact_identity(
    verify,
) -> None:
    candidate = _candidate()
    artifact_digest = candidate.digest()
    complete = verify(candidate)

    candidate.deferred = ["private/B001: first", "private/B002: second"]
    incomplete = verify(candidate)
    reordered = verify(_candidate(*reversed(candidate.deferred)))

    assert incomplete.candidate == complete.candidate == artifact_digest
    assert (
        incomplete.metrics["deferred_ledger_digest"] != complete.metrics["deferred_ledger_digest"]
    )
    assert (
        incomplete.metrics["deferred_ledger_digest"] != reordered.metrics["deferred_ledger_digest"]
    )
    public_record = json.dumps(
        {"metrics": incomplete.metrics, "detail": incomplete.detail}, sort_keys=True
    )
    assert "private/B001" not in public_record
    assert "private/B002" not in public_record


def test_the_verifier_registers_under_its_recipe_name() -> None:
    from recast.registry import REGISTRY

    assert isinstance(REGISTRY.get("verifier", "static.complete")(), CandidateCompletenessVerifier)


def test_the_exploratory_translate_recipe_does_not_enable_the_promotion_gate() -> None:
    assert "static.complete" not in {stage.plugin for stage in TranslateRecipe().stages({})}
