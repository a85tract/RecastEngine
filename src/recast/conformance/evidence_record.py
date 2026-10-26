"""What an ``EvidenceStore`` must have written: a ``recast.evidence.v1`` record.

The engine owns this format; ``Evidence.to_record`` produces it and this module
holds the reading a store is held to, as checks rather than a JSON Schema so the
conformance suite needs no schema library. The two live in the same package and
change together.

This is deliberately not CC-Test's acceptance record (its ``evidence-manifest.v1``). That document
describes a released product version and is assembled by CC-Test's own
``make_manifest.py`` from the run summary; a per-verdict record cannot honestly
fill its required fields and, until 2026-09-24, one that claimed to failed
CC-Test's schema on nineteen counts. Keeping the two names distinct is the fix.

Returns problems rather than raising, so a failing check can name all of them
at once instead of one per run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recast.model import Confidence, Evidence

__all__ = ["check_record"]

_TOP_LEVEL = (
    "schema",
    "evidence_class",
    "unit",
    "recipe",
    "executor",
    "artifact",
    "reference",
    "environment",
    "cases",
    "verdict",
    "artifacts",
    "timestamp",
)

_EVIDENCE_CLASSES = frozenset({"complete", "reconstructed"})
_VERDICT_KEYS = ("verifier", "confidence", "passed", "candidate", "metrics", "detail")


def check_record(document: Any) -> list[str]:
    """Problems with ``document`` as a ``recast.evidence.v1`` record. Empty means valid."""
    if not isinstance(document, Mapping):
        return [f"record is {type(document).__name__}, not a mapping"]

    problems = [f"missing key {key!r}" for key in _TOP_LEVEL if key not in document]
    if problems:
        return problems

    if document["schema"] != Evidence.SCHEMA:
        problems.append(f"schema is {document['schema']!r}, expected {Evidence.SCHEMA!r}")
    if document["evidence_class"] not in _EVIDENCE_CLASSES:
        problems.append(
            f"evidence_class {document['evidence_class']!r} is not one of "
            f"{sorted(_EVIDENCE_CLASSES)}"
        )
    for key in ("unit", "recipe", "executor"):
        if not (isinstance(document[key], str) and document[key]):
            problems.append(f"{key} is empty; a record has to say what it is about")
    for key in ("artifact", "reference", "environment", "artifacts"):
        if not isinstance(document[key], Mapping):
            problems.append(f"{key} is {type(document[key]).__name__}, not a mapping")
    # An environment nobody filled in is the single most common way a bit-exact
    # claim stops being reproducible, so an empty one is a defect and not a
    # matter of taste. See ``Evidence.environment``.
    if isinstance(document["environment"], Mapping) and not document["environment"]:
        problems.append("environment is empty; the run recorded nothing about where it ran")
    if not isinstance(document["cases"], Sequence) or isinstance(document["cases"], (str, bytes)):
        problems.append(f"cases is {type(document['cases']).__name__}, not a list")
    if not (isinstance(document["timestamp"], str) and document["timestamp"]):
        problems.append("timestamp is empty; a record of a run has to say when it ran")

    problems += _check_verdict(document["verdict"])
    return problems


def _check_verdict(verdict: Any) -> list[str]:
    if not isinstance(verdict, Mapping):
        return [f"verdict is {type(verdict).__name__}, not a mapping"]

    problems = [f"verdict missing key {key!r}" for key in _VERDICT_KEYS if key not in verdict]
    if problems:
        return problems

    confidences = {c.value for c in Confidence}
    if verdict["confidence"] not in confidences:
        problems.append(
            f"verdict.confidence {verdict['confidence']!r} is not one of {sorted(confidences)}"
        )
    if not isinstance(verdict["passed"], bool):
        problems.append(f"verdict.passed is {type(verdict['passed']).__name__}, not a bool")
    # The two say the same thing twice, and a reader may believe either.
    elif verdict["confidence"] in confidences:
        expected = verdict["confidence"] != Confidence.FAILED.value
        if verdict["passed"] != expected:
            problems.append(
                f"verdict.passed is {verdict['passed']} but confidence is {verdict['confidence']!r}"
            )
    if not (isinstance(verdict["verifier"], str) and verdict["verifier"]):
        problems.append("verdict.verifier is empty")
    if not isinstance(verdict["metrics"], Mapping):
        problems.append(f"verdict.metrics is {type(verdict['metrics']).__name__}, not a mapping")
    return problems
