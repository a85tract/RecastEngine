"""The producer side of CC-Test ``evidence-manifest.v1``.

RecastEngine does not own this schema -- CC-Test does, and ``Evidence.to_manifest``
is the only place the two vocabularies meet. What this module holds is the
reading of v1 that the engine writes against, expressed as checks so a store
can be held to it. Keeping it here rather than vendoring CC-Test's JSON Schema
is deliberate: a store's obligation is to write back what it was handed, and
that is checkable without a schema library or a network call. When CC-Test
revises the schema, this moves with ``to_manifest`` and not before.

Returns problems rather than raising, so a failing check can name all of them
at once instead of one per run.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from recast.model import Confidence

__all__ = ["check_manifest"]

_TOP_LEVEL = (
    "schema_version",
    "evidence_class",
    "artifact",
    "reference",
    "cc_test",
    "environment",
    "cases",
    "result",
    "timestamp",
    "notes",
)

_EVIDENCE_CLASSES = frozenset({"complete", "reconstructed"})
_RESULT_KEYS = ("verdict", "passed", "verifier", "metrics", "detail")


def check_manifest(document: Any) -> list[str]:
    """Problems with ``document`` as an ``evidence-manifest.v1``. Empty means valid."""
    if not isinstance(document, Mapping):
        return [f"manifest is {type(document).__name__}, not a mapping"]

    problems = [f"missing key {key!r}" for key in _TOP_LEVEL if key not in document]
    if problems:
        return problems

    if document["schema_version"] != 1:
        problems.append(f"schema_version is {document['schema_version']!r}, expected 1")
    if document["evidence_class"] not in _EVIDENCE_CLASSES:
        problems.append(
            f"evidence_class {document['evidence_class']!r} is not one of "
            f"{sorted(_EVIDENCE_CLASSES)}"
        )
    for key in ("artifact", "reference", "cc_test", "environment"):
        if not isinstance(document[key], Mapping):
            problems.append(f"{key} is {type(document[key]).__name__}, not a mapping")
    # An environment nobody filled in is the single most common way a bit-exact
    # claim stops being reproducible, so an empty one is a defect and not a
    # matter of taste. See ``Evidence.environment``.
    if isinstance(document["environment"], Mapping) and not document["environment"]:
        problems.append("environment is empty; the run recorded nothing about where it ran")
    if not isinstance(document["cases"], Sequence) or isinstance(document["cases"], (str, bytes)):
        problems.append(f"cases is {type(document['cases']).__name__}, not a list")
    if not isinstance(document["notes"], str):
        problems.append(f"notes is {type(document['notes']).__name__}, not a string")
    if not (isinstance(document["timestamp"], str) and document["timestamp"]):
        problems.append("timestamp is empty; a record of a run has to say when it ran")

    problems += _check_result(document["result"])
    return problems


def _check_result(result: Any) -> list[str]:
    if not isinstance(result, Mapping):
        return [f"result is {type(result).__name__}, not a mapping"]

    problems = [f"result missing key {key!r}" for key in _RESULT_KEYS if key not in result]
    if problems:
        return problems

    verdicts = {c.value for c in Confidence}
    if result["verdict"] not in verdicts:
        problems.append(f"result.verdict {result['verdict']!r} is not one of {sorted(verdicts)}")
    if not isinstance(result["passed"], bool):
        problems.append(f"result.passed is {type(result['passed']).__name__}, not a bool")
    # The two say the same thing twice, and a reader may believe either.
    elif result["verdict"] in verdicts:
        expected = result["verdict"] != Confidence.FAILED.value
        if result["passed"] != expected:
            problems.append(
                f"result.passed is {result['passed']} but verdict is {result['verdict']!r}"
            )
    if not isinstance(result["metrics"], Mapping):
        problems.append(f"result.metrics is {type(result['metrics']).__name__}, not a mapping")
    return problems
