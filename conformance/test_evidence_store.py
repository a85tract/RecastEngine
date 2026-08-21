"""EvidenceStore: append-only, and what it writes is a CC-Test manifest.

Append-only is not a preference. CC-Test's Layer-2 check asks whether an
existing package has been altered, and a store that lets one URI mean two
different documents makes that question unanswerable. A content-addressed
store satisfies this by construction -- different content lands under a
different URI -- and so does a store that refuses the second write. Both are
conforming; silently replacing the first document is not.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recast.conformance.manifest import check_manifest


def test_putting_the_same_record_twice_is_one_record(
    evidence_store_case: Any, sample_evidence: Any, scratch: Path
) -> None:
    """Re-running a recipe over unchanged inputs must not fork the record."""
    store = evidence_store_case.build(scratch)
    evidence = sample_evidence()
    first = store.put(evidence)
    second = store.put(sample_evidence())
    assert first == second, (
        "two puts of identical evidence returned different URIs; the same claim "
        "is now recorded twice and a reader cannot tell which is current"
    )


def test_a_uri_never_denotes_two_documents(
    evidence_store_case: Any, sample_evidence: Any, scratch: Path
) -> None:
    store = evidence_store_case.build(scratch)
    original = sample_evidence()
    uri = store.put(original)

    altered = sample_evidence()
    altered.verdict.metrics = {"bit_exact": 0, "total_points": 8}
    altered.verdict.detail = "the same run, described differently"
    try:
        other = store.put(altered)
    except Exception:  # refusing the write is the other conforming answer
        return

    assert other != uri, (
        f"altered evidence was written under the existing URI {uri!r}; an "
        "append-only store must either address it separately or refuse it"
    )
    if evidence_store_case.read_manifest is not None:
        kept = evidence_store_case.read_manifest(store, uri)
        assert kept["result"]["metrics"] == {"bit_exact": 8, "total_points": 8}, (
            "the first document changed under its own URI"
        )


def test_what_was_written_is_an_evidence_manifest(
    evidence_store_case: Any, sample_evidence: Any, scratch: Path
) -> None:
    if evidence_store_case.read_manifest is None:
        pytest.skip(
            f"{evidence_store_case.name!r} declares no read_manifest, so the suite "
            "cannot see what it wrote -- unexercised, not passed"
        )
    store = evidence_store_case.build(scratch)
    uri = store.put(sample_evidence())
    problems = check_manifest(evidence_store_case.read_manifest(store, uri))
    assert not problems, "not an evidence-manifest.v1:\n" + "\n".join(problems)
