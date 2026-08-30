"""``static.complete``: reject a Candidate with unresolved deferred work.

Transforms are allowed to return partial Candidates.  That is useful during
discovery, but it is not a claim that the resulting artifact is ready for
promotion.  This verifier is the explicit policy boundary between the two:
recipes that promote an artifact add it as a gate; exploratory recipes can
leave it out.

It is deliberately strict.  A project-specific proof that some unreachable
boundary is safe belongs in a separately named project verifier; weakening
``complete`` would make a passing Verdict mean two different things.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from recast.model import Candidate, Confidence, Unit, Verdict
from recast.plugins.executor import Executor
from recast.plugins.verifier import StaticVerifier

__all__ = ["CandidateCompletenessVerifier", "factory"]


_LEDGER_DOMAIN = b"recast.deferred-ledger.v1\0"
_LEDGER_SCHEMA = "recast.deferred-ledger.v1"


def _ledger_digest(entries: Sequence[object]) -> tuple[str, int]:
    """Bind the ordered deferred ledger without copying its contents to Evidence.

    The length prefix makes the encoding unambiguous (``["ab", "c"]`` and
    ``["a", "bc"]`` differ).  A malformed entry receives an opaque marker:
    malformed Candidates always fail this verifier, and neither ``repr`` nor a
    class name should turn private plugin data into a public evidence record.
    """
    digest = hashlib.sha256(_LEDGER_DOMAIN)
    malformed = 0
    for entry in entries:
        if isinstance(entry, str):
            encoded = entry.encode()
            digest.update(b"s")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        else:
            malformed += 1
            digest.update(b"x")
    return digest.hexdigest(), malformed


class CandidateCompletenessVerifier(StaticVerifier):
    """Require ``Candidate.deferred`` to be empty, with no policy bypass."""

    name = "static.complete"
    provides = Confidence.SAMPLED

    def check(
        self,
        unit: Unit,
        candidate: Candidate,
        workspace: Path,
        executor: Executor,
        config: dict[str, Any],
    ) -> Verdict:
        del unit, workspace, executor, config

        ledger_digest, malformed = _ledger_digest(candidate.deferred)
        deferred_total = len(candidate.deferred)
        metrics: dict[str, Any] = {
            "deferred_total": deferred_total,
            "deferred_malformed": malformed,
            "deferred_ledger_schema": _LEDGER_SCHEMA,
            "deferred_ledger_digest": ledger_digest,
        }

        if deferred_total:
            malformed_detail = (
                f"; {malformed} malformed entr{'y' if malformed == 1 else 'ies'}"
                if malformed
                else ""
            )
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"candidate declares {deferred_total} unresolved deferred "
                f"entr{'y' if deferred_total == 1 else 'ies'}{malformed_detail}; "
                f"ledger sha256 {ledger_digest}",
            )

        return self._verdict(
            candidate,
            Confidence.SAMPLED,
            metrics,
            f"candidate declares no deferred work; ledger sha256 {ledger_digest}",
        )

    def _verdict(
        self,
        candidate: Candidate,
        confidence: Confidence,
        metrics: dict[str, Any],
        detail: str,
    ) -> Verdict:
        return Verdict(
            unit=candidate.unit,
            candidate=candidate.digest(),
            verifier=self.name,
            confidence=confidence,
            metrics=metrics,
            detail=detail,
        )


def factory(**_config: Any) -> CandidateCompletenessVerifier:
    return CandidateCompletenessVerifier()
