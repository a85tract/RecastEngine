"""Stores: where records live, and who is allowed to read them.

Two stores, because SciRecast has two audiences:

    EvidenceStore   correctness evidence -> CC-Test, append-only, public
    FindingStore    vulnerabilities      -> Sec-Track, restricted, embargoed

Keeping them as separate ABCs rather than one store with a flag is deliberate.
It makes ``publish an evidence package`` and ``file a 0-day`` different calls
against different objects, so the dangerous one cannot be reached by passing the
wrong argument to the safe one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from recast.errors import AccessViolation
from recast.model import Access, Evidence, Finding

_ACCESS_ORDER = {Access.PUBLIC: 0, Access.INTERNAL: 1, Access.EMBARGOED: 2}


class EvidenceStore(ABC):
    """Append-only record of what was validated.

    Append-only is a requirement, not an implementation detail: CC-Test's
    Layer-2 check includes "has an existing package been altered?", and a store
    that permits overwrite makes that check unenforceable.
    """

    name: str
    max_access: Access = Access.PUBLIC

    @abstractmethod
    def put(self, evidence: Evidence) -> str:
        """Store and return a stable URI. Must reject a re-put under a
        different digest for an existing key."""

    @abstractmethod
    def get(self, uri: str) -> Evidence: ...

    @abstractmethod
    def query(self, **selectors: Any) -> Iterable[Evidence]: ...


class FindingStore(ABC):
    """Restricted record of vulnerabilities under coordinated disclosure."""

    name: str
    max_access: Access = Access.EMBARGOED

    def guard(self, finding: Finding) -> None:
        """Refuse to write a record more sensitive than this store can hold.

        Called by ``put`` implementations before any I/O. The check runs on
        every write rather than at configuration time because a Finding's access
        can be raised mid-pipeline by an adjudicator.
        """
        if _ACCESS_ORDER[finding.access] > _ACCESS_ORDER[self.max_access]:
            raise AccessViolation(
                f"finding {finding.uid!r} is {finding.access.value}; "
                f"store {self.name!r} holds at most {self.max_access.value}"
            )

    @abstractmethod
    def put(self, finding: Finding) -> str: ...

    @abstractmethod
    def get(self, uid: str) -> Finding: ...

    @abstractmethod
    def query(self, **selectors: Any) -> Iterable[Finding]: ...
