"""Filesystem-backed stores. Content-addressed, append-only.

Deliberately boring. Evidence is JSON on disk laid out so a CC-Test pull request
can be opened directly from it, and so ``git`` provides the audit trail rather
than a database nobody else can inspect.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from recast.errors import RecastError
from recast.model import Access, Evidence, Finding
from recast.plugins.store import EvidenceStore, FindingStore


def _digest(payload: dict[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


@dataclass
class FilesystemEvidenceStore(EvidenceStore):
    """Writes CC-Test evidence manifests under ``root/<unit>/<digest>.json``."""

    root: Path
    name: str = "fs-evidence"
    max_access: Access = Access.PUBLIC
    cc_test: dict[str, Any] | None = None
    """``{version, commit}`` of the CC-Test revision whose schema is being met."""

    def put(self, evidence: Evidence) -> str:
        manifest = evidence.to_manifest(
            cc_test=self.cc_test or {"version": "unknown", "commit": "unknown"},
            timestamp=evidence.meta.get("timestamp", ""),
        )
        digest = _digest(manifest)
        path = self.root / evidence.unit.replace(":", "_").replace("/", "_") / f"{digest}.json"
        if path.exists():
            return path.as_uri()  # identical content, idempotent
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        return path.as_uri()

    def get(self, uri: str) -> Evidence:
        raise NotImplementedError("P2: rehydrate Evidence from a CC-Test manifest")

    def query(self, **selectors: Any) -> Iterable[Evidence]:
        raise NotImplementedError("P2: index manifests by unit/recipe/confidence")


@dataclass
class FilesystemFindingStore(FindingStore):
    """A local stand-in for Sec-Track.

    Refuses to operate on a world-readable directory. That check is here because
    the realistic accident is not a wrong access class -- it is writing an
    embargoed finding into a repository checkout that later gets pushed.
    """

    root: Path
    name: str = "fs-findings"
    max_access: Access = Access.EMBARGOED

    def __post_init__(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self.root.stat().st_mode & 0o077:
            raise RecastError(
                f"{self.root} is group/world accessible; refusing to store embargoed findings"
            )

    def put(self, finding: Finding) -> str:
        self.guard(finding)
        path = self.root / f"{finding.uid}.json"
        path.write_text(
            json.dumps(
                {
                    "uid": finding.uid,
                    "unit": finding.unit,
                    "scanner": finding.scanner,
                    "title": finding.title,
                    "cwe": finding.cwe,
                    "severity": finding.severity.value,
                    "disclosure": finding.disclosure.value,
                    "access": finding.access.value,
                    "location": finding.location,
                    "exploitability": finding.exploitability,
                    "evidence": finding.evidence,
                    "upstream": finding.upstream,
                },
                indent=2,
                sort_keys=True,
            )
        )
        path.chmod(0o600)
        return path.as_uri()

    def get(self, uid: str) -> Finding:
        raise NotImplementedError("P3: rehydrate from Sec-Track record format")

    def query(self, **selectors: Any) -> Iterable[Finding]:
        raise NotImplementedError("P3: index by cwe/severity/disclosure")
