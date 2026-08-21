"""FindingStore: refuses what it cannot hold, and does not leave it readable.

The realistic accident is not a mislabelled record. It is an embargoed finding
written into a checkout that later gets pushed, or into a directory the rest of
the machine can read. Both are unrecoverable in the way row 5 of the disclosure
ledger describes, so both are checked here rather than left to review.
"""

from __future__ import annotations

import stat
from pathlib import Path
from typing import Any

import pytest

from recast.errors import AccessViolation
from recast.model import Access

# The contract's ordering, restated rather than imported. A change to the
# engine's order has to be a deliberate change to this suite as well, which is
# what stops it from being loosened by accident.
_ORDER = {Access.PUBLIC: 0, Access.INTERNAL: 1, Access.EMBARGOED: 2}


def test_guard_rejects_what_the_store_cannot_hold(
    finding_store_case: Any, sample_finding: Any, scratch: Path
) -> None:
    store = finding_store_case.build(scratch)
    for access in Access:
        finding = sample_finding(uid=f"CONF-{access.value}", access=access)
        too_sensitive = _ORDER[access] > _ORDER[store.max_access]
        if too_sensitive:
            with pytest.raises(AccessViolation):
                store.guard(finding)
        else:
            store.guard(finding)


def test_put_consults_guard_before_writing(
    finding_store_case: Any, sample_finding: Any, scratch: Path
) -> None:
    """A store whose ``put`` skips ``guard`` has a check nothing calls.

    Lowering the ceiling on the instance is how the suite reaches this for a
    store built to hold everything: what is being checked is that ``put`` asks,
    not what the answer happens to be for the store's configured ceiling.
    """
    store = finding_store_case.build(scratch)
    store.max_access = Access.PUBLIC
    with pytest.raises(AccessViolation):
        store.put(sample_finding(access=Access.EMBARGOED))


def test_nothing_it_wrote_is_group_or_world_readable(
    finding_store_case: Any, sample_finding: Any, scratch: Path
) -> None:
    store = finding_store_case.build(scratch)
    store.put(sample_finding(access=store.max_access))

    exposed = [
        f"{path} is {stat.filemode(path.stat().st_mode)}"
        for path in sorted(scratch.rglob("*"))
        if path.stat().st_mode & 0o077
    ]
    assert not exposed, "embargoed material left readable beyond its owner:\n" + "\n".join(exposed)
