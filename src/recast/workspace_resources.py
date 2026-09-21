"""Stage inputs that live in the run's workspace, declared without a host path.

A recipe sometimes needs a file the *caller* supplies: a recording to replay, a
fixture to compare against. Writing the host path into the stage's config makes
it part of the portable artifacts -- the semantic config, the bundle digest, the
frozen verification plan -- so the same work on another machine, or in another
lease, is a different identity for a reason that is not a difference.

So a stage may instead declare **where in the workspace** its input is:

    Stage("oracle", "dump-replay", config={"dumps": {"$workspace": "vendor/recording"}})

That declaration is portable: it names a relative location and nothing else.
The run resolves it against the workspace it was given -- the one operational
value the caller already passes as a keyword -- at the moment the stage is
invoked, and never earlier. The config that travels keeps the declaration; only
the call sees a path.

What is refused, and why:

* an absolute path, or one containing ``..``: the declaration would then reach
  outside the workspace the caller chose, which is the whole point of it;
* anything other than a single ``$workspace`` key in the marker, so the form
  cannot grow options by accident;
* a non-string relative location.

This says nothing about whether the resolved location exists. A stage that
needs its input to be there refuses on its own terms; a missing recording is
still a missing recording.
"""

from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from recast.errors import ConfigError

__all__ = ["WORKSPACE_MARKER", "workspace_resource", "resolve_workspace_resources"]

#: The single key that marks a workspace-relative declaration.
WORKSPACE_MARKER = "$workspace"


def workspace_resource(relative: str) -> dict[str, str]:
    """The portable declaration for ``relative`` inside the run's workspace."""

    _check(relative)
    return {WORKSPACE_MARKER: relative}


def _check(relative: Any) -> str:
    if not isinstance(relative, str) or not relative.strip():
        raise ConfigError(
            f"a {WORKSPACE_MARKER} declaration must name a relative location, not {relative!r}"
        )
    asked = PurePosixPath(relative)
    if asked.is_absolute() or ".." in asked.parts:
        raise ConfigError(
            f"{WORKSPACE_MARKER} {relative!r} must stay inside the workspace: an absolute or "
            "climbing location would defeat the declaration"
        )
    return relative


def _is_marker(value: Any) -> bool:
    return isinstance(value, dict) and set(value) == {WORKSPACE_MARKER}


def resolve_workspace_resources(value: Any, workspace: Path | None) -> Any:
    """Replace every workspace declaration in ``value`` with a real path.

    Containers are walked, so a declaration may sit anywhere a stage's config
    puts one. Without a workspace the declaration is left exactly as it is:
    a caller that has no workspace has no business being handed a path.
    """

    if _is_marker(value):
        relative = _check(value[WORKSPACE_MARKER])
        if workspace is None:
            return value
        return str(Path(workspace) / relative)
    if isinstance(value, dict):
        return {key: resolve_workspace_resources(item, workspace) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        resolved = [resolve_workspace_resources(item, workspace) for item in value]
        return type(value)(resolved) if isinstance(value, tuple) else resolved
    return value
