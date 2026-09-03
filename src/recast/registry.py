"""Plugin discovery with stable, path-free origin records.

Two ways in:

* ``importlib.metadata`` entry points -- how installed packages register.
  Out-of-tree plugins arrive this way, by being pip-installed.
* ``register()`` -- how tests and in-process extensions register.

An installed entry point carries its normalized distribution identity and exact
group/name/value.  An in-process registration is deliberately marked
``local``/``unverified``; it cannot impersonate installed distribution
provenance.  Individual import failures remain isolated, while ambiguous names
fail discovery for the whole kind before any candidate is loaded.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint, entry_points
from typing import Any, Final, Literal

from recast.errors import PluginError, PluginNotFound

KINDS: tuple[str, ...] = (
    "frontend",
    "transform",
    "oracle",
    "verifier",
    "scanner",
    "adjudicator",
    "executor",
    "store",
    "agent",
    "recipe",
    "engine",
)

ENTRY_POINT_GROUP = "recast.{kind}s"
PLUGIN_ORIGIN_SCHEMA: Final = "recast.plugin-origin.v1"

_ENTRY_POINT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$")
_ENTRY_POINT_VALUE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?::[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)?"
    r"(?:\s+\[[A-Za-z0-9_.-]+(?:\s*,\s*[A-Za-z0-9_.-]+)*\])?$"
)
_DISTRIBUTION_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_DISTRIBUTION_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.!+_-]{0,255}$")


@dataclass(frozen=True, slots=True)
class PluginOrigin:
    """Stable attribution for one registry address; never a filesystem location.

    ``verification`` means only that the attribution came from installed Core
    Metadata.  It is not a signature or a claim that the plugin code is safe.
    """

    schema: Literal["recast.plugin-origin.v1"] = field(default=PLUGIN_ORIGIN_SCHEMA, init=False)
    source: Literal["distribution", "local"]
    verification: Literal["distribution_metadata", "unverified"]
    distribution_name: str | None
    distribution_version: str | None
    group: str
    name: str
    value: str | None

    def __post_init__(self) -> None:
        groups = {ENTRY_POINT_GROUP.format(kind=kind) for kind in KINDS}
        if self.group not in groups or not _ENTRY_POINT_NAME.fullmatch(self.name):
            raise PluginError("plugin origin has an invalid registry address")
        if self.source == "local":
            if (
                self.verification != "unverified"
                or self.distribution_name is not None
                or self.distribution_version is not None
                or self.value is not None
            ):
                raise PluginError("local plugin origin must remain explicitly unverified")
            return
        if self.source != "distribution" or self.verification != "distribution_metadata":
            raise PluginError("plugin origin source and verification are inconsistent")
        if (
            not isinstance(self.distribution_name, str)
            or not _DISTRIBUTION_NAME.fullmatch(self.distribution_name)
            or not isinstance(self.distribution_version, str)
            or not _DISTRIBUTION_VERSION.fullmatch(self.distribution_version)
            or self.distribution_version != self.distribution_version.lower()
            or not isinstance(self.value, str)
            or not _ENTRY_POINT_VALUE.fullmatch(self.value)
        ):
            raise PluginError("distribution plugin origin is not normalized and path-free")

    def as_dict(self) -> dict[str, str | None]:
        return {
            "schema": self.schema,
            "source": self.source,
            "verification": self.verification,
            "distribution_name": self.distribution_name,
            "distribution_version": self.distribution_version,
            "group": self.group,
            "name": self.name,
            "value": self.value,
        }


@dataclass
class Registry:
    """Name -> plugin factory and immutable origin, per kind.

    ``discover_installed=False`` creates an isolated inventory for unit tests;
    every factory registered into it still receives an unverified local origin.
    """

    _plugins: dict[str, dict[str, Any]] = field(default_factory=dict)
    _loaded: set[str] = field(default_factory=set)
    _broken: dict[str, str] = field(default_factory=dict)
    _origins: dict[str, dict[str, PluginOrigin]] = field(default_factory=dict)
    discover_installed: bool = True

    def register(self, kind: str, name: str, factory: Any, *, replace: bool = False) -> None:
        if kind not in KINDS:
            raise PluginError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")
        if not isinstance(name, str) or not _ENTRY_POINT_NAME.fullmatch(name):
            raise PluginError(f"invalid {kind} plugin registry name")
        bucket = self._plugins.setdefault(kind, {})
        if name in bucket and not replace:
            raise PluginError(
                f"{kind} {name!r} is already registered; pass replace=True to override"
            )
        bucket[name] = factory
        self._origins.setdefault(kind, {})[name] = _local_origin(kind, name)

    def get(self, kind: str, name: str) -> Any:
        self._discover(kind)
        bucket = self._plugins.get(kind, {})
        if name not in bucket:
            raise PluginNotFound(kind, name, tuple(bucket))
        return bucket[name]

    def names(self, kind: str) -> tuple[str, ...]:
        self._discover(kind)
        return tuple(sorted(self._plugins.get(kind, {})))

    def origin(self, kind: str, name: str) -> PluginOrigin:
        """Return the immutable origin for an exact registry address."""

        self._discover(kind)
        bucket = self._plugins.get(kind, {})
        if name not in bucket:
            raise PluginNotFound(kind, name, tuple(bucket))
        try:
            return self._origins[kind][name]
        except KeyError as exc:  # only possible after unsupported private-state mutation
            raise PluginError(f"{kind} {name!r} has no registry origin") from exc

    def origins(self, kind: str) -> dict[str, PluginOrigin]:
        """Return a defensive name -> origin snapshot for one plugin kind."""

        self._discover(kind)
        bucket = self._origins.get(kind, {})
        return {name: bucket[name] for name in sorted(bucket)}

    def broken(self) -> dict[str, str]:
        """Plugins that failed to load, name -> reason. Surfaced by ``recast doctor``."""
        return dict(self._broken)

    def _discover(self, kind: str) -> None:
        if kind in self._loaded:
            return
        if kind not in KINDS:
            raise PluginError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")
        if not self.discover_installed:
            self._loaded.add(kind)
            return
        group = ENTRY_POINT_GROUP.format(kind=kind)
        try:
            points = tuple(entry_points(group=group))
        except Exception as exc:
            raise PluginError(f"could not inspect installed entry-point group {group!r}") from exc

        candidates: list[EntryPoint] = []
        for index, ep in enumerate(points, start=1):
            if ep.group != group or not _ENTRY_POINT_NAME.fullmatch(ep.name):
                self._broken[f"{kind}:<invalid-entry-point-{index}>"] = (
                    "PluginError: installed entry point has an invalid group or name"
                )
                continue
            candidates.append(ep)

        by_name: dict[str, list[EntryPoint]] = {}
        for ep in candidates:
            by_name.setdefault(ep.name, []).append(ep)
        duplicates = sorted(name for name, matches in by_name.items() if len(matches) != 1)
        if duplicates:
            rendered = ", ".join(repr(name) for name in duplicates)
            raise PluginError(f"ambiguous {kind} entry-point name(s) in {group!r}: {rendered}")
        candidates.sort(key=lambda ep: ep.name)

        local_conflicts = sorted(set(by_name).intersection(self._plugins.get(kind, {})))
        if local_conflicts:
            rendered = ", ".join(repr(name) for name in local_conflicts)
            raise PluginError(
                f"installed {kind} entry point conflicts with an explicit local registration: "
                f"{rendered}"
            )

        discovered: list[tuple[str, Any, PluginOrigin]] = []
        for ep in candidates:
            try:
                origin = _distribution_origin(ep, group)
                factory = ep.load()
            except Exception as exc:  # a third-party import must not break the CLI
                self._broken[f"{kind}:{ep.name}"] = f"{type(exc).__name__}: {exc}"
                continue
            discovered.append((ep.name, factory, origin))

        late_conflicts = sorted(
            name for name, _factory, _origin in discovered if name in self._plugins.get(kind, {})
        )
        if late_conflicts:
            rendered = ", ".join(repr(name) for name in late_conflicts)
            raise PluginError(
                f"installed {kind} entry point conflicted with a registration during discovery: "
                f"{rendered}"
            )
        plugin_bucket = self._plugins.setdefault(kind, {})
        origin_bucket = self._origins.setdefault(kind, {})
        for name, factory, origin in discovered:
            plugin_bucket[name] = factory
            origin_bucket[name] = origin
        self._loaded.add(kind)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        for kind in KINDS:
            for name in self.names(kind):
                yield kind, name


REGISTRY = Registry()


def register(kind: str, name: str, factory: Any, *, replace: bool = False) -> None:
    """Register into the process-wide registry."""
    REGISTRY.register(kind, name, factory, replace=replace)


def get(kind: str, name: str) -> Any:
    return REGISTRY.get(kind, name)


def origin(kind: str, name: str) -> PluginOrigin:
    """Return one process-wide plugin origin."""

    return REGISTRY.origin(kind, name)


def origins(kind: str) -> dict[str, PluginOrigin]:
    """Return a process-wide origin snapshot for one kind."""

    return REGISTRY.origins(kind)


def _local_origin(kind: str, name: str) -> PluginOrigin:
    return PluginOrigin(
        source="local",
        verification="unverified",
        distribution_name=None,
        distribution_version=None,
        group=ENTRY_POINT_GROUP.format(kind=kind),
        name=name,
        value=None,
    )


def _distribution_origin(ep: EntryPoint, group: str) -> PluginOrigin:
    if ep.group != group or not _ENTRY_POINT_NAME.fullmatch(ep.name):
        raise PluginError("installed entry point has an invalid address")
    if not _ENTRY_POINT_VALUE.fullmatch(ep.value):
        raise PluginError("installed entry point has an invalid path-free value")
    distribution = ep.dist
    if distribution is None:
        raise PluginError("installed entry point has no distribution attribution")
    try:
        raw_name = distribution.metadata["Name"]
    except KeyError as exc:
        raise PluginError("installed entry point distribution has no valid name") from exc
    raw_version = distribution.version
    if not isinstance(raw_name, str):
        raise PluginError("installed entry point distribution has no valid name")
    normalized_name = re.sub(r"[-_.]+", "-", raw_name.strip()).lower()
    if not _DISTRIBUTION_NAME.fullmatch(normalized_name):
        raise PluginError("installed entry point distribution has no valid normalized name")
    if not isinstance(raw_version, str) or not _DISTRIBUTION_VERSION.fullmatch(raw_version.strip()):
        raise PluginError("installed entry point distribution has no valid version")
    return PluginOrigin(
        source="distribution",
        verification="distribution_metadata",
        distribution_name=normalized_name,
        distribution_version=raw_version.strip().lower(),
        group=group,
        name=ep.name,
        value=ep.value,
    )
