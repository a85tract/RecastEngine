"""Plugin discovery.

Two ways in, and they are equivalent:

* ``importlib.metadata`` entry points -- how installed packages register.
  Out-of-tree plugins arrive this way, by being pip-installed.
* ``register()`` -- how tests and in-process extensions register.

Discovery is lazy and failure-isolated: a broken third-party plugin makes that
one plugin unavailable, it does not make ``recast --help`` stop working.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

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
)

ENTRY_POINT_GROUP = "recast.{kind}s"


@dataclass
class Registry:
    """Name -> plugin factory, per kind."""

    _plugins: dict[str, dict[str, Any]] = field(default_factory=dict)
    _loaded: set[str] = field(default_factory=set)
    _broken: dict[str, str] = field(default_factory=dict)

    def register(self, kind: str, name: str, factory: Any, *, replace: bool = False) -> None:
        if kind not in KINDS:
            raise PluginError(f"unknown plugin kind {kind!r}; expected one of {KINDS}")
        bucket = self._plugins.setdefault(kind, {})
        if name in bucket and not replace:
            raise PluginError(
                f"{kind} {name!r} is already registered; pass replace=True to override"
            )
        bucket[name] = factory

    def get(self, kind: str, name: str) -> Any:
        self._discover(kind)
        bucket = self._plugins.get(kind, {})
        if name not in bucket:
            raise PluginNotFound(kind, name, tuple(bucket))
        return bucket[name]

    def names(self, kind: str) -> tuple[str, ...]:
        self._discover(kind)
        return tuple(sorted(self._plugins.get(kind, {})))

    def broken(self) -> dict[str, str]:
        """Plugins that failed to load, name -> reason. Surfaced by ``recast doctor``."""
        return dict(self._broken)

    def _discover(self, kind: str) -> None:
        if kind in self._loaded:
            return
        self._loaded.add(kind)
        group = ENTRY_POINT_GROUP.format(kind=kind)
        for ep in entry_points(group=group):
            try:
                factory = ep.load()
            except Exception as exc:  # a third-party import must not break the CLI
                self._broken[f"{kind}:{ep.name}"] = f"{type(exc).__name__}: {exc}"
                continue
            self._plugins.setdefault(kind, {}).setdefault(ep.name, factory)

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
