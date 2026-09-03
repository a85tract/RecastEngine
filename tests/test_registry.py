"""Fail-closed plugin discovery and path-free origin attribution."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any

import pytest

import recast.registry as registry_module
from recast.errors import PluginError, PluginNotFound
from recast.registry import PluginOrigin, Registry


class _Distribution:
    def __init__(self, name: str, version: str, *, private_path: str = "/private/wheel") -> None:
        self.metadata = {"Name": name}
        self.version = version
        self.private_path = private_path


class _EntryPoint:
    def __init__(
        self,
        *,
        group: str,
        name: str,
        value: str,
        factory: Any = object,
        distribution: _Distribution | None = None,
        failure: Exception | None = None,
    ) -> None:
        self.group = group
        self.name = name
        self.value = value
        self.dist = distribution
        self.factory = factory
        self.failure = failure
        self.load_count = 0

    def load(self) -> Any:
        self.load_count += 1
        if self.failure is not None:
            raise self.failure
        return self.factory


def _install_inventory(
    monkeypatch: pytest.MonkeyPatch,
    *points: _EntryPoint,
) -> None:
    def selected(*, group: str) -> tuple[_EntryPoint, ...]:
        return tuple(point for point in points if point.group == group)

    monkeypatch.setattr(registry_module, "entry_points", selected)


def _entry_point(
    name: str,
    *,
    value: str = "example.plugins:factory",
    distribution_name: str = "Example.Plugin_Name",
    distribution_version: str = "1.0RC1",
) -> _EntryPoint:
    return _EntryPoint(
        group="recast.transforms",
        name=name,
        value=value,
        distribution=_Distribution(distribution_name, distribution_version),
    )


def test_local_registration_has_explicit_unverified_origin() -> None:
    registry = Registry(discover_installed=False)
    registry.register("transform", "example.local", object)

    origin = registry.origin("transform", "example.local")

    assert origin == PluginOrigin(
        source="local",
        verification="unverified",
        distribution_name=None,
        distribution_version=None,
        group="recast.transforms",
        name="example.local",
        value=None,
    )
    assert registry.origins("transform") == {"example.local": origin}
    copied = registry.origins("transform")
    copied.clear()
    assert registry.origin("transform", "example.local") == origin
    with pytest.raises(FrozenInstanceError):
        origin.name = "mutated"  # type: ignore[misc]


def test_distribution_origin_is_normalized_stable_and_path_free(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _entry_point("example.translate")
    assert point.dist is not None
    point.dist.private_path = "/do/not/expose/site-packages"
    _install_inventory(monkeypatch, point)
    registry = Registry()

    assert registry.get("transform", "example.translate") is object
    origin = registry.origin("transform", "example.translate")

    assert origin.as_dict() == {
        "schema": "recast.plugin-origin.v1",
        "source": "distribution",
        "verification": "distribution_metadata",
        "distribution_name": "example-plugin-name",
        "distribution_version": "1.0rc1",
        "group": "recast.transforms",
        "name": "example.translate",
        "value": "example.plugins:factory",
    }
    rendered = json.dumps(origin.as_dict(), sort_keys=True)
    assert "/do/not/expose" not in rendered
    assert "private_path" not in rendered


def test_duplicate_entry_point_name_aborts_before_loading_any_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry_point("example.ambiguous", value="first.plugin:factory")
    second = _entry_point(
        "example.ambiguous",
        value="second.plugin:factory",
        distribution_name="Second-Plugin",
    )
    _install_inventory(monkeypatch, first, second)
    registry = Registry()

    with pytest.raises(PluginError, match="ambiguous transform entry-point name"):
        registry.names("transform")
    with pytest.raises(PluginError, match=r"example\.ambiguous"):
        registry.names("transform")

    assert first.load_count == 0
    assert second.load_count == 0
    assert "transform" not in registry._loaded
    assert registry._plugins.get("transform", {}) == {}


def test_identical_duplicate_entry_points_are_still_ambiguous(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _entry_point("example.same")
    second = _entry_point("example.same")
    _install_inventory(monkeypatch, first, second)

    with pytest.raises(PluginError, match=r"example\.same"):
        Registry().get("transform", "example.same")

    assert first.load_count == second.load_count == 0


def test_installed_entry_point_cannot_silently_shadow_explicit_registration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _entry_point("example.collision")
    _install_inventory(monkeypatch, point)
    registry = Registry()
    local = object()
    registry.register("transform", "example.collision", local)

    with pytest.raises(PluginError, match="conflicts with an explicit local registration"):
        registry.get("transform", "example.collision")

    assert point.load_count == 0
    assert registry._plugins["transform"]["example.collision"] is local
    assert registry._origins["transform"]["example.collision"].verification == "unverified"
    assert "transform" not in registry._loaded


def test_distinct_local_and_installed_addresses_keep_distinct_origins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _entry_point("example.installed")
    _install_inventory(monkeypatch, point)
    registry = Registry()
    registry.register("transform", "example.local", object)

    assert registry.names("transform") == ("example.installed", "example.local")
    origins = registry.origins("transform")
    assert origins["example.installed"].source == "distribution"
    assert origins["example.local"].verification == "unverified"


def test_replace_after_discovery_is_explicitly_local_and_unverified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    point = _entry_point("example.replace")
    _install_inventory(monkeypatch, point)
    registry = Registry()
    assert registry.get("transform", "example.replace") is object

    replacement = object()
    registry.register("transform", "example.replace", replacement, replace=True)

    assert registry.get("transform", "example.replace") is replacement
    assert registry.origin("transform", "example.replace").as_dict() == {
        "schema": "recast.plugin-origin.v1",
        "source": "local",
        "verification": "unverified",
        "distribution_name": None,
        "distribution_version": None,
        "group": "recast.transforms",
        "name": "example.replace",
        "value": None,
    }


def test_plugin_with_unattributable_or_path_like_origin_stays_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_distribution = _entry_point("example.no-dist")
    missing_distribution.dist = None
    path_value = _entry_point("example.path", value="/private/plugin.py:factory")
    _install_inventory(monkeypatch, missing_distribution, path_value)
    registry = Registry()

    assert registry.names("transform") == ()
    with pytest.raises(PluginNotFound):
        registry.origin("transform", "example.no-dist")
    assert set(registry.broken()) == {"transform:example.no-dist", "transform:example.path"}
    rendered = json.dumps(registry.broken(), sort_keys=True)
    assert "/private/plugin.py" not in rendered


def test_one_import_failure_does_not_hide_a_distinct_attributed_plugin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    broken = _entry_point("example.broken")
    broken.failure = RuntimeError("optional dependency is unavailable")
    working = _entry_point("example.working")
    _install_inventory(monkeypatch, broken, working)
    registry = Registry()

    assert registry.names("transform") == ("example.working",)
    assert registry.origin("transform", "example.working").source == "distribution"
    assert registry.broken() == {
        "transform:example.broken": "RuntimeError: optional dependency is unavailable"
    }


def test_isolated_registry_remains_convenient_for_unit_tests() -> None:
    registry = Registry(discover_installed=False)
    registry.register("oracle", "f2py-golden", object)

    with pytest.raises(PluginNotFound, match="f2py-golden"):
        registry.get("oracle", "typo")
    assert registry.names("oracle") == ("f2py-golden",)


def test_local_registry_name_cannot_smuggle_a_path_into_origin() -> None:
    registry = Registry(discover_installed=False)

    with pytest.raises(PluginError, match="invalid transform plugin registry name"):
        registry.register("transform", "/private/plugin.py", object)


def test_plugin_origin_rejects_inconsistent_or_path_like_public_values() -> None:
    with pytest.raises(PluginError, match="explicitly unverified"):
        PluginOrigin(
            source="local",
            verification="unverified",
            distribution_name="pretend-package",
            distribution_version=None,
            group="recast.transforms",
            name="example.local",
            value=None,
        )
    with pytest.raises(PluginError, match="normalized and path-free"):
        PluginOrigin(
            source="distribution",
            verification="distribution_metadata",
            distribution_name="example-package",
            distribution_version="../private",
            group="recast.transforms",
            name="example.installed",
            value="example.plugin:factory",
        )
