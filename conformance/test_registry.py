"""What has to hold across every plugin installed here, not one at a time.

These take no case fixture: they read the process-wide registry, so they cover
whatever is installed alongside the engine. That is the point -- the failure
they look for is one no single plugin can see, because it only exists in the
presence of another.
"""

from __future__ import annotations

from typing import Any

from recast.registry import KINDS, REGISTRY


def _identify(factory: Any) -> tuple[str | None, Any]:
    """The name a plugin answers to, and the implementation behind it.

    Constructing is the reliable way to read ``name``, because a factory may be
    a function rather than the class. When construction needs arguments -- the
    filesystem stores take a root -- the factory is the class itself and the
    attribute is right there on it.
    """
    try:
        instance = factory()
    except Exception:
        name = getattr(factory, "name", None)
        return (name if isinstance(name, str) else None), factory
    return getattr(instance, "name", None), type(instance)


def test_every_plugin_reports_a_name() -> None:
    """It lands in Evidence. A record that does not say what produced it is a
    claim with no author."""
    nameless = [
        f"{kind}:{registered}"
        for kind in KINDS
        for registered in REGISTRY.names(kind)
        if not _identify(REGISTRY.get(kind, registered))[0]
    ]
    assert not nameless, "registered with no usable name attribute: " + ", ".join(nameless)


def test_one_name_means_one_implementation() -> None:
    """Two plugins of a kind may share a name only if they *are* the same thing.

    Sharing is legitimate and in use: ``recast.frontends`` carries both
    ``fortran`` and the domain extension's ``cesm``, and the second is not a
    new analysis -- it is the engine's own ``FortranFrontend`` constructed with
    CAM's kind table. One implementation, two ways to ask for it, and what
    differs between the two is recorded in ``Facts.provenance`` as the
    configuration it actually is.

    What must not happen is two *different* implementations answering to one
    name, because then the name in a Verdict, a Candidate, or a provenance
    record no longer says which code ran, and nothing downstream can recover
    it. The registry name stays unique on its own -- ``Registry.register``
    refuses a silent override -- so this is the half that is not already
    guaranteed.
    """
    collisions: list[str] = []
    for kind in KINDS:
        seen: dict[str, list[tuple[str, Any]]] = {}
        for registered in REGISTRY.names(kind):
            name, implementation = _identify(REGISTRY.get(kind, registered))
            if name:
                seen.setdefault(name, []).append((registered, implementation))
        for name, entries in seen.items():
            implementations = {implementation for _, implementation in entries}
            if len(implementations) > 1:
                collisions.append(
                    f"{kind} name {name!r} is claimed by "
                    + ", ".join(
                        f"{registered!r} ({getattr(impl, '__qualname__', impl)})"
                        for registered, impl in sorted(entries)
                    )
                )
    assert not collisions, (
        "two different implementations answer to one name, so a record naming it "
        "cannot say which ran:\n" + "\n".join(collisions)
    )
