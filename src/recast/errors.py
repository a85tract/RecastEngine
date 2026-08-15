"""Engine exceptions.

Split by who is at fault, because that determines what the caller should do:
a ``ConfigError`` means fix the recipe, a ``PluginError`` means fix the plugin,
an ``AccessViolation`` means stop the pipeline and involve a human.
"""

from __future__ import annotations


class RecastError(Exception):
    """Base for everything this package raises."""


class ConfigError(RecastError):
    """The recipe, config, or CLI invocation is wrong. Fixable by the operator."""


class PluginError(RecastError):
    """A plugin misbehaved: bad registration, unmet contract, wrong return type."""


class PluginNotFound(PluginError):
    def __init__(self, kind: str, name: str, available: tuple[str, ...] = ()) -> None:
        hint = f" Available: {', '.join(sorted(available))}." if available else ""
        super().__init__(f"no {kind} plugin named {name!r}.{hint}")
        self.kind = kind
        self.name = name


class PrerequisiteError(RecastError):
    """A Transform declared ``requires`` that the Facts do not satisfy."""


class OracleUnavailable(RecastError):
    """The reference could not be materialized. Verdicts must be FAILED, not skipped."""


class AccessViolation(RecastError):
    """An attempt to write a record into a store not cleared to hold it.

    Never catch this to continue. It is raised when an embargoed finding was
    about to reach a less-restricted destination, and the correct response is to
    stop and audit how it got routed there.
    """
