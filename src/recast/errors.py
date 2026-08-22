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


class ScannerUnavailable(RecastError):
    """A Scanner or Adjudicator could not run at all. The stage is INCOMPLETE.

    The counterpart to ``OracleUnavailable``, raised for the same reason and
    answered differently. ``scan`` returns an iterable, so a scanner whose tool
    is missing yields nothing -- which is byte-for-byte what a clean scan
    yields. A security gate that cannot tell "found nothing" from "never ran"
    reports the second as the first, which is the failure this engine refuses
    everywhere else.

    Distinct from an uninstalled *optional plugin*, which the runner reports as
    ``skipped``. That is a declaration the operator made when they left it out
    of the environment; this is a plugin that is installed, was asked, and could
    not answer.
    """


class AccessViolation(RecastError):
    """An attempt to write a record into a store not cleared to hold it.

    Never catch this to continue. It is raised when an embargoed finding was
    about to reach a less-restricted destination, and the correct response is to
    stop and audit how it got routed there.
    """
