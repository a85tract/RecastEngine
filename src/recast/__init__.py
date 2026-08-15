"""RecastEngine -- an agentic engine for modernizing legacy scientific software.

Four workloads over one spine: translate a language, refactor an architecture,
port to an accelerator, audit for vulnerabilities. What differs between them is
which plugin fills each slot; the spine does not change.

    >>> from recast.recipes import BUILTIN
    >>> sorted(BUILTIN)
    ['audit', 'port', 'refactor', 'translate']
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

from recast.errors import (
    AccessViolation,
    ConfigError,
    PluginError,
    PluginNotFound,
    RecastError,
)
from recast.model import (
    Access,
    Candidate,
    Confidence,
    Disclosure,
    Evidence,
    Facts,
    Finding,
    OracleRef,
    Patch,
    Severity,
    Unit,
    Verdict,
)

__all__ = [
    "Access",
    "AccessViolation",
    "Candidate",
    "Confidence",
    "ConfigError",
    "Disclosure",
    "Evidence",
    "Facts",
    "Finding",
    "OracleRef",
    "Patch",
    "PluginError",
    "PluginNotFound",
    "RecastError",
    "Severity",
    "Unit",
    "Verdict",
    "__version__",
]
