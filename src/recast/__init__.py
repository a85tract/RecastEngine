"""RecastEngine -- an agentic engine for modernizing legacy scientific software.

Four workloads over one spine: translate a language, refactor an architecture,
port to an accelerator, audit for vulnerabilities. What differs between them is
which plugin fills each slot; the spine does not change.

    >>> from recast.recipes import BUILTIN
    >>> sorted(BUILTIN)
    ['audit', 'port', 'refactor-todo', 'translate']
"""

from __future__ import annotations

__version__ = "0.0.1.dev0"

from recast.engines import (
    ArtifactContract,
    TranslationEngine,
    python_jax_engine,
    python_numba_engine,
)
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
from recast.observe import RunEvent, RunEventAction, RunEventEntity, RunObserver

WORKSPACE_DIRNAME = ".recast"
"""Where per-machine state that is not a run's output lives.

Only the embargoed finding store uses it now, at ``~/.recast/findings``. It
stays in a ``Frontend``'s skip list because trees carrying a pre-``output/``
run still have one, and reading a previous run's generated code back in turns
the engine's output into its input.
"""

OUTPUT_DIRNAME = "output"
"""The directory a run's candidates and evidence are written under.

One level down is the source project's name -- ``output/toy_physics/`` -- so
that runs over different trees stay apart and a person looking for what the
engine produced has one place to look. It deliberately does not live inside
the tree it was produced from: generated code sitting in the source is one
``git add -A`` from being committed as if it were source, and a discovery pass
that reads it back finds units the previous run created.

``config["output"]`` overrides the whole path; ``config["workspace"]``
overrides only the per-recipe half.
"""

__all__ = [
    "OUTPUT_DIRNAME",
    "WORKSPACE_DIRNAME",
    "Access",
    "AccessViolation",
    "ArtifactContract",
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
    "RunEvent",
    "RunEventAction",
    "RunEventEntity",
    "RunObserver",
    "Severity",
    "TranslationEngine",
    "Unit",
    "Verdict",
    "__version__",
    "python_jax_engine",
    "python_numba_engine",
]
