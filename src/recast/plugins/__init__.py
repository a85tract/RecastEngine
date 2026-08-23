"""The extension contract.

Everything in RecastEngine's core is written against these ABCs. Nothing in the
core imports a concrete Fortran, domain, JAX, or scheduler implementation.

That constraint is the whole point: it is what lets a domain extension supply
the domain knowledge, and what lets an extension add capability at runtime instead
of forking the engine.

Stability: these signatures follow SemVer from 1.0. A minor release must not
break a plugin that satisfied the previous minor.
"""

from recast.plugins.adjudicator import Adjudicator
from recast.plugins.agent import AgentCall, AgentProvider, AgentResult
from recast.plugins.executor import Executor, Job, JobResult
from recast.plugins.frontend import Frontend
from recast.plugins.oracle import Oracle
from recast.plugins.recipe import Recipe, Stage
from recast.plugins.scanner import Scanner
from recast.plugins.store import EvidenceStore, FindingStore
from recast.plugins.transform import Transform
from recast.plugins.verifier import StaticVerifier, Verifier

__all__ = [
    "Adjudicator",
    "AgentCall",
    "AgentProvider",
    "AgentResult",
    "EvidenceStore",
    "Executor",
    "FindingStore",
    "Frontend",
    "Job",
    "JobResult",
    "Oracle",
    "Recipe",
    "Scanner",
    "Stage",
    "StaticVerifier",
    "Transform",
    "Verifier",
]
