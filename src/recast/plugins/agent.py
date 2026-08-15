"""AgentProvider: the LLM boundary.

The open-source engine ships a single-agent loop and a provider-agnostic
interface. Multi-agent orchestration, budget and policy control, retrieval over
a validated corpus, and batch patch adjudication are RecastRuntime's.

Every call is recorded. A non-deterministic transform is acceptable only if the
run that produced it can be reconstructed, so ``AgentCall`` carries the model,
the prompt digest, and the sampling parameters into ``Candidate.notes``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentCall:
    """One request to a model, in a form that can be replayed and audited."""

    task: str
    """What the engine wanted, e.g. ``synthesize-rule``, ``repair-deferred-site``."""

    prompt: str
    context: dict[str, Any] = field(default_factory=dict)
    tools: tuple[str, ...] = ()
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass
class AgentResult:
    text: str
    model: str
    prompt_digest: str
    usage: dict[str, int] = field(default_factory=dict)
    stop_reason: str = ""


class AgentProvider(ABC):
    """A model that can be asked for structured help.

    Implementations must not silently retry with a different model. If a
    fallback occurs, ``AgentResult.model`` reports the model that actually
    answered -- otherwise the provenance recorded in the Candidate is a lie.
    """

    name: str
    default_model: str = ""

    @abstractmethod
    def complete(self, call: AgentCall) -> AgentResult: ...

    def budget_remaining(self) -> int | None:
        """Output tokens left, if the provider enforces a budget. ``None`` = unmetered."""
        return None
