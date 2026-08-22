"""AgentProvider: the LLM boundary.

The engine ships a single-agent loop and a provider-agnostic interface.
Multi-agent orchestration, budget and policy control, retrieval over a validated
corpus, and batch patch adjudication are left to plugins.

Every call is recorded. A non-deterministic transform is acceptable only if the
run that produced it can be reconstructed, so ``AgentCall`` carries the model,
the prompt digest, and the sampling parameters into ``Candidate.notes``.

What the contract permits, stated because the security review asked: a
provider's ``AgentResult.text`` can become the body of a deferred site, which
is emitted into the candidate module, which a Verifier imports into this
process. So a model's output runs here, with the operator's privileges, and
nothing in this contract sandboxes it. The gates downstream are about
correctness -- does it compute what the original computed -- not about what
else it does on the way. That is the same trust the engine extends to the
operator's own patches in ``config["patches"]``, and the operator who enables
a provider is the boundary. An implementation that wants less trust puts the
sandbox in the Executor the Verifier is handed, which is where execution was
always meant to be confined.
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
