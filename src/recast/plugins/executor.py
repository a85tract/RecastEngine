"""Executor: where work actually runs.

The engine ships ``local`` and nothing else -- enough to translate and verify a
kernel on one machine. Batch schedulers (PBS, Slurm), cross-cluster submission,
relay/resume of multi-day runs, and queue arbitrage belong in executor plugins;
they plug in here without the core changing.

A recipe names its executor through config rather than hardcoding one, because
the name of a real one (``pbs-<site>``, say) is site knowledge and does not
belong in this repository. ``Oracle`` and ``Verifier`` receive the resolved
instance as an argument; see ``recipe.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Job:
    """A single piece of work to run somewhere."""

    argv: Sequence[str]
    cwd: Path
    env: dict[str, str] = field(default_factory=dict)
    timeout_s: float | None = None

    resources: dict[str, Any] = field(default_factory=dict)
    """Scheduler hints: ``{"nodes": 4, "ranks": 512, "gpus": 1, "queue": "main"}``.

    ``local`` ignores everything here except as a refusal check -- it declines
    jobs it cannot honestly satisfy rather than silently running them at the
    wrong scale.
    """

    label: str = ""


@dataclass
class JobResult:
    returncode: int
    stdout: str
    stderr: str
    artifacts: dict[str, Path] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.returncode == 0


class Executor(ABC):
    """Run Jobs. Knows nothing about what they compute."""

    name: str
    supports_batch: bool = False

    @abstractmethod
    def submit(self, job: Job) -> str:
        """Enqueue and return a handle. Must not block on completion."""

    @abstractmethod
    def wait(self, handle: str, timeout_s: float | None = None) -> JobResult:
        """Block until the job finishes or the timeout expires."""

    def run(self, job: Job) -> JobResult:
        """Convenience: submit then wait."""
        return self.wait(self.submit(job), job.timeout_s)

    def cancel(self, handle: str) -> None:
        return None
