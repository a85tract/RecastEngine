"""The local executor. Runs Jobs as subprocesses on this machine.

This is the whole of the in-tree execution story, and that is intentional: it is
enough to translate a kernel, build an f2py oracle, and run a differential gate
on a laptop or a login node. PBS/Slurm submission, cross-cluster routing, and
relay/resume of multi-day runs belong in executor plugins.
"""

from __future__ import annotations

import subprocess
import uuid
from dataclasses import dataclass, field
from typing import Any

from recast.plugins.executor import Executor, Job, JobResult


@dataclass
class LocalExecutor(Executor):
    """Fire-and-collect subprocess execution."""

    name: str = "local"
    supports_batch: bool = False

    _pending: dict[str, Job] = field(default_factory=dict)
    _done: dict[str, JobResult] = field(default_factory=dict)

    def submit(self, job: Job) -> str:
        self._refuse_unsatisfiable(job)
        handle = f"local-{uuid.uuid4().hex[:12]}"
        self._pending[handle] = job
        return handle

    def wait(self, handle: str, timeout_s: float | None = None) -> JobResult:
        if handle in self._done:
            return self._done[handle]
        job = self._pending.pop(handle)
        try:
            proc: subprocess.CompletedProcess[str] = subprocess.run(  # noqa: S603
                list(job.argv),
                cwd=job.cwd,
                env={**job.env} or None,
                capture_output=True,
                text=True,
                timeout=timeout_s or job.timeout_s,
                check=False,
            )
            result = JobResult(proc.returncode, proc.stdout, proc.stderr)
        except subprocess.TimeoutExpired as exc:
            # A timed-out child may hand back either str or bytes depending on
            # how far it got; normalize so partial output is still reportable.
            partial = exc.stdout or b""
            text = partial.decode(errors="replace") if isinstance(partial, bytes) else partial
            result = JobResult(124, text, f"timeout after {exc.timeout}s")
        except OSError as exc:
            result = JobResult(127, "", str(exc))
        self._done[handle] = result
        return result

    @staticmethod
    def _refuse_unsatisfiable(job: Job) -> None:
        """Decline what this executor cannot honestly deliver.

        A 512-rank bit-for-bit gate silently downgraded to one rank on a laptop
        would produce a passing Verdict that means nothing. Refusing is the only
        safe behaviour; the operator picks a real executor or a smaller case.
        """
        ranks = int(job.resources.get("ranks", 1) or 1)
        nodes = int(job.resources.get("nodes", 1) or 1)
        if ranks > 1 or nodes > 1:
            raise RuntimeError(
                f"local executor cannot run {nodes} node(s) x {ranks} rank(s) "
                f"for job {job.label!r}; use a batch executor"
            )


def factory(**_config: Any) -> LocalExecutor:
    return LocalExecutor()
