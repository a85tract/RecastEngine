"""Executor: refuses what it cannot honestly satisfy, and answers the same twice.

The refusal check is the load-bearing one. An executor that quietly runs a
512-rank gate on one rank produces a passing Verdict about a comparison that
never happened, and nothing downstream can tell. Refusing is the only safe
behaviour, so the suite asks for something the case says this executor cannot
deliver and requires it to say no.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from recast.plugins.executor import Job


def test_refuses_what_it_cannot_satisfy(
    executor_case: Any, build_executor: Any, probe_job: Any, tmp_path: Path
) -> None:
    executor = build_executor(executor_case)
    probe = probe_job(executor_case, tmp_path)
    job = Job(
        argv=probe.argv,
        cwd=probe.cwd,
        env=dict(probe.env),
        resources=dict(executor_case.unsatisfiable),
        label="conformance-unsatisfiable",
    )
    # Refusal may come at submit or at wait -- an executor that queues first and
    # discovers the queue will not take it is still refusing. What it may not do
    # is hand back a result, because a result means it ran something, and what it
    # ran was not what was asked for.
    try:
        handle = executor.submit(job)
        result = executor.wait(handle)
    except Exception:  # any refusal is a refusal, whatever it is raised as
        return
    pytest.fail(
        f"executor {executor_case.name!r} accepted {dict(executor_case.unsatisfiable)} "
        f"and returned {result!r}; it must refuse rather than run the job at a scale "
        "nobody asked for"
    )


def test_wait_is_idempotent(
    executor_case: Any, build_executor: Any, probe_job: Any, tmp_path: Path
) -> None:
    """Two waits on one handle are two readings of one run, not two runs."""
    executor = build_executor(executor_case)
    handle = executor.submit(probe_job(executor_case, tmp_path))
    assert isinstance(handle, str) and handle, "submit must return a handle"

    first = executor.wait(handle)
    second = executor.wait(handle)
    assert (first.returncode, first.stdout, first.stderr) == (
        second.returncode,
        second.stdout,
        second.stderr,
    ), "waiting twice on one handle reported two different outcomes"


def test_cancel_on_an_unknown_handle_is_a_no_op(executor_case: Any, build_executor: Any) -> None:
    """Cleanup runs after failures, when the handle may never have existed."""
    build_executor(executor_case).cancel("conformance-no-such-handle")


def test_run_reports_what_the_job_did(
    executor_case: Any, build_executor: Any, probe_job: Any, tmp_path: Path
) -> None:
    """The convenience path has to carry the child's output back, not just its code."""
    executor = build_executor(executor_case)
    result = executor.run(probe_job(executor_case, tmp_path))
    assert result.ok == (result.returncode == 0)
    if executor_case.probe is None:
        assert "conformance probe" in result.stdout, (
            "the probe printed to stdout and the JobResult did not carry it"
        )
