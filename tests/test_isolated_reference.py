"""The compiled reference in a process of its own (#21)."""

from __future__ import annotations

from pathlib import Path

import pytest

from recast.oracle.isolated import IsolatedModule, ReferenceAborted, ReferenceRaised

FAKE = """\
import os, sys
import numpy as np


def w_probe(x):
    if x < 0:
        sys.stderr.write("STOP negative argument\\n")
        sys.stderr.flush()
        os._exit(3)  # what a Fortran ERROR STOP does to the process
    print("a line the reference prints")  # unit 6 goes to fd 1
    return x * 2.0


def w_inout(a, k):
    a[0] = 7.0  # written in place, as an intent(inout) array is
    return k + 1


def w_bad(a):
    raise ValueError("wrong shape")
"""


@pytest.fixture
def stage(tmp_path: Path) -> Path:
    (tmp_path / "fake_ref.py").write_text(FAKE)
    return tmp_path


def test_calls_cross_the_process_and_inout_arrays_come_back(stage: Path) -> None:
    import numpy as np

    module = IsolatedModule(stage, "fake_ref")
    try:
        assert module.w_probe(2.0) == 4.0
        a = np.zeros(3)
        assert module.w_inout(a, 1) == 2
        assert a.tolist() == [7.0, 0.0, 0.0], "the caller's array carries the write"
        assert "w_probe" in dir(module)
        with pytest.raises(AttributeError):
            _ = module.w_missing
        with pytest.raises(ReferenceRaised, match="wrong shape"):
            module.w_bad(a)
        assert module.restarts == 0
    finally:
        module.close()


def test_a_reference_that_ends_its_process_raises_and_the_next_call_starts_again(
    stage: Path,
) -> None:
    """The process ends -- an ERROR STOP -- and the caller gets an exception
    naming the exit status and what the reference wrote, not a dead
    interpreter. The call after it is answered by a fresh worker."""
    module = IsolatedModule(stage, "fake_ref")
    try:
        with pytest.raises(ReferenceAborted) as caught:
            module.w_probe(-1.0)
        message = str(caught.value)
        assert "exit 3" in message and "STOP negative argument" in message, message
        assert module.w_probe(3.0) == 6.0
        assert module.restarts == 1
    finally:
        module.close()


def test_what_the_reference_prints_never_reaches_the_reply(stage: Path) -> None:
    """A PRINT from the reference goes to fd 1, where the reply stream was;
    the worker moves the stream aside first, so the print lands in the log."""
    module = IsolatedModule(stage, "fake_ref")
    try:
        for _ in range(3):
            assert module.w_probe(1.0) == 2.0
        assert "a line the reference prints" in module._tail()
    finally:
        module.close()
