"""The compiled reference in a process of its own.

A Fortran ``error stop`` in an f2py module is ``exit()`` in the process that
imported it: the verifier, the run, every other unit's verdict. CLUBB_core
has dozens of them, on inputs a generated draw reaches easily, and until
this module the answer was one process per unit outside the engine (#21).

``IsolatedModule`` stands in for the imported module. It starts a worker
process that imports the extension, forwards each call as a pickled
message and reads the result back -- arrays included, and the arguments
after the call, so an ``intent(inout)`` array the wrapper wrote in place is
written into the caller's array here as it would have been in-process.
When the worker ends instead of answering, the call raises
``ReferenceAborted`` naming the exit status and the tail of what the worker
wrote (the ``error stop`` message, the Fortran runtime's line), the worker
is gone, and the next call starts a fresh one: the verifier records the
draw as declined by the reference and draws again, and the run goes on.

The worker moves the process's stdout aside before it imports anything:
libgfortran writes unit 6 to file descriptor 1 directly, and a PRINT in the
reference would otherwise land in the middle of a pickled reply. Everything
the reference prints goes to the worker's log with its stderr.
"""

from __future__ import annotations

import contextlib
import os
import pickle
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from recast.errors import RecastError

__all__ = ["IsolatedModule", "ReferenceAborted", "ReferenceRaised"]


class ReferenceAborted(RecastError):
    """The reference process ended during a call -- an ``error stop``, a
    signal -- instead of answering."""


class ReferenceRaised(RecastError):
    """The reference raised a Python exception in its own process: what an
    in-process f2py call would have raised (a shape it cannot take)."""


_WORKER = r'''
import importlib, os, pickle, struct, sys

def _read():
    head = sys.stdin.buffer.read(8)
    if len(head) < 8:
        return None
    (size,) = struct.unpack("!Q", head)
    return pickle.loads(sys.stdin.buffer.read(size))

# The protocol keeps the original stdout; fd 1 goes to stderr so that a
# PRINT from the reference (libgfortran writes fd 1 directly) cannot land
# inside a reply.
_out = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr  # Python's own prints, unbuffered where the log is

def _write(obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    _out.write(struct.pack("!Q", len(data)) + data)
    _out.flush()

stage, name = sys.argv[1], sys.argv[2]
sys.path.insert(0, stage)
try:
    module = importlib.import_module(name)
except BaseException as error:  # the parent reports it, not this process
    _write(("failed", f"{type(error).__name__}: {error}"))
    raise SystemExit(1)
_write(("ready", sorted(n for n in dir(module) if not n.startswith("__"))))
while True:
    message = _read()
    if message is None:
        break
    fn, args, kwargs = message
    try:
        result = getattr(module, fn)(*args, **kwargs)
    except Exception as error:
        _write(("raised", f"{type(error).__name__}: {error}"))
        continue
    _write(("ok", result, args, kwargs))
'''


class IsolatedModule:
    """A stand-in for an imported f2py module whose functions run in a
    worker process. ``getattr(module, name)`` is a callable for every name
    the worker's module has; anything else is an ``AttributeError``, as it
    would be on the module."""

    def __init__(self, stage: Path, module_name: str, log_dir: Path | None = None) -> None:
        self._stage = Path(stage)
        self._module_name = module_name
        self._log_dir = Path(log_dir) if log_dir is not None else self._stage
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: Any = None
        self._names: set[str] | None = None
        self.restarts = 0
        """How many times the worker had to be started again after it ended."""

    # -- lifecycle ----------------------------------------------------------

    def _start(self) -> None:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._log = tempfile.NamedTemporaryFile(
            "w+b", prefix=f"{self._module_name}-worker-", suffix=".log", dir=self._log_dir
        )
        self._proc = subprocess.Popen(  # noqa: S603 -- our own worker, our own argv
            [sys.executable, "-c", _WORKER, str(self._stage), self._module_name],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._log,
            # libgfortran buffers a preconnected unit written to a file
            # until exit; the log has to show what the reference printed
            # before it stopped.
            env={**os.environ, "GFORTRAN_UNBUFFERED_PRECONNECTED": "y"},
        )
        reply = self._read()
        if reply is None or reply[0] != "ready":
            why = reply[1] if reply else "no reply"
            self._end()
            raise ReferenceAborted(
                f"reference {self._module_name!r} did not start: {why}"
                + self._tail(" -- worker log:\n")
            )
        self._names = set(reply[1])

    def _end(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        for stream in (proc.stdin, proc.stdout):
            try:
                if stream is not None:
                    stream.close()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

    def close(self) -> None:
        """End the worker. The next call starts a new one."""
        self._end()
        if self._log is not None:
            self._log.close()
            self._log = None

    def __del__(self) -> None:  # pragma: no cover - interpreter shutdown order
        with contextlib.suppress(Exception):
            self.close()

    # -- the protocol ----------------------------------------------------

    def _read(self) -> Any:
        assert self._proc is not None and self._proc.stdout is not None
        head = self._proc.stdout.read(8)
        if len(head) < 8:
            return None
        (size,) = struct.unpack("!Q", head)
        data = self._proc.stdout.read(size)
        if len(data) < size:
            return None
        return pickle.loads(data)  # noqa: S301 -- our own worker's reply

    def _tail(self, prefix: str = "", lines: int = 12) -> str:
        if self._log is None:
            return ""
        try:
            self._log.flush()
            self._log.seek(0)
            text = self._log.read().decode(errors="replace")
        except OSError:
            return ""
        kept = [line for line in text.splitlines() if line.strip()][-lines:]
        return prefix + "\n".join(kept) if kept else ""

    def _call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        if self._proc is None:
            if self._names is not None:
                self.restarts += 1
            self._start()
        assert self._proc is not None and self._proc.stdin is not None
        data = pickle.dumps((name, args, kwargs), protocol=pickle.HIGHEST_PROTOCOL)
        try:
            self._proc.stdin.write(struct.pack("!Q", len(data)) + data)
            self._proc.stdin.flush()
            reply = self._read()
        except (BrokenPipeError, OSError):
            reply = None
        if reply is None:
            code = self._proc.poll()
            tail = self._tail(" -- last lines the reference wrote:\n")
            self._end()
            raise ReferenceAborted(
                f"reference {self._module_name}.{name} ended the process"
                f" (exit {code if code is not None else '?'}) instead of answering{tail}"
            )
        if reply[0] == "raised":
            raise ReferenceRaised(f"{self._module_name}.{name}: {reply[1]}")
        _, result, args_after, kwargs_after = reply
        # What the wrapper wrote in place comes back as new arrays; the
        # caller holds the originals, and reads INOUT results from them.
        for before, after in zip(args, args_after, strict=True):
            _copy_back(before, after)
        for key, after in kwargs_after.items():
            _copy_back(kwargs.get(key), after)
        return result

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        if self._names is None:
            self._start()
        assert self._names is not None
        if name not in self._names:
            raise AttributeError(f"module {self._module_name!r} has no attribute {name!r}")

        def bound(*args: Any, **kwargs: Any) -> Any:
            return self._call(name, *args, **kwargs)

        bound.__name__ = name
        return bound

    def __dir__(self) -> list[str]:
        if self._names is None:
            self._start()
        return sorted(self._names or ())


def _copy_back(before: Any, after: Any) -> None:
    """An array the worker returned as modified, written into the array the
    caller passed. Scalars and everything immutable are left alone: the
    wrapper returns scalar INOUTs, as f2py does."""
    if before is None or after is None or before is after:
        return
    if hasattr(before, "shape") and hasattr(after, "shape") and hasattr(before, "__setitem__"):
        try:
            if before.shape == after.shape:
                before[...] = after
        except (TypeError, ValueError):
            pass
