"""Fake external tools, for checking a Scanner that wraps one.

A Scanner that shells out to gitleaks or syft has two halves: deciding what to
point the tool at, and reading what came back. Substituting a fake *plugin*
checks neither -- it replaces the code under test. Substituting a fake *tool*,
on PATH, leaves the plugin's argv construction, its subprocess handling and its
report parsing in the run, which is where the interesting failures are.

The technique is borrowed from ``tests/run.sh`` in ``hpc-devsecops``
(a85tract/CESM-CC-Test, by Chien-Wei Huang), which generates stub scanners and
asserts on the gate's exit codes. Nothing is copied -- that harness is shell and
this is a pytest fixture -- but the idea that the honest place to fake a wrapped
tool is PATH rather than the wrapper is theirs, and it is a better test than
faking the wrapper.

Ships here rather than inside ``conformance/`` because an out-of-tree scanner
author needs it more than this repository does: nothing in the engine wraps an
external security tool, and every plugin that does will live somewhere else.
"""

from __future__ import annotations

import json
import os
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["fake_tool", "on_path"]

_STUB = '''\
#!{python}
"""A fake {name}, written by recast.conformance.fake_tool."""
import sys

argv = sys.argv[1:]
report = None
for flag in {flags!r}:
    if flag in argv:
        index = argv.index(flag)
        if index + 1 < len(argv):
            report = argv[index + 1]
        break
    for item in argv:
        if item.startswith(flag + "="):
            report = item.split("=", 1)[1]
            break
    if report is not None:
        break

payload = {payload!r}
if report is not None and payload is not None:
    with open(report, "w") as handle:
        handle.write(payload)
elif payload is not None:
    sys.stdout.write(payload)

sys.stderr.write({stderr!r})
sys.exit({exit_code!r})
'''


def fake_tool(
    directory: Path,
    name: str,
    *,
    sarif: dict[str, Any] | None = None,
    payload: str | None = None,
    exit_code: int = 0,
    stderr: str = "",
    report_flags: tuple[str, ...] = ("--report-path", "-o", "--output"),
) -> Path:
    """Write an executable ``name`` into ``directory`` and return its path.

    It writes ``sarif`` (or the raw ``payload``) to whichever path follows one
    of ``report_flags`` in its argv, falling back to stdout when the caller's
    tool does not take one, and exits with ``exit_code``.

    ``payload`` is deliberately a string rather than only a document, because
    the case worth checking most is the one a document cannot express: output
    that is not valid JSON at all. A converter that answers "clean" to that is
    the bug this whole apparatus is for.
    """
    if sarif is not None and payload is not None:
        raise ValueError("pass sarif or payload, not both")
    body = json.dumps(sarif) if sarif is not None else payload
    path = directory / name
    path.write_text(
        _STUB.format(
            python=sys.executable,
            name=name,
            flags=report_flags,
            payload=body,
            stderr=stderr,
            exit_code=exit_code,
        )
    )
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


@contextmanager
def on_path(directory: Path, *, only: bool = False) -> Iterator[None]:
    """Put ``directory`` at the front of PATH for the duration.

    ``only`` replaces PATH outright, which is how you check the other half:
    what a Scanner does when the tool is not installed anywhere. That case has
    no fake to write -- the absence *is* the fixture.
    """
    previous = os.environ.get("PATH", "")
    os.environ["PATH"] = str(directory) if only else f"{directory}{os.pathsep}{previous}"
    try:
        yield
    finally:
        os.environ["PATH"] = previous
