"""Carry a candidate's probes over to the reference source, mechanically.

The paper's supervisor agent adds ``gate.h`` and a block of ``GATE_*`` calls
to the golden program once, then the exact same block to the candidate. When
the instrumented golden was not kept, the gate re-derives it from the
candidate's: the ``GATE_`` statements are copied verbatim and placed in the
reference source at the line that precedes them in the candidate -- the same
``free``, ``printf`` or closing brace, matched with whitespace and ``std::``
normalised, three lines of context first, then one, then the following line.
Ambiguous or absent anchors are a refusal, never a guess.

The probes only *observe* buffers; nothing about the reference program's
behaviour changes, which is what keeps it the oracle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = ["ProbeBlock", "ProbeInjectionError", "extract_probes", "has_probes", "inject"]

_INCLUDE_GATE = re.compile(r'^\s*#\s*include\s+"([^"]*gate\.h)"')
_GATE_CALL = re.compile(r"^\s*GATE_[A-Z0-9_]+\s*\(")
_ANY_INCLUDE = re.compile(r"^\s*#\s*include\b")


class ProbeInjectionError(Exception):
    """The candidate's probes could not be placed in the reference source unambiguously."""


@dataclass(frozen=True)
class ProbeBlock:
    lines: tuple[str, ...]
    """The ``GATE_*`` statements, complete (a call may span lines)."""

    anchor_before: str | None
    """The nearest preceding non-blank line in the candidate, stripped."""

    anchor_after: str | None
    """The nearest following non-blank line, stripped."""

    context_before: tuple[str, ...] = ()
    """Up to three preceding non-blank lines, oldest first. Tried first: a
    run of three lines is unique where a lone ``}`` is not."""


def has_probes(source: str) -> bool:
    """Whether a source already carries ``GATE_`` calls."""
    return any(_GATE_CALL.match(ln) for ln in source.splitlines())


def _norm(line: str) -> str:
    return re.sub(r"\s+", "", line.replace("std::", ""))


def _statement_end(lines: list[str], start: int) -> int:
    depth = 0
    for i in range(start, len(lines)):
        depth += lines[i].count("(") - lines[i].count(")")
        if depth <= 0 and lines[i].rstrip().endswith(";"):
            return i
    raise ProbeInjectionError(f"unterminated GATE statement at line {start + 1}")


def extract_probes(candidate_source: str) -> tuple[str | None, list[ProbeBlock]]:
    """The gate include line and every contiguous block of probe statements."""
    lines = candidate_source.splitlines()
    include = next((ln for ln in lines if _INCLUDE_GATE.match(ln)), None)
    blocks: list[ProbeBlock] = []
    i = 0
    while i < len(lines):
        if not _GATE_CALL.match(lines[i]):
            i += 1
            continue
        start = i
        end = _statement_end(lines, i)
        j = end + 1
        while j < len(lines):
            if lines[j].strip() == "":
                j += 1
                continue
            if _GATE_CALL.match(lines[j]):
                end = _statement_end(lines, j)
                j = end + 1
                continue
            break
        preceding = [lines[k].strip() for k in range(start - 1, -1, -1) if lines[k].strip()][:3]
        after = next(
            (lines[k].strip() for k in range(end + 1, len(lines)) if lines[k].strip()), None
        )
        blocks.append(
            ProbeBlock(
                tuple(lines[start : end + 1]),
                preceding[0] if preceding else None,
                after,
                tuple(reversed(preceding)),
            )
        )
        i = end + 1
    return include, blocks


def _unique_match(lines: list[str], anchor: str) -> int | None:
    want = _norm(anchor)
    hits = [i for i, ln in enumerate(lines) if _norm(ln) == want]
    if len(hits) == 1:
        return hits[0]
    # A print whose text drifted between the two sources: match on the call
    # up to its first argument (``printf("Average kernel execution time``).
    head = want.split(",", 1)[0]
    if "(" in head and len(head) > 12:
        hits = [i for i, ln in enumerate(lines) if _norm(ln).startswith(head)]
        if len(hits) == 1:
            return hits[0]
    return None


def _unique_run(lines: list[str], context: tuple[str, ...]) -> int | None:
    """Index of the last line of the one place ``context`` occurs as consecutive
    non-blank lines, or None if it occurs zero or several times."""
    if not context:
        return None
    nonblank = [(i, _norm(ln)) for i, ln in enumerate(lines) if ln.strip()]
    stripped = [t for _, t in nonblank]
    want = tuple(_norm(c) for c in context)
    n = len(want)
    hits = [k for k in range(len(stripped) - n + 1) if tuple(stripped[k : k + n]) == want]
    return nonblank[hits[0] + n - 1][0] if len(hits) == 1 else None


def inject(reference_source: str, candidate_source: str) -> str:
    """The reference source with the candidate's probes in the corresponding places."""
    include, blocks = extract_probes(candidate_source)
    if not blocks:
        raise ProbeInjectionError("candidate carries no GATE_ statements")
    lines = reference_source.splitlines()
    if has_probes(reference_source):
        raise ProbeInjectionError("reference source already carries GATE_ statements")

    placements: list[tuple[int, tuple[str, ...]]] = []
    for block in blocks:
        at: int | None = None
        hit = _unique_run(lines, block.context_before)
        if hit is not None:
            at = hit + 1
        if at is None and block.anchor_before is not None:
            hit = _unique_match(lines, block.anchor_before)
            if hit is not None:
                at = hit + 1
        if at is None and block.anchor_after is not None:
            hit = _unique_match(lines, block.anchor_after)
            if hit is not None:
                at = hit
        if at is None:
            raise ProbeInjectionError(
                "no unique anchor in the reference source for the probe block after "
                f"{block.anchor_before!r} / before {block.anchor_after!r}"
            )
        placements.append((at, block.lines))
    for at, body in sorted(placements, key=lambda p: p[0], reverse=True):
        lines[at:at] = list(body)

    if include is not None and not any(_INCLUDE_GATE.match(ln) for ln in lines):
        last_include = max((i for i, ln in enumerate(lines) if _ANY_INCLUDE.match(ln)), default=-1)
        # Spelled bare: the gate SDK is on every build's include path, and the
        # reference is not always staged at the candidate's depth.
        lines.insert(last_include + 1, '#include "gate.h"')
    return "\n".join(lines) + "\n"
