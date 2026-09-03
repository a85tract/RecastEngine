"""A lexical scan of C/C++ sources: what a serial kernel is made of.

Counts, not semantics -- ``for`` and ``while`` occurrences, the deepest
nesting by brace counting, allocations, a timer, OpenMP pragmas, includes.
It is what the analysis stage of a translation planner records before
writing any pragma, and it is honest about being a regex: ``provenance``
says so, and a transform that needs a parse brings its own.
"""

from __future__ import annotations

import re

__all__ = ["Scan", "scan"]

_FUNCTION = re.compile(
    r"^[ \t]*(?:static\s+|inline\s+|extern\s+)*"
    r"(?:const\s+)?[A-Za-z_][\w:<>,\s\*&]*?[\s\*&]+"
    r"([A-Za-z_]\w*)\s*\([^;{)]*\)\s*(?:const\s*)?\{",
    re.MULTILINE,
)
_LOOP = re.compile(r"\b(for|while)\s*\(")
_ALLOC = re.compile(r"\b(malloc|calloc|aligned_alloc|new\s+\w+\s*\[|std::vector<)")
_TIMER = re.compile(
    r"\b(std::chrono|clock\(\)|omp_get_wtime|gettimeofday|clock_gettime|timer_start)\b"
)
_PRAGMA = re.compile(r"^\s*#\s*pragma\s+omp\b", re.MULTILINE)
_TARGET = re.compile(r"^\s*#\s*pragma\s+omp\s+target\b", re.MULTILINE)
_INCLUDE = re.compile(r'^\s*#\s*include\s+[<"]([^>"]+)[>"]', re.MULTILINE)
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "else", "do", "catch"}


class Scan:
    """What one body of source showed the scanner."""

    def __init__(self, text: str) -> None:
        self.functions: list[str] = sorted(
            {m.group(1) for m in _FUNCTION.finditer(text)} - _KEYWORDS
        )
        self.loops: int = len(_LOOP.findall(text))
        self.loop_depth: int = _loop_depth(text)
        self.allocations: int = len(_ALLOC.findall(text))
        self.timed_region: bool = bool(_TIMER.search(text))
        self.omp_pragmas: int = len(_PRAGMA.findall(text))
        self.target_regions: int = len(_TARGET.findall(text))
        self.includes: list[str] = sorted(set(_INCLUDE.findall(text)))
        self.lines: int = text.count("\n") + 1


def _loop_depth(text: str) -> int:
    """Deepest nesting of ``for``/``while`` bodies, by brace counting."""
    depth = best = 0
    stack: list[int] = []
    brace = 0
    for m in re.finditer(r"\b(?:for|while)\s*\(|[{}]", text):
        tok = m.group(0)
        if tok == "{":
            brace += 1
        elif tok == "}":
            brace -= 1
            while stack and stack[-1] >= brace:
                stack.pop()
                depth -= 1
        else:
            stack.append(brace)
            depth += 1
            best = max(best, depth)
    return best


def scan(text: str) -> Scan:
    return Scan(text)
