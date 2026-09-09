"""The one place that touches fparser.

fparser2 is an optional dependency (``recast-engine[fortran]``). Nothing in
``recast.fortran.frontend`` imports this module at import time, so the plugin
still *registers* on an installation without it -- you find out at ``analyze``
time, with a message that names the extra, rather than at ``recast doctor``
time with a broken entry point.

Parsing a large Fortran module is the expensive part of this frontend, and the
``Frontend`` contract asks implementations to cache expensive analysis keyed on
source content. That cache lives here so every stage of a run agrees on one
parse tree per source revision.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any

from fparser.common.readfortran import FortranFileReader, FortranStringReader

# f03 is the alias every module in this package uses for the node classes;
# the lowercase name is worth more at the ~200 call sites than the rule is.
from fparser.two import Fortran2003 as f03  # noqa: N813
from fparser.two import Fortran2008 as f08  # noqa: N813
from fparser.two.parser import ParserFactory

__all__ = ["STD", "digest", "f03", "f08", "parse", "parser", "walk"]

STD = "f2008"
"""Fortran standard the parser is built for. Recorded in ``Facts.provenance``."""

_parsers: dict[str, Any] = {}
_trees: dict[tuple[str, str], Any] = {}

# ``implicit none (type, external)`` is the Fortran 2018 spelling of the
# ordinary ``implicit none``: the parenthesised list only says *which* of the
# two implicit rules the statement turns off, and ``(type)`` -- present here
# whenever the statement appears at all -- is the one that matters, the ban on
# implicit typing. fparser2 0.2.5 tops out at Fortran 2008 and rejects the
# spec-list form outright (fftpack's ``fftpack_kind`` module opens with it),
# taking the whole file down and with it the ``rk`` kind every submodule
# host-associates from it. Normalising the statement back to a bare
# ``implicit none`` before the reader sees it costs nothing the analysis reads:
# the spec-list carries no type information.
_IMPLICIT_NONE_SPEC = re.compile(r"(?im)^([ \t]*implicit[ \t]+none)[ \t]*\([^)]*\)[ \t]*$")


def walk(node: Any, types: Any = object) -> list[Any]:
    """Every node of the given types under ``node``, in source order.

    A stack rather than fparser's recursion: a physics kernel nests forty deep in
    places, and the recursive walk raises RecursionError on it before any
    rule gets a chance to refuse. Accepts a node, or a list or tuple of them,
    as fparser's does -- some nodes hold their children in bare containers.
    """
    if isinstance(types, type):
        types = (types,)
    found: list[Any] = []
    stack: list[Any] = [node]
    while stack:
        item = stack.pop()
        if isinstance(item, (list, tuple)):
            stack.extend(reversed(item))
            continue
        if item is None:
            continue
        if isinstance(item, types):
            found.append(item)
        children = getattr(item, "children", None)
        if children and isinstance(children, (list, tuple)):
            stack.extend(reversed(children))
    return found


def digest(path: Path) -> str:
    """SHA-256 of a source file. The cache key, and the provenance record."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parser(std: str = STD) -> Any:
    """The parser for a standard, created once.

    Creating one has a side effect beyond returning it: fparser sets up the
    match patterns its ``Fortran2003`` classes need, so constructing a node
    from source text fails until this has run at least once. Anything building
    nodes directly -- a rule under test, a fixture -- has to call it, which is
    why it is not hidden inside ``parse``.
    """
    existing = _parsers.get(std)
    if existing is None:
        existing = _parsers[std] = ParserFactory().create(std=std)
    return existing


def parse(path: Path, *, std: str = STD) -> Any:
    """Parse a Fortran source file into an fparser2 AST.

    Cached on ``(content digest, std)``: re-analyzing the same revision under
    the same standard reuses the tree, and editing the file invalidates it
    without anyone having to remember to.
    """
    key = (digest(path), std)
    tree = _trees.get(key)
    if tree is None:
        text = path.read_text(errors="replace")
        normalized = _IMPLICIT_NONE_SPEC.sub(r"\1", text)
        if normalized != text:
            # A rewritten source has to be read as a string; the include path
            # the file reader derives from the file's own directory is handed
            # over explicitly so a source with an ``include`` still resolves it.
            reader: Any = FortranStringReader(normalized, include_dirs=[str(path.parent)])
        else:
            reader = FortranFileReader(str(path))
        # fparser's reader answers a malformed line -- ``end subroutine`` naming
        # the wrong procedure, say -- by logging it and calling ``sys.exit(1)``,
        # which took the whole discovery of a tree down with the one file
        # (E3SM's ``external_models/emi/.../clm_varctl.F90``). Told not to
        # exit, it logs the same message, ignores the line, and the parse
        # goes on to raise or succeed on its own terms.
        reader.exit_on_error = False
        tree = _trees[key] = parser(std)(reader)
    return tree
