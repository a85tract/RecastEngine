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
from pathlib import Path
from typing import Any

from fparser.common.readfortran import FortranFileReader

# f03 is the alias every module in this package uses for the node classes;
# the lowercase name is worth more at the ~200 call sites than the rule is.
from fparser.two import Fortran2003 as f03  # noqa: N813
from fparser.two import Fortran2008 as f08  # noqa: N813
from fparser.two.parser import ParserFactory
from fparser.two.utils import walk

__all__ = ["STD", "digest", "f03", "f08", "parse", "parser", "walk"]

STD = "f2008"
"""Fortran standard the parser is built for. Recorded in ``Facts.provenance``."""

_parsers: dict[str, Any] = {}
_trees: dict[tuple[str, str], Any] = {}


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
        tree = _trees[key] = parser(std)(FortranFileReader(str(path)))
    return tree
