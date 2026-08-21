"""Frontend: turn a source tree into Units and Facts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from recast.model import Facts, Unit


class Frontend(ABC):
    """Reads source. Never writes it.

    One Frontend per source language or per source model. ``recast-fortran``
    ships in-tree as the reference implementation; a domain extension layers
    its conventions (CCPP metadata, CIME cases) on top of it.

    **A recipe may declare several, and they do not chain.** Each reads the
    tree independently and their Unit sets are unioned, which is how a project
    written in more than one language is walked in one run: the Unit remembers
    which frontend found it and that one analyzes it. No Frontend sees
    another's Facts -- that is why ``analyze`` takes none -- so layering one
    analysis on another happens *inside* a Frontend, by wrapping or
    subclassing, not between two of them.

    Two frontends claiming the same ``uid`` is refused rather than resolved.
    The Unit would carry one of their Facts with no record of whose, and the
    run would be reproducible only by accident of declaration order.

    It also has to skip ``recast.WORKSPACE_DIRNAME``. That directory is the
    engine's own output -- workspaces, oracle builds, evidence -- and a
    frontend that reads a previous run's generated code back has turned output
    into input.
    """

    name: str
    languages: tuple[str, ...] = ()

    @abstractmethod
    def discover(self, root: Path) -> Iterable[Unit]:
        """Enumerate the Units in a source tree.

        Must be deterministic and side-effect free. Ordering is not
        significant; the engine topologically sorts using ``Facts.callgraph``.
        """

    @abstractmethod
    def analyze(self, unit: Unit, root: Path) -> Facts:
        """Extract everything a Transform might need, without transforming.

        Expensive analyses (whole-program call graphs, code property graphs)
        should be cached by the implementation keyed on source content hash.
        """

    def preprocess(self, unit: Unit, root: Path) -> Unit:
        """Optional hook: resolve ``#ifdef``/``#include`` before analysis.

        Default is identity. Implementations that need it must record the flags
        they used in ``Facts.provenance`` -- a translation is only reproducible
        if the preprocessor invocation is.
        """
        return unit
