"""Frontend: turn a source tree into Units and Facts."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path

from recast.model import Facts, Unit


class Frontend(ABC):
    """Reads source. Never writes it.

    One Frontend per source language or per source model. ``recast-fortran``
    ships in-tree as the reference implementation; a CESM extension layers
    CESM conventions (CCPP metadata, CIME cases) on top of it.
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
