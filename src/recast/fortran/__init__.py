"""The reference Frontend implementation: Fortran.

Ships in-tree, but it is a plugin like any other -- it registers through the
``recast.frontends`` entry point and the engine reaches it only through the
``Frontend`` contract. A domain extension layers its own conventions on top of this
one rather than replacing it.

Importing this package is free. fparser2 is an optional dependency
(``recast-engine[fortran]``) and is not touched until a ``discover`` or
``analyze`` call actually needs a parse tree.
"""

from __future__ import annotations

from recast.fortran.frontend import FortranFrontend, UnparsableSource, factory

__all__ = ["FortranFrontend", "UnparsableSource", "factory"]
