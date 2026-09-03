"""C/C++ kernel programs as Units: a frontend, and the build and staging helpers
the executable oracle and the probe/benchmark verifiers share.

A *kernel* here is a directory with a program in it -- a benchmark's
``main.cpp`` and its Makefile, its input files beside it -- built and run as a
whole. That is the unit a serial-to-offload translation is judged at, and it
is coarser than the Fortran frontend's subprogram: nothing in this package
parses C. ``scan`` counts what a translation planner looks at first (loops,
their nesting, allocations, a timed region) and says so in ``provenance``.

How a kernel builds is not the engine's to know. A frontend that does know
(a benchmark suite's layout lives in a case repository) writes a *build spec*
into ``Unit.attrs`` -- argument vectors with ``{cc}``-style placeholders --
and ``build`` renders them with the operator's toolchain table at run time.
"""

from __future__ import annotations
