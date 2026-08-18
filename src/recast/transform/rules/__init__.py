"""Decisions that are neither a fact about the source nor a spelling.

A Fortran array starts at 1 and a Python one at 0, so every subscript shifts.
That is not something the frontend can report -- the source says nothing about
it -- and not something only one backend needs, because NumPy, JAX, CuPy and C
all index from zero with an exclusive upper bound. It belongs between them.

Rules produce plans, not text. A plan names what has to happen to each part of
a construct and refers to the source nodes without rendering them, so a backend
renders once and does not re-derive the decision. The test of whether something
belongs here rather than in a backend is whether a second backend would want to
copy it.
"""

from __future__ import annotations

from recast.errors import RecastError


class NoRule(RecastError):
    """This construct has no mechanical rewrite.

    Distinct from ``semantics.Unanalyzable``, which means the source's meaning
    could not be settled. Here the meaning is clear and no rule covers it --
    a negative-stride slice off a non-unit lower bound is perfectly good
    Fortran with no correct one-line equivalent. Both end the same way, as a
    site in ``Candidate.deferred`` for the agent layer, but a report that says
    which of the two happened is the difference between "teach the frontend"
    and "write the rule".
    """
