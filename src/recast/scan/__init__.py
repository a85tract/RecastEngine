"""Scanners: the cyber half of CC-Test, in the engine.

In-tree rather than in a distribution of their own, on the argument the
``Scanner`` contract already makes for itself: every one of these runs against
*any* git repository, and a check that needs no domain knowledge is engine
territory, the same way ``recast/fortran/`` is. The line is written at the top
of ``plugins/scanner.py`` -- what knows it is a climate model goes to the domain
extension, what knows which machine it runs on is an ``Executor``, and neither
of those is here.

The shape of each scanner is ``hpc-devsecops``'s (a85tract/CESM-CC-Test, by
Chien-Wei Huang): gitleaks over the sources, SARIF in and Findings out, a
missing tool reported as a missing tool rather than as a clean scan. Nothing of
that script was ported -- it is shell, this is Python -- but the decisions about
what to run, what to read back and what its silence is allowed to mean are
theirs, and ``NOTICE`` says so.

Two of the four families ship today: ``secret`` and ``composition``. ``audit``
(LLM source audit) and ``dynamic`` (sanitizer builds) are declared by the
``audit`` recipe and refused by name at ``recast plan`` until they exist, which
is the honest state: a stub that registers and raises is worse than an absence
the runner can report.
"""
