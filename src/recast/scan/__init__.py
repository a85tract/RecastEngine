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

Two of the four families ship here: ``secret`` and ``composition``. The other
two are the domain extension's, by the maintainer's decision (2026-08-21), and
for two different reasons. ``dynamic`` (sanitizer builds) needs a compiler and
a build, and which compiler a CAM build expects is domain knowledge.
``audit`` (LLM source audit) is an advanced capability that stays out of the
public repository. Both remain declared by the ``audit`` recipe as optional
stages, which is how an out-of-tree plugin attaches: ``recast plan`` reports
them as ``opt`` until the extension is installed, and never as a stub.
"""
