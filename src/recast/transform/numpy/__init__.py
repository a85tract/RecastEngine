"""The reference translation backend: Fortran to NumPy.

Behind the ``translate`` extra, because it is the one in-tree package that
needs NumPy -- both to run and, more to the point, because the code it emits
imports it.

Importing this package is free; nothing under it is loaded until a Transform
actually runs.
"""
