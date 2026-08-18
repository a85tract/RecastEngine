"""Transforms shipped in-tree, and the rules they share.

``recast.transform.rules`` will hold decisions that are target-independent but
are not facts about the source -- a Fortran array starts at 1 and a Python one
at 0, so every index shifts. ``recast.transform.numpy`` is the reference
backend and the only part that knows a target language exists.

Nothing here is imported by the engine core. A Transform arrives through the
``recast.transforms`` entry point like any other plugin. See
``docs/splitting-the-translator.md``.
"""
