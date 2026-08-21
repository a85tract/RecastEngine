"""The JAX backend for the ``port`` recipe.

``backend`` is the emitter, migrated byte-faithfully from the collection and
held there by ``tools/jax_diff.py``. ``runtime`` is the ``_f_*`` shim library
the emitted module imports, and it is written into the Candidate beside the
module rather than imported from here, so a ported kernel runs on a GPU node
that has never heard of this engine.
"""
