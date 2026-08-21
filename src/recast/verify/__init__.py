"""Verification primitives shared by every recipe.

P2 lands the four already proven in CESM-language-translator:
high-precision expression equivalence (mpmath), ULP-distance audit,
read/write-set cross-check, and SymPy notarization.

``tolerance`` is the fifth and does not come from there. It is P4's, and it
exists because a backend can be correct and still never be bit-exact -- XLA
lowers transcendentals to its own implementations and fuses multiply-add, so
the strongest honest claim about a JAX port is a bound, not equality. Its
source was never held to a bit gate either, which is why what carried over is
the argument rather than the answer.
"""
