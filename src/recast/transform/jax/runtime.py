"""The ``_f_*`` runtime a jaxized module needs, in JAX rather than NumPy.

Migrated from ``13_jax_backend/jax_shim.py``, behaviour intact and layout
reformatted to this repository's style, so a diff against the origin is not
what holds it: that is done by the anchor check in
``tests/test_jax_backend.py`` and, for the numbers, by the ULP gate every
ported kernel has to pass. Counterpart of the
shim library ``recast.transform.numpy.runtime`` inlines into every translated
module, and deliberately not identical to it:

  - **no strict libm.** ``jnp.exp``/``log``/``power`` lower to XLA's own
    implementations, which differ from glibc by ULPs. That is why this backend
    gates at the ULP tier and never at bit-exactness.
  - ``_f_min``/``_f_max`` reproduce the gfortran SSE ``minsd``/``maxsd`` NaN
    order exactly -- the left operand's NaN absorbed, the right's propagated --
    which matches the NumPy shim bit for bit on every non-transcendental path.

Its twenty anchors are a strict subset of the NumPy runtime's forty-four, and
that is the property worth keeping: two backends held to one set of anchors
rather than drifting into separate notions of correct. The twenty-four it does
not implement are string, bit and pointer intrinsics a numeric kernel does not
reach.

Importing this enables float64, which JAX does not do by default and which has
to happen before any array is created. That is also why nothing in the engine
imports this module: ``backend.emit_runtime`` reads its text off disk instead,
so emitting JAX code never requires JAX to be installed.
"""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
from jax import lax  # noqa: E402

# A star-import from the generated module must see the underscore names.
__all__ = [
    "_f_adjustl",
    "_f_dim",
    "_f_epsilon",
    "_f_huge",
    "_f_int_div",
    "_f_len_trim",
    "_f_max",
    "_f_min",
    "_f_mod",
    "_f_modulo",
    "_f_nint",
    "_f_sign",
    "_f_tiny",
    "_f_trim",
    "_f_vceil",
    "_f_vdot",
    "_f_vexp",
    "_f_vfloor",
    "_f_vlog",
    "_f_vlog10",
    "_f_vmax",
    "_f_vmin",
    "_f_vpow",
    "_fstr_eq",
    "jax",
    "jnp",
    "lax",
]


def _f_min(*xs):
    """gfortran MIN, SSE minsd fold order: per step the FIRST operand's
    NaN is absorbed, the second's propagates (x != x is the NaN test —
    valid for int operands too, unlike jnp.isnan)."""
    r = xs[0]
    for b in xs[1:]:
        r = jnp.where(r != r, b, jnp.where(r < b, r, b))
    return r


def _f_max(*xs):
    r = xs[0]
    for b in xs[1:]:
        r = jnp.where(r != r, b, jnp.where(r > b, r, b))
    return r


def _f_vmin(a, b):
    return jnp.where(jnp.isnan(a), b, jnp.where(a < b, a, b))


def _f_vmax(a, b):
    return jnp.where(jnp.isnan(a), b, jnp.where(a > b, a, b))


def _f_sign(a, b):
    """Fortran SIGN(a,b): real b -> copysign (-0.0 aware); integer b ->
    value compare with b == 0 giving +|a|. dtype dispatch is static at
    trace time."""
    b_ = jnp.asarray(b)
    if jnp.issubdtype(b_.dtype, jnp.floating):
        return jnp.copysign(jnp.abs(a), b_)
    return jnp.where(b_ >= 0, jnp.abs(a), -jnp.abs(a))


def _f_dim(x, y):
    """Fortran DIM(x,y) = max(x-y, 0)."""
    return jnp.maximum(jnp.asarray(x) - y, 0)


def _f_mod(a, p):
    """Fortran MOD: truncated, sign follows a."""
    a_, p_ = jnp.asarray(a), jnp.asarray(p)
    q = jnp.trunc(a_ / p_).astype(jnp.result_type(a_, p_))
    return a_ - q * p_


def _f_modulo(a, p):
    """Fortran MODULO: floored, sign follows p."""
    return jnp.mod(jnp.asarray(a), p)


def _f_int_div(a, b):
    """Fortran integer division truncates toward zero."""
    a_, b_ = jnp.asarray(a), jnp.asarray(b)
    return jnp.trunc(a_ / b_).astype(jnp.result_type(a_, b_))


def _f_nint(x):
    """Fortran NINT: round half away from zero."""
    x_ = jnp.asarray(x)
    r = jnp.where(x_ >= 0, jnp.floor(x_ + 0.5), jnp.ceil(x_ - 0.5))
    return r.astype(jnp.int32)


def _f_vexp(x):
    return jnp.exp(x)


def _f_vlog(x):
    return jnp.log(x)


def _f_vlog10(x):
    return jnp.log10(x)


def _f_vpow(a, b):
    return jnp.asarray(a) ** b


def _f_vdot(a, b):
    """Fortran DOT_PRODUCT accumulates in order; sequential fori_loop
    keeps the fold order (XLA may still contract the FMA)."""
    af, bf = jnp.ravel(a), jnp.ravel(b)

    def body(i, s):
        return s + af[i] * bf[i]

    return lax.fori_loop(0, af.shape[0], body, jnp.float64(0.0))


def _f_vceil(x):
    return jnp.ceil(x).astype(jnp.int32)


def _f_vfloor(x):
    return jnp.floor(x).astype(jnp.int32)


def _f_huge(x):
    d = jnp.asarray(x).dtype
    if jnp.issubdtype(d, jnp.floating):
        return jnp.finfo(d).max
    return jnp.iinfo(d).max


def _f_tiny(x):
    return jnp.finfo(jnp.asarray(x).dtype).tiny


def _f_epsilon(x):
    return jnp.finfo(jnp.asarray(x).dtype).eps


def _fstr_eq(a: str, b: str) -> bool:
    """Fortran character equality: pad the shorter operand with blanks.
    Characters are static under tracing -- plain Python strings."""
    return a.rstrip(" ") == b.rstrip(" ")


def _f_trim(s: str) -> str:
    """Fortran TRIM: strip trailing blanks only."""
    return s.rstrip(" ")


def _f_len_trim(s: str) -> int:
    return len(s.rstrip(" "))


def _f_adjustl(s: str) -> str:
    return s.lstrip(" ").ljust(len(s))
