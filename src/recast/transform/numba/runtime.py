"""The ``_f_*`` runtime an njit module needs, compiled rather than interpreted.

Relayed from the ``HEADER`` string constant in ``pipeline/numbaize.py``,
behaviour intact. It lives here as real code for the same reason the NumPy
runtime does -- a string constant is not linted, not type-checked and not
testable, and these are the definitions of ``sign`` and ``mod``, where Python's
answer differs from Fortran's.

Counterpart of ``recast.transform.numpy.runtime``, and deliberately not
identical to it:

  - **no strict libm.** The NumPy runtime routes ``exp``/``log``/``pow``
    through ``ctypes`` into the system libm so the bit-exact gates can be met.
    Nothing of the sort survives ``@njit``: a kernel calls what numba lowers,
    and numba lowers to its own implementations. This is the throughput
    backend, and its honest ceiling is a tolerance gate.
  - ``_f_vdot`` accumulates in order, in a Python loop, precisely to *avoid*
    numba's ``np.dot`` -- which dispatches to BLAS and rounds differently.
    Fortran's ``DOT_PRODUCT`` accumulates in order, and so does the NumPy
    runtime's shim, so this is the spelling that keeps the two agreeing.
  - ``fastmath`` is off on every decorator here and in every emitted kernel.
    It is not a tuning knob: it licenses the compiler to reassociate
    floating-point arithmetic, which changes the numbers a gate is checking.

Its fourteen anchors are a strict subset of the NumPy runtime's forty-four,
which is the property worth keeping -- two backends held to one set of anchors
rather than drifting into separate notions of correct. The thirty it does not
implement are the intrinsics a numeric kernel does not reach, plus those numba
cannot compile.

Nothing in the engine imports this module: the emitter reads its text off disk,
so translating to Numba never requires numba to be installed. That is the same
rule the JAX runtime follows and for the same reason.
"""

import math
from typing import Any

import numpy as np
from numba import njit, vectorize

# A star-import from the generated module must see the underscore names.
__all__ = [
    "_f_int_div",
    "_f_max",
    "_f_min",
    "_f_mod",
    "_f_sign",
    "_f_trim",
    "_f_vdot",
    "_f_vexp",
    "_f_vlog",
    "_f_vlog10",
    "_f_vmax",
    "_f_vmin",
    "_f_vpow",
    "_fstr_eq",
    "math",
    "njit",
    "np",
    "vectorize",
]


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_min(a: Any, b: Any) -> Any:
    """gfortran MIN NaN semantics (left operand's NaN absorbed)."""
    return b if (a != a) else (a if a < b else b)


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_max(a: Any, b: Any) -> Any:
    return b if (a != a) else (a if a > b else b)


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vmin(a: Any, b: Any) -> Any:
    return np.where(np.isnan(a), b, np.where(a < b, a, b))


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vmax(a: Any, b: Any) -> Any:
    return np.where(np.isnan(a), b, np.where(a > b, a, b))


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_sign(a: Any, b: Any) -> Any:
    return abs(a) if b >= 0 else -abs(a)


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_mod(a: Any, p: Any) -> Any:
    """Fortran MOD truncates toward zero where Python's ``%`` floors."""
    return a - int(a / p) * p


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_int_div(a: Any, b: Any) -> Any:
    """Fortran integer division truncates where Python's ``//`` floors."""
    return int(a / b)


@njit(cache=True, fastmath=False, error_model="numpy")
def _fstr_eq(a: Any, b: Any) -> Any:
    """Fortran compares CHARACTER blank-padded to the longer operand."""
    return a.rstrip(" ") == b.rstrip(" ")


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_trim(s: Any) -> Any:
    return s.rstrip(" ")


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vexp(x: Any) -> Any:
    return np.exp(x)


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vlog(x: Any) -> Any:
    return np.log(x)


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vlog10(x: Any) -> Any:
    return np.log10(x)


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vpow(a: Any, b: Any) -> Any:
    return a ** b


@njit(cache=True, fastmath=False, error_model="numpy")
def _f_vdot(a: Any, b: Any) -> Any:
    # Fortran DOT_PRODUCT accumulates in order (matches the strict-libm numpy
    # shim); numba's np.dot dispatches to BLAS and rounds differently.
    af, bf = np.ravel(a), np.ravel(b)
    s = 0.0
    for i in range(af.size):
        s += af[i] * bf[i]
    return s
