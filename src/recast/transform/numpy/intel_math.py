"""Intel's libimf, for translations held to an ``ifx``-built reference.

``ifx`` links its transcendentals from libimf, and libimf's ``exp``,
``log`` and ``pow`` are not glibc's: they differ by an ULP on some
arguments, and an ULP is the difference between a bit-exact gate passing
and failing. So a translation under the ``ifx`` profile calls the same
library, through this module, rather than ``math``.

Migrated from the pipeline's ``intel_math.py`` with one change. The pipeline
loads the library at import, so a machine without one cannot import the
translated module at all. Here the library is loaded on the first call,
with the same refusal: the emitted module can be imported, inspected and
have its signatures read anywhere, and fails -- naming the fix -- only when
a number is asked for. Which library is a site fact and comes from the
environment: ``RECAST_LIBIMF`` (or the pipeline's ``CTP_LIBIMF``) names the
file; failing both, ``libimf.so`` is asked of the dynamic linker, which finds
it wherever an Intel oneAPI environment has been sourced.
"""

from __future__ import annotations

import ctypes
import os
from collections.abc import Callable
from typing import Any

import numpy as np

__all__ = ["library"]

_UNARY = (
    "exp",
    "log",
    "log10",
    "sqrt",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "erf",
    "erfc",
    "fabs",
)
_BINARY = ("pow", "atan2")

_library: ctypes.CDLL | None = None


def library() -> ctypes.CDLL:
    """The loaded libimf, found on first use."""
    global _library
    if _library is not None:
        return _library
    candidates = [
        os.environ.get("RECAST_LIBIMF"),
        os.environ.get("CTP_LIBIMF"),
        "libimf.so",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        if os.path.sep in candidate and not os.path.exists(candidate):
            continue
        try:
            _library = ctypes.CDLL(candidate)
        except OSError:
            continue
        break
    if _library is None:
        raise RuntimeError(
            "Cannot load Intel libimf, which the ifx profile's transcendentals "
            f"come from. Searched: {[c for c in candidates if c]}. Point RECAST_LIBIMF "
            "at your libimf.so, or source an Intel oneAPI environment; on a "
            "machine without Intel libraries, translate with profile 'gfortran'."
        )
    for name in _UNARY:
        function = getattr(_library, name, None)
        if function is not None:
            function.restype = ctypes.c_double
            function.argtypes = [ctypes.c_double]
    for name in (*_BINARY, "tgamma"):
        function = getattr(_library, name, None)
        if function is not None:
            function.restype = ctypes.c_double
            function.argtypes = [ctypes.c_double] * (1 if name == "tgamma" else 2)
    return _library


def _unary(name: str) -> Callable[[Any], float]:
    def call(x: Any) -> float:
        return float(getattr(library(), name)(float(x)))

    call.__name__ = name
    return call


def _binary(name: str) -> Callable[[Any, Any], float]:
    def call(x: Any, y: Any) -> float:
        return float(getattr(library(), name)(float(x), float(y)))

    call.__name__ = name
    return call


exp = _unary("exp")
log = _unary("log")
log10 = _unary("log10")
sqrt = _unary("sqrt")
sin = _unary("sin")
cos = _unary("cos")
tan = _unary("tan")
asin = _unary("asin")
acos = _unary("acos")
atan = _unary("atan")
sinh = _unary("sinh")
cosh = _unary("cosh")
tanh = _unary("tanh")
erf = _unary("erf")
erfc = _unary("erfc")
fabs = _unary("fabs")
gamma = _unary("tgamma")
pow = _binary("pow")  # the pipeline's name, which emitted code uses
atan2 = _binary("atan2")


def _vunary(scalar: Callable[[Any], float]) -> Callable[[Any], Any]:
    """libimf element by element: the loop is the point, as with ``_f_vexp``."""

    def call(values: Any) -> Any:
        array = np.asarray(values, dtype=np.float64)
        out = np.empty_like(array)
        flat_in, flat_out = array.ravel(), out.ravel()
        for i in range(flat_in.size):
            flat_out[i] = scalar(float(flat_in[i]))
        return out.reshape(array.shape)

    return call


def _vbinary(scalar: Callable[[Any, Any], float]) -> Callable[[Any, Any], Any]:
    def call(left: Any, right: Any) -> Any:
        a, b = np.broadcast_arrays(
            np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        )
        out = np.empty_like(a)
        flat_a, flat_b, flat_out = a.ravel(), b.ravel(), out.ravel()
        for i in range(flat_a.size):
            flat_out[i] = scalar(float(flat_a[i]), float(flat_b[i]))
        return out.reshape(a.shape)

    return call


vexp = _vunary(exp)
vlog = _vunary(log)
vlog10 = _vunary(log10)
vsqrt = _vunary(sqrt)
vsin = _vunary(sin)
vcos = _vunary(cos)
vtanh = _vunary(tanh)
verf = _vunary(erf)
verfc = _vunary(erfc)
vpow = _vbinary(pow)
