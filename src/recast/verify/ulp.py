"""ULP distance: how far apart two float64 values really are.

Migrated from the source pipeline's ``pipeline/highprec_verify.py``. The
tolerance ladder's rungs are defined in these terms -- bit-exact is a ULP
distance of zero, and a claimed ``max_ulp`` in a Verdict's metrics is what
makes a near-miss reviewable instead of a shrug.

Standard library only, deliberately. This is core vocabulary for any
verifier that compares floating point, and the core must work without NumPy
installed; arrays arriving from a NumPy-using verifier still compare fine,
one ``float()`` at a time.
"""

from __future__ import annotations

import math
import struct
from collections.abc import Iterable, Sequence
from typing import Any

__all__ = ["ulp_audit", "ulp_distance"]


def _as_ordered_int(value: float) -> int:
    """Reinterpret a float64's bits as an integer that orders like the float.

    IEEE-754 doubles of one sign compare like their bit patterns; folding the
    negative half over makes the whole line monotone, so a ULP distance is a
    plain integer subtraction.
    """
    (bits,) = struct.unpack("<q", struct.pack("<d", value))
    return int(bits) if bits >= 0 else -(int(bits) & 0x7FFFFFFFFFFFFFFF)


def ulp_distance(a: float, b: float) -> float:
    """How many representable doubles lie between ``a`` and ``b``.

    0 is bit-exact (two NaNs count: both sides refused the same way), 1 is
    the smallest possible disagreement, and NaN against a number is
    ``inf`` -- there is no meaningful distance to something that is not a
    value.
    """
    a, b = float(a), float(b)
    a_nan, b_nan = math.isnan(a), math.isnan(b)
    if a_nan and b_nan:
        return 0
    if a_nan or b_nan:
        return math.inf
    if a == b:  # covers +0.0 vs -0.0, which are 2**63 bit-patterns apart
        return 0
    return abs(_as_ordered_int(a) - _as_ordered_int(b))


def ulp_audit(
    values_a: Iterable[Any],
    values_b: Iterable[Any],
    *,
    dominant: Sequence[bool] | None = None,
) -> dict[str, Any]:
    """Compare two equal-length sequences of float64, ULP by ULP.

    The returned metrics are the reviewable form of a differential claim:
    ``{"total_points": ..., "bit_exact": ..., "max_ulp": ...,
    "nan_mismatch": ..., "ulp_histogram": {...}}``. A verdict built on these
    can be argued with; a verdict that says "close enough" cannot.

    ``dominant`` marks the elements a gate should hold to a ULP bound, and
    adds ``dominant_points`` and ``max_ulp_dominant`` to the result. Which
    elements those are is not decidable here: it depends on the array's shape,
    and this module is deliberately shape-agnostic and NumPy-free. The caller
    that knows the shape decides; see ``recast.verify.tolerance``.
    """
    a = [float(x) for x in values_a]
    b = [float(x) for x in values_b]
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} vs {len(b)}")
    if dominant is not None and len(dominant) != len(a):
        raise ValueError(f"dominance mask is {len(dominant)} long, values are {len(a)}")

    distances = [ulp_distance(x, y) for x, y in zip(a, b, strict=True)]
    histogram: dict[int | str, int] = {}
    for d in distances:
        key: int | str = "inf" if math.isinf(d) else int(d)
        histogram[key] = histogram.get(key, 0) + 1

    finite = [d for d in distances if not math.isinf(d)]
    audit: dict[str, Any] = {
        "total_points": len(distances),
        "bit_exact": sum(1 for d in distances if d == 0),
        "max_ulp": int(max(finite)) if finite else 0,
        "nan_mismatch": sum(1 for d in distances if math.isinf(d)),
        "ulp_histogram": {
            k: histogram[k] for k in sorted(histogram, key=lambda x: (isinstance(x, str), x))
        },
    }
    if dominant is not None:
        marked = [d for d, keep in zip(distances, dominant, strict=True) if keep]
        finite_marked = [d for d in marked if not math.isinf(d)]
        audit["dominant_points"] = len(marked)
        audit["max_ulp_dominant"] = int(max(finite_marked)) if finite_marked else 0
    return audit
