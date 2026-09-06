"""What MINPACK's inputs look like, where a uniform draw is not one it takes.

The bit-exact gate draws every argument uniformly and sizes every free
extent the same. Two of MINPACK's routines take a packed workspace whose
length is a function of the order -- an extent no draw lands on and no
engine rule can read off the source without lying about it -- so this
profile says how the length follows the order. A shaped draw is an
assertion that the reference takes it and is judged as one: never redrawn.
"""

from __future__ import annotations

import numpy as np


def prepare(unit, subprogram, inputs, rng):
    """One trial's draw by argument name, shaped into the routine's domain."""
    if subprogram == "dogleg":
        # ``r`` holds the upper triangle of an n-by-n matrix by columns:
        # lr = n(n+1)/2.
        n = int(inputs["n"])
        lr = n * (n + 1) // 2
        inputs["lr"] = np.int32(lr)
        inputs["r"] = np.asfortranarray(rng.uniform(-1.0, 1.0, size=lr))
        return inputs
    if subprogram == "r1updt":
        # ``s`` holds the lower trapezoid of an m-by-n matrix by columns,
        # n <= m: ls = n(2m - n + 1)/2.
        m = int(inputs["m"])
        n = min(int(inputs["n"]), m)
        ls = n * (2 * m - n + 1) // 2
        inputs["n"] = np.int32(n)
        inputs["ls"] = np.int32(ls)
        inputs["s"] = np.asfortranarray(rng.uniform(-1.0, 1.0, size=ls))
        inputs["v"] = np.asfortranarray(rng.uniform(-1.0, 1.0, size=n))
        return inputs
    return None
