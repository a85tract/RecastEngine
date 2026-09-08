"""The operator's statement about this tree's domain: the tridiagonal solve
takes a diagonally dominant band, which no uniform draw is. The band is
read the way Python reads it -- ``lhs[-1]`` is its last row, the
subdiagonal at +1 -- the way CLUBB's own profile reads the last altitude
of a sorted grid as ``xlist[-1]``."""

from __future__ import annotations

from typing import Any

import numpy as np


def prepare(unit: str, name: str, inputs: dict[str, Any], rng: Any) -> dict[str, Any] | None:
    if unit == "fortran:tridiag_lu_solver" and name == "tridiag_lu_solve_single_rhs_lhs":
        lhs = np.asarray(inputs["lhs"])
        lhs[1] = np.abs(lhs[0]) + np.abs(lhs[-1]) + 1.0  # the diagonal, row 0 of (-1:1)
        inputs["lhs"] = np.asfortranarray(lhs)
        return inputs
    return None
