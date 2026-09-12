"""The operator's statement about this tree's domain: the tridiagonal solve
takes a diagonally dominant band, which no uniform draw is. The band is
read the way Python reads it -- ``lhs[-1]`` is its last row, the
subdiagonal at +1 -- the way CLUBB's own profile reads the last altitude
of a sorted grid as ``xlist[-1]``."""

from __future__ import annotations

from typing import Any

import numpy as np


def prepare(unit: str, name: str, inputs: dict[str, Any], rng: Any) -> dict[str, Any] | None:
    if unit == "fortran:fill_window" and name in ("fill_window_column", "first_hole"):
        # A window walked in the grid's direction: the step is 1 or -1, and
        # the edges are ordered the way the step walks them -- half the
        # draws descend, the case no recorded CLUBB run reaches.
        nzm = int(np.asarray(inputs["field"]).shape[0])
        direction = int(rng.choice([-1, 1]))
        lo, hi = sorted(int(v) for v in rng.integers(1, nzm + 1, size=2))
        inputs["grid_dir_indx"] = direction
        inputs["k_start"], inputs["k_end"] = (lo, hi) if direction > 0 else (hi, lo)
        return inputs
    if unit == "fortran:tridiag_lu_solver" and name == "tridiag_lu_solve_single_rhs_lhs":
        lhs = np.asarray(inputs["lhs"])
        lhs[1] = np.abs(lhs[0]) + np.abs(lhs[-1]) + 1.0  # the diagonal, row 0 of (-1:1)
        inputs["lhs"] = np.asfortranarray(lhs)
        return inputs
    return None
