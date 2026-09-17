"""The operator's statement about this tree's domain: ``filter`` holds
``numf`` distinct column indices in ``1..ncol``; each column's window
``jtop(ci):jbot(ci)`` lies inside ``lbj:ubj`` and holds at least three
levels, so every diagonal of the pentadiagonal band is laid; and the
band is diagonally dominant, which no uniform draw is -- LAPACK's pivot
search would still run, but the system it solved would say nothing
about a heat-conduction step."""

from __future__ import annotations

from typing import Any

import numpy as np


def prepare(unit: str, name: str, inputs: dict[str, Any], rng: Any) -> dict[str, Any] | None:
    if unit == "fortran:band_solve" and name in ("band_diagonal", "band_diagonal_flat"):
        ncol = int(inputs["ncol"])
        numf = int(inputs["numf"])
        lbj, ubj = int(inputs["lbj"]), int(inputs["ubj"])
        inputs["filter"] = np.asarray(rng.permutation(ncol)[:numf] + 1, dtype=np.int32)
        jtop = np.zeros(ncol, dtype=np.int32)
        jbot = np.zeros(ncol, dtype=np.int32)
        for c in range(ncol):
            jtop[c] = rng.integers(lbj, lbj + 3)
            jbot[c] = rng.integers(jtop[c] + 2, ubj + 1)
        inputs["jtop"], inputs["jbot"] = jtop, jbot
        b = np.array(inputs["b"], dtype=np.float64, order="F")
        b[:, 2, :] = 10.0 + np.abs(b[:, 2, :])  # the diagonal (band 3 of 5) dominates
        inputs["b"] = np.asfortranarray(b)
        return inputs
    return None
