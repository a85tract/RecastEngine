"""The operator's statement about this tree's domain: ``filter`` holds
``num`` distinct patch indices in ``1..np``, and ``ncan`` a layer count in
``1..nlev`` per patch. A uniform draw is neither."""

from __future__ import annotations

from typing import Any

import numpy as np


def prepare(unit: str, name: str, inputs: dict[str, Any], rng: Any) -> dict[str, Any] | None:
    if unit == "fortran:physics_mod" and name in ("warm_flat", "fill_flat"):
        np_ = int(np.asarray(inputs["inst__tleaf"]).shape[0])
        num = int(inputs["num"])
        inputs["filter"] = np.asarray(rng.permutation(np_)[:num] + 1, dtype=np.int32)
        inputs["inst__ncan"] = np.asarray(rng.integers(1, 4, size=np_), dtype=np.int32)
        return inputs
    return None
