"""``differential.tolerance``: the gate for a backend that cannot be bit-exact.

Migrated from the gate in ``CESM-Agent-Produced-Scripts`` ``13_jax_backend/
diffutil.py``, whose rationale rests on ``ulp_analysis_beljaars.py`` in the
same collection. Both are agent-produced and were never held to a bit gate --
unlike the translator this engine's other verifiers came from -- so what is
carried over here is the *argument*, restated, rather than an answer taken on
trust.

The argument. Some backends cannot agree bit for bit and never will: XLA lowers
transcendentals to its own implementations rather than to libm, and fuses
multiply-add. That is a property of the target, not a defect in the
translation, so a gate that demands bit-exactness of a JAX port rejects
correct work forever. But a plain relative tolerance is too blunt in the other
direction, because it hides the failure that matters. Hence two tiers:

    dominant elements   ULP distance <= ``ulp_gate``    (default 32)
    every element       relative diff <= ``rel_gate``   (default 1e-12)

An element is *dominant* when its reference value is within ``dominant_at`` of
the largest in its row (default 1e-3). The reason for the split is
conditioning: ``exp`` and ``pow`` amplify a few ULP of input difference
linearly in the magnitude of the argument, and they do it on exactly the
elements that are exponentially negligible in the answer. Holding those to a
ULP bound fails a correct port on values nobody reads; holding *only* the
whole array to a relative tolerance lets a real defect in a dominant value
hide inside a tolerance sized for the negligible ones. So the tier that
matters is strict -- 32 ULP is around 7e-15 relative, tighter than the 1e-12
that catches everything else.

Note the ladder this can climb. If every element, dominant or not, lands
inside the ULP bound, the claim is stronger than a tolerance and is awarded as
``ULP_BOUNDED``. It is only when the negligible tail drifts past that bound,
and the relative tolerance is what excuses it, that the verdict is
``TOLERANCED``. A gate that reported one number for both would throw away the
distinction it exists to make.
"""

from __future__ import annotations

from typing import Any

from recast.model import Candidate, Confidence, Verdict
from recast.verify.bitexact import BitexactVerifier

__all__ = ["ToleranceVerifier", "factory"]

REL_GATE = 1e-12
ULP_GATE = 32
DOMINANT_AT = 1e-3


class ToleranceVerifier(BitexactVerifier):
    """Two tiers: a ULP bound where it matters, a relative bound everywhere."""

    name = "differential.tolerance"
    provides = Confidence.ULP_BOUNDED
    """The strongest this can ever award, not what it usually does. A proven
    bound on every element's ULP distance is what ``ULP_BOUNDED`` means, and
    this gate reaches it whenever the negligible tail happens to stay inside
    the bound too."""

    dominant_at: float | None = DOMINANT_AT

    def _award(
        self,
        candidate: Candidate,
        totals: dict[str, Any],
        per_subprogram: dict[str, Any],
        metrics: dict[str, Any],
        config: dict[str, Any],
    ) -> Verdict:
        rel_gate = float(config.get("rel_gate", REL_GATE))
        ulp_gate = int(config.get("ulp_gate", ULP_GATE))
        worst_rel = metrics["max_rel"]
        worst_ulp = totals["max_ulp"]
        worst_ulp_dominant = totals.get("max_ulp_dominant", worst_ulp)
        metrics = {**metrics, "rel_gate": rel_gate, "ulp_gate": ulp_gate}

        if totals["bit_exact"] == totals["points"]:
            return self._verdict(
                candidate,
                Confidence.BIT_EXACT,
                metrics,
                f"{totals['points']} points across {len(per_subprogram)} "
                f"subprogram(s), all bit-exact -- this backend did not need the tiers",
            )
        if worst_ulp_dominant > ulp_gate:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"a dominant element is {worst_ulp_dominant} ULP out (gate {ulp_gate}); "
                "that is where a translation defect shows, not where conditioning does",
            )
        if worst_rel > rel_gate:
            return self._verdict(
                candidate,
                Confidence.FAILED,
                metrics,
                f"max_rel={worst_rel:.3e} exceeds rel_gate={rel_gate:g}, with dominant "
                f"elements inside {ulp_gate} ULP -- the disagreement is in the tail, "
                "and it is still too large to excuse",
            )
        if worst_ulp <= ulp_gate:
            return self._verdict(
                candidate,
                Confidence.ULP_BOUNDED,
                metrics,
                f"every one of {totals['points']} points within {worst_ulp} ULP (gate {ulp_gate})",
            )
        return self._verdict(
            candidate,
            Confidence.TOLERANCED,
            metrics,
            f"{totals.get('dominant_points', 0)} dominant point(s) within "
            f"{worst_ulp_dominant} ULP (gate {ulp_gate}); the tail reaches {worst_ulp} ULP "
            f"but max_rel={worst_rel:.3e} is inside rel_gate={rel_gate:g}",
        )


def factory(**_config: Any) -> ToleranceVerifier:
    return ToleranceVerifier()
